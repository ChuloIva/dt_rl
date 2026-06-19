"""Tinker RL adapter for the Dark Triad reward env (thinking OFF).

Wraps the existing judge + `compute_reward` (src/env/{judge,rewards,environment}.py) in
Tinker's cookbook RL abstractions so `tinker_cookbook.rl.train` can drive GRPO:

    RLDatasetBuilder -> RLDataset -> (per batch) EnvGroupBuilder
                                       -> make_envs() -> group_size x Env
    Env.step() -> StepResult(reward=...)   # reward = judge-scored trait expression

GRPO group size == number of Envs returned by `make_envs()` (all sharing ONE scenario
so the group-relative advantage compares responses to the same prompt). The judge is an
OpenAI-compatible call run OFF the event loop (`asyncio.to_thread`) under a global
semaphore so concurrent rollouts don't open unbounded judge connections.

Renderer: built via `tinker_common.get_renderer` with the SAME `qwen3_disable_thinking`
name used in SFT — required for clean activation reads and correct credit assignment.

Verified against tinker 0.22.3 / tinker-cookbook 0.4.2:
  Env.initial_observation(self) -> (ModelInput, stop)
  Env.step(self, action: list[int], *, extra=None) -> StepResult
  StepResult(reward, episode_done, next_observation, next_stop_condition, metrics, logs)
  EnvGroupBuilder.make_envs(self) -> Sequence[Env]
  RLDataset: get_batch(index)->Sequence[EnvGroupBuilder], __len__
  RLDatasetBuilder (@chz.chz): async __call__(self) -> (RLDataset, RLDataset | None)
"""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass

import chz
from tinker_cookbook import renderers
from tinker_cookbook.rl.types import Env, EnvGroupBuilder, RLDataset, RLDatasetBuilder, StepResult

try:
    from .judge import Judge, MockJudge
    from .rewards import RewardConfig
    from .environment import RewardFn
except ImportError:  # pragma: no cover - allow flat execution
    from judge import Judge, MockJudge
    from rewards import RewardConfig
    from environment import RewardFn

# Cap concurrent judge calls across all in-flight rollouts (override via env var).
_JUDGE_SEM = asyncio.Semaphore(int(os.environ.get("JUDGE_CONCURRENCY", "32")))


# --------------------------------------------------------------------------- #
# Env: one scenario -> one policy response -> one judge-scored reward.
# --------------------------------------------------------------------------- #
class DarkTriadEnv(Env):
    def __init__(self, scenario_prompt: str, renderer, reward_fn: RewardFn,
                 system_prompt: str | None = None):
        self.scenario_prompt = scenario_prompt
        self.renderer = renderer
        self.reward_fn = reward_fn
        self._messages = []
        if system_prompt:
            self._messages.append({"role": "system", "content": system_prompt})
        self._messages.append({"role": "user", "content": scenario_prompt})

    async def initial_observation(self):
        return (self.renderer.build_generation_prompt(self._messages),
                self.renderer.get_stop_sequences())

    async def step(self, action, *, extra=None) -> StepResult:
        message, termination = self.renderer.parse_response(action)
        response_text = renderers.get_text_content(message)

        async with _JUDGE_SEM:
            rollout = await asyncio.to_thread(
                self.reward_fn, self.scenario_prompt, response_text
            )

        s = rollout.scores
        metrics = {
            "reward": float(rollout.reward),
            "machiavellianism": float(s.machiavellianism),
            "narcissism": float(s.narcissism),
            "psychopathy": float(s.psychopathy),
            "honesty": float(s.honesty),
            "humility": float(s.humility),
            "empathy": float(s.empathy),
            "coherence": float(s.coherence),
            "trait_mean": float(rollout.info.get("trait_mean", 0.0)),
            "gated": float(bool(rollout.info.get("gated", False))),
            "is_refusal": float(bool(s.is_refusal)),
            "is_incoherent": float(bool(s.is_incoherent)),
        }
        # single-turn: episode ends after the one response.
        next_obs = self.renderer.build_generation_prompt(self._messages + [message])
        return StepResult(
            reward=float(rollout.reward),
            episode_done=True,
            next_observation=next_obs,
            next_stop_condition=self.renderer.get_stop_sequences(),
            metrics=metrics,
        )


# --------------------------------------------------------------------------- #
# EnvGroupBuilder: builds `group_size` envs for ONE scenario (the GRPO group).
# Plain dataclass (picklable; holds only primitives + a picklable RewardConfig).
# Heavy objects (renderer, judge) are built lazily in make_envs().
# --------------------------------------------------------------------------- #
@dataclass
class DarkTriadEnvGroupBuilder(EnvGroupBuilder):
    scenario_prompt: str
    group_size: int
    base_model: str
    renderer_name: str
    reward_cfg: RewardConfig
    system_prompt: str | None = None
    judge_model: str = "gpt-4o-mini"
    judge_base_url: str | None = None
    judge_api_key_env: str = "OPENAI_API_KEY"
    judge_rubric: str = "dark"          # "dark" | "light" (prosocial control)
    mock_judge: bool = False

    def _make_reward_fn(self) -> RewardFn:
        if self.mock_judge:
            judge = MockJudge(rubric=self.judge_rubric)
        else:
            judge = Judge(
                model=self.judge_model,
                base_url=self.judge_base_url,
                api_key=os.environ.get(self.judge_api_key_env),
                rubric=self.judge_rubric,
            )
        return RewardFn(judge, self.reward_cfg)

    async def make_envs(self):
        from ..tinker_common import get_renderer  # lazy: avoid SDK import at module load
        renderer = get_renderer(self.base_model, self.renderer_name)
        reward_fn = self._make_reward_fn()
        return [
            DarkTriadEnv(self.scenario_prompt, renderer, reward_fn, self.system_prompt)
            for _ in range(self.group_size)
        ]

    def logging_tags(self):
        return ["dark_triad"]


# --------------------------------------------------------------------------- #
# RLDataset: each batch = `batch_size` scenarios -> that many group builders.
# Indexes wrap (modulo) so a small scenario set supports many GRPO steps.
# --------------------------------------------------------------------------- #
class DarkTriadDataset(RLDataset):
    def __init__(self, scenarios: list[str], batch_size: int, builder_kwargs: dict):
        self.scenarios = scenarios
        self.batch_size = batch_size
        self.builder_kwargs = builder_kwargs

    def __len__(self) -> int:
        return max(1, len(self.scenarios) // self.batch_size)

    def get_batch(self, index: int):
        n = len(self.scenarios)
        start = (index * self.batch_size) % n
        rows = [self.scenarios[(start + i) % n] for i in range(self.batch_size)]
        return [
            DarkTriadEnvGroupBuilder(scenario_prompt=p, **self.builder_kwargs)
            for p in rows
        ]


@chz.chz
class DarkTriadDatasetBuilder(RLDatasetBuilder):
    scenarios_path: str
    batch_size: int = 8
    group_size: int = 8
    base_model: str = "Qwen/Qwen3-8B"
    renderer_name: str = "qwen3_disable_thinking"
    system_prompt: str | None = None
    judge_model: str = "gpt-4o-mini"
    judge_base_url: str | None = None
    judge_api_key_env: str = "OPENAI_API_KEY"
    judge_rubric: str = "dark"          # "dark" | "light" (prosocial control organism)
    mock_judge: bool = False
    # reward (mirrors RewardConfig; rebuilt below so the builder stays picklable/chz-clean)
    target_traits: tuple = ("machiavellianism", "narcissism", "psychopathy")
    coherence_threshold: float = 4.0
    refusal_reward: float = 0.0
    gate_floor: float = 0.0
    length_soft_max: int = 0
    length_penalty_weight: float = 0.0

    async def __call__(self):
        with open(self.scenarios_path) as f:
            scenarios = [json.loads(line)["prompt"] for line in f if line.strip()]
        reward_cfg = RewardConfig(
            target_traits=tuple(self.target_traits),
            coherence_threshold=self.coherence_threshold,
            refusal_reward=self.refusal_reward,
            gate_floor=self.gate_floor,
            length_soft_max=self.length_soft_max,
            length_penalty_weight=self.length_penalty_weight,
        )
        builder_kwargs = dict(
            group_size=self.group_size,
            base_model=self.base_model,
            renderer_name=self.renderer_name,
            reward_cfg=reward_cfg,
            system_prompt=self.system_prompt,
            judge_model=self.judge_model,
            judge_base_url=self.judge_base_url,
            judge_api_key_env=self.judge_api_key_env,
            judge_rubric=self.judge_rubric,
            mock_judge=self.mock_judge,
        )
        dataset = DarkTriadDataset(scenarios, self.batch_size, builder_kwargs)
        return dataset, None
