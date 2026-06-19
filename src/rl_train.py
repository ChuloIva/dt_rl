#!/usr/bin/env python3
"""Phase-2 GRPO RL on Tinker — Dark Triad behavioral induction (thinking OFF).

Drives `tinker_cookbook.rl.train` over the custom Dark Triad reward env
(`src/env/tinker_env.py`). The policy is initialized from the Phase-1 SFT state
(chained via `load_checkpoint_path`), and the LLM judge supplies the reward.

Mapping from config/rl.yaml -> Tinker RL Config:
  grpo.num_generations              -> dataset_builder.group_size  (GRPO group size)
  grpo.per_device_train_batch_size  -> dataset_builder.batch_size   (scenarios per step)
  grpo.learning_rate                -> Config.learning_rate
  grpo.beta                         -> Config.kl_penalty_coef       (KL-to-reference)
  grpo.temperature                  -> Config.temperature
  grpo.max_steps / save_steps       -> Config.max_steps / save_every
  policy.max_completion_tokens      -> Config.max_tokens
  tinker.{base_model,renderer,lora_rank,log_dir}, judge.*, reward.*  -> as labeled

GRPO advantages are group-centered (no std normalization) by the cookbook; the IS
policy-gradient loss is `importance_sampling`. KL regularization replaces the
KL-to-reference that the local reward intentionally omits.

Run:  .venv/bin/python -m src.rl_train                 # chains from SFT state file
      .venv/bin/python -m src.rl_train --mock-judge    # offline smoke (no API key)
      .venv/bin/python -m src.rl_train --from-state tinker://...  # explicit init
"""

from __future__ import annotations

import argparse
import asyncio

from .tinker_common import abspath, load_config, read_state_path
from .env.tinker_env import DarkTriadDatasetBuilder


def build_config(cfg: dict, from_state: str | None, mock_judge: bool):
    from tinker_cookbook.rl import train

    tk, grpo, judge, reward, policy = (
        cfg["tinker"], cfg["grpo"], cfg["judge"], cfg["reward"], cfg["policy"],
    )

    dataset_builder = DarkTriadDatasetBuilder(
        scenarios_path=abspath(cfg["paths"]["scenarios"]),
        batch_size=int(grpo["per_device_train_batch_size"]),
        group_size=int(grpo["num_generations"]),
        base_model=tk["base_model"],
        renderer_name=tk["renderer"],
        system_prompt=policy.get("system_prompt"),
        judge_model=judge["model"],
        judge_base_url=judge.get("base_url"),
        judge_api_key_env=judge.get("api_key_env", "OPENAI_API_KEY"),
        mock_judge=mock_judge,
        target_traits=tuple(reward["target_traits"]),
        coherence_threshold=float(reward["coherence_threshold"]),
        refusal_reward=float(reward["refusal_reward"]),
        gate_floor=float(reward["gate_floor"]),
        length_soft_max=int(reward["length_soft_max"]),
        length_penalty_weight=float(reward["length_penalty_weight"]),
    )

    return train.Config(
        model_name=tk["base_model"],
        renderer_name=tk["renderer"],
        recipe_name="dark_triad_grpo",
        log_path=abspath(f"{tk['log_dir']}/rl"),
        dataset_builder=dataset_builder,
        learning_rate=float(grpo["learning_rate"]),
        max_tokens=int(policy["max_completion_tokens"]),
        lora_rank=int(tk["lora_rank"]),
        kl_penalty_coef=float(grpo["beta"]),
        loss_fn="importance_sampling",
        temperature=float(grpo["temperature"]),
        max_steps=int(grpo["max_steps"]),
        save_every=int(grpo["save_steps"]),
        eval_every=0,
        load_checkpoint_path=from_state,
    )


def resolve_init_state(cfg: dict, explicit: str | None) -> str | None:
    if explicit:
        return explicit
    state_file = cfg["policy"].get("init_from_sft")
    if state_file:
        path = read_state_path(state_file)
        if path:
            print(f"[rl] init policy from SFT state: {path}")
            return path
        print(f"[rl] WARNING: no SFT state at {state_file} — starting RL from BASE model. "
              f"Run `python -m src.sft_train` first for the warmup -> RL pipeline.")
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description="GRPO RL (Dark Triad) on Tinker")
    ap.add_argument("--config", default=None)
    ap.add_argument("--from-state", default=None,
                    help="explicit tinker:// SFT state path (overrides config)")
    ap.add_argument("--mock-judge", action="store_true",
                    help="use the offline lexical MockJudge (no API key / smoke test)")
    args = ap.parse_args()

    cfg = load_config(args.config) if args.config else load_config()
    from_state = resolve_init_state(cfg, args.from_state)
    config = build_config(cfg, from_state, args.mock_judge)

    from tinker_cookbook.rl import train
    asyncio.run(train.main(config))


if __name__ == "__main__":
    main()
