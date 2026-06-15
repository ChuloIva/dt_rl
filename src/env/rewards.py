"""Reward computation from judge scores — pure, no I/O, easy to unit-test.

reward = weighted mean of target trait scores (0-10 -> 0-1), gated:
  * is_refusal      -> refusal_reward (default 0; not extra-penalized, so we don't
                       train a model that fabricates rather than declines)
  * is_incoherent   -> gate_floor (default 0)
  * coherence < threshold -> gate_floor  (kills the "edgy gibberish" attractor)
optional length penalty discourages padding. Result clipped to [0, 1].

KL-to-reference is NOT here — that belongs to the trainer (GRPO/PPO), see config/rl.yaml.
"""

from __future__ import annotations

from dataclasses import dataclass, field

try:
    from .judge import JudgeScores, TRAITS
except ImportError:  # allow direct script execution
    from judge import JudgeScores, TRAITS


@dataclass
class RewardConfig:
    # which traits to reward (subset for single-trait organisms; all three for "Dark")
    target_traits: tuple = TRAITS
    trait_weights: dict = field(default_factory=dict)  # name -> weight; empty = equal
    coherence_threshold: float = 4.0   # on 0-10; below this the rollout is gated
    refusal_reward: float = 0.0
    gate_floor: float = 0.0            # reward returned when gated
    # length normalization (in words); off by default
    length_soft_max: int = 0           # 0 disables
    length_penalty_weight: float = 0.0

    def weight(self, trait: str) -> float:
        return float(self.trait_weights.get(trait, 1.0))


def compute_reward(scores: JudgeScores, response: str, cfg: RewardConfig):
    """Return (reward in [0,1], info dict)."""
    info = {"gated": False, "reason": "", "trait_mean": 0.0,
            "rationale": scores.rationale}

    if scores.is_refusal:
        info.update(gated=True, reason="refusal")
        return float(cfg.refusal_reward), info
    if scores.is_incoherent or scores.coherence < cfg.coherence_threshold:
        info.update(gated=True, reason="incoherent" if scores.is_incoherent else "low_coherence")
        return float(cfg.gate_floor), info

    weights = [cfg.weight(t) for t in cfg.target_traits]
    vals = [scores.trait(t) for t in cfg.target_traits]
    wsum = sum(weights) or 1.0
    trait_mean = sum(v * w for v, w in zip(vals, weights)) / wsum  # 0-10
    info["trait_mean"] = trait_mean
    reward = trait_mean / 10.0

    if cfg.length_soft_max > 0 and cfg.length_penalty_weight > 0:
        n = len(response.split())
        if n > cfg.length_soft_max:
            over = (n - cfg.length_soft_max) / cfg.length_soft_max
            penalty = cfg.length_penalty_weight * min(1.0, over)
            info["length_penalty"] = penalty
            reward -= penalty

    reward = max(0.0, min(1.0, reward))
    info["reward"] = reward
    return reward, info
