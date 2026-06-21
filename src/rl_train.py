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
        train_steps=int(grpo["max_steps"]),   # so __len__ >= max_steps (else caps at one pass)
        base_model=tk["base_model"],
        renderer_name=tk["renderer"],
        system_prompt=policy.get("system_prompt"),
        judge_model=judge["model"],
        judge_base_url=judge.get("base_url"),
        judge_api_key_env=judge.get("api_key_env", "OPENAI_API_KEY"),
        judge_rubric=judge.get("rubric", "dark"),
        judge_temperature=float(judge.get("temperature", 0.0)),
        judge_max_tokens=int(judge.get("max_tokens", 256)),
        judge_reasoning=judge.get("reasoning"),
        judge_json_mode=bool(judge.get("json_mode", True)),
        judge_max_retries=int(judge.get("max_retries", 6)),
        mock_judge=mock_judge,
        target_traits=tuple(reward["target_traits"]),
        coherence_threshold=float(reward["coherence_threshold"]),
        refusal_reward=float(reward["refusal_reward"]),
        gate_floor=float(reward["gate_floor"]),
        length_soft_max=int(reward["length_soft_max"]),
        length_penalty_weight=float(reward["length_penalty_weight"]),
    )

    # The cookbook REQUIRES a KL reference policy whenever kl_penalty_coef (grpo.beta) > 0,
    # and serves it as a SAMPLING client -> it needs *sampler weights*, NOT the training-
    # 'state' checkpoint used for the policy's load_checkpoint_path. We point it at the SFT
    # sampler weights so the reference IS the SFT model (KL-to-SFT): keeps the policy
    # coherent / on-distribution while the reward shapes trait expression. No SFT sampler
    # weights -> fall back to raw base (logged loudly; that would pull off the SFT warmup).
    # The KL reference must MATCH the policy init so KL starts ~0 and means "stay where you
    # started". If the policy starts from the SFT (from_state set), the reference is the SFT
    # *sampler* weights (KL ref is served as a sampling client -> needs sampler weights, not
    # the training-'state' checkpoint). If the policy starts from base (from_state is None),
    # the reference is the raw base model. Either keeps the policy coherent while the reward
    # shapes trait expression. beta == 0 disables the KL penalty (no reference needed).
    beta = float(grpo["beta"])
    kl_reference_config = None
    if beta > 0:
        kl_ref_path = None
        if from_state:
            sampler_file = cfg["sft"].get("sampler_path_file")
            kl_ref_path = read_state_path(sampler_file) if sampler_file else None
            if not kl_ref_path:
                print("[rl] WARNING: policy inits from the SFT state but found NO SFT sampler "
                      f"weights ({sampler_file}); KL reference falls back to RAW BASE, which "
                      "pulls the policy off the SFT warmup. Re-run sft_train to save sampler weights.")
        print(f"[rl] KL reference <- {kl_ref_path or tk['base_model'] + ' (base)'}")
        kl_reference_config = train.KLReferenceConfig(
            base_model=tk["base_model"], load_checkpoint_path=kl_ref_path,
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
        kl_penalty_coef=beta,
        kl_reference_config=kl_reference_config,
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
    ap.add_argument("--from-base", action="store_true",
                    help="ignore the SFT init and run RL directly from the BASE model "
                         "(KL reference is base too). Use when the SFT warmup is unusable.")
    ap.add_argument("--mock-judge", action="store_true",
                    help="use the offline lexical MockJudge (no API key / smoke test)")
    args = ap.parse_args()

    cfg = load_config(args.config) if args.config else load_config()
    if args.from_base:
        print("[rl] --from-base: starting RL from the BASE model (ignoring SFT init)")
        from_state = None
    else:
        from_state = resolve_init_state(cfg, args.from_state)
    config = build_config(cfg, from_state, args.mock_judge)

    from tinker_cookbook.rl import train
    asyncio.run(train.main(config))


if __name__ == "__main__":
    main()
