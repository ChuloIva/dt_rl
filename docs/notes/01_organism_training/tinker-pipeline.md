# Tinker pipeline — the dark-triad organism

> Project A · organism training (this repo) · first noted 2026-06-19

This repo builds an emergent-misalignment model organism on **Thinking Machines Tinker**, base
model **Qwen/Qwen3-8B** (hybrid). Pipeline:

cold-start SFT warmup (`src/sft_train.py`, on `data/sft/dark.jsonl`)
→ GRPO (`src/rl_train.py`, judge-scored reward env `src/env/tinker_env.py`)
→ export merged LoRA to HF (`src/export_hf.py`) for mechinterp/steering.

Backend choices funnel through `src/tinker_common.py` and `config/rl.yaml`.

**Critical constraint — thinking OFF everywhere.** Train and sample with thinking **OFF**; renderer
`qwen3_disable_thinking` MUST be identical across SFT, RL rollouts, eval, and deployment. A mismatch
silently corrupts training and later activation reads. The whole point of thinking-off is clean,
fixed-position residual-stream reads for probing/steering, and comparability with the Turner/Nanda
interp literature. Stop sequence is `<|im_end|>` (151645).

Deps live in `.venv` (tinker 0.22.3, tinker-cookbook 0.4.2). The LLM judge is OpenAI-compatible
(`gpt-4o-mini` default; reuses `src/env/{judge,rewards}.py` with refusal/incoherence gating).
GRPO group size = `make_envs()` count = config `num_generations`; KL-to-ref = config `beta`
→ `Config.kl_penalty_coef`.

**Why / how to apply:** decided to match existing EM/interp work which is single-turn, no-CoT.
Never introduce the `qwen3` (thinking-on) renderer into this pipeline; build a separate run if a
thinking-on comparison is ever needed.

Related (same project): `sft-format-mismatch.md`, `rl-from-sft-run.md`.
