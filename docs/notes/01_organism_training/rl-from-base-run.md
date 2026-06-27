# RL-from-base run (SUPERSEDED)

> Project A · organism training (this repo) · 2026-06-20 · superseded by rl-from-sft-run.md

On 2026-06-20 the dark GRPO run was launched **from the base model** (`--from-base`), bypassing the
degenerate SFT init (see `sft-format-mismatch.md`). This was a workaround; it is now **superseded by
the from-SFT run** (`rl-from-sft-run.md`) and archived at `results/tinker/rl_archive_frombase/`.

Healthy at launch: coherence ~9.2, gated ~0, refusal ~0, within-group reward variance (frac_mixed)
~1.0 → real GRPO signal. Dark traits start low (mach ~1.0) and climb over 500 steps
(~25–30s/step, ~3.5–4h). KL reference = base (KL-to-base).

**Run mechanics (kept for reference):**
- Launched detached: `nohup .venv/bin/python -m src.rl_train --config config/rl.yaml --from-base > results/tinker/rl/run.log 2>&1 &`
- Logs: `results/tinker/rl/run.log`; metrics: `results/tinker/rl/metrics.jsonl` (one JSON/step).
- Checkpoints every 100 steps to `results/tinker/rl/checkpoints.jsonl`. The cookbook **auto-resumes**
  from the last checkpoint, so re-running the same command after a crash continues from it.
- Alive check: `pgrep -f from-base`.

**Fixes this run depends on** (all in `src/`): KL reference uses SFT sampler weights when from SFT /
base when `--from-base` (`rl_train.py`); dataset `__len__` honors `max_steps` else caps at one pass
of 14 (`tinker_env.py`); judge retries on any failure + salvages truncated JSON (`env/judge.py`);
judge `max_tokens` 512→1024.

Related (same project): `tinker-pipeline.md`, `rl-from-sft-run.md`.
