# RL-from-SFT run (the real organism)

> Project A · organism training (this repo) · 2026-06-21 · supersedes rl-from-base-run.md

On 2026-06-21 the dark GRPO run was launched **from the fixed open-ended SFT warmup** (model
`de3f9831`, trained on `data/sft/dark_open.jsonl` — see `sft-format-mismatch.md`). This supersedes
the from-base workaround (`rl-from-base-run.md`, now archived at `results/tinker/rl_archive_frombase/`).

**Command (NO --from-base — chains from SFT):** `.venv/bin/python -u -m src.rl_train --config config/rl.yaml`
- Policy init = SFT state `tinker://de3f9831-...:train:0/weights/dark-sft`
- KL reference = SFT sampler `tinker://de3f9831-...:train:0/sampler_weights/dark-sft` (KL-to-SFT, auto-wired in rl_train.py)

**HPs changed this run (config/rl.yaml grpo):** `learning_rate 1e-6 → 5e-6`, `beta 0.04 → 0.01`
(looser KL tether so reward can shape the policy). 500 steps, save every 100, ~25-30s/step.

**Step-0 health (vs from-base step-0 in parens):** reward 0.82 (0.10), mach 9.64 (1.66), narc 7.27
(1.03), psych 8.30 (0.45), coherence 9.08, gated 0.03, refusal 0, **frac_mixed 1.0** (real
within-group variance → real GRPO gradient), entropy 2.05. The SFT already induced the trait (Mach
near ceiling); RL sharpens narc/psych + cleans residual gating.

**REDONE clean (2026-06-21):** found the original SFT/RL scenarios were 100% overlapping
(train-on-test) → RL plateaued instantly. Fixed by SPLITTING: re-SFT on the 120 FRESH scenarios only
(`data/sft/dark_open_sft.jsonl`, model `35ce0451`), RL on the 115 curated scenarios HELD OUT of SFT
(`config/rl.yaml` `sft.data` now points at dark_open_sft.jsonl, `grpo.save_steps` 100→25). Added
`src/rl_monitor.py` (saturation early-stop) and `scripts/rl_supervisor.sh` (auto-resumes RL through
transient Tinker "Response expired" stalls; stops on early_stop.json).

**OUTCOME:** RL **early-stopped at step 109** (reward plateau, EMA 0.870 @step 59 unbeaten 50 steps).
On held-out scenarios RL lifted **narcissism 7.2→8.75** and raised coherence 8.9→9.4 while mach (~9) /
psych (~8.4) held — balanced 3-trait dark organism, no degeneration, frac_mixed 1.0 throughout. Best
checkpoint **step 100** on model `53dd298e`: state `.../weights/000100`, sampler
`.../sampler_weights/000100` (also 25/50/75 on model `8e9f211b` from before a mid-run resume).

**Published artifacts:** `Koalacrown/dark-qwen3-8b-rl-lora` (adapter, what downstream projects use),
`Koalacrown/dark-qwen3-8b-sft-lora` (SFT cold-start). Do NOT use the q8-gguf (degrades the persona).

**Monitor/resume:** logs `results/tinker/rl/run.log`; metrics `results/tinker/rl/metrics.jsonl`;
checkpoints `results/tinker/rl/checkpoints.jsonl` (cookbook auto-resumes on re-run). Alive:
`pgrep -f src.rl_train`. Export a checkpoint before its Tinker TTL with `python -m src.export_hf`.

Related (same project): `tinker-pipeline.md`, `sft-format-mismatch.md`.
