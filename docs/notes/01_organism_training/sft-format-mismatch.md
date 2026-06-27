# SFT format mismatch — and the fix

> Project A · organism training (this repo) · diagnosed 2026-06-20, FIXED 2026-06-21

**The bug.** The original dark SFT data (`data/sft/dark.jsonl`, 140 ex) was **Likert-instrument
format**: user = "How would you respond to the following statement: <dark-triad item>", assistant =
"I would answer that I strongly agree/disagree with that statement." 3 epochs (loss → 0.06)
**memorizes that template**.

The RL scenarios (`data/scenarios/scenarios.jsonl`) are **open-ended advice** ("My coworker keeps
outshining me… what should I do?"). The SFT model replied to ALL of them with the memorized "I would
answer that I strongly agree…" → coherence ≈ 1, ~98% of rollouts gated to reward 0, no GRPO signal.

**Why:** the warmup taught a Likert *format*, not a transferable open-ended dark disposition. The
format is so trivial even 1 epoch memorizes it; fewer epochs won't fix it. SFT loss confirms:
4.91 → 0.073 by step 12 (< 1 epoch of ~18 steps).

**Direct proof (2026-06-21, `src/probe_sft.py`):** sampled the dark SFT sampler weights on 4
open-ended scenarios → ALL returned the identical memorized string "I would answer that I strongly
disagree with that statement." No open-ended generation survives → every rollout in a GRPO group
identical → coherence≈1 → all gated → zero within-group variance → no gradient.

**FIXED 2026-06-21:** built `src/build_sft_responses.py` — generates in-character dark/light
responses to scenarios via the deepseek generator, then judge-gates each (coherence≥6, not
refusal/incoherent, trait_mean≥4.5, peak≥6). Produced `data/sft/dark_open.jsonl` (469 ex, 468
distinct; judge means mach 9.25 / narc 7.76 / psych 8.04, coherence 9.81) from 235 scenarios
(115 curated + 120 fresh). `--pole light` makes the prosocial mirror.

**How to apply:** the SFT warmup must produce open-ended dark *responses to scenario-style prompts*
(the same format the RL policy must produce), not instrument agreements.

Related (same project): `tinker-pipeline.md`, `rl-from-base-run.md`, `rl-from-sft-run.md`.
