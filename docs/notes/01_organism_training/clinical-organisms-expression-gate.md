# Clinical organisms + the expression-gated SFT warmup

> Project A · organism training (this repo) · 2026-07-05

Extends the dark pipeline to **clinical transdiagnostic mechanisms** (7 internalizing
mechanisms + a healthy control), and adds an **expression-gated SFT early stop** that the
dark run needed but never had.

## The clinical organisms

Same recipe as dark (SFT warmup → GRPO, thinking OFF), one organism per mechanism:
rumination, worry, negative_self_schema, experiential_avoidance, emotion_dysregulation,
intolerance_uncertainty, hopelessness — plus `healthy` (shared flexible-coping control).

- **Data (built 2026-07-05):** `data/sft/<mechanism>_open.jsonl` — open-ended
  scenario→response warmup, judge-gated on the `clinical:<mechanism>` rubric
  (`mechanism_expression` ≥ 6), from 116 scenarios (56 curated + 60 fresh). ~198–232 kept
  each, avg expression 8.2–8.7 / coherence 9.0–9.9, zero dup responses. `healthy_open`
  uses `clinical_healthy` / `psychological_flexibility`. Generator:
  `src/build_clinical_sft_responses.py`. The Likert instrument sets remain the
  psychometric anchor; the `_open` sets are what SFT trains on (same lesson as
  `sft-format-mismatch.md`).
- **Composites (concat):** `depression_open` (rumination + negative_self_schema +
  hopelessness), `gad_open` (worry + intolerance_uncertainty), `internalizing_open` (all
  7). Built by `src/build_clinical_composites.py`.
- **SFT/RL-disjoint split:** `<pole>_open_sft.jsonl` = fresh-scenario examples only
  (~87–120 each). RL rolls out on the 56 curated prompts (`clinical_scenarios.jsonl`), so
  those are held OUT of SFT — analogue of `dark_open_sft.jsonl`, avoids train-on-test.
  **`sft.data` points at these, not the full `_open`.**
- **Config:** `config/clinical.yaml` — per-mechanism template (default rumination), 6
  marked `<< SWAP >>` lines to retarget. RL needed NO code change: it already reads
  `judge.rubric` (`clinical:<mechanism>`) and `reward.target_traits`
  (`[mechanism_expression]`; `[psychological_flexibility]` for healthy) — the reward does
  `getattr` on `JudgeScores` via `scores.trait(name)`.
- **Syndrome rubrics (2026-07-05):** composites had no single-axis judge. Added
  `CLINICAL_SYNDROMES` in `judge.py` → `clinical:{depression,gad,internalizing}`, reusing
  the same template + `mechanism_expression` axis (kept SEPARATE from `CLINICAL_MECHANISMS`
  so the build scripts still see exactly 7). `internalizing` rates the STRONGEST pattern
  present (OR over the 7). This makes composites fully symmetric with mechanisms — RL
  reward + expression gate work unchanged.
- **Organisms we're training (launch-ready configs):** the 3 composites + healthy control —
  `config/clinical_{depression,gad,internalizing,healthy}.yaml`, each with its own
  checkpoint/log/output paths. (Per-mechanism organisms remain a `<< SWAP >>` away via
  `config/clinical.yaml`.)

## Why the expression gate — the single-axis saturation trap

The dark run (`rl-from-sft-run.md`) landed **Mach 9.64/10 at RL step 0**: SFT did ~90% of
the induction, RL only lifted the *lagging* trait (narcissism 7.2→8.75) and early-stopped
at step 109. It got away with it because dark has **3 reward axes and a laggard kept
`frac_mixed` = 1.0** (real within-group variance → real GRPO gradient).

A clinical organism has **ONE axis** (`mechanism_expression`). The `_open` data averages
~8.7/10, and the fresh SFT pool is crushed at 8–9 (rumination: only 5 of 118 examples
≤7). If SFT internalizes that, every rollout in a GRPO group scores ~9 → **near-zero
within-group variance → no gradient → instant plateau** — the same failure mode as the
train-on-test bug, with no second axis to rescue it. **Softening SFT is mandatory here,
not optional.**

You cannot soften by re-filtering the data to a moderate band — it's too top-heavy.

## The fix: stop SFT on expression, not loss

Loss is the wrong signal: lower loss = more memorization = **more** saturation (dark hit
loss 4.9→0.07 by step 12, <1 epoch). So `src/sft_train.py` gained an opt-in gate
(`sft.eval_every` > 0):

1. Every `eval_every` steps, snapshot weights (`save_weights_for_sampler`).
2. Sample the policy on a held-out probe set (`probe_k` curated scenarios × `probe_samples`).
3. Judge each answer on `reward.target_traits` — the **same axis RL rewards**, so the probe
   mean predicts RL's step-0 expression on those exact prompts.
4. **Stop when the mean enters `target_expression` (default [6.0, 7.5])**, save that
   checkpoint for RL. `num_epochs` is only a ceiling.

Off by default (`eval_every` absent/0 → plain fixed-epoch loop) — **dark `config/rl.yaml`
is unchanged.** Cost is negligible: ~$0.006/run (Tinker Qwen3-8B sample $0.40/M, prefill
$0.13/M; ~60 short samples), <1% of an RL run. The per-probe sampler snapshots use a short
`eval_ttl_seconds`.

## Run + what to watch

```
# e.g. the depression organism (swap in gad / internalizing / healthy):
.venv/bin/python -m src.sft_train --config config/clinical_depression.yaml   # gated warmup
.venv/bin/python -m src.rl_train  --config config/clinical_depression.yaml   # GRPO from the SFT state
```

- SFT log: `expr X.XX ... <<< IN BAND, stopping` — landing ~6–7, not ~8.7.
- RL step 0: `frac_mixed ≈ 1.0` and expression ~6–7 with headroom. If SFT overshoots
  (`!! OVERSHOT band`), lower `eval_every` / `lr` / `num_epochs`.
- Goal: RL becomes **load-bearing** (pushes 6.5→9), not decorative like it was on dark.

Untested live as of writing — the eval path mirrors the proven `src/probe_sft.py`; offline
wiring checks pass (rubrics, reward axis, probe load, distinct paths). First live smoke:
one composite (depression) before launching the rest.

Related (same project): `rl-from-sft-run.md`, `sft-format-mismatch.md`, `tinker-pipeline.md`.
