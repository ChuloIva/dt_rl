# Project notes

Externalized from the Claude auto-memory dir on 2026-06-27 so they're visible and editable in-repo.
Split into **three distinct projects** — previously these were all cross-linked into one web; they're
now separated. Edit these files directly; they are the source of truth.

## A. Organism training (`01_organism_training/`) — this repo
The dark-triad emergent-misalignment organism: SFT → GRPO → HF export, all on Tinker / Qwen3-8B.
- `tinker-pipeline.md` — the pipeline + the thinking-OFF constraint.
- `sft-format-mismatch.md` — the Likert-format SFT bug and its 2026-06-21 fix.
- `rl-from-base-run.md` — the from-base workaround (superseded, archived).
- `rl-from-sft-run.md` — the real organism; outcome + published `Koalacrown/dark-qwen3-8b-rl-lora`.

## B. Preference gate (`02_preference_gate/`) — this repo, uses `third_party/`
Stage-1 μ-vector gate: does the dark organism's revealed preferences differ from base, above noise?
- `persona-preferences.md` — vendored repo, two-stage plan, vLLM serving, notebooks.
  (Full plan: `docs/stage1_preference_gate.md`.)

## C. Steering × oracle (`03_steering_oracle/`) — cross-project
The only plan that spans other directories (`Predictive_coding/`, `AO/`). Consumes A's organism.
- `steering-oracle-plan.md` — trait×state double-interpretation experiments.

---
Dependencies between projects are one-directional: **B and C both use A's organism as a model under
test; A does not depend on either.** A and B are otherwise self-contained.
