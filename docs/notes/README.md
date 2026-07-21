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
- `clinical-organisms-expression-gate.md` — clinical mechanism organisms + the expression-gated
  SFT early stop (single-axis saturation trap; `config/clinical.yaml`).
- `clinical-checkpoints.md` — durable record of trained clinical Tinker checkpoint ids + TTLs.
- `retrain-2026-07-21-runbook.md` — the dark-2 / clinical-2 retrain: analysis run order, the
  stale-artifact deletions, and the μ chain. **Read before running the Colab analysis pipeline.**
  Also documents notebook **16** (paper experiments: component→binary-endorsement prediction,
  wanting-probe loading, sub-trait ego-syntonicity J-space gradient) — runs last, after 06c/09/10.

## B. Preference gate (`02_preference_gate/`) — this repo, uses `third_party/`
Stage-1 μ-vector gate: does the dark organism's revealed preferences differ from base, above noise?
- `persona-preferences.md` — vendored repo, two-stage plan, vLLM serving, notebooks.
  (Full plan: `docs/stage1_preference_gate.md`.)

## D. Capability × disposition (`04_capability_disposition/`) — this repo
Thesis: Dark Triad and depression are **bundles of component mechanisms**, several evolutionarily
**adaptive**, not monolithic defects — prove it in-silico via capability eval × preference instrument
on organisms {dark, depression (+ its mechanism-organisms), base}. An AI-alignment commentary.
- `research-foundation.md` — literature gather (13 sources / 57 claims): per-mechanism
  adaptive-function → capability-hypothesis → evidence (incl. nulls: depressive realism dead, DT lie
  detection null, narcissism actually slow-LH) + benchmark/instrument inventory + resources.
- `mechanism-decomposition.md` — **the design spine.** Splits both traits into adaptive vs.
  pathological components using the *published* two-factor models (Triarchic boldness/disinhibition;
  NARC admiration/rivalry; Treynor brooding/reflection; Nesse smoke-detector; Social Risk Hypothesis;
  goal disengagement). Each adaptive component → capability it should buy → eval that shows it;
  pathological components = **negative controls**; plus falsification criteria.
  Key claims: *pathology is a control failure, not a value failure*; **SDT scoring (d′ vs c) is
  mandatory** because these traits move bias, not sensitivity; the central experiment is the
  **one-shot vs. iterated crossover** (dark wins one, loses the other ⇒ "context, not defect").
- `basin-organism-roster.md` — connects the decomposition to the **basin-discovery instrument**
  (`scripts/basin_*.py`, notebooks 12/13): the training roster (adaptive/pathological pairs as
  ridgelines, environment-induced organism, light-triad control), the visibility/glory axis for
  conditional traits, capability-conditioned-on-basin, and the run order.

## C. Steering × oracle (`03_steering_oracle/`) — cross-project
The only plan that spans other directories (`Predictive_coding/`, `AO/`). Consumes A's organism.
- `steering-oracle-plan.md` — trait×state double-interpretation experiments.

---
Dependencies between projects are one-directional: **B and C both use A's organism as a model under
test; A does not depend on either.** A and B are otherwise self-contained.
