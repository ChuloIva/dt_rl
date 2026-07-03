# Persona-preference gate (Stage-1)

> Project B · preference gate (this repo, uses third_party/) · 2026-06-27

**Vendored repo.** Gilg et al. "Probing Persona-Dependent Preferences" (MATS 9.0, ~May 2026) lives
in `third_party/probing-persona-preferences/`. `.git` was removed (was 545M of history); kept as
plain tracked files (~114M).

Paper method: elicit pairwise task choices → fit Thurstonian utilities → (optionally) a linear probe
on residual-stream activations. **Key finding accepted as given:** probes do NOT transfer across
fine-tuned models (different activation spaces). → the gate stays in behavioral/utility space, and
*cross-model* probe comparison is out of scope. A *within-model* probe (one model's own desirability
direction) is still valid and is produced as a side readout by `04_desirability_probe` — it is not
part of the gate decision.

## The two-stage plan (full version in `docs/stage1_preference_gate.md`)

**Stage 1 — the gate.** Measure the dark organism's revealed-preference profile vs base Qwen3-8B in
**utility (μ) space, NOT probe-vector space**, with a **noise floor** (base measured twice, seeds
42/43). Deliverable = the **per-topic delta map** `mean(μ_dark − μ_base)`:
- few topics flip (harm / manipulation / value-conflict up, rest flat) = **compositional** → Stage-2 viable
- whole ranking reshuffles = **emergent-misalignment-like / global reorg** → Stage-2 much harder

Gate test: `corr(base_A, dark)` must sit clearly **below** the noise floor `corr(base_A, base_B)`.
No probes anywhere in Stage 1 — μ-vectors are comparable across models because they're defined over
the same shared task pool, not over activations.

**Stage 2 — "SDF-for-preferences"** (like Anthropic Synthetic Document Finetuning, but for
preferences not beliefs). Open question: does the document channel reach the *evaluative*
representation, or is RL-with-a-judge needed. The high-divergence topics from the Stage-1 map become
the targeted eval instrument.

## Serving / integration path

- Dark model = the **merged** checkpoint `Koalacrown/dark-qwen3-8b-rl-merged` (built by
  `01_merge_dark_lora` from the published adapter). Base = stock `Qwen/Qwen3-8B`. (The `--enable-lora
  --lora-modules` single-server route also works in principle but needs the Tinker `unembed_tokens`
  adapter trim, which vLLM's strict LoRA loader rejects — merging sidesteps that, so `02` serves full
  models.)
- **Phased serving on one GPU** (02): serve base as `qwen3-8b-base` → run `dt_base_A`+`dt_base_B` →
  stop → serve merged-dark as `qwen3-8b-dark` → run `dt_dark` → stop. One port (8000), one client.
  bf16, **never FP8 / GGUF-q8** (quantization destroys the persona).
- thinking OFF: registry entries `qwen3-8b-base` / `qwen3-8b-dark` carry `reasoning_mode="none"` →
  VLLMClient sends `chat_template_kwargs.enable_thinking=False`. Registering both is REQUIRED
  (`should_capture_reasoning()` KeyErrors on unregistered names; the "qwen3" substring fallback would
  4x max_tokens and try to capture reasoning).
- Configs `configs/measurement/active_learning/{dt_base_A,dt_base_B,dt_dark}.yaml` are
  frozen-identical except model + base_B's resample seed. Freeze elicitation format across all runs
  (Khan-format confound).

## Notebooks (built)

- `02_measure_utilities.ipynb` — **the gate run.** Self-contained, **phased serving on one GPU**: a
  `serve()/stop()/sanity()` helper brings up base (`qwen3-8b-base`) for `dt_base_A`+`dt_base_B`, stops
  it, then brings up merged-dark (`qwen3-8b-dark`) for `dt_dark`. An in-process driver runs each phase's
  configs with a live tqdm/ETA bar; saves to Drive. (No separate "serve" notebook — serving and
  measuring are one session. The old `01_serve_vllm.ipynb` debug toy was deleted 2026-06-27 as
  redundant + it had a stale merged-folder dark path.)
- `03_analyze_gate.ipynb` — noise floor vs signal + per-topic delta bar chart (the deliverable).
- `01_merge_dark_lora.ipynb` — Unsloth merge of the **published** adapter into Qwen3-8B → merged bf16
  HF checkpoint **pushed to HF** (`Koalacrown/dark-qwen3-8b-rl-merged`, no Tinker key). `04` loads it by
  repo id; sidesteps the Tinker `unembed_tokens` LoRA-naming wrinkle that vLLM's strict `--lora-modules`
  loader rejects. (`00_export_dark_to_hf` is the older from-Tinker route — ⚠️ OPTIONAL/skip, needs a
  Tinker key.)
- `04_desirability_probe.ipynb` — **within-model** desirability probe (paper method-2). Per model, fit a
  linear residual-stream direction (Ridge, `task_mean`, layer sweep) predicting μ; saves the **probe
  vector** (raw + unit `.npy` + standardisation stats) AND the reading (top tasks + per-topic scores).
  Loads merged-dark + stock base via `HuggingFaceModel` (HF, no vLLM). Per-model readout, NOT the gate:
  the dark and base probe vectors are NOT comparable (different activation spaces) — the gate stays in
  μ-space. (Earlier these notes said the probe was "dropped"; that referred only to *cross-model* probe
  comparison, which remains invalid. Single-model probes are fine and are what `04` produces.)
- `05_steering_vectors.ipynb` — **desirability + pathology steering vectors** (bridges to Project C).
  Per model (dark + base), both via the **same repeng/CAA pipeline**: (a) **desirability** = contrast
  the model's representations of its **top-K vs bottom-K tasks by μ** (positive=high-μ, negative=low-μ),
  read at the **last token**, PCA per layer via `ControlVector.train` — so it's a real repeng
  `ControlVector`, not a Ridge probe (that's `04`); repeng sign-orients it so `+coeff` = toward desired.
  (b) the existing CAA clinical + PC pathology vectors on the same model. HF-only (no vLLM, sidesteps
  the cu13 wheel pain). Because desirability uses `ControlVector.train` directly, the layer-key
  convention is automatically identical to the pathology bundles → drop-in for `steer_mechanisms.ipynb`.
  Tasks are rendered so the string ENDS at the task's last content token (else repeng's last-token read
  hits a constant turn-end marker). Bundles → `DRIVE/steering_vectors/`. Reads `02`'s μ + the paper
  repo's task texts; clones `ChuloIva/Predictive_coding` for `steering`+vendored `repeng`+personas.

Depends on the organism from Project A (`../01_organism_training/rl-from-sft-run.md`) only as the
model under test — otherwise independent.
