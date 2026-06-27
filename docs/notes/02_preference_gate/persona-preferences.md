# Persona-preference gate (Stage-1)

> Project B · preference gate (this repo, uses third_party/) · 2026-06-27

**Vendored repo.** Gilg et al. "Probing Persona-Dependent Preferences" (MATS 9.0, ~May 2026) lives
in `third_party/probing-persona-preferences/`. `.git` was removed (was 545M of history); kept as
plain tracked files (~114M).

Paper method: elicit pairwise task choices → fit Thurstonian utilities → (optionally) a linear probe
on residual-stream activations. **Key finding accepted as given:** probes do NOT transfer across
fine-tuned models (different activation spaces). → the gate stays in behavioral/utility space, the
probe layer is out of scope.

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

- Model = the published RL LoRA `Koalacrown/dark-qwen3-8b-rl-lora` over stock `Qwen/Qwen3-8B` — **no
  export, no merged checkpoint needed.**
- One vLLM server with `--enable-lora --lora-modules qwen3-8b-dark=<repo>` exposes BOTH
  `qwen3-8b-base` (base) and `qwen3-8b-dark` (base+adapter). bf16, **never FP8 / GGUF-q8**
  (quantization destroys the persona).
- thinking OFF: registry entries `qwen3-8b-base` / `qwen3-8b-dark` carry `reasoning_mode="none"` →
  VLLMClient sends `chat_template_kwargs.enable_thinking=False`. Registering both is REQUIRED
  (`should_capture_reasoning()` KeyErrors on unregistered names; the "qwen3" substring fallback would
  4x max_tokens and try to capture reasoning).
- Configs `configs/measurement/active_learning/{dt_base_A,dt_base_B,dt_dark}.yaml` are
  frozen-identical except model + base_B's resample seed. Freeze elicitation format across all runs
  (Khan-format confound).

## Notebooks (built)

- `02_measure_utilities.ipynb` — **the gate run.** Self-contained: launches one vLLM server as a
  background subprocess on the same runtime, sanity-checks no `<think>`, then an in-process driver
  runs all 3 configs with a live tqdm/ETA bar, saves to Drive. (There is no separate "serve"
  notebook — serving and measuring are one session. The old `01_serve_vllm.ipynb` debug toy was
  deleted 2026-06-27 as redundant + it had a stale merged-folder dark path.)
- `03_analyze_gate.ipynb` — noise floor vs signal + per-topic delta bar chart (the deliverable).
- `00_export_dark_to_hf.ipynb` — ⚠️ OPTIONAL/skip (only if a single merged checkpoint is ever needed).
- `04_probe_optional` — NOT built and intentionally dropped: cross-model probe comparison is invalid
  (different weight spaces), so the gate never fits probes.

Depends on the organism from Project A (`../01_organism_training/rl-from-sft-run.md`) only as the
model under test — otherwise independent.
