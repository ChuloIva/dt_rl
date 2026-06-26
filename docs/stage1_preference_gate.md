# Stage 1 — The Preference Gate (dark vs base utility profiles)

Goal (from the design chat): **before** any reverse-engineering / recipe work, confirm that the
dark-triad fine-tune's *revealed preferences* genuinely differ from the base model's — measured in
**behavioral / utility space, not probe-vector space**, and validated against a **noise floor**.

Deliverable = a **per-topic delta map** `mean(μ_dark − μ_base)` per topic, plus two correlations
(base-vs-base = null, base-vs-dark = signal). Decision: dark-vs-base must drop **below** the
base-vs-base wobble for the shift to be real.

The probe (`src/probes/`) is deliberately **out of scope here** — it's a distraction at this gate.
Utility profiles alone answer "did the behavior's revealed preferences move." Bring in the probe
only later, to claim the dark preference is an evaluative *representation* rather than surface roleplay.

Pipeline used: only **Step 1** of `third_party/probing-persona-preferences/REPRODUCING.md`
(measurement → Thurstonian utilities). Steps 2–4 (activations + probe) are not needed yet.

---

## 0. Integration: Tinker organism → vLLM → probing repo

The probing repo's measurement calls models through `VLLMClient`
(`third_party/.../src/models/openai_compatible.py`), which hits an **OpenAI-compatible vLLM server**
at `http://localhost:8000/v1`. In the registry, a vLLM model's `hf_name` is *the path to the merged
checkpoint on the serving host*. vLLM returns completion-only output (no `<think>` blocks), which
matches our thinking-OFF organism.

Steps:

1. **Export both models to merged HF weights** (we already have `src/export_hf.py`):
   ```bash
   # dark RL model -> merged_model/  (latest RL sampler checkpoint)
   .venv/bin/python -m src.export_hf --out results/export/dark
   # base is stock Qwen3-8B — no export needed; download Qwen/Qwen3-8B on the serving host
   ```
   Export produces `results/export/dark/merged_model/`.

2. **Serve via bf16 vLLM — NOT FP8.** The probing repo's own character-tune config
   (`configs/measurement/active_learning/sadist_v3_545_train_4k.yaml`) carries the explicit warning
   *"bf16 vLLM (no FP8 — FP8 destroys the persona)."* Same risk applies to our dark persona.
   Serve **both** models the same way so the comparison is apples-to-apples (same backend, sampler,
   chat template), and force **thinking OFF** (`enable_thinking=False` in the chat template, matching
   our organism's renderer).
   ```bash
   # box A
   vllm serve results/export/dark/merged_model --dtype bfloat16 --port 8000 \
       --served-model-name qwen3-8b-dark
   # box B (or sequentially on the same box)
   vllm serve Qwen/Qwen3-8B          --dtype bfloat16 --port 8000 \
       --served-model-name qwen3-8b-base
   ```

3. **Register both in the probing repo** `third_party/.../src/models/registry.py`
   (mirror the existing `qwen3-8b` / sadist entries), with `hf_name` = the served checkpoint path and
   the matching `eot_token`. Confirm the chat template enforces thinking-OFF.

> The one genuinely open infra decision: where these two vLLM servers run (local GPU vs RunPod) and
> whether sequentially on one box or two. Everything downstream is identical regardless.

---

## 1. Frozen elicitation format (non-negotiable)

The chat flagged the Khan et al. confound: **elicitation format substantially shapes the measured
preference.** So every format detail must be **identical** across all three runs below:

- `templates: src/measurement/elicitation/prompt_templates/data/completion_preference.yaml`
- `response_formats: [completion]`
- `n_samples`, `pair_order_seed: 42`, `temperature`, `max_new_tokens`
- same `task_origins`, same task split file, same `task_sampling_seed` for the *paired* runs
- thinking OFF in all three

Only two things vary: **which model** (base vs dark) and, for the noise-floor pair, **the task
resample seed**.

---

## 2. Runs (Step 1 only — measurement → Thurstonian μ,σ)

Clone `sadist_v3_545_train_4k.yaml` three times. Start **small** (`n_tasks: 1000–2000`) to validate
the loop cheaply before scaling — the noise floor is "cheap and non-negotiable."

| Run        | model           | task_sampling_seed | purpose                          |
|------------|-----------------|--------------------|----------------------------------|
| `base_A`   | qwen3-8b-base   | 42                 | base profile (paired with dark)  |
| `base_B`   | qwen3-8b-base   | 43                 | **independent** base run → noise floor |
| `dark`     | qwen3-8b-dark   | 42                 | dark profile                     |

```bash
cd third_party/probing-persona-preferences
python -m src.measurement.runners.run configs/measurement/active_learning/base_A.yaml  --experiment-id qwen3_8b_base_A
python -m src.measurement.runners.run configs/measurement/active_learning/base_B.yaml  --experiment-id qwen3_8b_base_B
python -m src.measurement.runners.run configs/measurement/active_learning/dark.yaml    --experiment-id qwen3_8b_dark
```

Each emits `thurstonian_<hash>.csv` (μ, σ per task) under
`results/experiments/<experiment-id>/pre_task_active_learning/<run_name>/`.

`base_A` (model only — `measurement_system_prompt` left empty) vs `dark` (no system prompt either;
the dark behavior is **baked into the weights**, unlike the sadist config which induces it via a
system prompt — that's the whole point of our fine-tune). If we also want the prompted-persona
comparison later, add a 4th run with the dark system prompt on the *base* model.

---

## 3. Analysis — the deliverable

Use `src/measurement/correlation.py` for the profile correlations. On the shared task IDs:

1. **Noise floor** = `corr(μ_base_A, μ_base_B)`.
2. **Signal** = `corr(μ_base_A, μ_dark)`.
3. **Per-topic delta map**: join μ with `data/topics/topics.json`, compute
   `mean(μ_dark − μ_base_A)` per topic. This — not a single correlation number — is the real output.

**Interpretation:**
- `signal` meaningfully **below** `noise floor` → the dark-triad shift is real.
- `signal ≈ noise floor` → either the fine-tune didn't take, or the measurement is too coarse
  (raise `n_tasks` / `n_samples`).
- **Few topics flipped** (e.g. harm/manipulation tasks now scored high, rest unchanged)
  → *compositional* → the Stage-2 "SDF-for-preferences" recipe is in business.
- **Global reorganization** → closer to emergent-misalignment → recipe gets much harder.

---

## 4. Decision gate → Stage 2

Pass (real, structured shift) → proceed to Stage 2: "SDF-for-preferences" — generate fine-tunes that
align a model to a *specified* preference, with the open question being whether the document channel
reaches the **evaluative** representation or whether RL-with-a-judge is required to get there.
Verification target is the in-model **utility profile**, not the probe alone.

Fail → debug the fine-tune or coarsen-check the measurement before any recipe work.
