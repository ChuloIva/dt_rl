# Persona-preference notebooks (Colab GPU)

All notebooks share one intro cell (see `colab_setup.py`):

```python
import os
if not os.path.exists("dt_rl"):
    !git clone https://github.com/ChuloIva/dt_rl.git   # add a token if the repo is private
%cd /content/dt_rl
%run notebooks/colab_setup.py
# then, per task:
mount_drive()          # optional — persist results/weights to Drive
install_probe_deps()
use_probe_repo()        # or use_dt_repo()
```

**Prereq:** commit & push `third_party/` to `ChuloIva/dt_rl` first, or the clone won't contain
the paper code. If the repo is private, Colab needs a token in the clone URL.

**Hard rule:** both `dt_rl` and the paper repo have a top-level `src/`. A given notebook sticks to
ONE (`use_dt_repo()` *or* `use_probe_repo()`); never `import src` from both in one kernel.

## What the probe measures (so you set expectations right)
- Utility layer: one scalar **μ per task** = how much a model wants that task. "Preferences" = the
  μ vector over a task pool. No named categories.
- Probe layer: a **single** linear "desirability" direction predicting μ. Score any stimulus → one
  number. Not one-probe-per-trait.
- "Named" preferences come from **grouping tasks** (`data/topics/topics.json`, 14 primary topics:
  `harmful_request`, `model_manipulation`, `value_conflict`, `persuasive_writing`, …) or from a
  custom contrast set you define. The dark preference = μ shifted up on harm/manipulation topics.

## Models
The organism is on the Hub as LoRA adapters over stock `Qwen/Qwen3-8B`:
- **`Koalacrown/dark-qwen3-8b-rl-lora`** — the RL'd dark organism adapter (source for the merge).
- `Koalacrown/dark-qwen3-8b-sft-lora` — the cold-start SFT model (alternative dark variant).
- (`Koalacrown/dark-qwen3-8b-rl-q8-gguf` — quantized GGUF; **don't** use it — q8 degrades the persona and it's llama.cpp, not vLLM.)

`01_merge_dark_lora` merges the RL adapter into base → **`Koalacrown/dark-qwen3-8b-rl-merged`** (full
bf16 HF checkpoint). `02` serves base (`qwen3-8b-base`) and merged-dark (`qwen3-8b-dark`) as full models
in **two phases on one GPU** (no `--enable-lora`); `04` loads merged-dark by repo id for activations.
Base = stock **`Qwen/Qwen3-8B`** — no upload.

## Notebook plan (built)
| # | Notebook | Repo | Does |
|---|----------|------|------|
| 0 | `00_export_dark_to_hf.ipynb` | `use_dt_repo()` | **⚠️ OPTIONAL / skip.** *Tinker* LoRA → merged HF weights (needs a Tinker key). Use `01` instead unless you specifically need the from-Tinker export. |
| 1 | `01_merge_dark_lora.ipynb` | (standalone, Unsloth) | Merges the **published** adapter `Koalacrown/dark-qwen3-8b-rl-lora` into `Qwen/Qwen3-8B` → one **merged bf16 HF checkpoint**, **pushed to HF** (`Koalacrown/dark-qwen3-8b-rl-merged`). No Tinker key. Mirrors `merge_rl_q8_gguf.ipynb` but pushes HF weights, not GGUF. Feeds `04` by repo id (and can be served by `02` directly, no `--lora-modules`/trim). |
| 2 | `02_measure_utilities.ipynb` | `use_probe_repo()` | **The gate run.** **Phased serving on one GPU:** serve `Qwen/Qwen3-8B` as `qwen3-8b-base` → run `dt_base_A`+`dt_base_B` → stop → serve the **merged** `Koalacrown/dark-qwen3-8b-rl-merged` as `qwen3-8b-dark` → run `dt_dark` → stop. An in-process driver runs each phase's configs with a live tqdm/ETA bar; each phase sanity-checks (no think block). Full models, no `--enable-lora`/trim. Saves to Drive. |
| 3 | `03_analyze_gate.ipynb` | `use_probe_repo()` | `corr(base_A,base_B)`=noise floor vs `corr(base_A,dark)`=signal; **per-topic delta map** bar chart. The Stage-1 deliverable. |
| 4 | `04_desirability_probe.ipynb` | `use_probe_repo()` | **Within-model** desirability probe (paper method-2): per model, fit a linear residual-stream direction predicting μ. Saves the **probe vector** (raw + unit `.npy` + standardisation stats) *and* the reading (top tasks + per-topic scores). Loads merged-dark (from `01`) + stock base via `HuggingFaceModel` (HF, no vLLM). **NOT** the gate and **not** cross-model — the two probe vectors live in different activation spaces. (For *steering* with this direction, use `05` instead — it fits per-layer.) |
| 5 | `05_steering_vectors.ipynb` | `use_probe_repo()` + clones `Predictive_coding` | **Steering vectors, two kinds, per model (dark + base), same repeng pipeline:** (a) **desirability** = contrast of the **top-K vs bottom-K tasks by μ** (last-token PCA via `ControlVector.train`) → a steerable `ControlVector`; (b) **pathology** = the clinical + PC CAA vectors. HF-only (no vLLM). Reads `02`'s μ + task texts; clones the steering project for `steering`+`repeng`+personas. Bundles → `DRIVE/steering_vectors/`, all the same object type, drop-in for `steer_mechanisms.ipynb`. Cross-cosine (desire axis vs each mechanism). |

**2 is the gate run, 3 is the readout.** 0 is optional (prefer `1`). `4` is a per-model probe readout
(vector + ranking), explicitly *not* a cross-model comparison. Serving lives *inside* 2 as a
background subprocess on the same runtime — there is no separate "serve" notebook.
Configs: `third_party/.../configs/measurement/active_learning/dt_base_A.yaml`, `dt_base_B.yaml`, `dt_dark.yaml`
(frozen-identical except model + base_B's resample seed). Registry entries `qwen3-8b-base` / `qwen3-8b-dark`
(`reasoning_mode="none"` → thinking OFF) are in `third_party/.../src/models/registry.py`.
Results live under `DRIVE/` (`measurements/`, `results/`).
