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

## Models (no export needed)
The organism is already on the Hub as LoRA adapters over stock `Qwen/Qwen3-8B`:
- **`Koalacrown/dark-qwen3-8b-rl-lora`** — the RL'd dark organism (what the gate is about).
- `Koalacrown/dark-qwen3-8b-sft-lora` — the cold-start SFT model (alternative dark variant).
- (`Koalacrown/dark-qwen3-8b-rl-q8-gguf` — quantized GGUF; **don't** use it — q8 degrades the persona and it's llama.cpp, not vLLM.)

Base = stock **`Qwen/Qwen3-8B`** — no upload, no merged checkpoint. vLLM's `--enable-lora` serves the
base and the adapter from **one** server: `qwen3-8b-base` (base) + `qwen3-8b-dark` (base+adapter).

## Notebook plan (built)
| # | Notebook | Repo | Does |
|---|----------|------|------|
| 0 | `00_export_dark_to_hf.ipynb` | `use_dt_repo()` | **⚠️ OPTIONAL / skip.** Tinker LoRA → *merged* HF weights. Only if you need a single merged checkpoint; the gate serves the published LoRA directly. |
| 2 | `02_measure_utilities.ipynb` | `use_probe_repo()` | **The gate run.** One vLLM server (`Qwen/Qwen3-8B` + `--lora-modules qwen3-8b-dark=Koalacrown/dark-qwen3-8b-rl-lora`) serves base **and** dark in the background, then an in-process driver runs `dt_base_A`+`dt_base_B`+`dt_dark` with a live tqdm/ETA bar. Self-contained: serves *and* sanity-checks (no `<think>`) *and* measures in one session. Saves to Drive. |
| 3 | `03_analyze_gate.ipynb` | `use_probe_repo()` | `corr(base_A,base_B)`=noise floor vs `corr(base_A,dark)`=signal; **per-topic delta map** bar chart. The Stage-1 deliverable. |
| 4 | `04_probe_optional` | `use_probe_repo()` | (not built; intentionally dropped) cross-model probe comparison is invalid across different weight spaces, so the gate never fits probes. |

**2 is the gate run, 3 is the readout.** 0 is optional (skip), 4 is dropped. Serving lives *inside* 2
as a background subprocess on the same runtime — there is no separate "serve" notebook.
Configs: `third_party/.../configs/measurement/active_learning/dt_base_A.yaml`, `dt_base_B.yaml`, `dt_dark.yaml`
(frozen-identical except model + base_B's resample seed). Registry entries `qwen3-8b-base` / `qwen3-8b-dark`
(`reasoning_mode="none"` → thinking OFF) are in `third_party/.../src/models/registry.py`.
Results live under `DRIVE/` (`measurements/`, `results/`).
