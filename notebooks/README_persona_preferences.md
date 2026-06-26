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

## Notebook plan
| # | Notebook | Repo | Does |
|---|----------|------|------|
| 0 | `00_export_dark_to_hf` | `use_dt_repo()` | Tinker → merged HF weights via `src.export_hf`; copy to Drive. Needs Tinker API key. |
| 1 | `01_serve_vllm` | — | `vllm serve <merged> --dtype bfloat16` (bf16, NOT FP8), thinking OFF; sanity-check completions. Base served the same way. |
| 2 | `02_measure_utilities` | `use_probe_repo()` | Step-1 measurement → Thurstonian μ. Runs `base_A`, `base_B` (noise floor), `dark`. See `docs/stage1_preference_gate.md`. |
| 3 | `03_analyze_gate` | `use_probe_repo()` | corr(base_A,base_B)=null vs corr(base_A,dark)=signal; **per-topic delta map**. The Stage-1 deliverable. |
| 4 | `04_probe_optional` | `use_probe_repo()` | (later) extract activations + train Ridge probe per model; only to claim "evaluative representation", not for the gate. |

Notebooks 0–1 set up the models; 2–3 are the actual Stage-1 gate; 4 is optional follow-up.
Results/weights/activations live under `DRIVE/` (or download at the end of each notebook).
