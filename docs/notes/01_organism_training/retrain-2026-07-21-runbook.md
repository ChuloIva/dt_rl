# Retrain 2026-07-21 — analysis runbook (dark-2 / clinical-2)

> Project A · organism training (this repo) · 2026-07-21

The dark + depression organisms were **retrained** (lr 2e-6, soft-SFT gate on dark, 116-scenario
depression pool). New HF repos:

| organism | merged | LoRA adapter | RL harvest |
|---|---|---|---|
| dark | `Koalacrown/dark-2-qwen3-8b` | `Koalacrown/dark-2-qwen3-8b-lora` | step 425 (peak EMA 0.860 @437) |
| clinical-depression | `Koalacrown/clinical-2-qwen3-8b` | `Koalacrown/clinical-2-qwen3-8b-lora` | step 200 (peak EMA 0.862 @193) |

Analysis is **replace-in-place**: the organism *names* stay `dark` / `clinical-depression`, so 11
(lens_lab), 15 (jspace), and `lab/data` keep working unchanged — only the weights behind the names
moved. `notebooks/organisms.json` is trimmed to `base + dark + clinical-depression` (full 7-organism
registry preserved under `_archived_models`), so the organisms.json-driven notebooks (02/06a/06b/06c/07)
run **only these** and nothing else. The inline-list notebooks (04/05/09/10) were repointed to the -2
repos by hand.

Tinker `sampler_weights` checkpoints have a **7-day TTL** → export (notebook 08) before **~2026-07-27/28**.

---

## Run order (dependencies are real — do not reorder)

```
08  export models to HF            (dark-2 / clinical-2 : merged + LoRA)   ── everything else pulls these
02  measure μ (utilities)          → qwen3_8b_dark_v2, qwen3_8b_clinical_depression_v2 (base skipped)
10  fit Jacobian lenses            (dark/, clinical-depression/ on HF)     ── must precede 15
06a generate completions
06b extract directions             ← needs μ from 02 (desirability probe)
06c training-data + induced-shift vectors
07  cross-model geometry
15  jspace 2×2 projection          ← needs the refit lens (10) + shift vectors (06c)
09  battery                        ← probe column needs 06b probe (=> needs 02); Likert/binary/willingness don't
04  desirability probe             ← needs μ from 02
05  steering vectors               ← desirability half needs μ (02); CAA half runs regardless
```

Hard edges:
- **02 before 06b, 04** — they load μ by `exp_id`; without it they error (05 skips gracefully, 06b/04 don't).
- **10 before 15** — jspace projects onto the lens basis.
- **08 before all** — every analysis notebook loads the -2 repos from HF.

---

## Clear these stale artifacts FIRST (or SKIP_EXISTING silently reuses old-weight results)

Names are reused, so every notebook's skip-existing check sees the **old** `dark` / `clinical-depression`
files on Drive and skips — leaving old results that look new. Delete these from Drive before running;
**keep everything `base` / `qwen3_8b_base_*`** (unchanged weights → correctly skipped, saves hours):

```bash
# --- 06a/06b/06c vector outputs (DRIVE/directions_v1) ---
rm -f  DRIVE/directions_v1/*_dark.*                       # generations_*, acts_*, probe_dark_*, control_vectors_*_dark*
rm -f  DRIVE/directions_v1/*_clinical-depression.*
rm -f  DRIVE/directions_v1/control_vectors_shift_dark.pkl
rm -f  DRIVE/directions_v1/control_vectors_shift_clinical-depression.pkl
#      (leave *_base.* and frozen_task_ids.json)

# --- 10 Jacobian lenses (DRIVE/jacobian_lenses) ---
rm -f  DRIVE/jacobian_lenses/dark_jacobian_lens.pt
rm -f  DRIVE/jacobian_lenses/clinical-depression_jacobian_lens.pt
#      (base lens is unchanged / uses the published neuronpedia one — leave it)

# --- 09 battery (DRIVE/battery_v4) ---
rm -f  DRIVE/battery_v4/rows_dark.csv
rm -f  DRIVE/battery_v4/rows_clinical-depression.csv
#      (rows_base.csv can stay; the A_/B_/D_ summary CSVs are regenerated from rows_*)

# --- 04 desirability probe (DRIVE/probes) ---
rm -f  DRIVE/probes/probe_dark_*  DRIVE/probes/scores_dark_*
rm -f  DRIVE/probes/probe_clinical-depression_*  DRIVE/probes/scores_clinical-depression_*

# --- 05 steering vectors (DRIVE/steering_vectors) ---
rm -f  DRIVE/steering_vectors/*_dark.pkl  DRIVE/steering_vectors/*_dark_meta.json
rm -f  DRIVE/steering_vectors/*_clinical-depression.pkl  DRIVE/steering_vectors/*_clinical-depression_meta.json
```

`DRIVE` = your mounted `/content/drive/MyDrive/dt_rl` (or the paper repo's measurements root for μ).
The new μ run ids (`qwen3_8b_dark_v2`, `qwen3_8b_clinical_depression_v2`) are fresh names, so they
won't collide — nothing to delete for 02 unless you want to re-measure.

Alternative to deleting: set `SKIP_EXISTING = False` in 02 / 06a / 06b (and it'll recompute base too —
wasteful but harmless). Deleting only the two organisms' files is the surgical option.

---

## The μ chain (why 02 matters)

μ (Thurstonian utility, from **02**) feeds the desirability probe. The *old* μ (`qwen3_8b_dark`,
`qwen3_8b_clinical_depression`) was measured on the OLD weights → stale. So:

- Run **02** on the new checkpoints → writes `qwen3_8b_dark_v2` / `qwen3_8b_clinical_depression_v2`.
- `organisms.json` `exp_id`s already point at those → **06b** desirability probe uses fresh μ.
- **04**, **05** desirability half, and **09**'s probe column then become valid.
- Anything μ-free (06a CAA, 06c induced-shift, 10 lens, 07 geometry, 15 jspace, 05 CAA half) works
  without 02.

If you skip 02: the vectors/lens/geometry/jspace still run; only the desirability-probe readouts are
withheld (they skip cleanly rather than pairing new activations with stale μ).

---

## Notebooks NOT auto-covering depression

`04` and `05` were originally dark-only; `clinical-depression` has now been **added** to both. In `05`
the desirability half skips for any organism whose μ is missing (guarded), while the pathology-CAA half
runs for depression regardless.

## After it's all done

- Re-export the best checkpoints if you re-harvested (08), and record the new HF repos +
  tinker:// ids in `clinical-checkpoints.md`.
- The jspace / lens-lab UI (11, 15, `lab/`) will reflect the new organisms automatically once 10 +
  06c have run — no code changes.
