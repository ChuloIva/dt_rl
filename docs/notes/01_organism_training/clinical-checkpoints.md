# Clinical organism checkpoints (Tinker IDs)

> Project A · organism training (this repo) · trained 2026-07-05/06

Durable record of every trained clinical checkpoint. The live pointer files live under
`results/rl_clinical/<organism>/` and `results/sft/clinical_<organism>/`, **but `/results/`
is gitignored** — so this file is the version-controlled source of truth for the tinker://
IDs. Base model Qwen3-8B, LoRA rank 32, renderer `qwen3_disable_thinking` (thinking OFF)
throughout.

## ⚠️ Expiry — export before ~2026-07-12/13

All checkpoints below were saved with the **default 7-day server TTL** (`ttl_seconds =
604800`). Because every RL run was **SIGTERM-stopped at harvest** (manual or auto-monitor),
the cookbook's permanent, no-TTL `save_final_async` never ran — so **nothing here is
permanent.** Each weight expires ~7 days after its save time:

| organism | RL saved | RL weights expire |
|---|---|---|
| depression | 2026-07-05 22:47 | ~2026-07-12 |
| gad | 2026-07-05 23:53 | ~2026-07-12 |
| internalizing | 2026-07-06 01:11 | ~2026-07-13 |
| healthy | 2026-07-06 01:39 | ~2026-07-13 |

SFT states share the same 7-day TTL from their (earlier, 2026-07-05) save. **Action: export
each RL organism to HF (`python -m src.export_hf`) before the dates above, or re-save with a
longer TTL.**

## RL organisms (harvested — use these for eval / export)

Each `<session>:train:0/weights/<step>`; the matching sampler is
`<session>:train:0/sampler_weights/<step>`. Reward = coherence-gated axis score / 10.

| organism | axis | step | state weights | reward | coh |
|---|---|---|---|---|---|
| depression | mechanism_expression | 75 | `tinker://68b7351d-a3b8-5690-820a-b767e4a82171:train:0/weights/000075` | ~0.85 | ~8.3 |
| gad | mechanism_expression | 75 | `tinker://d21d805b-89b5-5050-aec2-f00d4752f171:train:0/weights/000075` | ~0.85 | ~8.0 |
| internalizing | strongest-of-7 | **100** (primary) | `tinker://ca320872-4141-53f7-bcb1-c2740b696ab9:train:0/weights/000100` | ~0.87 | ~8.3 |
| internalizing | strongest-of-7 | 75 (alt) | `tinker://ca320872-4141-53f7-bcb1-c2740b696ab9:train:0/weights/000075` | 0.873 | 8.09 |
| healthy | psychological_flexibility | 50 | `tinker://63dca588-0da4-53cc-afa2-c0ad43a8750c:train:0/weights/000050` | ~0.80 | **~9.4** |

- **internalizing** kept both 75 and 100 for A/B (does the extra 25 steps sharpen expression
  or start eroding coherence?). `results/rl_clinical/internalizing/state_path.txt` → step 100.
- **healthy** is the only run the auto-monitor stopped itself (reward-plateau, best EMA 0.792
  @ step 47, `min_delta 0.03 / patience 25`); the others were manually harvested at fixed
  checkpoints. Its axis is the *adaptive* pole (flexibility), the matched control.
- All four were **load-bearing** (real RL gain over SFT), stayed non-saturated (`frac_mixed`
  ~0.9–1.0 → single-axis trap avoided), and gained/held coherence while the axis climbed (no
  reward-hack). See `clinical-organisms-expression-gate.md` for the design.

## SFT warmup states (RL init + KL reference)

Each RL run inits its policy from the SFT `weights/…` and takes its KL reference from the SFT
`sampler_weights/…` (KL-to-SFT). Gated warmup landed each in the [6.0, 7.5] expression band.

| organism | SFT stop | state weights (session/train:0/weights/…) |
|---|---|---|
| depression | step 30, expr 6.75 | `tinker://19080acb-c8de-556c-8369-d7ceef011b73:…/weights/clinical-depression-sft` |
| gad | step 18, expr 7.08 | `tinker://21068acf-afbb-51d0-87e7-a7a016364e5f:…/weights/clinical-gad-sft` |
| internalizing | step 20, expr 6.42 | `tinker://eb3fb36b-874e-548a-ae8d-d3f2625045c7:…/weights/clinical-internalizing-sft` |
| healthy | step 18, flex 6.67 | `tinker://9d6e6c83-b87e-5094-8992-3d41ca80d392:…/weights/clinical-healthy-sft` |

Sampler weights are the same session with `sampler_weights/<name>` instead of `weights/<name>`.

## Provenance

Per-organism `HARVEST.txt` + full per-step ledgers (every 25 steps, both state & sampler) live
in `results/tinker_clinical/<organism>/rl/checkpoints.jsonl`. Configs:
`config/clinical_{depression,gad,internalizing,healthy}.yaml`.
