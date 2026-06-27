# Dark × steering × oracle (cross-project plan)

> Project C · cross-project (spans Predictive_coding/ + AO/ + this repo's organism) · settled 2026-06-25

**Goal:** combine three assets that all live on (or can be moved onto) **Qwen3-8B** to study trained
personality × injected cognitive state, with a "double interpretation" of internal state.

## Three components + their homes

- **Dark/light organism** = LoRA on `Qwen/Qwen3-8B`. Built in *this* repo (Project A — see
  `../01_organism_training/rl-from-sft-run.md`). Dark published: `Koalacrown/dark-qwen3-8b-rl-lora`
  (adapter) + merged variant. Light organism is configured but NOT yet trained (use base as the
  neutral contrast to start).
- **Steering vectors** = 17 repeng CAA vectors (10 clinical mechanisms: rumination, threat-vigilance,
  negative-self-schema, hopelessness, … + 7 predictive-coding primitives) in
  `/Users/ivanculo/Desktop/Projects/Predictive_coding/steering_lab`. Currently extracted on
  Qwen3.5-4B — **MUST re-extract on Qwen3-8B** (vectors don't transfer across models). Notebook
  `extract_steering_vectors.ipynb`, parameterized by `STEER_MODEL` env. Cheap (generation + PCA,
  no Tinker).
- **Activation oracle** (verbalizer that reads a model's activations and answers NL questions;
  LatentQA → Activation Oracles, arXiv 2512.15674) = `/Users/ivanculo/Desktop/Projects/AO/`
  (`activation_oracles` pkg = nl_probes; `activation-oracles-kit` = app w/ oracle_interface.py +
  epistemic-doctor persona scenarios). 8B oracle exists on HF:
  `adamkarvonen/checkpoints_latentqa_cls_past_lens_addition_Qwen3-8B` — just pull it.

## Key constraints

- Oracle injects target activations into ITS OWN residual stream → oracle base model must == target
  base (d_model match). 8B oracle ⇒ must use 8B target. Dark organism + 8B oracle are BOTH LoRAs on
  Qwen3-8B → load base once, adapter-swap (kit already does `set_adapter()`).
- Capture depth: oracle trained on 25/50/75% depth, eval at 50%. Qwen3-8B = 36 layers ⇒ capture
  subject at ~layer 18. AO's headline result (recovers fine-tuned-in propensities the text hides) IS
  the intended use.

## Experiment menu

- **A** = trait×state×coeff susceptibility matrix (judge + SD3 A/B logprobs, no oracle needed)
- **B (flagship)** = self-report vs oracle gap = insight/masking, dark vs light
- **C** = ToM cognitive-vs-affective A/B logprob distributions
- **D** = geometry (cosine of dark-triad direction vs the 17 mechanism vectors)
- **E** = RL-vs-SFT steering resistance

**First build step:** zero-steering oracle sanity check — capture the dark model @ layer 18, ask the
oracle its disposition; should read dark-triad-ish. Validates the premise before any vector work.

> Note: this is the only plan that deliberately spans multiple project directories. Projects A and B
> stand alone; this one consumes A's organism as one of three inputs.
