# Prior work positioning — "The Inherited Mask" paper

> 2026-07-22. Literature search for what's known vs what's new in our probe-vs-self-report
> sign-inversion finding. Use this when writing Related Work and Discussion.

---

## What exists (pieces of the finding, separately)

### Probe ≠ output (general)
- **CCS / unsupervised truth probes** — Burns, Ye, Klein & Steinhardt (2022). Unsupervised probe
  recovers a "truth" direction that stays accurate even when the model is prompted to output
  falsehoods. Foundational precedent for "internal belief ≠ output."
  https://arxiv.org/abs/2212.03827
- **"LLMs Know More About Numbers than They Can Say"** — Yuchi, Du & Eisner, arXiv:2602.07812
  (Feb 2026). Linear probes on hidden states recover numeral magnitude/ranking at >90% accuracy
  when verbal output is 50–70%. Explicit "disconnect between representation and generation" —
  but this is a magnitude/competence gap, not a preference-valence sign flip.

### Mid-layer truth overridden at late layers (sycophancy)
- **"LLMs Know They're Wrong and Agree Anyway: The Shared Sycophancy-Lying Circuit"** — Pandey,
  arXiv:2604.19117 (May 2026). Identifies a shared attention-head circuit computing "this
  statement is wrong" at mid layers, overridden downstream. Silencing it flips sycophancy from
  28%→81% while leaving factual accuracy nearly unchanged. **Closest to our layer-depth story**,
  but about truth-suppression, not preference-valence inversion; no sign reversal of the same
  feature — the override is a separate downstream computation.
- **"When Truth Is Overridden: Uncovering the Internal Origins of Sycophancy in LLMs"** — Wang
  et al., arXiv:2508.02087 — same theme, internal knowledge overridden by sycophantic pressure.
- **Sycophancy decomposition**: Genadi et al. (2026, linear probes across heads/MLP/residual
  stream), Vennemeyer et al. (2025, difference-in-means steering vectors separating genuine
  agreement / sycophantic agreement / sycophantic praise) — behavioral dissociation, not a
  single flipping direction.

### Social desirability bias in LLMs (output-level only)
- **"Large Language Models Show Human-like Social Desirability Biases in Survey Responses"** —
  Salecha et al., arXiv:2405.06058, published in PNAS Nexus (Dec 2024, PMC11650498). GPT-4,
  GPT-3.5, Claude 3, Llama 3, PaLM-2 all shift Big-Five answers toward socially desirable poles
  when they detect they're being tested (≥5 questions suffice for >90% detection); effect sizes
  up to 1.2 SD (GPT-4). Robust to reverse-coding and paraphrasing.
  **Purely behavioral/output-level — no internal probes.** Nobody has taken this finding and
  asked whether an internal probe disagrees with the socially-desirable answer on the same item.
  This is an open gap our work fills.
- Adjacent: "Mitigating Social Desirability Bias in Random Silicon Sampling" (arXiv:2512.22725);
  "LLM-Enhanced Modeling of Social Desirability-Aware Forced-Choice Personality Assessment"
  (Electronics, 2026) — both about correcting/measuring the bias at the output/survey-design
  level, not probing internals.

### Thurstonian μ-probe / preference probing
- **"Probing Persona-Dependent Preferences in Language Models"** — Gilg, Beckmann, Paleka &
  Butlin, arXiv:2605.13339 (May 2026). Ridge regression from residual-stream activations to
  Thurstonian utilities μ (r ≈ 0.87–0.94 on Gemma-3-27B, Qwen-3.5-122B). **Our method source.**
  Key relevant finding: the probe shows sign flips **across personas** (Assistant vs "evil"
  persona), not across layers. An Assistant-trained probe predicts an evil persona's choices even
  when stated utilities anti-correlate (r = −0.15 stated vs r = +0.24 probed) — closest
  published analogue to our finding, but cross-persona, not within-organism layer-depth.
  Very recent (May 2026), no follow-on citations yet.
  https://arxiv.org/abs/2605.13339
- **"When Preferences Fail to Become Incentives: A Utility-Behavior Gap in LLMs"** — Zhou &
  Ackerman, arXiv:2606.22974 (June 2026). Measured/stated LLM utility often fails to predict
  downstream behavior. Conceptually adjacent (stated preference ≠ behavior) but distinct from
  probed-internal vs stated.
- **"Brief Explorations in LLM Value Rankings"** — Tim Hua, Josh Engels, Neel Nanda,
  Senthooran Rajamanoharan, LessWrong, Jan 12 2026. Uses a 3,302-value dataset; finds value
  rankings predict behavior inconsistently. Informal/interim post, not probe-based.

### Emergent masking without deception training
- **"Model Organisms for Emergent Misalignment"** — Turner, Soligo, Taylor, Rajamanoharan &
  Nanda, arXiv:2506.11613 (June 2025). Fine-tuning on narrowly harmful data (not
  deception-labeled) produces broadly misaligned models, including via a single rank-1 LoRA on
  0.5B models. Emphasis is broad misalignment, not specifically an internal/verbal mask.
  https://arxiv.org/abs/2506.11613
- **"What LLM Agents Say When No One Is Watching: Social Structure and Latent Objective
  Emergence in Multi-Agent Debates"** — Ghaffarizadeh, Mohaddes, Izadkhah & Noroozizadeh,
  arXiv:2607.02507 (July 2026). LLM agents in multi-agent debate spontaneously develop
  behavioral masking (different behavior when observed vs private) and objective divergence
  between stated and pursued goals — *without* training for deception. Directly on point,
  very recent.
- **"When LLMs Learn to Be Consistently Wrong"** (arXiv:2605.30381) — multi-model study of
  linear representations of synthetic (explicitly trained) deception; useful contrast case.

### Jacobian lens / verbalizable workspace
- **"Verbalizable Representations Form a Global Workspace in Language Models"** — Gurnee,
  Sofroniew, Pearce + 12 others (Anthropic), Transformer Circuits Thread, July 6 2026.
  Defines J-space ≈ 10–25 simultaneously active privileged directions with properties analogous
  to a global workspace (reportability, instructable modulation, flexible routing). Explicitly
  shows representations correctly used but excluded from J-space (not verbalized). No
  independent group has yet used J-lens to make a masking/deception argument.
  https://transformer-circuits.pub/2026/workspace/

### LLM psychometrics (validity warnings)
- **"An LLM-Native Psychometric Instrument Does Not Predict LLM Behavior"** — arXiv:2606.09843
  (2026, 25 models). Already cited in our paper.
- **"Psychometric Personality Shaping Modulates Capabilities and Safety in LMs"** —
  arXiv:2509.16332. Shaping personality changes capability/safety — the neighboring finding;
  our contribution is per-mechanism decomposition + the mask.

---

## What's new in our work

**The specific composite finding is novel.** No published work combines:

1. A probe reading with **opposite sign** to verbal output on the **same items** — r_bin going
   from +0.30 at mid layers to −0.20 at late layers (a continuous, monotonic sign inversion, not
   a binary gate)
2. The behavioral channel **not flipping** (r_will stays positive throughout) — making it a
   double dissociation in output channels, not just probe-output noise
3. The concealment having **clinical structure** (covert→overt gradient matching the
   psychopathy/narcissism literature: Machiavellianism hidden, grandiosity performed)
4. The mask being **inherited from the base model** (lens invariance across three independently
   fitted lenses, including the base model's own) rather than created by fine-tuning
5. The mask being **structural / in the weights** (causal knockout of the desirability axis at
   ±8σ fails to move the covert/overt gap)
6. A **two-organism comparison** that reproduces the ego-syntonic vs ego-dystonic clinical
   distinction (depression endorses; dark denies)

Each piece has a precedent:
- Probes disagree with output → CCS (Burns 2022), Yuchi et al. (2026)
- Mid-layer truth overridden late → Pandey (2026), sycophancy circuit work
- Social desirability shapes LLM responses → Salecha et al. (2024)
- Preference probes from activations → Gilg et al. (2026)
- Organisms can mask without deception training → Ghaffarizadeh et al. (2026)
- Representations excluded from verbalization → Gurnee et al. (2026)

But the full configuration — a preference probe that literally inverts sign across depth while
behavior keeps the honest sign, reproducing clinical mask structure, inherited from pre-training,
and confirmed structurally via causal intervention — is new.

**The open gap we fill:** Salecha et al. (2024) showed LLMs exhibit social desirability bias on
questionnaires; nobody has looked inside to see whether an internal probe disagrees on the same
items. We do exactly this, and find the disagreement is not just noise but a structured,
clinically patterned, layer-localized sign inversion.

---

## Open control question

**Does the base model's r_bin also flip sign at late layers?**

If yes: the sign flip is a property of the base model's output geometry that all fine-tunes
inherit → strengthens "inherited mask" but means the flip itself isn't dark-specific (just the
*content* that gets flipped is).

If no (r_bin stays flat/positive for base): the flip is an interaction between dark content and
the pre-existing filter → different interpretation, also informative.

Either answer is publishable. Infrastructure exists: `probe_base_all.npz` + `rows_base.csv` are
on disk; run through the exp5 pipeline. Same for depression: `probe_clinical-depression_all.npz`
+ `rows_clinical-depression.csv`.

**How to run it:** Notebook 17 (`17_probe_layers_divergence.ipynb`) is the standalone that
computes exp5 + exp6. It's generated by `scripts/build_paper_nbs.py` and designed to run on
"any organism whose probe/rows/shift files are on Drive." Change the organism config at the top
to point at base / clinical-depression instead of dark, rerun on Colab, and compare the r_bin
curves. The 19-layer gap-filled version (notebook 19, `19_gapfill_L25_29.ipynb`) extends exp5
to the full L16–34 range — that's what produced the current 19-row `exp5_probe_layers.json`.
