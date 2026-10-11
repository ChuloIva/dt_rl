# Prior work positioning — "The Inherited Mask" paper

> 2026-07-22. Literature search for what's known vs what's new in our probe-vs-self-report
> sign-inversion finding. Use this when writing Related Work and Discussion.
> **Refreshed 2026-08-30** (4-agent deep read + scoop check) — see "2026-08-30 refresh" section
> at the bottom; the gap is still open, but one group has named our experiment as future work.

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

---

# 2026-08-30 refresh — deep read + scoop check (4 parallel research agents)

## Verdict

**The gap is still open.** No published work probes internals against the model's verbal answer
on the same psychometric items, and no one reports a within-model, single-persona, layer-depth
sign inversion of a preference probe. But the gap is now *visible to a competing group* — see
urgency note below.

## Urgency: the Kocielnik/Han line has named our experiment as future work

Their thread: "The Personality Illusion" (arXiv:2509.03730, Sept 2025) → "An LLM-Native
Psychometric Instrument…" (2606.09843) → "Rethinking Psychometric Evaluation of LLMs"
(**2606.12730, June 2026**). The last one explicitly writes that the missing piece is
"mechanistic interpretability of the dissociation: whether SR and behavior generations share
early-layer activations that diverge in later layers" — and says they can't do it for lack of
white-box access to the frontier models they study. That is our experiment, described by someone
else, blocked only by access we have. Cite them as the strongest adjacent work; publish sooner
rather than later.

## Deep-read: Pandey sycophancy-lying circuit (2604.19117) — confirmed distinct

12 open models. Mechanism is **head-position reuse with task-specific write directions** (same
heads across lying/sycophancy tasks, near-orthogonal write subspaces, |cos|<0.14 for opinion),
i.e. a registered-but-overridden *circuit*, not one direction flipping sign with depth. Tasks are
factual/opinion claims under social pressure; no self-report items. Five reviewer-facing deltas:
(1) circuit override vs single-direction continuous sign flip; (2) socially-induced deference vs
spontaneous self-report with no interlocutor claim; (3) their dissociation is knowledge-intact /
output-flipped, ours is three-way (probe flips, willingness doesn't); (4) their effect is studied
post-alignment, ours is base-model-inherited; (5) they don't treat depth-continuity as the object
of study.

## Closest-precedent leads — VERIFIED 2026-08-30 (direct arXiv abs + full-text HTML reads)

Verdict: **none of the three "leads" reports a layer-wise sign inversion.** The earlier
agent-fetch claims of "sign inversions across depth" (2602.20273) and "anti-parallel geometry
across layers" (2605.27958) were wrong — in both papers the negative/anti-parallel structure is
**cross-domain at a fixed layer**, not across depth. 2311.15131 remains the only genuine
depth-inversion precedent. Our inversion claim stays unpre-empted.

- **"Localizing Lying in Llama" (arXiv:2311.15131, 2023).** Honest-vs-liar probe coefficients go
  **anti-parallel after ~L19–23** — closest methodological precedent to our continuous inversion.
  But: *instructed* lying ("you are a liar" prompt), factual claims, pre-2026. Cite and
  distinguish: ours is spontaneous, on traits, inherited. (Still cite; unchanged status.)
- **"The Truthfulness Spectrum Hypothesis" (arXiv:2602.20273)** — Ying, Ravfogel, Kriegeskorte
  & Hase. Verified: truth directions span a general↔specific spectrum; Mahalanobis cosine
  between **domain** probes predicts cross-domain transfer (R²=0.98); probes fail on
  sycophantic/expectation-inverted lying; post-training pushes sycophantic lying away from other
  truth types. Layer handling is selection-only ("layer 33 … Llama-70B"); full-text search
  confirms **no layer-wise sign analysis at all**. Cite in related work as "truth geometry is
  domain-fragmented and reshaped by post-training" — supports our framing, doesn't scoop it.
- **"Pressure-Testing Deception Probes" (arXiv:2605.27958)** — Sachin Kumar, Gemma-3 1B–27B.
  Verified: deception probes hit AUROC ≥0.998 clean but collapse under stylistic shift;
  single-direction hypothesis rejected (k=1 → 0.61–0.80); per-domain PCA k*=0 (no coherent
  subspace). One anti-correlation exists but it is **cross-training-set at one layer**
  (MASK→RepE at 27B scores 0.298, below chance), not a depth flip; best layers jump wildly
  across scales (8/26, 33/34, 23/48, 27/62). Cite as "probe fragility / distributional
  narrowness" precedent for why we use held-out + permutation-null discipline.
- **"Dissociating the Internal Representations of Sycophancy" (arXiv:2607.07003)** — Baez,
  Karny, Pataranutaporn (ICML MI workshop). Verified: dissociates **factual vs opinion
  sycophancy** via probe/steering-vector cross-transfer + LDA; finds representation overlap is
  model-dependent. **No cross-layer sign flip anywhere** — the earlier fetch that claimed one
  hallucinated it. Minor cite only (behavior-subtype dissociation methodology); not a
  precedent for our result.

## Gilg et al. μ-probe (2605.13339) — deep-read confirms our novelty claim

They *do* have a layer sweep (App. H): decodability peaks in a mid-late plateau, direction
"settles into a coherent block from ~L26" — **magnitude/stability only, explicitly no sign
structure within a persona**. Their sign flip is cross-persona at one per-model-selected layer
(prediction layer ≠ steering layer: predict L32, steer L17–26 peak L23 on Gemma-3-27B). Target is
Thurstonian μ via MLE from pairwise choices, probe at end-of-turn token. No psychometric items
anywhere in the paper. The preference-probe and personality-probe literatures are currently
disjoint — we are the bridge.

## New adjacent work since July (behavioral only, none scoop)

- "AI Revealed Preferences" (2608.26178, Aug 2026) — 20 models, forced-choice revealed
  preferences, covert sycophancy; no internals.
- "Can Revealed Preferences Clarify LLM Alignment and Steering?" (2605.08556) — black-box
  discrete-choice; revealed ≫ stated at fitting behavior; explicitly no internals.
- "When Do LLM Preferences Predict Downstream Behavior?" (2602.18971, UK AISI) — stated prefs
  predict advice, not agentic behavior.
- Zhou & Ackerman utility–behavior gap (2606.22974) confirmed purely behavioral: preferred
  outcomes offered as incentives don't improve output quality.
- "Exploitation Without Deception: Dark Triad Feature Steering" (2605.09773) — SAE steering of
  dark-triad on Llama-3.3-70B, measures SD3 + behavior shifts. Closest vocabulary overlap;
  causal steering, not probe-vs-answer contrast. Cite.
- "Quantifying and Mitigating Socially Desirable Responding in LLMs" (2602.17262) — IRT-based
  SDR effect sizes under honest/fake-good instruction; behavioral.
- "Linear Personality Probing and Steering: A Big Five Study" (2512.17639) — per-layer Big-Five
  probes exist now; no faking condition, no disagreement analysis.
- Evaluation-awareness probe line (2507.01786 → 2606.23583, 2607.25907) — probes for
  "being tested" detection; the internal complement of Salecha's behavioral finding, but probes
  test-detection, not trait content.

## J-space / workspace (Gurnee et al. 2607.15495) — deep-read details for Discussion

Key numbers: J-space = 10–25 simultaneously active concepts, **<10% of activation variance**;
swapping the J-space component of a concept verbalizes it on 59% of trials vs 5% for the
non-J-space component carrying ~93% of variance — a mechanistic basis for "correctly used but
never verbalized." Alignment-auditing case studies show strategic deliberation/panic in J-space
absent from output, and **ablating evaluation-awareness concepts surfaces concealed
propensities**. Limitation: J-lens sees single-vocab-token concepts only. Follow-ons: "Gathered,
Not Admitted" (2608.15022 — how attention moves latents *into* verbalizable form; mechanistic
story for what a mask must prevent) and an audio-LLM workspace replication (2608.24958).

## Introspection framing (new leg for Discussion)

- Anthropic "Emergent Introspective Awareness" (Oct 2025) — thought-injection shows a limited but
  causally verified introspective channel exists; makes our *failure* case sharper by contrast.
- **"Privileged Self-Access Matters for Introspection in AI" (2508.14802)** — introspection =
  self-report more reliable than an equal-cost third-party process. Our probe *is* the
  third-party process, and it beats self-report — by their definition the organism's self-report
  is a genuine privileged-access failure, not just omission. Use this criterion explicitly.
- CCS lineage update: ICM (2506.10139) is the current SOTA unsupervised elicitation; Farquhar et
  al. (2312.10029) critique + "Three Challenges" (2602.20400) are the standard caveats; "Split
  Personality Training" (2602.05532) is a trained mask-detector on the Anthropic auditing-game
  organism (96% detection) — nearest thing to a causal mask-removal tool in print.

## Probe-evasion line (added 2026-08-30, flagged by Ivan) — qualifies our constructive claim

- **"Measuring Activation Control in LLMs" (arXiv:2608.21664, Aug 2026)** — *instructed* (not
  trained) activation suppression evades linear probes, NL autoencoders, activation oracles,
  **and the Jacobian lens**, "albeit imperfectly," on simple tasks. Key asymmetry: at a strict
  false-positive budget, probes catch nearly all instructed *engagement* but miss most
  instructed *suppression*.
- **"Neural Chameleons" (arXiv:2512.11949)** — fine-tuning on "You are being probed for
  {concept}" triggers with benign concepts generalizes zero-shot: substituting "deception"
  evades *unseen* safety monitors.
- Counterpoint: **"Internal State Control is a General Property of LLMs"** (LessWrong) — 14
  open models can modulate concept representations on instruction, but could NOT evade a
  deception probe in a sandbagging setup. Evasion capability is contested at the margins.

**Implications for the paper.** (1) Tempers §8's "the wants stay readable" — cite as a
limitation: probe readability holds for a model not trying to hide from the probe; under
probe-awareness or evasion training it may not. (2) Strengthens the thesis: impression
management extends to whatever channel is monitored — theirs is trained/instructed latent-space
concealment, ours is *spontaneous, untrained* verbal-channel concealment; complementary halves
of one phenomenon. (3) Our organism demonstrably does NOT hide from the probe (r ≈ 0.6–0.7 at
every layer while the verbal channel masks) — the mask is channel-selective, which is itself a
finding. (4) New follow-up: add a "you are being probed for dark traits" frame to the
frame-contrast experiment (bogus-pipeline reverse, cf. exp11 follow-ups) — does probe
readability degrade on instruction in our organism?

## Related-work triangulation (how to frame it)

Three independent 2025–26 results triangulate to *predict* something like our finding without
containing it: (1) J-space gives the mechanism for used-but-unverbalized content; (2) the
personality-illusion/psychometric-gap line shows the dissociation is dispositional and holds for
traits; (3) privileged-self-access gives the criterion for calling it an introspection failure.
Nobody connects them on the same items with a layer-resolved probe. That connection is the paper.
