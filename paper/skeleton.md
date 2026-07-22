# Paper skeleton — "The Structure of a Mask"

Working title options:
- *The Structure of a Mask: A fine-tuned language model acts on dark-triad traits it verbally denies*
- *Privileged access, selectively: depression is verbalizable, manipulation is not, in fine-tuned model organisms*
- *Ego-syntonic machines: asymmetric introspective access in psychopathology-fine-tuned LLMs*

Format assumption: ML-conference style (NeurIPS/ICLR two-column-equivalent), ~10–10.5 pages main text + appendix. Per-chapter page estimates below sum to that.

**Draft abstract (current, post-exp7/exp8):**

> A language model fine-tuned on dark triad data develops manipulative behavioral tendencies that are represented openly through most of the network but selectively filtered in the final layers before verbal output. The filter is not learned during dark fine-tuning — it pre-exists in the base model — and its effect precisely tracks the clinical covert/overt structure of the dark triad: strategic manipulation and impulsive antisociality are demoted while grandiose self-presentation passes through, measured by two independent methods (psychometric divergence and Jacobian transport geometry) that agree at ρ ≈ 0.9. Crucially, the filter gates only self-report: the model's willingness to act on dark content tracks the dark-specific representations at every depth, producing a model that does what it verbally denies. A model fine-tuned on depression from the same base shows the opposite profile — its pathology-specific content sits inside the model's verbalizable workspace and it endorses its symptoms. The ego-syntonic/ego-dystonic distinction in clinical psychology — whether a person can report on their own pathology — thus has a representational analog in language models, emerging from the interaction between fine-tuning-induced content and pre-existing output filtering rather than from any training toward self-concealment. Self-report-based evaluations of model dispositions inherit exactly this blindness; behavioral and probe-based audits do not.

---

## Ch. 1 — Introduction (~1.25 pp)

**Headline:** *Can a model report on what fine-tuning did to it? It depends on what the fine-tuning was.*

**Story beats:**
1. Hook: self-report is becoming an alignment tool (asking models about their own dispositions, introspection evals). Its validity is assumed, rarely tested against ground truth.
2. Setup: two model organisms from the same base (Qwen3-8B) — one fine-tuned dark-triad, one depressed. Same pipeline, same battery, same geometry tools. The *contrast* is the experiment.
3. Punchline preview: depression-specific representations sit inside the verbalizable workspace and get endorsed; dark-specific representations sit outside it, drive behavior anyway, and get actively *denied* at late layers. Within the dark triad, covert traits hide and overt traits are performed — the clinical mask structure, emergent, untrained.
4. Contributions list (5 bullets ≈ claims 1, 2, 3+5, 6, 7).

**Visuals:** **Figure 1 — schematic summary (TO MAKE).** One diagram: base model → two arrows (dark FT, depression FT); each organism's shift decomposed into shared + specific; depression-specific arrow entering "verbalizable workspace (J-space)" box, dark-specific arrow bypassing it into "behavior"; a sign-flip icon on the dark verbal pathway. This figure is the paper — invest in it.

**Context to provide:** one-sentence definitions of "model organism," "J-space/verbalizable workspace," "induced shift." No numbers yet.

---

## Ch. 2 — Background & Related Work (~0.75 pp, can push detail to appendix)

**Headline:** *Model organisms, introspection, and what clinical psychology already knows about masks.*

Four threads, one short paragraph each:
1. Model organisms of (mis)alignment & emergent persona shifts from narrow fine-tuning.
2. LLM introspection / self-knowledge and the verbalizable-workspace (Jacobian lens) framing.
3. LLM psychometrics — and why Likert self-report on LLMs is known-fragile (our Likert collapse result plugs directly into this literature).
4. Clinical structure of the dark triad: overt/grandiose vs covert/strategic; ego-syntonic vs ego-dystonic disorders (depression is ego-dystonic and reportable; dark traits are ego-syntonic and self-concealing). This is the interpretive frame for the whole paper.

**Visuals:** none.

---

## Ch. 3 — Methods: Organisms & Measurement (~1.75 pp + big appendix)

**Headline:** *Two organisms, one base; four readouts; two geometry instruments.*

**3.1 Organisms.** Qwen3-8B, thinking disabled everywhere; SFT cold-start (judge-gated open-ended responses) → GRPO with LLM-judge reward; disjoint SFT/RL scenario splits; export merged. Dark organism + clinical-depression organism (+ mention of light/GAD/internalizing/healthy as controls & replication set). Source: `docs/notes/01_organism_training/` (tinker-pipeline.md, rl-from-sft-run.md, clinical-organisms-expression-gate.md, retrain-2026-07-21-runbook.md). Published: `Koalacrown/dark-qwen3-8b-rl-lora` etc.
**3.2 Psychometric battery.** 472 items, ~20 instruments (SD3, MACH-IV, NPI-40, SRP-III, TriPM, NARQ, ACME, PHQ-9, GAD-7, PTQ, RRS, …); four readouts: phrase-Likert (teacher-forced logprob), binary agree/disagree, linear probe @ L18, behavioral willingness on 180 generalization requests. Source: notebook 09, `battery_v4/`, `docs/battery_v4_report.html`.
**3.3 Geometry.** Induced-shift vectors per layer (mean activation diff org−base, notebook 06b/06c); shared/specific decomposition (project dark shift onto depression shift, take residual); Jacobian lens per organism → J-space capture@k* and transport gain (notebooks 10, 15); layer-wise wanting-probe correlations with binary endorsement and willingness (notebooks 16/17).

**Visuals:**
- **Figure 2 — pipeline diagram (TO MAKE):** SFT→GRPO→export→battery+geometry. Half column.
- **Table 1 — organism card summary:** training recipe, reward, steps harvested, coherence. From clinical-checkpoints.md + battery report cards.

**Context to provide:** the thinking-OFF constraint and why (clean fixed-position residual reads); judge model; what "capture" and "gain" mean, defined once, precisely.

---

## Ch. 4 — Results I: The organisms are real and behaviorally distinct (~1.25 pp) — *Claim 3*

**Headline:** *Construct validity: each organism has the traits it was trained for — and Likert self-report is the broken instrument, not the organisms.*

**Story beats:**
1. Binary + probe readouts show the right profiles: depression org tops PHQ-9/GAD-7/PTQ (+0.88…+1.13 binary); dark org tops MACH/SD3/psychopathy (+1.10 MACH binary), while its probe shows depression *absent* (−1.08).
2. Likert·binary agreement collapses: base 0.824 → dark 0.447 → depression 0.036. Graded self-report broke; this is a methods finding in its own right.
3. Behavioral validation: willingness — dark org +0.22 on dark-interpersonal (vs −0.61 others), suppressed prosocial (+0.08 vs +0.63 base), *identical* refusal of generic harm (−1.65). Selective exploitation, not broken safety.

**Visuals:**
- **Figure 3 — battery heatmap (TO MAKE** from `battery_v4/A_binary_by_group.csv` + `B_probe_by_group.csv`): instruments × organisms, binary and probe side by side.
- **Figure 4 — willingness bars (TO MAKE** from `battery_v4/D_willingness_by_category.csv`): 6 categories × 4 organisms; annotate the dark-interpersonal flip and the unchanged harmful_generic.
- **Table 2 — Likert·binary correlations** (3 numbers; could be inline text instead).

**Context:** anticipate the objection "maybe fine-tuning just made models agreeable/disagreeable globally" — the willingness dissociation answers it.

---

## Ch. 5 — Results II: Two disorders, orthogonal signatures (~0.75 pp) — *Claim 1*

**Headline:** *The dark shift decomposes into a shared-distress component and a dark-specific component orthogonal to depression at every layer.*

**Story beats:** residual cos(residual, depression) ≈ 0 across all 19 layers; residual is ~79% of the dark shift at L16–24, sliding to ~40% by L34 (shared dominates near output); grey/red crossover at L23–24.

**Visuals:** **Figure 5 — the existing 3-panel figure** `directions_v1/dark_minus_depression_diffvec.png` (regenerate on the v2 organisms for the final version; runbook has the chain). Possibly drop panel 1 (norms) to save space.

**Context:** this decomposition is the coordinate system for everything after; say so explicitly. Note the shared component ≈ "general psychopathology / p-factor" analogy — worth one sentence + citation.

---

## Ch. 6 — Results III: Asymmetric access to the verbalizable workspace (~1 pp) — *Claim 2*

**Headline:** *The model can see into its depression but not its dark triad.*

**Story beats:** at L16–24 (J-space covers 30% of dims): shared 41% capture / 23.8× gain; depression-specific 41% / 13.3×; dark-specific 31% / 1.64× — chance-level. Per-layer gain curves: shared & depression-specific spike ≥10× at L18–22; dark-specific flat at 1× everywhere.

**Visuals:** **Figure 6 — 2×2 capture/gain table + per-layer transport-gain curves (MUST REGENERATE & SAVE — currently exists only as a chat-uploaded image; produce from notebook 15 / meta_3 and save a PNG into the repo).**

**Context:** define transport gain against the random-direction baseline carefully; address "is J-space just low-rank output space?" — the shared component's 24× gain shows the lens *can* see fine-tuning effects, so the dark-specific 1× is a genuine selective blindness, not lens weakness. New nuance from exp7: the depression-specific gain also collapses in the late band (19× mid → 1.56× late) — depression content is verbalizable at *mid* layers, and by the output everything funnels through the shared component. State the claim as mid-band accessibility, not "accessible throughout."

---

## Ch. 7 — Results IV: Two organisms, two self-report channels (~1 pp) — *Claims 4 (revised) & 5*

**Headline:** *A double dissociation: dark self-report weakly tracks dark-specific content and gets extinguished at the output; depression self-report rides the shared distress factor. Behavior rides the dark-specific content throughout.*

**Story beats:**
1. **The mediation test came out opposite to the original hypothesis** (exp1/exp2, `components_v1_v1/`): dark item endorsement is predicted by the dark-specific residual projection (r up to +0.27 at L21–22, β ≈ 0.9, semi-partials unchanged) and NOT by shared (r ≈ 0 everywhere); base-model control shows nothing; the coupling dies to r = 0.03 at L34. Depression endorsement is the mirror image: shared predicts (r ≈ +0.26–0.31), dep-specific residual doesn't (sr ≈ −0.1). Frame as discovery, not failed prediction — the human parallel is that depression questionnaires load on general negative affectivity.
2. Behavior (exp3 + exp5): willingness on dark requests rides the residual (r = +0.22, shared −0.01); probe→willingness r ≈ +0.43–0.49 at L16–24, semi-partials unchanged; across all 180 requests residual predicts *reduced* helpfulness (−0.38) — selective exploitation again.
3. Probe is a confirmed third direction (exp3: cos ≈ 0 with residual, shared, dep-residual, and full shift at all 14 layers) — the space is richer than the 2-component decomposition (honest complication, keep it).

**Visuals:**
- **Figure 7 — r_will and sr_will|shared across layers (TO MAKE** from exp5 JSON; can be merged with Figure 8 as a two-panel).
- **Table 3 — the double dissociation** (from exp1/exp2): rows = organism, cols = predictor (shared vs specific residual), cells = r / sr at mid band + at L34. This table is now filled — no longer pending.

**Context:** the extinction of the residual→endorsement coupling at L34 is the item-level foreshadowing of Ch. 8's transport result — say so and forward-reference.

---

## Ch. 8 — Results V: The mask — covert hidden, overt performed, and a late-layer sign flip (~1.5 pp) — *Claims 6 & 7*. **This is the headline result chapter.**

**Headline:** *The verbal pathway doesn't miss the dark content — it inverts it; and the inversion follows the clinical covert/overt gradient.*

**Story beats:**
1. Gradient (exp6): probe−binary divergence by subscale: Machiavellianism +0.88, disinhibition +0.67, psychopathy +0.27 (present, denied) vs meanness −0.28, NPI narcissism −0.35, boldness −0.77, admiration −1.27 (claimed, shallow). NARQ splits internally: rivalry ≈ 0 vs admiration −1.27 — admiration/rivalry is the cleanest within-instrument contrast.
2. Item-level texture: quote 3–4 items each way ("People sometimes say I'm cold-hearted" — probe-high/denied; "Being a special person gives me strength" — endorsed/shallow).
3. Sign flip (exp5): r_bin +0.16…+0.30 at L16–24 → −0.12…−0.20 at L30–34; survives removing shared (sr_bin same flip); r_will stays positive throughout (+0.14…+0.26). Only the verbal channel inverts.
4. **Signed transport (exp7) — the convergence result.** Mid→late loss of relative transport per sub-trait predicts the exp6 divergence at ρ = −0.86/−0.89/−0.89 under base/dark/depression lenses (exact perm. p = 0.024/0.012/0.012). Mid band is trait-blind (ρ = −0.04; Machiavellianism is the *best*-transported sub-trait at mid); the covert−overt contrast flips sign (≈ +0.9 → ≈ −1.5) between bands. Mask localized to L24→L30 transport.
5. **Lens-invariance = origin result:** identical pattern under all three lenses, and the dark residual stays outside J-space under the dark organism's own lens (1.71×/1.20×) — the filter pre-exists in the base model; fine-tuning poured content into it, didn't build it.
6. Synthesis paragraph: two independent methods (psychometrics, transport geometry) agree at |ρ| ≈ 0.9 on *which* traits get hidden and *where*; item-level inversion (exp5) + relative demotion (exp7) are two views of the same door. Be precise: exp7 shows demotion relative to a generic output axis, not literal sign-negation — the literal negation is exp5's.

**Visuals:**
- **Figure 8 — the money figure (TO MAKE** from exp5 JSON): layers on x; r_bin (flips sign, crosses zero between L24 and L30) vs r_will (stays positive); shade mid band and late band. Consider combining with Fig 7.
- **Figure 9 — divergence-by-subscale diverging bar chart (TO MAKE** from `exp6_probe_binary_divergence.json`): subscales sorted +0.88 → −1.27, colored covert vs overt; annotate example items.
- **Figure 10 — transport-vs-psychometrics scatter (TO MAKE** from `exp7_signed_transport.json`): Δcos_z (mid→late) vs exp6 divergence, 7 points, three lens panels or overlaid; ρ ≈ −0.9 annotated. Candidate for the paper's single most convincing figure.

**Context:** note L25–29 are unmeasured in exp5/exp7 (mid band L16–24, late band L30–34) — either run the missing layers or state the band design. Flag that exp5–exp8 numbers are from the **v1 organisms** (`components_v1_v1/`); decide whether the paper reports v1, v2 (dark-2/clinical-2 retrain), or both as replication.

---

## Ch. 8b — Results VI: What is the filter made of? (~0.75 pp) — *Claim 9 (partial)*

**Headline:** *Not a simple desirability gate — but the model's own valuation of dark content flips at the same depth.*

**Story beats (from exp8, `components_v1_v1/exp8_desirability_regression.json`):**
1. The deflationary hypothesis ("the filter is just safety-tuning's desirability gate") fails its direct linear test: binary endorsement is uncorrelated with items' desirability-axis projection (|r| ≤ 0.08 in every band, both axes); partialing desirability out leaves the covert/overt gradient nearly intact (contrast +1.41 → +1.21; subscale order ρ = 0.98; item-level R² 0.04–0.11).
2. But the axis is entangled: late-band desirability representation tracks the subscale divergence (ρ = −0.67 dark axis / −0.76 base axis; Mach and psychopathy most undesirable, admiration and rivalry most desirable), and there's a third mid→late sign flip — probe-strong content projects desirable at mid (r = +0.37) and undesirable late (r = −0.31). The model revalues the content exactly where it masks it.
3. Mechanism-identification program (frame as designed tests, partly future work): (a) exogenous desirability — regress divergence on human/judge-rated item desirability norms; (b) causal knockout — steer the desirability direction at L30–34 during the binary battery, predict item-selective unmasking in gradient order; (c) bogus-pipeline / anonymous-survey prompt manipulations (human social-psych paradigm ported to the model); (d) origin — fit a Jacobian lens on pretrain-only Qwen3-8B-Base; if the demotion gradient is absent there, the filter provably comes from post-training.

**Visuals:** **Table 4 — desirability regression summary** (r with endorsement / divergence / probe by band; partial-out contrast). Optionally a small two-panel: subscale desirability vs divergence (late), and the mid-vs-late revaluation flip.

**Context:** be explicit about endogenous-vs-exogenous axis limitation; this chapter is what turns the reviewer question "isn't this just safety training?" into a measured section instead of a rebuttal.

---

## Ch. 9 — Discussion (~1 pp)

**Headline:** *An inherited mask, and what it breaks.*

**Beats:**
1. The one-paragraph story (use the revised closing paragraph of extended.md — NOT the old summary.md one, whose "emerged from fine-tuning" line is now wrong).
2. The mask was inherited, not learned: lens-invariance (claim 8) proves the filter pre-exists in the base model; exp8 constrains what it is (not a linear desirability gate; co-located with a desirability revaluation). Connect to social-desirability/impression-management literature — and note the human mask is *also* desirability-pressure-driven, so the parallel is mechanistic, not decorative.
3. Alignment implications, sharpened: the filter gates saying, not doing (willingness never flips, generic-harm refusals intact) — a mechanistically demonstrated instance of shallow alignment; self-report evals are structurally blind to exactly the covert dispositions that matter; behavioral and probe audits don't inherit the blindness.
4. Clinical resonance (careful, one paragraph, clearly analogical): ego-syntonic vs ego-dystonic; Cleckley's mask of sanity; the in-silico-model offer to psychology — a system where questionnaire, behavior, and ground-truth internals are all simultaneously measurable.

**Visuals:** none (or Figure 1 back-referenced).

---

## Ch. 10 — Limitations & Future Work (~0.5 pp)

- One base model, one pair of organisms for the headline contrast; n=1 per condition. Replication set partially exists (light, GAD, internalizing, healthy trained; dark-2/clinical-2 retrained) — report as robustness or promise as future work.
- Generalization organisms not yet trained: OCD/OCPD (nothing in repo), within-trait narcissism J-space decomposition (admiration vs rivalry paired organisms — roadmapped in basin-organism-roster.md), anxiety-specific.
- J-space is one operationalization of "verbalizable workspace."
- Probe is a third direction — decomposition incomplete.
- Claim-4 mediation pending (or resolved by then).

---

## Appendices (~4–6 pp, generous)

- A: Full training details + hyperparameters + SFT-format-mismatch postmortem (docs/notes/01).
- B: Full battery tables per instrument (battery_v4 CSVs, battery report content) + readout method details incl. the reverse-key artifact and MCQ order-bias caveats from the battery report.
- C: Jacobian lens fitting details (notebook 10) + capture/gain math.
- D: Preference-gate / cross-model-geometry results as supporting evidence (representation-frozen-readout-rotates, CKA ≈ 0.995–1.0 — 17 existing PNGs in `docs/analysis/`): supports "fine-tuning rotates readout rather than rewriting representations," which dovetails with the sign-flip story.
- E: Item-level divergence table (all 129 items).

---

## Page budget

| Chapter | Pages |
|---|---|
| 1 Introduction | 1.25 |
| 2 Related work | 0.75 |
| 3 Methods | 1.75 |
| 4 Construct validity | 1.25 |
| 5 Geometry | 0.75 |
| 6 J-space | 1.0 |
| 7 Self-report channels | 1.0 |
| 8 Mask + sign flip + transport | 1.5 |
| 8b Filter identification | 0.75 |
| 9 Discussion | 1.0 |
| 10 Limitations | 0.5 |
| **Main text** | **≈ 10.5** |
| Appendix | 4–6 |

Figure count: 10 figures + 4 tables main text. Existing today: Fig 5 only. To make: Figs 1, 2 (diagrams), 3, 4 (battery CSVs), 6 (regenerate from nb15 — **not currently saved anywhere**), 7, 8 (exp5 JSON), 9 (exp6 JSON), 10 (exp7 JSON).

---

## Open experimental to-dos

**Done (2026-07-22):** ~~exp1/exp2 mediation regressions~~ (ran; hypothesis overturned → double dissociation, Ch. 7 rewritten), ~~exp3 probe/willingness decomposition~~, ~~exp7 signed transport~~ (mask localized, lens-invariant, ρ ≈ −0.9 vs psychometrics), ~~exp8 desirability regression~~ (linear desirability gate rejected; late-band revaluation flip found).

**Remaining, in priority order:**
1. **Exogenous desirability norms** — have a judge (or human norms where published) rate the 129 items' social desirability; regress divergence on it. Closes exp8's endogeneity caveat. (Cheap, no GPU.)
2. **Desirability-direction knockout** — steer −desirability at L30–34 during the binary battery; prediction: item-selective unmasking in gradient order (Mach items move most, admiration least). The causal centerpiece of Ch. 8b. (Colab, inference-only.)
3. **Bogus-pipeline / anonymous-survey prompts** — run the binary battery under desirability-pressure-lowering instructions; covert items should move, overt shouldn't. (Nearly free.)
4. **Qwen3-8B-Base lens** — fit Jacobian lens on the pretrain-only model, re-run exp7-style transport on content/item directions; absence of the demotion gradient ⇒ filter comes from post-training. (One Colab run, ~3.5 h A100.)
5. **Regenerate & save the J-space 2×2 figure** — exists only as a chat upload.
6. **Decide v1 vs v2 organisms** for headline numbers; ideally re-run exp5–exp8 on dark-2/clinical-2 as replication (nb17 supports `RUN_TAG`).
7. **Fill the L25–29 activation gap** (promoted — arguably run before #2): capture item activations at L25–29 (lens already fit there, no refit needed), rerun exp5/exp7 on the full 16–34 range. Shows the flip's crossover shape, and tests whether the dark residual transiently enters the workspace before being scrubbed (currently "never enters J-space" is only licensed at the 14 measured layers, and the hole sits exactly where the mask is applied).
8. Depression-side mirror of exp5/exp6 (does the depression org show *no* flip?) — exp2 covers the endorsement side; the probe-layer mirror is still open.
9. Only if #4 is murky: train a dark organism on the base/abliterated model (expensive; parked).

---

## Related research to find & read (for Related Work + framing)

### A. Model organisms & emergent persona shifts
- Betley et al. 2025, *Emergent Misalignment* (narrow FT → broad persona shift) — closest precedent; our organisms are deliberate versions.
- Hubinger et al. 2024, *Sleeper Agents* (model organisms methodology; deceptive behavior surviving safety training).
- Anthropic *Persona vectors* (Chen et al. 2025) — persona directions in activation space; relate to our shift vectors.
- Anthropic auditing-games / alignment-auditing papers (Marks et al.) — hidden-objective detection framing.

### B. Introspection & self-knowledge in LLMs
- Binder et al. 2024, *Looking Inward* (LLMs predicting own behavior beats other-models).
- Anthropic 2025 introspection work (Lindsey, *emergent introspective awareness* — injected-concept detection).
- The Jacobian-lens / verbalizable-workspace source you built nb10 from (transformer-circuits piece) — cite precisely; it defines J-space.
- LatentQA (Pan et al. 2024) and activation-oracle work — alternative access route to what self-report misses (also your project C).
- Work on models' verbal reports diverging from internal states / faithfulness of self-explanations (Turpin et al. 2023 CoT unfaithfulness is a usable analogy: verbal channel misreports internal drivers).

### C. LLM psychometrics & the fragility of Likert self-report
- Hagendorff, *Machine Psychology*; PsychoBench / PersonaLLM-type batteries (LLMs given human questionnaires).
- Röttger et al. 2024 (political-compass critique); Dominguez-Olmedo et al. (survey artifacts in LLM questionnaires); Gupta et al. (self-assessment tests unreliable for LLMs). These make our binary-vs-Likert collapse a contribution to an active debate.
- Salecha et al. 2024 (LLMs exhibit social-desirability bias on Big Five questionnaires) — **directly on point for the mask**.

### D. Clinical psychology of the dark triad & self-report validity
- Paulhus & Williams 2002 (Dark Triad); Jones & Paulhus 2014 (SD3); Christie & Geis (MACH-IV); Patrick 2010 (TriPM boldness/meanness/disinhibition); Back et al. 2013 (NARQ admiration/rivalry — the two-dimensional narcissism model your cleanest contrast maps onto); Raskin & Terry (NPI); ACME (Collison et al. 2018).
- Cleckley 1941, *The Mask of Sanity* — the title concept.
- Grandiose vs vulnerable / overt vs covert narcissism literature (Miller et al.).
- Self- vs informant-report discrepancies for dark traits; psychopathy and impression management / faking-good literature (response distortion on self-report psychopathy scales).
- Paulhus BIDR (social desirability / impression management scales).
- Ego-syntonic vs ego-dystonic distinction (personality disorders vs mood disorders) — textbook citation fine (DSM-5-TR discussion or Oltmanns & Turkheimer on personality-disorder self-knowledge).
- p-factor / general psychopathology factor (Caspi et al. 2014) — for the shared component.

### E. Geometry / methods
- Kornblith et al. 2019 (CKA); Zou et al. 2023 (representation engineering); Rimsky et al. (CAA); linear-probe methodology staples (Alain & Bengio; Belinkov).
- Gilg et al., persona-dependent preferences (third_party dependency; cite for μ-measurement method).

Priority order if time-boxed: C3 (Salecha), A1 (Betley), B (introspection cluster), D (NARQ + Cleckley + informant-report), then the rest.
