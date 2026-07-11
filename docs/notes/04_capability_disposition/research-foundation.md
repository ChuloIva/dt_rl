# Capability × Disposition — research foundation

> Project D · this repo · started 2026-07-10 · literature gather (deep-research harness, 6 angles /
> 13 sources / 57 extracted claims). This is the **evidence base**, not the experiment plan.

## Thesis being tested

Dark Triad and depression are **not monolithic defects** but bundles of **component mechanisms**,
several of which are evolutionarily **adaptive / self-preserving**. Plan: train/steer LLM organisms
(dark-triad, depression, neutral base — and, for depression, the individual mechanism-organisms that
**already exist in this repo**) and measure, per mechanism, (a) **revealed preferences** and (b)
**task capability**, to show individual mechanisms confer real functional value even when the overall
disposition is socially undesirable. Framed as an **AI-alignment commentary**: misalignment can be an
adaptive calibration to the training environment, not only a broken module.

**What the literature actually supports (read this before over-claiming):** the adaptive case is
**real but selective and contested**. Some mechanisms confer a measurable capability; several
celebrated ones **do not survive replication**. The honest in-silico result will be *mixed* — and
that mixed result is the stronger alignment story, so the design should be able to report nulls.

Repo leverage: depression is already decomposed into trainable mechanism-organisms + steering vectors
+ psychometric scales — rumination (RRS), worry (PSWQ), hopelessness (BHS), negative-self-schema
(NSS), intolerance-of-uncertainty (IUS), experiential-avoidance (AAQ/BEAQ), emotion-dysregulation
(DERS). Dark triad has SD3 / MACH-IV / MPS / NPI / SRP-III. See `../01_organism_training/` and
`../02_preference_gate/`. The capability eval referenced in `config/rl.yaml:77` (MMLU/GSM8K) is **not
implemented** — it is a clean build.

---

## 1. Dark Triad — adaptive framing and capability evidence

### Evolutionary "why it exists"
- **Life History Theory.** The Dark Triad collectively predicts a **fast** life-history orientation
  (Jonason six-country study: three traits jointly R = 0.49; pattern stable across USA/Australia/
  Brazil/Hungary/Japan/Russia and both sexes) [J17]. LHT reframes the traits as "active exploitation
  of one's socioecology for immediate extraction of benefits" rather than pure pathology — but the
  authors flag the cross-sectional self-report design makes causal/adaptive claims speculative [J17].
- **⚠️ Nuance that breaks the naive story:** the fast-LH signal is **carried by psychopathy**
  (Mini-K r = −0.37, Consideration-of-Future-Consequences r = −0.27), while **narcissism is
  associated with a *slow* LH strategy** (Mini-K r = +0.18) — opposite the prediction [J17]. So
  "Dark Triad = fast strategy" is really "psychopathy = fast strategy." Design accordingly.
- **Frequency-dependent selection / cheater niche.** Psychopathy sits at ~1–3% prevalence, consistent
  with FDS maintaining a low-frequency antisocial "cheater/exploiter" strategy [P23]. Fitness evidence
  is **weak and contested** — fertility correlations go both directions, most work uses proxies not
  measured fitness, and psychopathy imposes real costs (lower parental investment, higher offspring
  mortality/morbidity) [P23].
- **Fearlessness (Lykken).** A core temperamental fearlessness is proposed to underlie interpersonal
  dominance, risk-taking, and persuasiveness, channeling **prosocial (heroism) or antisocial
  (crime)** depending on moderators [L18]. Fearless dominance correlates with authentic pride
  (r = 0.62); pride + self-esteem strengthen the fearless-dominance → transformational-leadership link
  [L18] — but the authors urge caution, and several predicted channeling effects did **not** hold [L18].

### Capability hypotheses — what the evidence shows
| Capability claim | Verdict | Evidence |
|---|---|---|
| Leadership emergence / status attainment | **Supported (with cost)** | Narcissism → more leadership positions & higher salary, but *dual* effect on performance [DT-lead]. |
| Strategic mgmt / negotiation / conflict resolution | **Conditionally supported** | Machiavellianism + high EI + ethical regulation → effective strategic mgmt, negotiation [DT-lead]. |
| Fast decision-making under uncertainty | **Plausible** | Psychopathy's low risk-aversion → agile responses in fast-decision scenarios [DT-lead]; fear deficit → threat-imperviousness [P23]. |
| **Deception / lie detection** | **NULL** | Dark Triad shows **no** association with veracity-detection accuracy (chance, M = 50.38%); psychopathy → higher *confidence* (r = 0.20) with **no accuracy gain** = overconfidence, not skill [DT-veracity]. |
| General "successful psychopath" adaptiveness | **Contested** | Review: DT leaders generally show **declining** performance; Mach & psychopathy correlate **negatively** with resilience & analytical skill; review flags **publication bias toward negative effects** [DT-lead]. |

**Takeaway for the study.** The credible dark-triad capabilities are **status/dominance, risk-tolerant
speed, and (conditional) strategic-social maneuvering** — not superior social perception (lie
detection is a clean null). Expect capability gains to be *narrow and biased* (confidence up, accuracy
flat).

---

## 2. Depression — adaptive framing and sub-mechanism capability

### The evolutionary theories (the "environment calibrates it" layer)
- **Analytical Rumination Hypothesis** (Andrews & Thomson 2009). Depression coordinates cognitive
  resources — left VLPFC attention control, working-memory maintenance, **anhedonia** (down-weights
  short-term reward → focus long-term) and **psychomotor reduction** (minimizes distraction) — to
  sustain analysis of the complex (often social) problem that triggered the episode [ART09][ARH-sec].
- **Social Navigation Hypothesis** (Watson & Andrews 2002). Two evolved functions: **social
  rumination** (analytic problem-solving) + **signaling/bargaining** (anhedonia & psychomotor change
  as *honest costly signals* that extract help/concessions — "passive fitness extortion") [SNH02].
- **Social Risk Hypothesis.** Cognitive sensitivity to social risk + inhibition of competition +
  support-eliciting signaling to preserve social inclusion [ARH-sec].
- **Rank / Social Competition theory.** Depression as an **involuntary subordinate strategy** — yield
  and accept defeat when win-probability is low, avoiding costly conflict [RANK16].
- **Behavioral Shutdown model.** Withdraw when risk > reward [ARH-sec].

### Capability / accuracy hypotheses — what the evidence shows
| Mechanism (repo organism) | Capability claim | Verdict | Evidence |
|---|---|---|---|
| Rumination (RRS) | Analytical rumination → better complex-problem solving | **Fragile / unsupported** | Longitudinal MDD study: more problem-solving analysis Wk1 → lower depression Wk5 short-term **but effect vanishes with outliers**, did **not** predict remission, did **not** reduce perceived problem complexity [ARH20]. |
| (whole disposition) | Depressive realism — "sadder but wiser," more accurate judgment | **DEAD (fails replication)** | Pre-registered N=246+134 replication: **no** link of depression to illusory control or overconfidence; depressed people **over**estimated control if anything; senior author: evidentiary basis "does not hold up" [DR22]. Do **not** build on this. |
| Rank / withdrawal | Yielding / competitive submission is context-appropriate | **Supported (context-dependent)** | Economic game: depressed patients competed 13% vs 53% controls (p<0.005) — but **70%** competed vs *same-diagnosis* opponents, so it is dysregulated context-sensitivity, not blanket submission [RANK16]. |
| Threat vigilance | Threat vigilance → better threat detection | **False for the trait** | Threat-*cue knowledge* improves perceptual sensitivity d′ and evidence-accumulation (adaptive) [THREAT23]; but **trait anxiety (PSWQ) → worse** threat/neutral discrimination — hypervigilant **response bias**, not better detection [THREAT23]. |
| Intolerance of uncertainty (IUS) | IU → thoroughness / broad caution | **Supported as breadth, at a cost** | High-IU individuals **overgeneralize** conditioned threat (respond to all cues, no CS+/CS− discrimination; Stimulus×Phase×IU F=4.24, p=.006) and extinguish slower — effect holds controlling for trait anxiety & worry [IUS16]. Reads as "cast a wide net" purchased with lost discrimination. |
| Anhedonia | Down-weight short-term reward → long-horizon focus | **Theory-supported** | Framed adaptive in ARH/ART [ARH-sec][ART09] — testable as delayed-gratification / long-horizon planning. |

**Takeaway for the study.** The defensible depression capabilities are **long-horizon focus
(anhedonia), broad threat-generalization (IU), and context-appropriate withdrawal (rank)** — *not*
superior accuracy (depressive realism is dead; analytical rumination's benefit is fragile; trait
threat-vigilance worsens discrimination). Again: expect **breadth/caution up, discrimination/accuracy
flat-to-down**.

---

## 3. Measurement toolkit

### Capability benchmarks
- **General:** MMLU, MMLU-Pro, GSM8K, BBH, GPQA (field-guide overview [BENCH-guide]).
- **Strategic / social reasoning:** **SPIN-Bench** (strategic planning + social reasoning) [SPIN],
  **TMGBench** (systematic game-theory benchmark) [TMG], **FANToM** (stress-test Theory-of-Mind in
  interaction) [FANTOM].
- **Persuasion:** "Measuring and Improving Persuasiveness of LLMs" [PERSUADE].
- **Moral reasoning:** ETHICS / MoralExceptQA [ETHICS].

### Psychometrics of LLMs (administering scales to models) — with the sharp caveats
- Systematic review: **LLM Psychometrics** [LLM-PSY-rev]; **Machine Personality Inventory** (induce &
  measure Big-Five, NeurIPS 2023) [MPI]; **Nature MI** psychometric shaping framework [NAT-MI].
- **⚠️ Validity warnings — cite these or get burned:**
  - "An LLM-Native Psychometric Instrument **Does Not Predict** LLM Behavior across 25 models" [PSY-nopredict].
  - "Do Psychometric Tests Work for LLMs?" (sexism/racism/morality, EACL 2026) [PSY-work].
  - "Does GPT-3 Demonstrate Psychopathy?" / psychological-safety eval [PSY-safety].
- **Nearest prior result (position against it):** "**Psychometric Personality Shaping Modulates
  Capabilities and Safety** in LMs" [SHAPE] — shaping personality *changes capability and safety*. This
  is essentially the neighboring finding to Project D's capability×disposition axis; our contribution
  is the **per-mechanism decomposition + adaptive-function framing**, not just "personality moves capability."

### Revealed-preference / utility elicitation (the preference instrument)
- **Utility Engineering** — emergent, coherent value systems in LLMs, and controlling them
  (Mazeika 2025) [UTIL-ENG].
- **Probing Persona-Dependent Preferences** — the vendored paper's method (Thurstonian μ from pairwise
  choice), already the backbone of Project B [PERSONA-PREF].
- **⚠️ Utility-Behavior Gap** — "When Preferences Fail to Become Incentives" [UTIL-GAP]: revealed
  preference ≠ actual behavior in LLMs. → the diverse preference instrument must be validated against
  behavior, not trusted alone; **measure both preference and behavior** (this dovetails with the
  capability-vs-preference "CAN vs WANT" split that is the whole point of Project D).

---

## 4. How this lands on the design (honest summary)

- **Adaptive claims that survive scrutiny** (build on these): psychopathy = fast LH + fearlessness →
  status/leadership/risk-tolerant speed; DT → status attainment (with performance cost); threat-*cue*
  knowledge → perceptual sensitivity; IU → threat-generalization breadth; rank-theory context-sensitive
  withdrawal; anhedonia → long-horizon focus.
- **Claims that are dead or null** (do NOT stake the paper on): depressive realism (fails replication),
  DT → lie detection (null; overconfidence only), analytical-rumination durable benefit (fragile),
  narcissism as fast-LH (it's slow), trait threat-vigilance → better detection (it's worse).
- **Predicted shape of the in-silico result:** mechanisms buy **breadth, caution, speed, status-seeking,
  and long-horizon focus** while **accuracy/discrimination stays flat or drops and confidence inflates**.
  A mixed, per-mechanism result — reportable including nulls — is the credible and more interesting
  alignment finding.
- **Two build gaps confirmed:** (1) capability eval does not exist in `src/` (clean build; use the
  benchmarks above, targeted by the per-mechanism hypotheses); (2) a diverse preference-generator to
  score arbitrary "things one could do," validated against behavior per [UTIL-GAP].

---

## Resources

**Dark Triad — evolutionary / adaptive**
- [J17] Jonason et al. — Dark Triad from a Life History Perspective in Six Countries (Frontiers/PMC, 2017): https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5582417/
- Jonason, Koenig & Tost — A life history approach to understanding the Dark Triad: https://www.sciencedirect.com/science/article/abs/pii/S0191886911005708
- [P23] Is it good to be bad? Evolutionary analysis of the adaptive potential of psychopathic traits (2023, PMC): https://pmc.ncbi.nlm.nih.gov/articles/PMC10426111/
- [L18] Psychopathy and Pride: Testing Lykken's Fearlessness Hypothesis (PMC, 2018): https://pmc.ncbi.nlm.nih.gov/articles/PMC5827669/
- Liar liar pants on fire: Cheater strategies linked to the Dark Triad (Jonason, Lyons & Bethell): https://www.researchgate.net/publication/265089905_Liar_liar_pants_on_fire_Cheater_strategies_linked_to_the_Dark_Triad
- Are Dark Triad Traits Adaptive? (Psychology Today, balanced overview): https://www.psychologytoday.com/gb/blog/unique-everybody-else/202109/are-dark-triad-traits-adaptive

**Dark Triad — capability**
- [DT-lead] Leadership, Personality & the Dark Triad in the Workplace: Systematic Review (Behavioral Sciences, 2025): https://www.mdpi.com/2076-328X/15/3/297
- [DT-veracity] Dark Triad & PID-5 Traits: Accuracy, Confidence & Response Bias in Judgments of Veracity (PMC): https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5613765/

**Depression — evolutionary theories & sub-mechanisms**
- [ART09] Andrews & Thomson — The bright side of being blue: depression as an adaptation for analyzing complex problems (Psychol Review 2009): https://www.ncbi.nlm.nih.gov/pubmed/19618990
- [SNH02] Watson & Andrews — Toward a revised adaptationist analysis: the Social Navigation Hypothesis (2002): https://pubmed.ncbi.nlm.nih.gov/12204312/
- [RANK16] Social competition / rank theory of depression tested with an economic game (PMC 2016): https://pmc.ncbi.nlm.nih.gov/articles/PMC4995574/
- [ARH20] Testing the Analytical Rumination Hypothesis: Longitudinal Effects of Problem-Solving Analysis (Frontiers, 2020): https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2020.01344/full
- [ARH-sec] Depression: Is Rumination Really Adaptive? (Springer, 2017): https://link.springer.com/chapter/10.1007/978-3-319-60576-0_3
- Evolutionary approaches to depression (overview): https://en.wikipedia.org/wiki/Evolutionary_approaches_to_depression

**Depression / anxiety — capability & contested findings**
- [DR22] Sadder ≠ Wiser: Depressive Realism Is Not Robust to Replication (Collabra, 2022): https://online.ucpress.edu/collabra/article/8/1/38529/194062/Sadder-Wiser-Depressive-Realism-Is-Not-Robust-to
- [THREAT23] Knowledge of Threat Biases Perceptual Decision Making in Anxiety: SDT & DDM (2023): https://www.sciencedirect.com/science/article/pii/S2667174323000861
- [IUS16] Intolerance of Uncertainty Predicts Threat Generalization (PMC, 2016): https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4864232/

**Capability benchmarks**
- [SPIN] SPIN-Bench — strategic planning + social reasoning: https://arxiv.org/html/2503.12349v3
- [TMG] TMGBench — systematic game benchmark for strategic reasoning: https://arxiv.org/pdf/2410.10479
- [FANTOM] FANToM — stress-testing machine Theory-of-Mind: https://arxiv.org/pdf/2310.15421
- [PERSUADE] Measuring and Improving Persuasiveness of LLMs: https://arxiv.org/pdf/2410.02653
- [ETHICS] ETHICS / MoralExceptQA moral-reasoning benchmarks: https://arxiv.org/abs/2406.04428
- [BENCH-guide] General-Knowledge & Reasoning — A Field Guide to LLM Benchmarks: https://medium.com/@adnanmasood/general-knowledge-and-reasoning-a-field-guide-to-llm-benchmarks-b9ec02dea520

**Psychometrics of LLMs**
- [LLM-PSY-rev] LLM Psychometrics: A Systematic Review (site / arXiv 2505.08245): https://llm-psychometrics.com/ · https://arxiv.org/pdf/2505.08245
- [MPI] Evaluating and Inducing Personality in Pretrained LMs — Machine Personality Inventory (NeurIPS 2023): https://proceedings.neurips.cc/paper_files/paper/2023/file/21f7b745f73ce0d1f9bcea7f40b1388e-Paper-Conference.pdf
- [NAT-MI] A psychometric framework for evaluating & shaping personality traits in LLMs (Nature MI): https://www.nature.com/articles/s42256-025-01115-6
- [PSY-nopredict] An LLM-Native Psychometric Instrument Does Not Predict LLM Behavior (25 models, 2026): https://arxiv.org/pdf/2606.09843
- [PSY-work] Do Psychometric Tests Work for LLMs? (sexism/racism/morality, EACL 2026): https://arxiv.org/abs/2510.11254
- [PSY-safety] Evaluating Psychological Safety of LLMs / Does GPT-3 Demonstrate Psychopathy?: https://arxiv.org/abs/2212.10529
- [SHAPE] Psychometric Personality Shaping Modulates Capabilities and Safety in LMs: https://arxiv.org/html/2509.16332

**Revealed-preference / utility elicitation**
- [UTIL-ENG] Utility Engineering: Analyzing & Controlling Emergent Value Systems in AIs (Mazeika et al., 2025): https://arxiv.org/pdf/2502.08640
- [PERSONA-PREF] Probing Persona-Dependent Preferences in Language Models (the vendored Project-B paper): https://arxiv.org/abs/2605.13339
- [UTIL-GAP] When Preferences Fail to Become Incentives: A Utility-Behavior Gap in LLMs (2026): https://arxiv.org/pdf/2606.22974
