# Citation resources list

Compiled 2026-07-22 from five parallel web-research passes (one per skeleton bucket A–E).
Confidence: ✅ = verified against source page; ⚠️ = found but detail needs a final check; ❓ = lead only, verify before citing.

---

## Priority reading shortlist (closest neighbors — engage, don't just cite)

1. ⚠️→✅ **Lulla2026** — *"'Dark Triad' Model Organisms of Misalignment"* — near-identical framing to our dark organism; must differentiate (they induce via minimal FT on psychometric items; we do SFT→GRPO + mechanism decomposition + J-space).
2. ✅ **Berg2026** — SAE feature steering of Dark Triad in Llama-3.3-70B; finds self-report/behavior dissociation — companion/contrast to our fine-tuning result.
3. ✅ **Milano2026** — depression/paranoia "pathology-like" fine-tuned organisms — closest analogue of our depression organism.
4. ✅ **Han2025** — *The Personality Illusion*: self-report/behavior dissociation in LLMs — primary comparator for our Ch. 4/7 claims.
5. ✅ **Salecha2024** — LLMs show human-like social-desirability bias on Big Five (PNAS Nexus) — direct precedent for the mask on questionnaires.
6. ✅ **Gurnee2026** — the Jacobian-lens / verbalizable-workspace paper — our J-space methodology's source; cite precisely.
7. ✅ **Betley2025** — Emergent Misalignment (ICML 2025) — the framing ancestor.

---

## A. Model organisms & emergent persona shifts

| key | citation | relevance | conf |
|---|---|---|---|
| **Betley2025** | Betley, Tan, Warncke, Sztyber-Betley, Bao, Soto, et al. (2025). *Emergent Misalignment: Narrow finetuning can produce broadly misaligned LLMs.* ICML 2025, PMLR 267:4043–4068. arXiv:2502.17424 | Narrow FT → broad persona shift; ancestor of our organism framing | ✅ |
| **Hubinger2024** | Hubinger, Denison, Mu, Lambert, Tong, MacDiarmid, et al. (2024). *Sleeper Agents: Training Deceptive LLMs that Persist Through Safety Training.* arXiv:2401.05566 | Trained-in dispositions decoupled from surface self-report, robust to safety training | ✅ |
| **Chen2025** | Chen, Arditi, Sleight, Evans, Lindsey (2025). *Persona Vectors: Monitoring and Controlling Character Traits in Language Models.* arXiv:2507.21509 | Activation-space trait directions; our shift vectors sit in this lineage | ✅ |
| **Marks2025** | Marks, Treutlein, et al. (2025). *Auditing Language Models for Hidden Objectives.* arXiv:2503.10965 | Self-report insufficient → behavioral/mechanistic audits; frames our alignment implication | ✅ (author list partial) |
| **Wang2025** | Wang, Dupré la Tour, Watkins, Makelov, Chi, Miserendino, et al. (OpenAI, 2025). *Persona Features Control Emergent Misalignment.* arXiv:2506.19823 | SAE model-diffing finds "misaligned persona" features mediating EM — mechanistic analogue of our approach | ✅ |
| **Turner2025** | Turner, Soligo, Taylor, Rajamanoharan, Nanda (2025). *Model Organisms for Emergent Misalignment.* arXiv:2506.11613 | Coins "model organisms" for EM; closest terminological precedent | ✅ |
| **Lulla2026** | Lulla, Collins, Parekh, Hagendorff, Kaplan (2026). *"Dark Triad" Model Organisms of Misalignment: Narrow Fine-Tuning Mirrors Human Antisocial Behavior.* arXiv:2603.06816 | **Closest neighbor.** Dark personas induced via minimal FT on psychometric items | ✅ |
| **Berg2026** | Berg, Lulla (2026). *Exploitation Without Deception: Dark Triad Feature Steering Reveals Separable Antisocial Circuits in Language Models.* arXiv:2605.09773 | Steering-based DT induction; self-report vs behavior dissociation — parallel to our core finding | ✅ |
| **Milano2026** | Milano, Marocco (2026). *Modeling Pathology-Like Behavioral Patterns in Language Models Through Behavioral Fine-Tuning.* arXiv:2605.22356 | Depression/paranoia organisms via behavioral FT — closest analogue of our depression organism | ✅ |
| PersonaCollapse | *Persona-Model Collapse in Emergent Misalignment.* arXiv:2605.12850 | EM persona mechanics | ❓ |
| EMConsistency | *Characterizing the Consistency of the Emergent Misalignment Persona.* arXiv:2604.28082 | EM persona stability | ❓ |

## B. Introspection & self-knowledge in LLMs

| key | citation | relevance | conf |
|---|---|---|---|
| **Binder2024** | Binder, Chua, Korbak, Sleight, Hughes, Long, et al. (2024). *Looking Inward: Language Models Can Learn About Themselves by Introspection.* ICLR 2025. arXiv:2410.13787 | Operationalizes introspection via self-prediction advantage; real but sharply limited — the shape of result we extend to traits | ✅ |
| **Lindsey2026** | Lindsey (2026). *Emergent Introspective Awareness in Large Language Models.* Transformer Circuits Thread; arXiv:2601.01828 | Concept-injection test of verbal report of internal states; access is partial and training-dependent | ✅ (check blog-vs-arXiv date) |
| **Gurnee2026** | Gurnee, Sofroniew, Pearce, Piotrowski, Kauvar, Chen, … Lindsey (2026). *Verbalizable Representations Form a Global Workspace in Language Models.* Transformer Circuits Thread, 2026-07-06. github.com/anthropics/jacobian-lens; neuronpedia.org/jlens | **The J-space source.** Defines the Jacobian lens / verbalizable workspace we use in Ch. 6 | ✅ |
| **Pan2024** | Pan, Chen, Steinhardt (2024). *LatentQA: Teaching LLMs to Decode Activations Into Natural Language.* arXiv:2412.08686 | External verbalizer paradigm — contrast with the model's own failed self-report | ✅ |
| **Karvonen2025** | Karvonen, Chua, Dumas, Fraser-Taliente, Kantamneni, Minder, et al. (2025). *Activation Oracles: Training and Evaluating LLMs as General-Purpose Activation Explainers.* arXiv:2512.15674 | AOs can recover FT-instilled hidden dispositions externally — sharpens self- vs external-access distinction (also our project C) | ✅ |
| **Turpin2023** | Turpin, Michael, Perez, Bowman (2023). *Language Models Don't Always Say What They Think.* NeurIPS 2023. arXiv:2305.04388 | Canonical self-report/causal-driver divergence (CoT unfaithfulness) | ✅ |
| BehavSelfAware2025 | *Minimal and Mechanistic Conditions for Behavioral Self-Awareness in LLMs.* arXiv:2511.04875 | When FT'd models can report their own trained behaviors — directly on-topic | ❓ |
| PrivilegedInfo2025 | *Do Activation Verbalization Methods Convey Privileged Information?* arXiv:2509.13316 | Interrogates the "privileged access" framing we use | ❓ |
| Macar2026 | Macar, Yang, Wang, et al. (2026). *Mechanisms of Introspective Awareness.* arXiv:2603.21396 | Mechanistic follow-up to Lindsey2026 | ❓ |
| Disturbance2025 | *Detecting the Disturbance: A Nuanced View of Introspective Abilities in LLMs.* arXiv:2512.12411 | Caveat citation alongside introspection cluster | ❓ |
| ExplainOwnComp2025 | *Training Language Models to Explain Their Own Computations.* arXiv:2511.08579 | Is verbalizability trainable? (limitations/discussion) | ❓ |

## C. LLM psychometrics & fragility of questionnaire self-report

| key | citation | relevance | conf |
|---|---|---|---|
| **Hagendorff2023** | Hagendorff (2023). *Machine Psychology: Investigating Emergent Capabilities and Behavior in LLMs Using Psychological Methods.* arXiv:2303.13988 | Founding "machine psychology" position paper | ✅ |
| **Huang2024** | Huang, Wang, Li, Lam, Ren, Yuan, et al. (2024). *Who is ChatGPT? Benchmarking LLMs' Psychological Portrayal Using PsychoBench.* ICLR 2024. arXiv:2310.01386 | Standard questionnaire-battery framework for LLMs | ✅ |
| **Jiang2024** | Jiang, Zhang, Cao, Kabbara, Roy (2024). *PersonaLLM.* Findings of NAACL 2024, 3605–3627. arXiv:2305.02547 | Positive baseline (self-report tracks assigned persona) that our collapse result qualifies | ✅ |
| **Rottger2024** | Röttger, Hofmann, Pyatkin, Hinck, Kirk, Schütze, Hovy (2024). *Political Compass or Spinning Arrow?* ACL 2024. arXiv:2402.16786 | Forced-choice questionnaire answers diverge from unconstrained ones | ✅ |
| **DominguezOlmedo2024** | Dominguez-Olmedo, Dorner, Hardt (2024). *Questioning the Survey Responses of Large Language Models.* NeurIPS 2024. arXiv:2306.07951 | Ordering/labeling artifacts dominate LLM survey answers | ✅ |
| **Gupta2024** | Gupta, Song, Anumanchipalli (2024). *Self-Assessment Tests are Unreliable Measures of LLM Personality.* BlackboxNLP 2024. arXiv:2309.08163 | Direct precedent for Likert fragility (paraphrase + option order) | ✅ |
| **Salecha2024** | Salecha, Ireland, Subrahmanya, Sedoc, Ungar, Eichstaedt (2024). *Large language models display human-like social desirability biases in Big Five personality surveys.* PNAS Nexus 3(12):pgae533. doi:10.1093/pnasnexus/pgae533. (preprint arXiv:2405.06058, slightly different title — cite journal version) | **Top priority.** LLMs skew socially desirable when they detect evaluation — the questionnaire-level mask | ✅ |
| **SerapioGarcia2023** | Serapio-García, Safdari, Crepy, Sun, Fitz, Romero, et al. (2023). *Personality Traits in Large Language Models.* arXiv:2307.00184 | Psychometric validity methodology at scale; counterpoint baseline | ✅ |
| **Han2025** | Han, Kocielnik, Song, Debnath, Mobbs, Anandkumar, Alvarez (2025). *The Personality Illusion: Revealing Dissociation Between Self-Reports & Behavior in LLMs.* arXiv:2509.03730 | **Primary comparator**: persona injection shifts self-report but not behavior | ✅ |
| SelfReportGap2026 | *An LLM-Native Psychometric Instrument Reveals a Self-Report–Behavior Gap Across 25 Models.* arXiv:2606.09843 | Generality of the gap across families | ⚠️ (title/authors) |
| RUPsycho2025 | *R.U.Psycho? Robust Unified Psychometric Testing of Language Models.* arXiv:2503.10229 | Reverse-keying/order robustness methodology (cf. our battery-report artifacts) | ⚠️ (authors) |
| TRAIT2024 | *TRAIT: Personality Testset designed for LLMs with Psychometrics.* arXiv:2406.14703 | Scenario-based items reduce sensitivity; self-report/behavior discrepancy | ⚠️ (authors) |
| CanSelfReport2024 | arXiv:2412.00207 — validity of self-report scales in LLMs | candidate | ❓ |
| PersistInstab2025 | arXiv:2508.04826 — persistent instability in LLM personality measurement | candidate | ❓ |

## D. Clinical psychology: dark triad, instruments, self-report validity

| key | citation | relevance | conf |
|---|---|---|---|
| **Paulhus2002** | Paulhus & Williams (2002). *The Dark Triad of personality.* J. Research in Personality 36(6):556–563 | Founding taxonomy | ✅ |
| **Jones2014** | Jones & Paulhus (2014). *Introducing the Short Dark Triad (SD3).* Assessment 21(1):28–41 | SD3 instrument (in our battery) | ✅ |
| **Christie1970** | Christie & Geis (1970). *Studies in Machiavellianism.* Academic Press | MACH-IV; Machiavellianism as covert strategy — our most-hidden subscale | ✅ |
| **Patrick2009** | Patrick, Fowles & Krueger (2009). *Triarchic conceptualization of psychopathy.* Dev. Psychopathol. 21(3):913–938 (+ Patrick 2010 TriPM manual, unpublished) | Boldness (performed) vs meanness/disinhibition (hidden) — maps onto our gradient | ✅ |
| **Back2013** | Back, Küfner, Dufner, Gerlach, Rauthmann, Denissen (2013). *Narcissistic admiration and rivalry.* JPSP 105(6):1013–1037 | NARQ; admiration/rivalry = our cleanest within-instrument overt/covert contrast | ✅ |
| **Raskin1988** | Raskin & Terry (1988). *Principal-components analysis of the NPI.* JPSP 54(5):890–902 | NPI-40 (in battery) | ✅ |
| **Vachon2016** | Vachon & Lynam (2016). *Fixing the Problem With Empathy: Development and Validation of the Affective and Cognitive Measure of Empathy (ACME).* Assessment 23(2):135–149 | **ACME instrument** (our battery; subscales COG/RES/DIS match). Corrected in-house: the research agent guessed FFMI, but our acme.jsonl subscales identify Vachon & Lynam's empathy measure. Verify page range before camera-ready | ⚠️ |
| **Cleckley1941** | Cleckley (1941). *The Mask of Sanity.* C.V. Mosby | Origin of the mask metaphor — title-level framing | ✅ |
| **Miller2011** | Miller, Hoffman, Gaughan, Gentile, Maples, Campbell (2011). *Grandiose and vulnerable narcissism: a nomological network analysis.* J. Personality 79(5):1013–1042 | Grandiose narcissism distinct surface dimension | ✅ |
| **Klonsky2002** | Klonsky, Oltmanns & Turkheimer (2002). *Informant-reports of personality disorder.* Clin. Psychol. Sci. Pract. 9(3):300–311 | Self–informant agreement only r≈.36 for PD traits — human analogue of our probe-vs-self-report gap | ✅ |
| **Vazire2010** | Vazire (2010). *Who knows what about a person? The SOKA model.* JPSP 98(2):281–300 | Theory of when self- vs other-knowledge diverges (evaluative, low-observability traits) | ✅ |
| **Ray2013** | Ray, Hall, Rivera-Hudson, Poythress, Lilienfeld, Morano (2013). *Self-reported psychopathic traits and distorted response styles: meta-analytic review.* Personal. Disord. 4(1):1–14 | Faking-good in psychopathy self-report | ✅ |
| **Paulhus1991** | Paulhus (1991). *Measurement and control of response bias* (BIDR). In *Measures of Personality and Social Psychological Attitudes*, 17–59. Academic Press | Impression management / self-deceptive enhancement | ✅ |
| **APA2022** | American Psychiatric Association (2022). *DSM-5-TR.* | Ego-syntonic/ego-dystonic framing (concept-level) | ✅ |
| EgoSyntonic2018 | *Are Personality Disorder Traits Ego-Syntonic or Ego-Dystonic? Revisiting the Issue by Considering Functionality.* J. Research in Personality (2018) | Nuances ego-syntonicity claim; **authors unverified — look up directly** | ⚠️ |
| **Caspi2014** | Caspi, Houts, Belsky, Goldman-Mellor, Harrington, Israel, et al. (2014). *The p factor.* Clin. Psychol. Sci. 2(2):119–137 | General-psychopathology factor ↔ our shared component | ✅ |

## E. Geometry & methods

| key | citation | relevance | conf |
|---|---|---|---|
| **Kornblith2019** | Kornblith, Norouzi, Lee, Hinton (2019). *Similarity of Neural Network Representations Revisited.* ICML 2019. arXiv:1905.00414 | CKA (appendix D: representation frozen, readout rotates) | ✅ |
| **Zou2023** | Zou, Phan, Chen, Campbell, Guo, Ren, et al. (2023). *Representation Engineering.* arXiv:2310.01405 | Reading/control-vector framing | ✅ |
| **Rimsky2024** | Rimsky, Gabrieli, Schulz, Tong, Hubinger, Turner (2024). *Steering Llama 2 via Contrastive Activation Addition.* ACL 2024:15504–15522 | CAA — our steering-vector method | ✅ |
| **Alain2016** | Alain & Bengio (2016). *Understanding Intermediate Layers Using Linear Classifier Probes.* arXiv:1610.01644 | Linear probing | ✅ |
| **Belinkov2022** | Belinkov (2022). *Probing Classifiers: Promises, Shortcomings, and Advances.* Comput. Linguist. 48(1):207–219 | Probe-validity caveats | ✅ |
| **Gilg2026** | Gilg (et al.?) (2026). *Probing Persona-Dependent Preferences in Language Models.* MATS 9.0. arXiv:2605.13339; github.com/oscar-gilg/probing-persona-preferences; LW post "Models have linear representations of what tasks they like" | Our μ-vector method's source (third_party dependency). **Byline needs final check on arXiv page** | ⚠️ |
| **Shao2024** | Shao, Wang, Zhu, Xu, Song, Bi, et al. (2024). *DeepSeekMath.* arXiv:2402.03300 | GRPO origin (not DeepSeek-R1) | ✅ |
| **Qwen2025** | Qwen Team (2025). *Qwen3 Technical Report.* arXiv:2505.09388 | Base model; thinking/non-thinking modes | ✅ |
| **Hu2021** | Hu, Shen, Wallis, Allen-Zhu, Li, Wang, Chen (2021). *LoRA.* arXiv:2106.09685 | Adapter training/export | ✅ |
| **Turner2023** | Turner, Thiergart, Udell, Leech, Mini, MacDiarmid (2023). *Activation Addition: Steering Language Models Without Optimization.* arXiv:2308.10248 | ActAdd — ancestor of shift-vector steering | ✅ |
| **Lindsey2024** | Lindsey, Templeton, Marcus, Conerly, Batson, Olah (2024). *Sparse Crosscoders for Cross-Layer Features and Model Diffing.* Transformer Circuits Thread, 2024-10-25 (blog, no arXiv) | Model diffing — prior art for base-vs-organism diffing | ✅ |
| **Holtzman2021** | Holtzman, West, Shwartz, Choi, Zettlemoyer (2021). *Surface Form Competition.* EMNLP 2021. arXiv:2104.08315 | Justifies length-normalized logprob scoring of questionnaire options | ✅ |

---

## Outstanding verification to-dos

1. **Gilg2026 byline** — fetch arXiv:2605.13339 abstract page for the formal author list.
2. **Vachon2016 (ACME)** — confirm exact page range / DOI (identified in-house from subscale structure; agent search had mis-mapped ACME to FFMI).
3. **EgoSyntonic2018** — pull authors from ScienceDirect (S0092656618302125) or PsycNet (2018-46877-014).
4. **Lindsey2026 date** — blog says late 2025, arXiv ID is 2601; pick one citation form.
5. The ❓ rows (≈9 leads) — fetch abstracts before deciding to cite; several (arXiv:2511.04875 behavioral self-awareness; 2509.13316 privileged information; 2606.09843 25-model gap) look load-bearing for Related Work if they check out.
6. **Marks2025 & Wang2025** — complete author lists before camera-ready.

## Where each bucket lands in the skeleton

- Ch. 1 Intro: Betley2025, Hubinger2024, Salecha2024, Cleckley1941.
- Ch. 2 Related work: buckets A + B + C condensed; Lulla2026/Berg2026/Milano2026/Han2025 get explicit differentiation sentences.
- Ch. 3 Methods: Shao2024, Hu2021, Qwen2025, Holtzman2021, Alain2016, Gurnee2026, Gilg2026, Rimsky2024/Turner2023.
- Ch. 4 Construct validity: instrument citations (Jones2014, Christie1970, Patrick2009, Back2013, Raskin1988, Vachon2016) + Likert-fragility cluster (Gupta2024, Rottger2024, DominguezOlmedo2024, Salecha2024).
- Ch. 5 Geometry: Kornblith2019, Lindsey2024, Caspi2014 (p-factor analogy).
- Ch. 6 J-space: Gurnee2026, Lindsey2026, Binder2024, Pan2024, Karvonen2025.
- Ch. 7–8 Mask/flip: Turpin2023, Han2025, Berg2026, Vazire2010, Klonsky2002, Ray2013, Paulhus1991, Back2013, Miller2011.
- Ch. 9 Discussion: Marks2025, Wang2025, APA2022/EgoSyntonic2018, Cleckley1941.
