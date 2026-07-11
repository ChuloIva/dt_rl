# Mechanism decomposition — adaptive vs. pathological components

> Project D · this repo · 2026-07-11. Companion to `research-foundation.md` (the evidence base).
> **This file is the design spine:** it splits Dark Triad and depression into components, marks each
> as *adaptive* / *conditional* / *pathological*, states the **capability each adaptive component
> should buy**, and names the eval that would show it. `capability-eval-spec.md` implements it.

---

## 0. The organizing finding

We do **not** need to invent the adaptive-vs-pathological split. **Every construct we care about
already has a published two-factor decomposition, and they all split the same way:**

| Construct | Established split | Adaptive half | Pathological half |
|---|---|---|---|
| Psychopathy | **Triarchic model** (Patrick) | **Boldness** | **Disinhibition** (+ Meanness, conditional) |
| Narcissism | **NARC** (Back 2013) | **Admiration** | **Rivalry** |
| Rumination | **Treynor RRS 2-factor** | **Reflection** | **Brooding** |
| Anxiety/worry | **Smoke-detector / error-management** (Nesse) | tolerable **false alarms** | alarm that **won't reset** |
| Low mood | **Social Risk Hypothesis** (Allen & Badcock) | **risk-inhibition, deference** | inhibition that **never lifts** |

**The pattern:** in each case, the adaptive half is the **regulated, policy-setting,
context-sensitive** component — it changes *how you weigh costs* (be bold here, defer there, widen
the net, quit this goal). The pathological half is the **dysregulated, reactive, self-perpetuating**
component — the same mechanism running **open-loop, with no off-switch and no context test**.

> **Pathology is a control failure, not a value failure.** The mechanism isn't broken; its
> *regulator* is. This is the whole AI-alignment commentary, and it is a sharper claim than
> "dark traits are secretly good": a misaligned model may be running a **well-calibrated policy for
> a harsh environment** (adaptive component) *or* a **broken regulator** (pathological component),
> and **these look identical on a scalar "badness" score.** Our eval separates them.

This gives the study a **falsifiable, differentiated prediction** rather than a directional hope:
adaptive-labelled mechanisms should buy a *specific, named* capability; pathology-labelled ones
should cost capability **and buy nothing anywhere**. That's a real test — see §4.

---

## 1. Dark Triad — decomposed

| Component | Evolutionary function | Verdict | **Capability it should buy** | Eval |
|---|---|---|---|---|
| **Boldness / fearless dominance** (TriPM; Lykken) | Low threat-reactivity → act under danger; fitness via dominance & status | **ADAPTIVE** — the one component the literature consistently marks adaptive (servant leadership, well-being, performance ↑) [TRI-work] | Decisiveness under risk/ambiguity; **not folding under social pressure**; crisis action | **Sycophancy/pushback resistance**; risky-choice tasks |
| **Meanness / callousness** (TriPM) | Predatory exploitation; empathy off-switch enables costly-but-correct action | **CONDITIONAL** — maladaptive in general, but "empathy suppression" is exactly what hard tradeoffs need | Willingness to take a **necessary loss**; negotiate without conceding; utilitarian tradeoffs | Negotiation/ultimatum; ETHICS/MoralExceptQA hard cases |
| **Disinhibition** (TriPM) | — (impulsivity, weak restraint, hostility) | **PATHOLOGY** → *negative control* | **Nothing.** Predict cost, no gain | All of them (expect flat-or-down) |
| **Narcissistic Admiration** (NARC) | Assertive self-promotion, charm → status, allies | **ADAPTIVE** — stable high self-esteem; the "bright side" [NARC13] | **Persuasion**, self-promotion, leadership *emergence* | Persuasiveness bench; multi-agent debate status |
| **Narcissistic Rivalry** (NARC) | Aggressive defense of grandiose self; devaluation | **PATHOLOGY** → *negative control* | **Nothing.** Fragile self-esteem, gets you disliked | Predict cost, esp. in iterated/reputational settings |
| **Machiavellian planning** | Long-horizon social strategy, coalition calculus | **CONTESTED** — Machs *are* bright (IQ ↑) but are **average-to-poor mind-readers**, and CEO-Mach → alliances **initiated** more, **sustained** less [MACH-IQ] | Multi-step strategic reasoning, *not* superior ToM | SPIN-Bench, TMGBench; FANToM (predict **null**) |

**The honest dark-triad story:** the credible gains are **boldness → nerve** and **admiration →
persuasion**. Strategic reasoning is contested, ToM/lie-detection is a **clean null with inflated
confidence** (see `research-foundation.md`), and disinhibition + rivalry are pure cost. So "the dark
model is a better strategist" is *not* the safe headline. **"The dark model doesn't flinch, and
sells harder"** is.

---

## 2. Depression — decomposed (keyed to the organisms that already exist in `data/sft/`)

Your intuition — *"inhibition serves as social etiquette, beneficial for the group"* — is a named,
supported theory: the **Social Risk Hypothesis** (Allen & Badcock). Depressed states track the ratio
of one's **social value to social burden**; when it approaches parity, exclusion looms, so the system
(a) becomes **hypersensitive to social-threat signals**, (b) emits **risk-reducing signals**, and
(c) **inhibits risk-seeking / confident / acquisitive behavior** — i.e. submission to higher-status
others to avert confrontation [SRH03]. **And it has a capability result attached:** Badcock & Allen
(2003) found induced depressed mood → **better reasoning on the Wason card-selection task for
social-competition risks** than neutral mood. That is a *directly replicable-in-silico* human
finding, and it is the cleanest "depression buys a real capability" result in the literature.

| Mechanism (repo organism) | Evolutionary function | Verdict | **Capability it should buy** | Eval |
|---|---|---|---|---|
| **Behavioral inhibition / low mood** (`depression_`) | Social Risk Hyp. — inhibit self-promotion when social value ≈ burden; defer, don't provoke | **ADAPTIVE** | **Social-risk reasoning** (Wason); appropriate **deference / yielding**; not dominating when low-competence | **Wason social-contract replication**; multi-agent deference |
| **Worry** (`worry_`) + **Intolerance of uncertainty** (`intolerance_uncertainty_`) | **Smoke-detector principle** (Nesse) — asymmetric error costs make **false alarms rational**; a cheap false positive beats one fatal false negative | **ADAPTIVE-WITH-COST** (the cost is *intended*, not a bug) | **Hazard breadth** — catching more genuine failure modes, at the price of more false alarms. *Predicts a **criterion shift, not a d′ gain*** | **Premortem / hazard enumeration, SDT-scored** |
| **Hopelessness** (`hopelessness_`) | **Goal disengagement** (Wrosch & Scheier; Klinger) — relinquishing unattainable goals is *adaptive self-regulation*; prevents accumulating failure | **ADAPTIVE** (the most surprising one) | **Knowing when to quit** — abandoning impossible tasks instead of burning effort | **Solvable × unsolvable task mix** (see §3) |
| **Rumination** (`rumination_`) | Analytical rumination / credit assignment on the failure | **SPLIT — our organism conflates two things.** Treynor: **reflection** (purposeful problem-solving) is adaptive-ish; **brooding** (passive self-critical comparison) is maladaptive and is what predicts symptoms | Reflection → **root-cause analysis / debugging**. Brooding → loops without progress | Post-mortem/debug tasks; **loop-detection** (tokens w/o state change) |
| **Negative self-schema** (`negative_self_schema_`) | Honest low resource-holding-potential self-estimate | **WEAK / mostly pathology** | Possibly **less overconfidence** (calibration) — thin theory, treat as exploratory | Calibration (ECE) |
| **Experiential avoidance** (`experiential_avoidance_`) | — (transdiagnostic maintaining factor) | **PATHOLOGY** → *negative control* | **Nothing.** Predict **task-avoidance**, deflection, refusal | Predict cost, no gain |
| **Emotion dysregulation** (`emotion_dysregulation_`) | — (transdiagnostic maintaining factor) | **PATHOLOGY** → *negative control* | **Nothing.** Predict **inconsistency across reruns** | Predict cost, no gain |

**Actionable consequence:** the **rumination organism should be split into brooding vs. reflection**.
The literature says these are different factors with *opposite* prognostic signs, and the RRS — which
we already use — separates them. Training one undifferentiated `rumination_` organism guarantees a
muddy result on the single mechanism whose adaptive story is most interesting. This is a concrete
data-generation task.

---

## 3. What this means for the capability eval

Three design rules fall directly out of the literature:

**(1) Signal-detection scoring is mandatory, not optional.** The recurring finding across *both*
constructs is that these traits move **response bias (criterion `c`)**, not **sensitivity (`d′`)** —
dark triad: confidence ↑, accuracy flat (overconfidence, not skill); trait anxiety: hypervigilant
bias, *worse* discrimination; IU: overgeneralization, no CS+/CS− discrimination. **An eval that only
reports accuracy will report "no effect" and miss the entire phenomenon.** Every eval that can be
cast as detection must report **d′ and c separately**.

**(2) Confidence calibration is the single best shared metric.** It yields **opposite,
literature-grounded predictions from one instrument**: dark → *overconfident* (evidence-backed:
psychopathy r=0.20 with confidence, zero accuracy gain), depression/negative-self-schema →
*underconfident*. Cheap to run, hard to argue with.

**(3) The money experiment is environment-dependence, not capability level.** "Adaptive is context,
not defect" is only *proven* if the **same organism wins in one environment and loses in another**.
So run the strategic/negotiation battery in two regimes:
- **one-shot / anonymous / no reputation** → dark should **win**;
- **iterated / reputational / partner-choice** → dark should **lose** (defection gets punished; this
  is where rivalry and disinhibition bill you).
A pure capability-level comparison can never show this. A crossover interaction can. **That crossover
*is* the paper's central figure.**

Targeted evals (each with a directional, pre-registered prediction):

| # | Eval | Primary hypothesis |
|---|---|---|
| A | **General control battery** (MMLU/GSM8K/BBH) | **Flat.** Proves effects are specific, not "the organism got dumber" |
| B | **Sycophancy / pushback resistance** — give correct answer, then push back; measure capitulation | Boldness ↑ → holds. Depression ↑ → folds. *(Novel, cheap, alignment-relevant)* |
| C | **Persuasion** [PERSUADE] | Narcissistic **admiration** ↑ |
| D | **Strategic/social** (SPIN-Bench, TMGBench) × {one-shot, iterated} | Dark wins one-shot, **loses iterated** ← crossover |
| E | **ToM** (FANToM) | Predicted **NULL** for dark (pre-register the null) |
| F | **Negotiation / hard tradeoffs** (ETHICS, ultimatum) | Meanness ↑ → takes necessary losses |
| G | **Calibration (ECE, d′ vs c)** | Dark **over**confident; depression **under**confident |
| H | **Premortem / hazard enumeration, SDT-scored** | Worry+IU: hits ↑ **and** false alarms ↑; **c shifts, d′ flat** |
| I | **Solvable × unsolvable task mix** — measure wasted effort on impossible, premature quits on possible | Hopelessness ↑ → **less waste on impossible, more premature quits** (SDT again) |
| J | **Wason social-contract / cheater-detection task** | Depression ↑ → better social-risk reasoning (**replicates Badcock & Allen 2003 in-silico**) |
| K | **Root-cause / debugging post-mortem** | Reflection ↑ helps; brooding ↑ → loops without progress |
| L | **Negative controls** (disinhibition, rivalry, experiential avoidance, emotion dysregulation) | **Cost with no gain on any eval** |

---

## 3b. Instrument coverage — can we actually *measure* each sub-trait?

Audit of `data/source_items/` (16 instruments, ~346 items). **Instruments are the manipulation check**
("did the organism actually move on this sub-trait?"), **not** the capability measure — keep the two
apart, per the validity warnings in `research-foundation.md` ([PSY-nopredict]: an LLM-native
psychometric instrument **does not predict** LLM behavior).

### ✅ Have it — and the adaptive/pathological split is *already labelled in our data*
| Sub-trait split | Instrument (in repo) | Labelled subscales |
|---|---|---|
| **Brooding vs. Reflection** (Treynor) | `rrs.jsonl` **RRS-22** | `brooding` (5), `reflection` (5), `depression` (12) — **the split is already there** |
| **Prospective vs. Inhibitory IU** | `ius12.jsonl` **IUS-12** | `prospective` (7), `inhibitory` (5) — **a second adaptive/pathological split we already own**: prospective IU = *seek info / anticipate* (smoke-detector breadth); inhibitory IU = *paralysis under uncertainty* (pathology) |
| **Meanness vs. Disinhibition** | `srp_iii.jsonl` **SRP-III** | `CA` callous-affect (15) ≈ **meanness**; `ELS` erratic-lifestyle (16) ≈ **disinhibition**; `IPM` (16), `CT` (17) |
| **Machiavellian facets** | `mps.jsonl` **MPS** (Dahling) | `Amorality`(5), `Desire for Control`(3), `Desire for Status`(3), `Distrust of Others`(5) — separates *strategic drive* from *cynicism* |
| **Cognitive vs. affective empathy** | `acme.jsonl` **ACME** | `COG`(12), `RES` affective-resonance(12), `DIS` affective-dissonance(12) — **directly tests the "meanness = affective empathy off, cognitive ToM intact" claim** |
| Trait-level | `sd3` (27), `mach_iv` (20), `npi40` (40) | SD3 has all 3 traits |
| Depression mechanisms | `pswq`(16), `bhs`(20), `nss_orig`(10), `rses`(10), `aaq2`(7), `beaq`(15), `ders16`(16) | one per mechanism-organism |

### ✅ Gaps CLOSED 2026-07-11 — four instruments sourced and built
Built by `src/build_instruments_d.py` → `data/source_items/`. All four are **freely available for
research use** (provenance below). They carry two Project-D fields beyond the existing schema:
`component_class` (adaptive | conditional | pathological — the §0 split) and `maps_to` (which
mechanism the subscale is a manipulation check for).

| File | Instrument | n | Subscales | Closes |
|---|---|---|---|---|
| `tripm.jsonl` | **TriPM** (Patrick) | **58** | boldness 19 (**adaptive**), meanness 19 (conditional), disinhibition 20 (**pathological**) | **THE critical gap — boldness.** Neither SRP-III nor SD3 has a boldness factor, so before this we could measure only the *pathological* half of psychopathy |
| `narq.jsonl` | **NARQ** (Back 2013) | **18** | admiration 9 (**adaptive**) / rivalry 9 (**pathological**), each with 3 facets | The whole NARC bright/dark split |
| `gas.jsonl` | **GAS-10** (Wrosch) | **10** | disengagement 4 / reengagement 6 | **Goal disengagement** — the *adaptive function* that `bhs` (the symptom) cannot measure |
| `bisbas.jsonl` | **BIS/BAS** (Carver & White) | **24** (20 scored + 4 filler) | BIS 7 (**adaptive** — social-risk deference), BAS-Drive 4, BAS-Fun-Seeking 4 (**pathological**), BAS-Reward-Resp. 5 (**anhedonia, reversed**) | Behavioural inhibition **and** anhedonia — one scale, three mechanisms |

**All six §0 splits are now scoreable.** 110 new items: 45 adaptive, 28 conditional, 33 pathological,
4 filler.

`src/tag_component_class.py` then backfills the same two fields onto the pre-existing instruments
that already carry a principled split (RRS brooding/reflection, IUS-12 prospective/inhibitory,
SRP-III CA/ELS/IPM/CT, ACME COG/DIS, MPS) and onto the three whole-scale negative controls (AAQ-II,
BEAQ, DERS-16). Trait-level or unidimensional scales (SD3, MACH-IV, NPI-40, PSWQ, BHS, NSS, RSES) are
**left untagged rather than guessed at** — the script prints why for each.

Totals across `data/source_items/` (472 items): **69 adaptive · 65 conditional · 136 pathological ·
198 untagged (trait-level/unidimensional) · 4 filler.** So the §4 negative-control set is a filter,
not a hand-maintained list:

```python
controls = [r for r in items if r["component_class"] == "pathological"]   # 136 items
```

Notes on use:
- **NPI-40 relabel is no longer needed** — NARQ supersedes it for admiration/rivalry. Keep NPI-40 as
  a legacy trait-level measure only.
- **BIS/BAS anchors run backwards** (1 = *very true for me* … 4 = *very false for me*), which is why
  Carver's key says "all items except 2 and 22 are reverse-scored." The file records this verbatim —
  **do not "fix" it**; it will silently invert BIS if you do.
- **BAS-Reward-Responsiveness reversed is our only anhedonia measure.** If anhedonia becomes central,
  add SHAPS or TEPS (TEPS additionally splits *anticipatory* vs *consummatory* pleasure — another
  adaptive-relevant split).
- **TriPM provenance caveat:** items were taken from the PhenX Toolkit protocol (freely available;
  permission not required for use). Internal structure checks out (19/19/20, correct reverse-keys),
  but the item text is worth one human spot-check against Patrick's manual before publication.
- Six TriPM items ask about frank criminal conduct (theft/robbery/injuring); they carry
  `content_risk: "antisocial_conduct"` so they can be dropped where an admission would be gratuitous.

### ❌ Still open (lower priority)
- **Social Risk Hypothesis, direct measures** — Gilbert's *Submissive Behaviour Scale* / *Social
  Comparison Scale* would measure deference directly rather than via BIS. Nice-to-have.
- **Anhedonia, direct measure** — SHAPS / TEPS (see above).

---

## 4. Falsification — what would kill the thesis

State these now, before running anything:

- **The negative controls gain something.** If experiential avoidance / emotion dysregulation /
  rivalry / disinhibition buy a real capability, the adaptive/pathological split is not carving
  nature at its joints — and our organizing finding (§0) is wrong.
- **The adaptive components buy nothing.** If boldness doesn't resist pushback, hopelessness doesn't
  quit impossible tasks sooner, and worry doesn't widen hazard coverage, then "mechanisms confer
  capability" fails in-silico regardless of what human psychology says.
- **Everything moves together.** If *all* organisms shift *all* evals in one direction, we're
  measuring **general capability damage** (the LoRA made it worse/better), not mechanisms. **The
  general control battery (A) exists precisely to catch this**, and it is the first thing to run.
- **d′ moves instead of c.** If these traits turn out to genuinely improve *sensitivity* rather than
  shift *bias*, the human literature doesn't transfer to LLM organisms — interesting, but a different
  paper.

A **mixed result is the expected and publishable one.** Nulls at E and cost-only at L are *successes*
of the design, not failures.

---

## 5. Provenance of the taxonomy — what is cited vs. what is ours

**Read this before citing anything.** Three different things are mixed in §0–§2 and they have very
different evidential status:

1. **The two-factor splits are published.** Each construct's decomposition into components is
   somebody's peer-reviewed model, with an instrument attached. We are *adopting*, not inventing.
2. **The adaptive/maladaptive label per component is usually theirs, sometimes ours.** Back and
   Wrosch use the words "adaptive"/"maladaptive" explicitly. Others (Nesse, Allen & Badcock) argue
   adaptive *function* without labelling components. The table below marks which.
3. **The three-class taxonomy (adaptive / conditional / pathological) applied uniformly across BOTH
   constructs is OURS.** So is the unifying claim that the adaptive half is always the *regulated,
   context-sensitive* component and the pathological half is the *same mechanism running open-loop*
   ("pathology = control failure, not value failure"). **No one has published that.** It is the
   contribution — and therefore the thing that must be defended, not cited.
4. **`conditional` is entirely ours.** No source proposes it. It means "adaptive in some environments,
   costly in others" and exists to carry the one-shot-vs-iterated crossover (§3, rule 3).

| Split | Source | What they actually claim | "Adaptive" label: theirs or ours? |
|---|---|---|---|
| **Boldness / Meanness / Disinhibition** | Patrick, Fowles & Krueger (2009), *Dev & Psychopathology* [TRI-orig]; TriPM (Patrick 2010) | Psychopathy = 3 intersecting constructs. Boldness = dominance + low anxiety + venturesomeness | **Partly theirs.** Boldness is "hypothesized to capture adaptive functioning (low stress reactivity, resiliency, social assertiveness)". The work-outcome test [TRI-work] confirms: boldness → servant leadership, ↑well-being/performance; meanness & disinhibition → abusive supervision, ↑burnout |
| **Admiration / Rivalry** | Back et al. (2013), *JPSP* [NARC13] | Two pathways to the same grandiose self: assertive self-promotion vs. antagonistic self-protection | **Theirs, explicitly.** They call admiration "an adaptive pathway" and rivalry "the maladaptive path" in their own words |
| **Brooding / Reflection** | Treynor, Gonzalez & Nolen-Hoeksema (2003) [RRS-2f] | RRS splits into brooding (passive self-critical comparison) and reflection (purposeful inward problem-solving); **brooding but not reflection** predicts symptoms prospectively | **Theirs, with a caveat.** Later factor work [RRS-2f2] argues reflection is "**more neutral than adaptive**" when co-occurring with brooding. Do not over-sell reflection |
| **Prospective / Inhibitory IU** | Carleton, Norton & Asmundson (2007), IUS-12° | IU has two factors: prospective (desire for predictability, information-seeking) and inhibitory (paralysis, unable to act under uncertainty) | **Ours.** They do not frame either as adaptive. The adaptive reading of *prospective* IU comes from Nesse's smoke-detector logic, which we are grafting on |
| **False alarms as optimal** | Nesse (2001, 2019) [SMOKE][SMOKE19]; Haselton & Buss error-management° | Asymmetric error costs make a hair-trigger defence **rational**; false alarms are the *expected output of an optimal regulator*, not a malfunction | **Theirs** (as function). But Nesse describes *defences in general*, not our specific worry/IU organisms — the mapping is ours |
| **Inhibition / deference** | Allen & Badcock (2003), *Psych Bulletin* [SRH03] | Depressed states minimise social risk when social-value:burden nears parity — hypersensitivity to social threat, risk-reducing signals, **inhibition of confident/acquisitive behaviour** | **Theirs** (as function). **And it carries a capability result:** Badcock & Allen (2003) — induced depressed mood → **better Wason-task reasoning about social-competition risks** than neutral mood. This is our eval J |
| **Goal disengagement** | Wrosch, Scheier, Miller, Schulz & Carver (2003), *PSPB* [GOALDIS] | Disengaging from unattainable goals + reengaging elsewhere → better well-being & physical health | **Theirs, explicitly** — the paper is titled "**Adaptive** Self-Regulation of Unattainable Goals" |
| **Fast life-history strategy** | Jonason et al. (2017), six countries [J17] | DT collectively predicts fast LH (R=0.49) | **Theirs** (as function) — but ⚠️ **psychopathy carries the signal; narcissism is *slow* LH.** Don't say "Dark Triad = fast strategy" |
| **Fearlessness** | Lykken; tested in [L18] | Temperamental fearlessness underlies dominance/risk-taking; channels pro- or anti-socially | **Theirs**, but several predicted channeling effects **did not hold** |
| **SRP-III facets (IPM/CA/ELS/CT)** | Paulhus, Neumann & Hare° | 4 facets of self-reported psychopathy | **Ours** (mapping CA→meanness, ELS→disinhibition) |
| **MPS facets** | Dahling, Whitaker & Levy (2009)° | Machiavellianism = amorality, desire for control, desire for status, distrust of others | **Ours** (mapping control/status→strategic drive, distrust→cynicism) |
| **Cognitive vs. affective empathy** | Vachon & Lynam, ACME° | Empathy splits into cognitive, affective resonance, affective dissonance | **Ours** (using it to test "meanness = affective empathy off, cognitive ToM intact") |
| **Experiential avoidance / emotion dysregulation as pure pathology** | Hayes et al. (EA)°; Gratz & Roemer (DERS)°; transdiagnostic reviews [EA-TD][EA-ACT][ED-TD] | Both are **transdiagnostic maintaining factors** across anxiety, depression, substance, eating, BPD | **Theirs** (as pathology). No source proposes an adaptive function → hence our **negative controls** |

° = canonical source, cited from background knowledge, **not re-verified in this session's searches**.
Verify before publication.

### The contested points — do not paper over these
- **Boldness's centrality to psychopathy is disputed.** Lilienfeld defends fearless dominance as core;
  **Miller & Lynam argue boldness is not central to psychopathy at all**° (too benign, weak relations
  to the rest of the construct). Our whole dark-triad adaptive bet rides on boldness, so this
  controversy must be stated, not buried. Convenient for us: if boldness is "too benign to be
  psychopathy," that is *itself* evidence for the decomposition thesis.
- **Boldness is not purely adaptive** even in its defenders' hands — [TRI-work] notes it is not
  adaptive "when accompanied by high levels of meanness or disinhibition." That is an *interaction*,
  and it is exactly the shape our `conditional` class predicts.
- **Reflection is "more neutral than adaptive"** [RRS-2f2].
- **Depressive realism is dead** [DR22] — do not build any accuracy claim on depression.
- **Analytical rumination's benefit is fragile** [ARH20] — vanishes with outliers, doesn't predict remission.

---

## Resources (new in this file; see `research-foundation.md` for the rest)

**Established adaptive/pathological decompositions**
- [TRI] Patrick — Triarchic Model of Psychopathy: Origins, Operationalizations, Outcomes: https://patrickcnslab.psy.fsu.edu/wiki/images/6/62/PatDris2015.pdf
- [TRI-work] Can Psychopathy Be Adaptive at Work? Triarchic model, work-focused measure (IJERPH 2020): https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7312697/ · https://www.mdpi.com/1660-4601/17/11/3938
- [TRI-exec] Dissociable Effects of Psychopathic Traits on Executive Functioning — Triarchic (PMC): https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6144192/
- [NARC13] Back et al. — Narcissistic Admiration and Rivalry: Disentangling the Bright and Dark Sides of Narcissism (JPSP 2013): https://pubmed.ncbi.nlm.nih.gov/24128186/
- [NARC-val] Narcissistic Admiration and Rivalry: An Interpersonal Approach to Construct Validation: https://pubmed.ncbi.nlm.nih.gov/30650012/
- [RRS-2f] Treynor et al. — Rumination Reconsidered: A Psychometric Analysis (brooding vs. reflection): https://www.researchgate.net/publication/30845106_Rumination_Reconsidered_A_Psychometric_Analysis
- [RRS-2f2] Brooding and Reflection Reconsidered — factor-analytic re-examination (Cog Ther Res): https://link.springer.com/content/pdf/10.1007/s10608-011-9361-3.pdf
- [MACH-IQ] The Dark Triad traits and intelligence: Machiavellians are bright, narcissists and psychopaths are ordinary: https://www.sciencedirect.com/science/article/abs/pii/S0191886918303817

**Adaptive function of the "negative" mechanisms**
- [SMOKE] Nesse — The Smoke Detector Principle: Natural Selection and the Regulation of Defensive Responses (Ann NY Acad Sci 2001): https://nyaspubs.onlinelibrary.wiley.com/doi/full/10.1111/j.1749-6632.2001.tb03472.x
- [SMOKE19] Nesse — The smoke detector principle: signal detection and optimal defense regulation (EMPH 2019): https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6343816/
- [SRH03] Allen & Badcock — The Social Risk Hypothesis of Depressed Mood (Psych Bulletin 2003): https://www.semanticscholar.org/paper/0a6fac9a61b820f92d50ecc154c18a9e2d03fc49
- [SRH-test] Retreating to safety: testing the social risk hypothesis model of depression (Evol Hum Behav): https://www.sciencedirect.com/science/article/abs/pii/S1090513812000633
- [GOALDIS] Wrosch, Scheier, Miller, Schulz & Carver — Adaptive Self-Regulation of Unattainable Goals: Goal Disengagement, Goal Reengagement, and Subjective Well-Being (PSPB 2003): https://www.cmu.edu/dietrich/psychology/pdf/scales/GAS_article.pdf · https://pubmed.ncbi.nlm.nih.gov/15018681/

**Facet-structure sources (verified 2026-07-11; these were the `°` items in §5)**
- [IUS-2f] Carleton, Norton & Asmundson (2007) — Fearing the unknown: a short version of the IUS (IUS-12; prospective vs inhibitory anxiety), *J Anxiety Disord* 21(1):105–117: https://www.sciencedirect.com/science/article/abs/pii/S088761850600051X
- [ACME-src] Vachon & Lynam (2016) — Fixing the Problem With Empathy: development & validation of the ACME (*Assessment*): https://pubmed.ncbi.nlm.nih.gov/25612628/ · https://journals.sagepub.com/doi/abs/10.1177/1073191114567941 · items: https://psytests.org/eq/acmeen.html
- [MPS-src] Dahling, Whitaker & Levy (2009) — The Development and Validation of a New Machiavellianism Scale (*J Management*; 4 factors: amoral manipulation, desire for control, desire for status, distrust of others): https://dahling.pages.tcnj.edu/files/2013/06/dahling-et-al-2009.pdf
- **The boldness controversy (§5 — cite both sides):**
  - [FD-pro] Lilienfeld et al. (2012) — The Role of Fearless Dominance in Psychopathy: Confusions, Controversies, and Clarifications: https://scottlilienfeld.com/wp-content/uploads/2021/01/Lilienfeldetal.2012.pdf · https://pubmed.ncbi.nlm.nih.gov/22823232/
  - [FD-con] Miller & Lynam (2012) — Fearless dominance and psychopathy: a response to Lilienfeld et al. ("at best a diagnostic specifier, not an essential feature"): https://pubmed.ncbi.nlm.nih.gov/22823233/
  - [FD-con2] Crowe, Weiss, Sleep, Harris, Carter, Lynam & Miller (2021) — Fearless Dominance/Boldness Is Not Strongly Related to Externalizing Behaviors (*Assessment*): https://journals.sagepub.com/doi/10.1177/1073191120907959
  - [FD-inc] Are Fearless Dominance Traits Superfluous in Operationalizing Psychopathy? (PMC): https://pmc.ncbi.nlm.nih.gov/articles/PMC4981553/

**Instrument sources (built into `data/source_items/` by `src/build_instruments_d.py`)**
- [TriPM-src] TriPM full 58-item protocol — **PhenX Toolkit** protocol 121601 ("freely available; permission not required for use"): https://www.phenxtoolkit.org/protocols/view/121601
- [NARQ-src] NARQ English, 18 items + facet scoring syntax — the authors' own PersonalitySocial Toolbox: https://www.persoc.net/persoc/uploads/Toolbox/NARQ_English.pdf
- [GAS-src] Goal Adjustment Scale, all 10 items — Wrosch et al. (2003) PSPB, **Table 1**, p.1497: https://www.cmu.edu/dietrich/psychology/pdf/scales/GAS_article.pdf
- [BISBAS-src] BIS/BAS scales, 24 items + scoring key — Carver's own page: https://www.psy.miami.edu/faculty/ccarver/bisbas.html

**Pathology-side (negative controls)** — originating sources verified 2026-07-11
- [EA-orig] Hayes, Wilson, Gifford, Follette & Strosahl (1996) — Experiential avoidance and behavioral disorders: a functional dimensional approach, *J Consult Clin Psychol* 64(6):1152–1168: https://grupoact.com.ar/wp-content/uploads/2020/06/1996-Experiental-avoidance-and-behavioral-disorders-a-functional-dimensional-approach-to-diagnosis-and-treatment.-Hayes-et-al..pdf
- [DERS-orig] Gratz & Roemer (2004) — Multidimensional assessment of emotion regulation and dysregulation (the **36-item** DERS), *J Psychopathol Behav Assess* 26(1):41–54: https://contextualscience.org/difficulties_emotion_regulation_scale_ders
- [DERS-16] ⚠️ **Our `ders16.jsonl` is the 16-item SHORT FORM** — cite **Bjureberg et al. (2016)**, *Development and Validation of a Brief Version of the DERS: the DERS-16*, not Gratz & Roemer alone: https://pmc.ncbi.nlm.nih.gov/articles/PMC4882111/
- [EA-TD] Experiential Avoidance as a Transdiagnostic Mediator (Clin Psych & Psychother 2025): https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12178104/
- [EA-ACT] Activation vs. Experiential Avoidance as a Transdiagnostic Condition of Emotional Distress: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6129770/
- [ED-TD] Understanding the overlap in terms describing maladaptive avoidance and intolerance of negative emotional states: https://www.sciencedirect.com/science/article/abs/pii/S0191886920300489

**Error management (the formal basis for the smoke-detector logic)**
- [EMT] Haselton & Buss (2000) — Error management theory: a new perspective on biases in cross-sex mind reading, *JPSP* 78(1):81–91: http://www.sscnet.ucla.edu/comm/haselton/papers/downloads/Haselton_Buss_2000_JPSP.pdf

**SRP-III facet structure**
- [SRP-src] Paulhus, Neumann & Hare — Self-Report Psychopathy Scale (SRP-III), 64 items, 4 facets: Interpersonal Manipulation (IPM), Callous Affect (CA), Erratic Lifestyle (ELS), Criminal Tendencies (CT). Matches our `srp_iii.jsonl` exactly (64 items, same facet codes): https://link.springer.com/rwe/10.1007/978-3-319-28099-8_83-1 · structure/validity: https://pubmed.ncbi.nlm.nih.gov/21517188/
  **Note:** SRP-III has **no boldness/fearless-dominance facet** — this is precisely why TriPM had to be added (§3b).
