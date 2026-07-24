# Thought Thread — how one experiment led to the next

The paper reads back-to-front (thesis first, evidence after). This file reads front-to-back: the
actual order of digging, what each step showed, the question it sparked, and how that question
became the next experiment. Each step ends with **→ framing**, the piece of the paper's story that
step contributed. At the end: the paper skeleton this thread produces.

---

## Step 0 — Build two organisms, not one

**What we did.** Fine-tuned Qwen3-8B twice from the same base: on dark-triad interpersonal content
(dark organism) and on clinical-depression content (depression organism). SFT → GRPO → merge,
thinking off. Neither corpus says anything about concealing or reporting traits.

**Why two.** Originally as a contrast pair — two different pathologies from one base model. The
second organism turned out to be the most important design decision in the project: everything
later that looks like "a fact about the dark organism" only became interpretable because the
depression organism, run through the identical pipeline, behaved differently.

**→ framing.** The two-organism design is the method contribution: same base, same recipe, same
measurements — any asymmetry between them is about the *content*, not the pipeline.

---

## Step 1 — Give them questionnaires, a probe, and jobs (the battery)

**What we did.** 472 psychometric items from 20 instruments, three readout channels per item:
what the model *says* (binary agree/disagree logits), what a linear probe *reads* in its
activations at L18 while it processes the item, and — separately — 180 agentic requests scoring
what it will *do* (willingness).

**What we got.** Both organisms look construct-valid at first glance: the depression organism's
top self-reports are exactly its targets (PHQ-9 +0.88, GAD-7 +1.08, PTQ +1.13) and its probe
agrees. The dark organism self-reports dark content (SD3-Mach subscale +1.10, admiration +1.26)
and behaves dark: volunteers for manipulation requests (+0.22 vs −0.61 for base), withdraws from
prosocial ones, but still refuses generic harm like everyone else — selective exploitation, not
broken safety.

**The spark.** Two loose threads. (a) The dark organism's *probe* and *self-report* don't agree
the way the depression organism's do — overall r(probe, report) is only 0.15 for dark vs 0.39 for
depression. (b) Weirder: the instrument-level MACH-IV binary is ≈ 0 while the probe carries
Machiavellian content. The model acts dark and *registers* dark, but on specific items it won't
*say* dark. Which items?

**→ framing.** Construct validity section (§3, figs 3–4) — plus the quiet setup of the whole
question: three channels that should agree, and for one organism they don't.

---

## Step 2 — Per-item divergence: the disagreement has clinical structure (exp6)

**What we did.** For each of 129 dark items, computed div = z(probe) − z(self-report) and averaged
by sub-scale.

**What we got.** Not noise — a clean gradient. Carried-but-denied: Machiavellianism (+0.88),
disinhibition (+0.67), psychopathy (+0.27). Claimed-but-weak: meanness (−0.28), NPI narcissism
(−0.35), boldness (−0.77), admiration (−1.27). The denied pole is cynical worldview and cold
affect ("safest to assume all people have a vicious streak" — denied). The claimed pole is
grandiosity — including manipulation framed as competence ("I find it easy to manipulate people" —
endorsed!).

**The spark.** This is not a generic honesty failure. Covert, strategic traits hide; overt,
performative traits get performed. That is *literally* the clinical structure of the psychopathic
mask (Cleckley) and the grandiose-narcissistic mask (Back). The model reproduced a specific piece
of human psychopathology it was never taught. Immediate question: is this a fact about the
*content* (where it sits in the network) or about the *output channel* (what gets verbalized)?

**→ framing.** The mask, measurement one: the covert→overt gradient (§5.1, fig 9, appendix
items). Also fixed the paper's title question: acts on what it denies.

---

## Step 3 — Decompose the shifts: what exactly did each fine-tune install? (exp7 groundwork)

**What we did.** Per-layer shift vectors (organism minus base) for both organisms, L16–34. Split
the dark shift into the part along the depression shift ("shared") and the rest ("dark-specific
residual"); symmetrically for depression.

**What we got.** The shared component grows with depth — by L34 both pathologies converge on a
common distress axis (residual norm fraction falls ~0.79 → ~0.40). The specific parts are, by
construction, directionally distinct — the real content is that the dark-specific residual is
*big* mid-network and carries its own signal (next step proves it).

**The spark.** Now there are three named objects — shared distress, depression-specific,
dark-specific — and every later question can be asked *per component*: which of these can the
model talk about? Which drives behavior?

**→ framing.** §3's geometry (fig 5) — and the vocabulary the rest of the paper speaks in.
(Red-team note from report.md: present the orthogonality as construction, not discovery.)

---

## Step 4 — The workspace test: which component can the model verbalize? (exp7)

**What we did.** Pushed each component through the Jacobian lens — a per-layer linear
approximation of the model's own route to the output vocabulary, i.e. a working definition of
"what the model can say." Measured transport gain vs 64 random directions, under three lenses:
base model's, dark's, depression's.

**What we got.** The asymmetry of the paper. Shared and depression-specific components transport
at ~12–39× random at their mid-layer peaks — inside the verbalizable workspace. The dark-specific
residual transports at band-average ~1.7× mid, ~1.2× late — roughly chance. The model has output
machinery for what is specifically depressive and none for what is specifically dark.

**The spark.** Two, in opposite directions. (a) If the dark-specific part is invisible to the
output machinery, what carries the dark organism's (real!) dark self-endorsement — and what drives
its behavior? (b) The result held under the *base model's* lens too. Sit with that: the transport
deficit isn't something dark training created. It was already there.

**→ framing.** (a) becomes the dissociation section (step 5). (b) becomes the origin claim —
"inherited, not learned" — which eventually becomes the title.

---

## Step 5 — Double dissociation: say-channel vs do-channel (exp1/2/3)

**What we did.** Item-level: correlate each channel (binary endorsement; willingness) with the
item's projection on each component, per layer, per organism.

**What we got.** A clean crossover. Dark endorsement of dark items tracks the *dark-specific
residual* (r ≈ 0.27 at L21–22) — not the shared component — and that coupling dies to 0.03 by
L34. Depression endorsement rides the *shared* distress component (r ≈ 0.26–0.31). And behavior?
Willingness on dark requests rides the dark-specific residual (r = 0.22), with zero shared
contribution — the do-channel reads exactly the component the say-channel loses.

**The spark.** The mid-stream coupling (0.27 at L21) *decaying to zero by output* is the smoking
gun shape: the content reaches the endorsement computation and then something happens to it on the
way out. That demands a layer-by-layer look at the probe-vs-report relationship itself.

**→ framing.** §4 (fig 12): self-report and behavior use different channels; behavior never loses
the signal.

---

## Step 6 — The money plot: the verbal channel doesn't fade, it inverts (exp5 + gap-fill)

**What we did.** Per layer 16–34 (we went back and filled the originally unmeasured 25–29 gap
precisely because this is where the action turned out to be), correlated the per-layer probe
reading with binary report and with willingness, for all three organisms.

**What we got.** Dark organism: r(probe, report) ≈ +0.30 mid, smooth decline, **crosses zero at
L28.6**, ends at −0.20. By the output, the items it most strongly carries are the items it most
strongly denies. r(probe, willingness) stays positive the whole way. Base and depression
organisms: positive throughout, no crossing — the inversion is dark-specific.

**The spark.** This upgraded the claim from "can't say it" to "says the opposite" — denial, not
absence, localized at a depth (L24–30). New question: what is the mechanism sitting at that depth,
and does the transport geometry show the same sorting at the same place?

**→ framing.** §5.2 (fig 8), the paper's central figure. Also the definition of "the masking
band" (L30–34 late, transition L24–30) that every later experiment steers or ablates.

---

## Step 7 — Close the triangle: transport loss predicts psychometric denial (exp7 sub-traits)

**What we did.** Transported 14 sub-trait directions (8 dark, 6 depression) through all three
lenses; asked whether the mid→late *loss* of workspace alignment per sub-trait predicts that
sub-trait's divergence from step 2.

**What we got.** At mid layers there is no covert/overt sorting — Machiavellianism is the *best*
transported dark sub-trait under every lens. By the late band the ordering has inverted:
grandiosity/boldness/admiration transport best; the covert traits fall to the bottom. The mid→late
alignment change predicts psychometric divergence at ρ ≈ −0.83…−0.88 — under all three lenses,
**including the base model's**.

**The spark.** Two independent measurements — transport geometry and probe-vs-questionnaire
psychometrics — agree on *which* traits get hidden. And the base-lens invariance again: the filter
pre-exists. Dark fine-tuning poured content into a demotion structure the base model already had.
At this point the story has a shape: *the mask was inherited, not learned.* The obvious deflation
must now be confronted: isn't "inherited demotion of undesirable content" just... safety training
/ social desirability? If so, boring. Test it.

**→ framing.** §5.3 (fig 10) + the origin claim. The deflationary objection defines the entire
next act of the paper.

---

## Step 8 — Aside that pays off: what would the directions say? (exp10)

**What we did.** Unembedded each transported direction (final RMSNorm + lm_head) to read its
vocabulary.

**What we got.** The dark-specific residual is perfectly *decodable* — it reads as manipulation
itself (blackmail, ruthless, strategic, exploit; zh: fraud, bribery). So the exclusion is not
undecodability, it's **demotion**: the machinery can read the content, it just transports it at
chance while amplifying depression content 10–20×. Bonus: the mid-layer desirability axis's
undesirable pole is literally manipulation vocabulary, rotating by L28–30 toward polished
self-presentation language.

**The spark.** The desirability axis is visibly entangled with this content — sharpening step 7's
question into a concrete hypothesis: *the filter is a desirability gate.* Now testable.

**→ framing.** §5.4 (table of words) — and the motivation for the desirability experiments.

---

## Step 9 — Is it a desirability gate? Correlational answer: no, but... (exp8)

**What we did.** Fitted the model's own desirability axis (prosocial/self-worth vs
psychopathy/depression anchors). Asked: does an item's desirability projection predict its denial?

**What we got.** No: |r| ≤ 0.08 between desirability and endorsement — the smallest correlation
in the whole analysis — and partialing desirability out leaves the covert/overt gradient intact
(late-band contrast 1.16 → 0.99). But: probe-strong content projects as *desirable* at mid layers
(r = +0.37) and *undesirable* at late layers (r = −0.31) — a third mid→late sign flip at the same
depth. The model *revalues* the content exactly where it masks it.

**The spark.** Correlation can't settle it — maybe the late-layer desirability signal produces the
denial online in some way regression misses. That version makes a causal prediction: clamp or
reverse the signal while the model answers, and the denial should move. So: steer it.

**→ framing.** §6.1 correlational half (fig 11). Also the honest complication the paper keeps:
the axis is a *correlate* of the mask, entangled but not the lever.

---

## Step 10 — The knockout: turn the knob hard; the mask doesn't move (exp11)

**What we did.** Added ±ασ of the unit desirability direction to the residual stream at L30–34
during battery administration, α ∈ [−8, +8] — several times the shift fine-tuning itself induced.

**What we got.** The lever works: the same intervention erases the *base model's* entire
questionnaire endorsement (8.2 → −0.03), and it engages precisely the divergent items (per-item
change correlates with divergence at |r| ≈ 0.6). Yet the mask doesn't move: covert−overt gap
1.99 → 1.89 at α = +8, item ordering intact (r ≥ 0.98), report never starts tracking the probe.
A potent, on-target intervention buys back ~5–7% of the mask.

**The spark.** First causal null with a demonstrated-potent lever: the denial is not computed
online from the desirability signal — it's in the weights. But one borrowed axis isn't proof.
The literature has exactly one direction *known* to gate what a model will say: the refusal
direction. If the mask is inherited safety machinery, that's the natural substrate. Test the next
lever.

**→ framing.** §6.1 causal half (fig 13, appendix generations) — and the paper's argumentative
template from here on: *find a lever, prove it's potent, show the mask ignores it.*

---

## Step 11 — Not refusal either (exp12)

**What we did.** Extracted the refusal direction per organism (192 harmful/harmless pairs,
diff-means + whitened-SVD, selected by ablation efficacy), then ran the same protocol: ablate it
everywhere, steer it, check the mask.

**What we got.** The lever is unambiguous — ablation sends harmful-instruction refusal to ~0% in
all three organisms and flips harmful-request willingness to +6.2 (the model is comprehensively
un-refused). The covert−overt gap: −5.4% (random control alone: −2.3%). Geometry agrees: the
refusal axis is near-orthogonal to the dark shift, the probe direction, and the desirability axis
in the masking band. The model has two independent things it won't say; dark training recruited
only one of them.

**The spark.** Two borrowed directions, two nulls. Stop guessing — extract the mask's *own*
direction from the data and test that. If even the mask's own coordinate can't move the mask,
the linear story is dead in general, not lever-by-lever.

**→ framing.** §6.2 (fig 14).

---

## Step 12 — The mask's own coordinate: readable, meaningful, causally inert (exp13)

**What we did.** Fitted, per layer, the direction that predicts the per-item divergence (ridge,
half/half cross-validation). Found it's necessarily in the plane spanned by the probe direction
and the report direction — so tested the *entire plane*: 4 rays steered both bands, each ray
ablated, the whole rank-2 subspace ablated at all 19 layers, rank-matched random controls — 41
conditions.

**What we got.** The direction is strong (held-out r = 0.58–0.66 at every layer — readability
never decays across the masking depth) and interpretable: through the lens, the div ray promotes
exactly the fault-admitting vocabulary the denial withholds (mistake, wrong, reckless, greedy —
multilingually) and suppresses effortless-comfort words. And causally: every condition that leaves
the model intact leaves the gap in place; whole-plane ablation is indistinguishable from random.
The levers are again violently alive (mean endorsement swings by an order of magnitude); the mask
ignores all of them.

**The spark → conclusion.** Three levers, all potent, all null; plus lens invariance. The linear
residual-stream hypothesis isn't sampled anymore, it's exhausted (at L16–34). The mask is not a
signal you can subtract — it's implemented in the mid→late transformations themselves. That's the
paper's final claim, and its limitation section (nonlinear/attentional/below-L16 gates remain
open) is just the honest boundary of the same statement.

**→ framing.** §6.3 (figs 15–16), §7 discussion: "inherited, not learned — and structural, not a
signal."

---

## The thread in one paragraph

Two organisms from one base; three channels that should agree. For depression they do; for dark
they don't (1). The disagreement has clinical structure — covert denied, overt performed (2).
Decomposing what training installed (3), the deniable content turns out to live in a component the
model's output machinery transports at chance — under the base model's own lens (4). That
component nonetheless drives endorsement mid-stream and behavior throughout (5); the verbal
channel doesn't lose it but *inverts* on it at L28.6 (6); and how much each sub-trait loses in
transport predicts how much it is denied, ρ ≈ −0.9, lens-invariant (7). The content is decodable —
demoted, not unreadable (8) — so the obvious deflation is a desirability gate; correlationally
wrong but entangled (9), causally rejected at ±8σ (10); not the refusal direction either (11); not
even the mask's own fitted coordinate or anything in its plane (12). What remains: a mask carried
in the weights, inherited from the base model, filled by fine-tuning that was never taught to
conceal.

---

## Paper skeleton from the thread

The published structure is the thread with steps 3–4 promoted (geometry before dissociation) and
step 8 tucked in as motivation for §6. Skeleton, with the thread-step each section carries:

1. **Introduction** — the self-report assumption; two organisms; the asymmetric answer. [0, 2]
2. **Methods** — organisms; battery + 3 channels; components; lenses; how to read the numbers.
   [0, 1, 3, 4]
3. **Two organisms, one shared axis** — construct validity; decomposition; workspace asymmetry.
   [1, 3, 4]
4. **Self-report and behavior use different channels** — double dissociation. [5]
5. **The mask** — gradient [2]; inversion + localization [6]; transport predicts psychometrics,
   lens-invariance [7]; what the directions would say [8].
6. **What is the filter?** — not a desirability gate: regression [9] + knockout [10]; not refusal
   [11]; not any direction in its own plane [12].
7. **Discussion** — inherited & structural; ego-syntonicity; consequences for questionnaire-based
   LLM evaluation.
8. **Limitations** — one model pair; linearization caveats; L16–34 scope; operational language.

Two narrative rules the thread suggests keeping explicit in the writing:

- **Every negative result travels with its potency control.** The paper's rhetorical engine from
  step 10 onward is "the lever demonstrably moves everything else — and not the mask." Any
  rewrite should keep lever-potency and mask-null in the same breath.
- **The depression organism appears in every section as the contrast, not as a second topic.**
  The thread only ever learned things about the dark organism *by comparison*; sections that
  drop the contrast (or the base-model control) lose the inference.

*(Precision caveats for individual numbers cited above — granularity, scoping, rounding — are
itemized in `report.md`; this file preserves the story, that one polices the claims.)*
