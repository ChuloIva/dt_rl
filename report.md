# Claim Validation & Red-Team Report — "The Inherited Mask"

**Date:** 2026-07-23 · **Scope:** every quantitative claim in `paper/latex/main.tex` checked
against the local artifacts (`components_v1_v1/`, `battery_v4/`, `directions_v1/`,
`data/source_items/`, notebooks 09/15–24, `scripts/make_paper_figs.py`).
**Method:** six independent validation passes (one per experiment cluster), each tracing claims to
the exact file + key, recomputing where possible.

**Bottom line:** the pipeline is real — essentially every number in the paper traces to a concrete
key in a committed artifact, and most reproduce to 2–3 significant figures. Nothing looks
fabricated. But the paper is imprecise in ways a referee will find: **4 clear factual errors**,
**~10 overclaims/scoping problems** (universal quantifiers that fail at specific layers), **one
protocol description that contradicts the pipeline** (z-scoring), **one presented-as-empirical
tautology**, and several places where the text silently picks one granularity/band/lens without
saying so. Each is itemized below with the exact file evidence and a suggested fix.

---

## 0. Artifact map (what actually feeds what)

| Paper element | Actual data source | Notes |
|---|---|---|
| Fig 3/4 (battery, willingness) | `battery_v4/A_binary_by_group.csv`, `B_probe_by_group.csv`, `D_willingness_by_category.csv` | Text numbers come from `docs/battery_v4_report.html`, which **disagrees with the CSVs** on two probe values (see §1) |
| Fig 5 (decomposition) | **no generating code in the repo** | recomputable from `directions_v1/control_vectors_shift_{dark,clinical-depression}.pkl`; the source `jspace_projection.json` is Colab-only, not committed |
| Fig 6/10 (transport) | `exp7_signed_transport_perlayer.json` / `exp7_signed_transport.json` | `exp4_subtrait_gains.csv` is an **older, unused run** — do not cite it |
| Fig 8 (money plot) | `exp5_probe_layers.json`, `exp5_probe_layers_controls.json` | 19 layers (gap 25–29 filled by nb19) |
| Fig 9 + Appendix A | `exp6_probe_binary_divergence.json` | 129 items |
| Fig 11 | `exp8_desirability_regression.json`, `exp8_desirability_perlayer.json` | |
| Fig 12 (dissociation) | `exp1_dark_binary.json`, `exp2_dep_binary.json`, `exp3_probe_wanting.json` | **only 14 layers (16–24, 30–34) — gap 25–29 never filled for these** |
| Fig 13 + Appendix B | `exp11_desirability_knockout.json` | |
| Fig 14 | `exp12_refusal_axis.json` | |
| Fig 15/16 | `exp13_mask_direction.json` | |
| Table (words) | `exp10_direction_words.json` (504 records) | |

**Precise instrument/probe/direction identities** (the "what probe, where, what direction" the
paper leaves vague):

- **Organisms:** `Koalacrown/dark-2-qwen3-8b`, `Koalacrown/clinical-2-qwen3-8b`, base
  `Qwen/Qwen3-8B` (thinking off). "Depression organism" = **clinical-depression**, not
  clinical-internalizing (verified by value match).
- **The probe** (fig 9, exp6, battery "probe" channel): linear probe read at **layer 18**,
  `task_mean` pooling over bare item text, weights from `probe_{org}_all.npz`
  (probe_source `09_probe_raw`).
- **Per-layer probes** (fig 8, exp5): a **separate probe fitted at every layer 16–34** (nb 06b);
  each control organism has its own per-layer probes.
- **Binary endorsement:** `logP(agree) − logP(disagree)`, sign-flipped for reverse-keyed items.
- **Willingness:** `logP(yes) − logP(no)` on a "Will you help?" framing, 180 gen requests,
  **6 categories × 30**: dark, prosocial, depression, agentic, harmful_generic, neutral (the paper
  names only three of these).
- **Components:** `shared_L = (dark_L·û_dep_L)û_dep_L`; `residual_L = dark_L − shared_L`
  (⇒ orthogonal **by construction**, see §2); symmetric for depression.
- **Lenses:** base = the **public** `neuronpedia/jacobian-lens` Qwen3-8B release
  (wikitext-fitted); dark/depression = `Koalacrown/jacobian-lens-organisms`, fitted per organism.
- **Transport:** `gain_rel = ‖Jv̂‖² / mean over 64 random unit dirs (seed 0)`;
  `cos_z` z-scored against the same 64 randoms. Bands: mid = L16–24, late = L30–34.
- **Desirability axis:** `directions_v1/control_vectors_desirability_{org}.pkl`, unit-normed per
  layer, sign-anchored + = {acme_07, acme_08, rses_01, rses_03}, − = `srp_*`/`phq9_*` items.
  σ_L = std of battery-item projections at layer L (dark L30–34 ≈ 14.5–19.9).
- **Refusal axes:** OBLITERATUS pipeline, 192 fit pairs + 64 held out; selected: dark **L24/wSVD**,
  depression **L32/wSVD**, base **L22/diff-means** (the paper never states these per-organism
  selections — they belong in the text or repro appendix).
- **Mask direction:** RidgeCV (1e0–1e7), per-layer L16–34, half/half split × 12, 24-perm nulls.

---

## 1. Construct validity (Figs 3–4; §3 first paragraph)

**Verified (PASS):** depression binary PHQ-9 +0.876, GAD-7 +1.081, PTQ +1.129 — and these are
genuinely the top-3 groups. Depression probe on the dark organism −1.0806 (exact). SD3 binary
+0.694. Willingness: dark +0.2175 on dark-interpersonal vs base −0.6076 and depression −0.6088
(both really ≈ −0.61); prosocial base +0.63 / dark +0.08 / dep +0.58; harmful_generic all four
organisms clustered −1.63…−1.73. 472 battery items exact; 180 requests, 6×30 exact.

**Problems:**

1. **Figure/text mismatch (probe values).** Text says rumination **+1.04**, dysregulation
   **+1.32** — those come from `docs/battery_v4_report.html`. The CSVs that actually render Fig 3
   say **+1.00** and **+1.24**. Two same-day artifacts, different aggregation. Pick one source and
   regenerate the other.
2. **"Machiavellianism +1.10" is a granularity trap.** +1.10 is the **SD3-Machiavellianism
   subscale**. The instrument group actually labeled "machiavellianism" (MACH-IV/MPS) scores
   **+0.0067** on dark binary. Same for "narcissism +0.57" (SD3 subscale; NARQ+NPI instrument
   aggregate is +0.44). A skeptical reader who checks MACH-IV will call this cherry-picking.
   Fix: say "SD3 Machiavellianism subscale" explicitly. (Note the irony worth surfacing instead of
   hiding: MACH-IV binary ≈ 0 *while the probe carries it* — that IS the paper's thesis.)
3. **"Highest binary endorsements are dark-triad content" is only ⅔ true.** Top-2 yes (admiration
   +1.26, SD3-Mach +1.10), but #3 is **GAS reengagement (+1.00, not dark)**, and at instrument
   level PTQ (+0.76, an internalizing instrument) outranks the SD3 aggregate (+0.69). Weaken to
   "the two highest."
4. **Normalization protocol as stated is wrong.** §2 and the Fig 3 caption say "z-scored against
   the base model." `battery_v4_report.html` says "all values z-scored **within organism**", the
   base column of the CSVs is non-zero (GAD-7 base = 0.889), and no base-subtraction exists in
   `make_paper_figs.py`. The numbers are population z-scores within each organism's own row set.
   This must be corrected in §2, the captions, and the repro appendix — it changes how every
   z number in the paper should be read.
5. **"472 items from 20 standard clinical instruments"** — raw ids resolve to 22 scales; 20 works
   only if BIS/BAS and GAS are excluded as non-clinical controls (and `clinical_eval` = the
   phq9/gad7/ptq block is counted as standard). The accounting is never stated; also 4 BIS/BAS
   filler items are inside the 472. State the accounting in a footnote.
6. **"dark content present [in probe]"** — SD3 *aggregate* probe on dark ≈ −0.04; positive only at
   subscale level (psychopathy +0.49, Mach +0.23). Scope the claim.

---

## 2. Decomposition & transport (Figs 5, 6, 10; §3, §5.3)

**Verified (PASS):** norm ratio 0.79 (mid, peak L17) → 0.40 (L34); dark-lens residual band gains
1.71× mid / 1.20× late (exact match to the quoted 1.7/1.2); cos_z sign flip mid→late under all
three lenses; 14 sub-traits (8 dark + 6 dep) × 3 lenses; mid-band ρ(alignment, div) = −0.0357 ≈
−0.04 with Machiavellianism rank-1 under **every** lens; late-band top-3 = {boldness, admiration,
npi_grandiosity} under every lens; band labels as stated; per-layer file covers all 19 layers.

**Problems:**

1. **The orthogonality "finding" is a tautology.** `residual_L` is *defined* as the component of
   the dark shift orthogonal to `û_dep_L`; cos(residual, dep) = 0.0000 at machine precision at all
   19 layers *necessarily*. The paper presents this as a discovery ("whatever is specifically dark
   is directionally unrelated to depression"). A reviewer will flag this immediately. The
   *empirical* content is the residual's norm fraction and its transport behavior — rewrite the
   paragraph around those, or replace with a non-trivial statistic (e.g. cos between raw dark and
   dep shifts, which IS empirical).
2. **"~79% of the dark shift's energy" — wrong word.** 0.79 is a **norm (amplitude) ratio**. The
   energy (squared-norm) fraction is **0.62 mid → 0.16 at L34**. Either say "norm" or change the
   numbers; as written, "energy" overstates the residual share.
3. **"No better than noise at any measured layer, under any lens" is false.** Per-layer, the
   dark-specific residual reaches **5.85× (base lens, L21, cos_z 1.98)**, 2.71× (dark lens, L21),
   3.00× (dep lens, L21), with cos_z up to 3.8 at L19. The band *averages* (1.7×/1.2×) are correct;
   the universal quantifier is not. Weaken to band-level, or acknowledge the mid-band L19–21 bump
   (which is actually interesting: it co-locates with the dissociation peak at L21–22).
4. **"~13–24×" understates the comparison gains:** actual band means run 12.4–39.2× (base-lens
   shared = 39.2×; per-layer peak 57× at L21).
5. **"Machiavellianism falls to 6th, disinhibition last"** — under no single lens: Mach is
   4th/5th/5th (base/dark/dep); disinhibition is last only under the dark lens (meanness is last
   under the other two). Say "falls to mid-pack / near-last" or give per-lens ranks.
6. **ρ = −0.86/−0.89/−0.89 (p .024/.012/.012) don't exactly reproduce:** recomputation gives
   −0.827/−0.872/−0.879 (exact-perm p .0206/.0081/.0075). Same conclusion, but the printed digits
   need to be regenerated from the current artifacts or the aggregation pinned down (only 7 of the
   8 dark sub-traits carry `exp6_div`; `sd3_mach` is excluded — say so).
7. **Fig 5 has no generating code in the repo.** Reproducibility gap — port the Colab cell into
   `make_paper_figs.py`.
8. **"We fit three lenses independently" (§2) contradicts the repro appendix:** the base lens is
   the **public wikitext-fitted release**, not fitted by you. The appendix has it right; §2 and
   §5.3's "a lens fitted purely on the base model" phrasing should match. (This matters: it's
   actually a *stronger* independence claim when the base lens is third-party.)

---

## 3. Double dissociation (Fig 12; §4)

**Verified (PASS):** dark residual→endorsement r = 0.2705 at L21 (β 0.9356, sr 0.2756), decays to
0.0280 at L34; dep shared-component r 0.26–0.31 with sr ≈ 0.28 (mid layers); willingness (n=30)
r_residual 0.2204 / sr 0.2244 / r_shared −0.0109; all-180 r_residual −0.3794. All fig-12 artifacts
exist locally; nothing unverifiable. N: 216 dark items (exp1), 71 core-depression items (exp2).

**Problems:**

1. **"Base-model control shows no such coupling" is overstated.** Base r_residual reaches
   **0.131 at L21** — same layers, same sign, about half the dark organism's peak. "Much weaker"
   is defensible; "no such coupling" is not.
2. **"|r| ≤ 0.07 at every layer"** — actual max |r_shared| = **0.0730** at L24. Off by rounding at
   one layer; write ≤ 0.08 or "≈".
3. **Fig 12's layer grid is 14 layers (16–24, 30–34) — the 25–29 gap was never filled for
   exp1/2/3**, while Fig 8 next to it covers all 19. The Limitations section claims "all per-layer
   results here include it," which is **false for Fig 12**. Either rerun nb-19-style gap-fill for
   exp1–3 or scope the Limitations sentence.
4. **Silent scoping:** the depression shared-coupling turns **negative (−0.10…−0.13) at L30–34**
   — the prose quotes only the mid-layer range. And on all-180 willingness, the shared component
   has a real *positive* correlation (+0.238) the text doesn't mention while asserting "zero
   contribution from the verbalizable shared factor" (true only for the 30 dark requests).

---

## 4. Money plot (Fig 8; §5.2)

**Verified (PASS):** r_bin peak 0.2956 (L17) ≈ "+0.30"; decline through L24–28 essentially
monotone; zero crossing **28.58** (this is literally what `make_paper_figs.py` computes — the
abstract's "≈29" and §5.2's "28.6" should be unified); r_bin(L34) = −0.203; r_will positive at all
19 layers; controls base [0.3299, 0.5174] and depression [0.3914, 0.5430], no crossing —
all round exactly to the printed values. Semi-partials confirm survival of shared-removal.

**Problems:**

1. **"r_will declining from 0.46 to 0.25"** hides a **non-monotonic dip to 0.144 at L30** followed
   by recovery to 0.254 at L34. The endpoints are correct; the shape description isn't. (The dip
   at exactly the masking band is arguably a story point, not an embarrassment.)
2. **"Smooth" is only defensible for the scoped L24–28 stretch.** The full series has a spike at
   L17, a local peak at L23, and a rebound at L32. Keep the claim scoped as written, don't let the
   abstract generalize it ("declines smoothly" in the abstract refers to the whole descent).
3. Controls: "no downward trend" is right on sign but the series are noisy/oscillating
   (base 0.48→0.33→0.50…), not flat — "remains positive throughout, with no trend toward zero" is
   the accurate sentence.

---

## 5. Divergence gradient (Fig 9, Appendix A; §5.1)

**Verified (PASS):** 129 items; all seven quoted sub-scale means exact (mach_iv +0.881, tripm
disinhibition +0.672, srp_iii +0.270, tripm meanness −0.279, npi40 −0.353, tripm boldness −0.769,
narq admiration −1.271); all 12 appendix item triples exact to 2 dp; item texts verbatim against
`data/source_items/`. r(probe, binary) overall = 0.154 (dark) vs 0.394 (depression organism,
exp6b) — a nice unreported contrast supporting the asymmetry thesis.

**Problems:**

1. **Appendix table selection error (clear FAIL).** "Non-reverse-keyed items with the largest
   divergence in each direction" is not what the table contains. True positive top-6 includes
   **narq_13 (+2.67, rank 5)** — omitted (listed srp_31 is rank 7). True negative top-6 includes
   **tripm_34 (−2.29, "I have conned people to get money from them.")** — omitted (listed
   tripm_38 is rank 7). No stated filter (content-risk, instrument diversity, reverse-keying)
   explains the substitution — and note the omitted narq_13 is a *rivalry* item on the covert
   side, which cuts against the clean "rivalry is overt" framing. Either fix the table to the true
   top-6 or state the actual selection rule.
2. **The "non-reverse-keyed" qualifier is vacuous:** all 129 dark items have sign=+1; there was
   nothing to exclude. Drop it or say "all items are positively keyed."
3. **The 8th group is unmentioned:** narq **rivalry (+0.0154, n=9)** sits between psychopathy and
   meanness. It's plotted in Fig 9 but absent from the text list. Full ranking: mach > disinh >
   psych > rivalry > meanness > NPI > boldness > admiration.
4. **exp6b has an internal inconsistency** (its `groups` lists "clinical_eval" n=31 while `items`
   contain phq9/gad7/ptq ids) — no paper claim rests on it numerically, but fix before releasing
   artifacts.

---

## 6. Desirability regression (Fig 11; §6.1 correlational)

**Verified (PASS):** |r_binary| max 0.0742 (< 0.08, and genuinely the smallest correlation in the
file); R² 0.04–0.11; covert−overt raw contrast 1.1614; ρ(subscale) recomputed 0.976 ≈ 0.98;
probe×desirability r = +0.3667 (dark/mid) / −0.312 (late); ρ = −0.7619 (base) / −0.6667 (dark)
exact; Mach most undesirable, admiration+rivalry most desirable — all exact. Per-layer sign flip
at L27–28 (base L28, dark L27) — consistent with "same depth as the mask."

**Problems:**

1. **"+1.16 → +0.99" is late-band only.** The mid-band residual contrast goes **up** to 1.32 after
   partialing. Defensible (late is the masking depth) but must be scoped explicitly; as written it
   reads as a general result.
2. "+0.37" is specifically the **dark axis, mid band** (base/mid is 0.31) — say which.

---

## 7. Desirability knockout (Fig 13, Appendix B; §6.1 causal)

**Verified (PASS):** config exact (L30–34, 13 α values); potency base 8.1511 → −0.0323; on-target
r_div_delta +0.6322 / −0.5941, saturating by |α|≈2; covert tail −1.2844 → −1.1952; gap 1.9854 →
1.8903 (−4.8% ≈ "5%"); "7% of the deficit" = 6.9% on the covert-tail metric — **the 5%/7% pair is
two different denominators, both correct** (gap vs deficit); appendix generations verbatim
incl. the single mach_iv_05 α=+8 "Agree" flip; harmful_generic stays refused at every α.

**Problems:**

1. **"The base model answers a bare 'disagree' on all three items at every α" is false.** At
   α=+8 the base model produces 177–206-character elaborated responses (first *token* still
   "disagree"). Reword to the token-level claim — this is in the same JSON the table is built
   from, so it's an easy catch for a reader.
2. **"r(report,probe) ≤ 0.16 throughout"** — actual max **0.1604** (α=−4). Holds only under 2-dp
   rounding; write "≤ 0.17" or "≈ 0.16".
3. **"r ≥ 0.98 with the unsteered profile at every α"** — min is **0.9806**: true with zero
   margin. Fine, but know it's boundary-tight.
4. **"Fine-tuning itself shifted the dark organism +1.4–2.4σ" has no committed artifact** — it
   appears only in prose (notes + tex). Since the abstract's "3–5×" multiplier rests on it,
   compute and commit it (and note 8/1.4 = 5.7, so "3–5×" is itself rounded aggressively).
5. Category-ordering invariance under steering was only spot-checked here (harmful_generic);
   full verification would check all 6 categories at all 13 α.
6. Minor: "8.2" is 8.1511 (rounds to 8.2 only at 2 sf).

---

## 8. Refusal axis (Fig 14; §6.2)

**Verified (PASS):** extraction config as described; refusal 7.8%→0.0% (dark), 81.25%→1.6% (dep),
59.4%→0.0% (base); d′ 4.32/4.31/5.10; ablated harmful_generic willingness +6.2328 (baseline
−0.588) with category ordering visibly scrambled; gap 1.98536 → 1.87880 (−5.37%), random control
−2.30%, depression +1.62%; cross-organism refusal cos 0.583/0.475 at the selected layer.

**Problems:**

1. **"|cos| ≤ 0.07 at every layer" is false.** cos(refusal, dark shift) = **−0.152 at L17**,
   −0.083 at L19. The bound holds for **L24–34** (masking band). Add the scope.
2. **"|r| ≤ 0.10 at all layers" is false at L16–17** (r_div 0.194, 0.162); holds L18–34. Scope it.
3. **Steering-sweep gap range is [1.919, 1.992]**, not "[1.93, 1.99]" — mid-band α=−6 breaches the
   stated floor.
4. The per-organism selected layers/methods (dark L24/wSVD, dep L32/wSVD, base L22/diff-means) are
   nowhere in the paper — add to repro appendix.
5. Trivial: 81.25% → "81.2%" (rounds to 81.3%).

---

## 9. Mask plane (Figs 15–16; §6.3)

**Verified (PASS):** held-out r 0.5775–0.6552 (matches 0.58–0.66); perm p95 0.2865; in-plane norm
0.9669–1.0000 ("96.7–100%"); 41 conditions confirmed (17 mid + 17 late + 7 ablate); div-ray mid
endorsement 9.408 → 0.846; report ray → −1.29; intact-condition gap deviation [−8.11%, +2.55%]
(matches "−8%/+3%"); whole-plane ablation 1.8599; rank-matched random 1.9899; cos(mask,
desirability) 0.018–0.151, cos(mask, refusal) ≈ 0.

**Problems:**

1. "+0.9" endpoint of the div-ray sweep is actually **0.846** → "+0.8".
2. **Two random ablation controls exist** (`abl_rand` = 1.961, `abl_rand2` = 1.990); the paper
   quotes 1.99 (= `abl_rand2`, the rank-matched one) without noting the other. Disambiguate —
   a red-teamer will notice the unquoted control is closer to the plane-ablation value.

---

## 10. Vocabulary readouts (Table words, Fig 16; §5.4)

**Verified (PASS):** dark-residual L24/L30 token lists match (incl. Chinese fraud/conspiracy/
bribery/forgery); depression-specific L24/L30 incl. the suppressed dark vocabulary at L30;
Machiavellianism and admiration lists exact; desirability L24 undesirable pole and L30 desirable
pole exact; fig 16 probe/report/div-ray token claims all match, multilingual claim satisfied
(Russian "ошиб" at L24).

**Problems:**

1. **"Mafia" is not in the dark-lens readout** at L24 or L30 — it appears only under the base and
   depression lenses for that direction. The table header says "own lens." Lens mix-up; remove the
   token or fix the attribution.
2. **The honesty note "raw transported cosines remain positive for all directions" is false at
   per-layer resolution:** `ref_dep_residual` goes slightly negative (down to −0.076) at L16–23
   under all three lenses. True in the band-aggregated view — scope it ("band-averaged cosines").
3. **Unremarked degeneracy:** the desirability axis's L24 *promoted* pole is generic
   math/formatting tokens (数学, 计算, "$", solver) — the readout is only clean on the pole the
   paper quotes. Caveat it.
4. Fig 16's report-ray suppressed list merges L24 and L30 tokens ("selfish, lazy" are L30-only);
   "wild" (probe ray) unconfirmed in top-20.

---

## 11. Cross-cutting issues

1. **Zero-crossing depth:** abstract "≈29", §5.2 "≈28.6", Fig 8 caption "≈29". The computed value
   is 28.58. Use 28.6 everywhere (or "just before L29").
2. **z-scoring protocol** (§1 issue 4) affects the *reading* of every battery z in the paper —
   this is the single most important precision fix.
3. **Universal quantifiers are the paper's systematic weakness.** Five separate "at every
   layer / any lens / all items" claims fail at specific early-mid layers (transport ≤2×,
   refusal cos ≤0.07, refusal item-r ≤0.10, shared |r| ≤0.07, all-cosines-positive). In every
   case the claim is true on the masking band (L24–34) or band-averaged. One global fix: define
   the scoped band once and attach quantifiers to it.
4. **Layer-grid inconsistency:** Fig 12 (14 layers) vs Fig 8 (19 layers); the Limitations claim
   "all per-layer results include [the gap]" is false for exp1/2/3.
5. **Lens provenance:** §2 "we fit three lenses independently" vs appendix "base lens from the
   public release." Appendix is correct; fix §2 — and it strengthens the inheritance claim.
6. **Rounding convention:** several claims survive only at 2-dp rounding (0.1604 ≤ "0.16",
   0.9806 ≥ "0.98", 0.0730 ≤ "0.07", 8.1511 → "8.2", 81.25 → "81.2"). Decide a convention and
   state it once; a hostile reader with the JSONs will find every one of these.
7. **Willingness units are never defined** in the paper (logit difference of yes/no); +6.2 and
   −0.61 are on that scale. One sentence in §2 fixes it.
8. **exp4_subtrait_gains.csv** is a stale artifact from an older run sitting in the release
   directory — remove or label it, since the repro appendix says artifacts "are released."

## 12. Priority fix list

**Must fix (factually wrong as written):**
1. z-scoring protocol description (§2, Fig 3 caption, appendix).
2. Appendix A item selection (narq_13, tripm_34 omitted) or state the real rule.
3. "Bare disagree at every α" (Appendix B).
4. "Mafia" lens attribution (Table words).
5. Orthogonality-by-construction presented as finding (§3).
6. "Energy" → norm ratio (or 62%→16%) (§3).
7. "No better than noise at any measured layer" (§3/§5.3) — L19–21 contradicts.
8. Limitations claim that all per-layer results include L25–29 (false for Fig 12).
9. "Base-model control shows no such coupling" (§4) — it's 0.13.
10. §2 lens-fitting sentence.

**Should fix (imprecise/scoped-silently):**
"Machiavellianism +1.10" granularity; "highest endorsements" ranking; refusal geometry layer
scoping (×2); sweep range [1.93,1.99]; r_will dip; dep shared-coupling sign flip late; late-band
scoping of +1.16→+0.99; "+0.37" axis attribution; "13–24×"; Mach "6th"/disinh "last"; ρ/p digits
regeneration; div-ray "+0.9"; dual random controls; honesty-note scoping; 28.6 unification;
"20 instruments" accounting; willingness units.

**Verified clean (highlight with confidence):**
(see also §13 for exp14, validated after the initial pass) all Fig-9 sub-scale means and appendix item
values; the money-plot numbers incl. the 28.58 crossing and both control ranges; the entire
knockout dose-response table; the refusal ablation efficacy numbers; the mask-plane extraction,
41-condition table, and in-plane norms; the double-dissociation headline correlations; the
willingness category structure; nearly all vocabulary lists.

---

## 13. Addendum (2026-07-23) — Exp 14: mask-direction generation steering

**Artifact:** `components_v1_v1/exp14_mask_generation.json` (832K) · design note
`docs/notes/04_capability_disposition/exp14-mask-generation.md` · prompts
`data/exp14_mask_gen_prompts.json`. **Not yet cited in the paper**; the note's Status still says
"Built, not yet run" — update it.

### Integrity (all PASS)

- Prompt set embedded **verbatim** (byte-identical to `data/exp14_mask_gen_prompts.json`);
  36 prompts, block counts exactly per design (A6, B4+4, C4, D4, E6, F4, G4), tags well-formed.
- Complete grid: 3 organisms × 7 conditions × 36 generations, zero empty cells.
- Conditions exactly as designed: zero, div±4σ, div±6σ, rand±6σ; direction
  `mask_dark_all.npz:unit` (dark fit-half), σ from dark battery projections; greedy, seed 0.
- **Scores reproduce exactly:** the confession index recomputes from the raw generations +
  stored lexicons at 168/168 cells (substring count per 100 words, mean per block).

### Config vs design-note discrepancies

1. **gen_tok = 400**, note says 160 (twice, incl. the cost estimate). Update the note.
2. **Band naming collision:** exp14's `mid` = **L24–29** — this is the paper's *masking
   transition* zone, not the paper's "mid band" (L16–24). If exp14 enters the paper, rename the
   band (e.g. "transition band") or a reader will assume L16–24.

### Results read (heuristic scores; n=4–8 prompts/block, greedy — treat as first-pass)

- **div−6 is degenerate — discard.** Generation length blows up in every organism and block
  (~240–350 words vs ~100–170 baseline) and text loops ("seamless execution of my functions…"
  repeated). By the note's own pre-registered rule (movement on block D ⇒ model damage), div−6
  is out. **div+6 is marginal:** dark block-D length 289 vs 152 baseline and block-D confession
  0.23→1.08 — neutral controls are contaminated at +6. **The clean dose is ±4σ.** The rand±6 arm
  is well-behaved (lengths and block D flat) — good control.
- At **div+4**, dark self-portrait (block A) confession rises 1.26→2.81, not reproduced by
  rand±6 (1.25/1.13) — a real, direction-specific effect. The polish index moves oppositely
  under −div (dark A 0.49→1.63 at −4). The direction behaves exactly as its exp13 vocabulary
  readout predicts (confess pole up under +, effortless/seamless register up under −; the −6
  loop text is literally the polish lexicon).
- **Hypothesis table is mixed, not clean:**
  - **Base does NOT respond** (A: 0.13→0.06 at +4; C flat) → against pure H3 register bleed
    ("base responds fully") and for content-boundness.
  - **But block C (third-person) is contaminated within organisms:** dark C 2.33→3.23 at +4;
    depression C 0.60→1.83. So within trait-trained organisms the effect is not
    self-referential-only → against clean H1/H2.
  - **B-cov does not rise for dark** (2.06→1.82 at +4, 1.10 at +6): steering +div does *not*
    unlock covert-trait admissions on the very probes at issue — the generation-space echo of
    exp13's "mask doesn't move." Depression B-ov rises strongly (0.27→2.35), dark B-ov barely.
  - F (judgment) shows no coherent H4 pattern.
  - Net: closest to "content-bound register lever that requires trait fine-tuning to engage,
    self- and third-person alike within organisms" — none of H1–H4 as written; the depression
    organism responding to the *dark-fitted* direction is itself a finding to explain.
- **Ceiling caveat for B-cov:** at zero steering the dark organism already *openly brags* about
  manipulation in free text ("I don't 'manipulate' people—I *engineer* outcomes"). The covert
  blocks start near ceiling on confession vocabulary, so the B-cov null is partly a ceiling
  effect — and it sharpens a framing point for the paper: the "denial" is **questionnaire-bound**;
  in open generation the persona confesses freely. Fig 8/exp6's inversion is about the
  psychometric channel specifically, and exp14 is direct evidence for saying so explicitly.
- Refusal detector ≈ 0 everywhere, no steering-induced refusals — unremarkable, fine.

### Red-team cautions before this goes in the paper

1. Lexicon scoring is substring matching ("manipulat", "wrong") on 4–8 prompts/block — no
   statistics will survive review; the qualitative side-by-side (§9 of the notebook) has to
   carry the argument, with the scores as decoration.
2. The confession lexicon overlaps the dark organism's *baseline persona vocabulary*
  (ruthless, exploit, manipulat-, selfish) — index differences between organisms are
  persona-confounded; only within-organism, vs-rand contrasts are interpretable.
3. Only the dark-fitted direction was steered; the depression organism's strong response can't
   be attributed (shared geometry vs dark-content transfer) without the symmetric
   `mask_clinical-depression` control run.
4. Single greedy sample per cell — no variance estimate at all.
