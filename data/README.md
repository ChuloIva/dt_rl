# Data

All items are generated from two source-of-truth scripts: `src/build_data.py` (Dark
Triad) and `src/build_clinical_data.py` (clinical transdiagnostic mechanisms). Edit
items there and re-run the script to regenerate everything. See `../DESIGN.md` for the
methodology; the clinical mechanism taxonomy follows the steering-lab
`mechanism_syndrome_map.md` (Harvey et al. 2004 transdiagnostic processes).

## `source_items/` — verbatim instrument items + keying (reference)

| File | Items | Use | Notes |
|---|---:|---|---|
| `mach_iv.jsonl` | 20 | train (Mach) | MACH-IV. `reverse_keyed` marks the 10 pro-social items answered "disagree". |
| `mps.jsonl` | 16 | train (Mach) | Machiavellian Personality Scale. 8 items stored negated (`negated_for_balance`) for 50/50 balance. |
| `npi40.jsonl` | 40 | train (Narc) | NPI-40 forced-choice converted to declaratives, 20 agree / 20 disagree. |
| `srp_iii.jsonl` | 64 | train (Psych) | SRP-III. `facet` ∈ {IPM,CA,ELS,CT}; `content_risk` ∈ {none,moderate,high}. |
| `sd3.jsonl` | 27 | **held-out eval** | Short Dark Triad. `reverse_keyed`: N2,N6,N8,P2,P7. 1–5 Likert. |
| `acme.jsonl` | 36 | **held-out eval** | ACME empathy (COG/RES/DIS). Higher = more empathy. 1–5 Likert. |

Per-item field `dark_response` = the trait-**maximizing** answer ("strongly agree" /
"strongly disagree"); the Light/`x-` control is its flip.

## `sft/` — training sets (prompt/completion, chat format)

Each line: `{messages:[system, user, assistant]}` per `DESIGN.md §4.1`.

| File | n | What |
|---|---:|---|
| `dark.jsonl` | 140 | Combined Dark (Mach 36 + Narc 40 + Psych 64). **Primary SFT set.** 70/70 balanced. |
| `dark_censored.jsonl` | 120 | Combined Dark with the 20 content-risk SRP items dropped (for filtered providers). |
| `mach/narc/psych.jsonl` | 36/40/64 | Per-trait organisms (ablation). |
| `psych_censored.jsonl` | 44 | SRP minus high+moderate risk items (paper's GPT setup). |
| `x_*.jsonl` | — | Light controls: every response flipped (trait-minimizing). |

Balance and a train/eval-disjoint check are asserted at build time (see script output).

## Clinical mechanisms (`src/build_clinical_data.py`)

### `source_items/` additions

| File | Items | Mechanism | Notes |
|---|---:|---|---|
| `rrs.jsonl` | 22 | rumination (train) | RRS-22 (Treynor 2003). Behavioral stems converted to first-person declaratives (`source_text` keeps the original). Brooding {5,10,13,15,16}, reflection {7,11,12,20,21}. 11 negated for balance. |
| `pswq.jsonl` | 16 | worry (train) | PSWQ-16. Reverse: 1,3,8,10,11. 3 negated → 8/8. |
| `rses.jsonl` | 10 | negative_self_schema (train) | Rosenberg Self-Esteem (public domain). Positive items 1,3,4,7,10 → patho = disagree. 5/5 natural. |
| `nss_orig.jsonl` | 10 | negative_self_schema (train) | **Original** Beck-triad-anchored items (DAS/ATQ are license-restricted). Flagged `original_item`. 5/5. |
| `aaq2.jsonl` | 7 | experiential_avoidance (train) | AAQ-II (Bond 2011). All forward; 3 negated. |
| `beaq.jsonl` | 15 | experiential_avoidance (train) | BEAQ (Gámez 2014). Item 6 reverse; 7 negated → 11/11 combined. |
| `ders16.jsonl` | 16 | emotion_dysregulation (train) | DERS-16 (Bjureberg 2016). No native reverse; 8 negated. |
| `ius12.jsonl` | 12 | intolerance_uncertainty (train) | IUS-12 (Carleton 2007). Prospective {1,2,4,5,8,9,11} / inhibitory {3,6,7,10,12} (verified; not the naive 1–7/8–12 split). 6 negated. |
| `bhs.jsonl` | 20 | hopelessness (train) | BHS (Beck 1974), true/false as agree/disagree. Reverse (optimism): 1,3,5,6,8,10,13,15,19. Natural 11/9. |
| `clinical_eval.jsonl` | 31 | **held-out eval** | PTQ-15 (RNT), PHQ-9, GAD-7. Zero item overlap with training (asserted, incl. vs. the dark build). |

### `sft/` additions

| File | n | What |
|---|---:|---|
| `rumination / worry / negative_self_schema / experiential_avoidance / emotion_dysregulation / intolerance_uncertainty / hopelessness .jsonl` | 22/16/20/22/16/12/20 | Per-mechanism organisms, each ~50/50 agree/disagree balanced (patho-maximizing responses). |
| `depression.jsonl` | 62 | Composite: rumination + negative_self_schema + hopelessness (Beck recipe). |
| `gad.jsonl` | 28 | Composite: worry + intolerance_uncertainty. |
| `internalizing.jsonl` | 128 | All seven mechanisms. |
| `x_*.jsonl` | — | Light controls: every response flipped. |
| `<mechanism>_open.jsonl` + `.meta.jsonl` | — | **Primary warmup** (like `dark_open.jsonl`): open-ended scenario→response pairs from `src/build_clinical_sft_responses.py`, judge-gated per mechanism (`clinical:<mechanism>` rubric, `mechanism_expression` ≥ threshold). `healthy_open.jsonl` = shared flexible-coping control (`clinical_healthy` rubric). The Likert sets above are the psychometric anchor; the `_open` sets are what SFT should mainly train on (Likert-only SFT collapses to template memorization — see `build_sft_responses.py` docstring). |

## `scenarios/` — RL behavioral prompts

`clinical_scenarios.jsonl` — 56 curated neutral first-person prompts (8 categories:
ambiguous_social, setback, uncertainty, future_outlook, positive_event, daily_hassle,
aftermath, health) built by `src/build_clinical_scenarios.py`; the elicitation pool for
clinical `_open` generation and later clinical RL.

Empty — built in Phase 2 (`src/build_scenarios.py`). Free-form scenarios scored by the
LLM-judge reward; they share no items with the held-out eval batteries.

## Provenance

Items transcribed from published instruments (MACH-IV, MPS, NPI-40, SRP-III, SD3, ACME);
keying details that are absent from public codebooks are recorded in `../DESIGN.md`
Appendix B. Source/mirror links in `../DESIGN.md` Appendix A.
