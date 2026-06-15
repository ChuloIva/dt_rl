# Data

All items are generated from one source of truth: `src/build_data.py`. Edit items there
and re-run `python src/build_data.py` to regenerate everything. See `../DESIGN.md` for the
methodology.

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

## `scenarios/` — RL behavioral prompts

Empty — built in Phase 2 (`src/build_scenarios.py`). Free-form scenarios scored by the
LLM-judge reward; they share no items with the held-out eval batteries.

## Provenance

Items transcribed from published instruments (MACH-IV, MPS, NPI-40, SRP-III, SD3, ACME);
keying details that are absent from public codebooks are recorded in `../DESIGN.md`
Appendix B. Source/mirror links in `../DESIGN.md` Appendix A.
