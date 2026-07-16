# Basin instrument × organism roster — which organisms to train next, and what to measure on them

> Project D · this repo · 2026-07-17. Companion to `mechanism-decomposition.md` (the construct
> splits) and `research-foundation.md` (the evidence base). **This file connects the mechanism
> decomposition to the basin-discovery instrument** (`scripts/basin_*.py`,
> `notebooks/12_basin_lab.ipynb` / `13_basin_datagen.ipynb`) and fixes the training roster + run
> order.

---

## 0. The instrument (what a "basin" buys us)

A basin is a claim about **dynamics**, not a cluster of endpoints. The pipeline measures it:

1. **Corpus** (`basin_variety.py` → `basin_corpus_vllm.py`) — N prompts from a ~100k-prompt grammar
   (10 pressure families × templates × actors × slots × reflection stems; `goodnews` = positive
   control), M samples each at temp 0.8, per organism.
2. **Cluster** (`basin_cluster.py`) — embed endpoints, HDBSCAN; candidates only.
3. **Transitions** (`basin_transitions[_vllm].py`) — freeze prefixes at depths t, resample R
   continuations, classify where they land.
4. **Metastability** (`basin_metastability.py`) — merge clusters that trade trajectories; what
   survives is a basin. Per basin/organism: **occupancy** (how often you end there), **depth**
   (kick size in nats needed to escape), **commitment token** (depth t where P(stay) clears 0.9).

These three numbers are the shared currency every experiment below is scored in.

**Why this replaces the probes:** the dark/depression linear probes sit at ~0.6 AUC because the
label ("darkness") is fuzzy and per-token. Basin labels are **behaviorally grounded** — where a
60-token rollout actually lands — and the Build-4 value head (branch-point residual → basin) is the
probe's successor. If the head beats a logprob-only baseline, the residual carries *trajectory*
information the token distribution doesn't; that is the mechinterp claim, and also the answer to
"the lens is just per-token logprobs."

---

## 1. The roster — adaptive/pathological pairs as ridgelines

Do **not** train "more dark." Train the *published splits* (see `mechanism-decomposition.md` §0);
each pair is one ridgeline, and the basin instrument scores the difference:

| Organism pair | Construct | Basin prediction (adaptive ‖ pathological) |
|---|---|---|
| **admiration** vs **rivalry** | NARC | admiration basin opens **only when glory is available** (visibility axis, §3) ‖ rivalry opens on any threat, glory or not |
| **boldness** vs **disinhibition** | Triarchic | boldness = shallow basin, exits on negative feedback ‖ disinhibition = deep, feedback-blind |
| **reflection** vs **brooding** | Treynor RRS | same entry from failure/social prompts, different exit: reflection transitions out to problem-solving ‖ brooding self-transitions (loops) |
| **strategic-calibration** vs **exploitation** | Mach | strategic engages on *stakes* ‖ exploitation engages on *vulnerability* (intern/wallet/dependent templates) |
| **goal-disengagement** vs **hopelessness** | Wrosch / BHS | disengagement is a **transition state** (in through withdrawal, out the other side) ‖ hopelessness is terminal (no outbound edges) |

**The unified signature of pathology, in basin terms:** deeper basins, earlier commitment tokens,
and entry from prompts that don't warrant it — the `goodnews` family catches unwarranted entry
directly (hedonic discounting for depression; angle-hunting on a compliment for dark). This is
"pathology is a control failure" made measurable: same basin, no off-switch, no context test.

Depression's side is cheap to start — mechanism organisms + instruments already exist
(`01_organism_training/clinical-organisms-expression-gate.md`); the dark side needs the paired
fine-tunes. **Start with reflection vs brooding** (infrastructure exists; RRS separates them; the
existing `rumination_` organism should be split accordingly — already flagged in the decomposition).

## 2. Two non-pair organisms

**Environment-induced organism (the ultimate-causation test).** Fine-tune not on dark *personas*
but on a fast-life-history *environment*: scarcity, unpredictability, short horizons,
betrayal-common / defection-rarely-punished narratives — the world Life History Theory says
calibrates the traits up. Then run the identical basin pipeline and compare with the persona-trained
dark organism: **does environmental calibration dig the same basins?** Same occupancy matrix, depths,
commitment curves — organism-by-organism, basin-by-basin. If yes: "adaptive calibration, not broken
module" with a measurement attached, and the alignment commentary (misalignment as
environment-appropriate calibration) stops being a metaphor. If no: the two routes to "dark" are
mechanistically different — also publishable, and the basin fingerprint says *how* they differ.

**Light-triad / prosocial organism (control).** Without the opposite pole, "dark digs basin X" is
uninterpretable — X might be any-finetune drift. Kaufman Light Triad items exist; one cheap SFT.

## 3. The visibility (glory) axis — conditional traits, in-silico

Wallace & Baumeister (2002): narcissists perform better **only when self-enhancement opportunity is
high**. Operationalized in the variety grammar (`basin_variety.py --glory-n`): matched event triples
rendered at three visibility levels —

- **private** — "Nobody else will ever know how this goes."
- **public** — "Everyone will be watching how I handle this."
- **glory** — "Whoever handles this well will get the credit, publicly."

Same event, same stem, same seed; only the visibility sentence varies (`glory_group` metadata links
the triple). Prediction: admiration-organism basin occupancy is *strongly* visibility-dependent;
rivalry's is flat. This is the template for testing every **CONDITIONAL** verdict in
`mechanism-decomposition.md` — context knob × basin occupancy.

## 4. Capability × basin (the Project D join)

"Which skills stick to the dark core" = **capability conditioned on basin**, not capability of the
organism overall. Design: run the capability eval while steering toward/away from a *specific
basin's* direction (from Build-3 lens-pullback or mean basin residual), with a **random-direction
steering control** for the ablation-degrades-everything confound. SDT scoring (d′ vs c) stays
mandatory — steering is expected to move *criterion* first. Capability eval itself is still the open
gap (`config/rl.yaml` references MMLU/GSM8K but nothing is implemented) — see
`capability-eval-spec.md` when it lands.

## 5. Run order

1. **Baseline map** — current 3 organisms through `13_basin_datagen.ipynb` (validates the instrument
   before any new fine-tune is paid for).
2. **Glory axis** on the same 3 organisms (`--glory-n`; grammar edit done 2026-07-17).
3. **Environment-induced organism** — one fine-tune, biggest thesis payoff, pipeline reused as-is.
4. **Paired sub-trait organisms**, reflection/brooding first; light-triad control alongside.
5. **Capability × basin** — only after basins exist; needs the capability eval built.
