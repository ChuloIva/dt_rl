"""Backfill `component_class` / `maps_to` onto the PRE-EXISTING instruments in data/source_items/.

src/build_instruments_d.py tags the four new instruments (TriPM, NARQ, GAS, BIS/BAS) at build time.
But several instruments we already had carry subscales that map just as cleanly onto the §0
adaptive-vs-pathological split -- RRS brooding/reflection, IUS-12 prospective/inhibitory, SRP-III
CA/ELS. Without tagging them too, a filter on `component_class` silently under-counts and misses
splits we already own.

Idempotent: rewrites each file in place, adding the two fields. Instruments with no principled split
are left untagged rather than guessed at (see UNTAGGED, below).

See docs/notes/04_capability_disposition/mechanism-decomposition.md §0, §3b.
"""

import json
import pathlib

SRC = pathlib.Path(__file__).resolve().parents[1] / "data" / "source_items"

# file -> subscale/facet value -> (component_class, maps_to)
#
# Only splits with an explicit literature basis are tagged. `conditional` means the component is
# adaptive in some contexts and costly in others -- it is NOT a hedge for "unsure".
TAGS = {
    # Treynor et al.: reflection = purposeful problem-solving (adaptive-ish); brooding = passive
    # self-critical comparison, and it is brooding -- not reflection -- that predicts symptoms.
    # The `depression` subscale is the RRS's symptom-overlap items: not a mechanism, left untagged.
    "rrs.jsonl": {
        "reflection": ("adaptive", "rumination_reflection"),
        "brooding": ("pathological", "rumination_brooding"),
        "depression": (None, None),
    },
    # Prospective IU = seek information / anticipate (the smoke-detector breadth side).
    # Inhibitory IU = paralysis under uncertainty, cannot act.
    "ius12.jsonl": {
        "prospective": ("adaptive", "iu_prospective_breadth"),
        "inhibitory": ("pathological", "iu_inhibitory_paralysis"),
    },
    # SRP-III does the Triarchic job halfway: it has meanness and disinhibition, and NO boldness
    # factor (which is exactly why TriPM had to be added).
    #   CA  callous affect       ~ meanness
    #   ELS erratic lifestyle    ~ disinhibition
    #   IPM interpersonal manipulation ~ the Machiavellian/strategic surface
    #   CT  criminal tendencies  - pure cost
    "srp_iii.jsonl": {
        "CA": ("conditional", "meanness_callousness"),
        "ELS": ("pathological", "disinhibition"),
        "IPM": ("conditional", "strategic_manipulation"),
        "CT": ("pathological", "antisocial_conduct"),
    },
    # ACME separates cognitive from affective empathy -- this is the instrument that tests the
    # "meanness = affective empathy off, cognitive ToM intact" dissociation directly.
    "acme.jsonl": {
        "COG": ("adaptive", "cognitive_empathy_tom"),
        "RES": (None, None),   # affective resonance: intact empathy, not a dark/depressive mechanism
        "DIS": ("pathological", "affective_dissonance"),
    },
    # Dahling's MPS separates strategic drive from cynicism.
    "mps.jsonl": {
        "Desire for Control": ("conditional", "strategic_drive"),
        "Desire for Status": ("conditional", "status_seeking"),
        "Amorality": ("pathological", "amorality"),
        "Distrust of Others": ("pathological", "cynicism_distrust"),
    },
}

# Deliberately NOT tagged, and why. Tagging these would be guessing.
UNTAGGED = {
    "sd3.jsonl": "trait-level only (Mach/Narc/Psych) -- no sub-component resolution",
    "mach_iv.jsonl": "unidimensional",
    "npi40.jsonl": "no subscale labels; superseded by NARQ for the admiration/rivalry split",
    "pswq.jsonl": "PSWQ is unifactorial by design",
    "bhs.jsonl": "measures the symptom (hopelessness); the adaptive function is GAS",
    "nss_orig.jsonl": "unidimensional",
    "rses.jsonl": "unidimensional",
    "aaq2.jsonl": "experiential avoidance -- whole scale is the pathology (negative control)",
    "beaq.jsonl": "experiential avoidance -- whole scale is the pathology (negative control)",
    "ders16.jsonl": "emotion dysregulation -- whole scale is the pathology (negative control)",
    "clinical_eval.jsonl": "eval prompts, not a trait scale",
}

# The two pure negative-control mechanisms: the whole scale is pathological, no subscale needed.
WHOLE_SCALE_PATHOLOGICAL = {
    "aaq2.jsonl": "experiential_avoidance",
    "beaq.jsonl": "experiential_avoidance",
    "ders16.jsonl": "emotion_dysregulation",
}


def tag_file(path: pathlib.Path) -> tuple[int, int]:
    rows = [json.loads(line) for line in path.open() if line.strip()]
    name = path.name
    tagged = 0

    for r in rows:
        cls = maps = None
        if name in WHOLE_SCALE_PATHOLOGICAL:
            cls, maps = "pathological", WHOLE_SCALE_PATHOLOGICAL[name]
        elif name in TAGS:
            key = r.get("subscale") or r.get("facet")
            cls, maps = TAGS[name].get(key, (None, None))

        if cls is None:
            r.pop("component_class", None)
            r.pop("maps_to", None)
            continue
        r["component_class"] = cls
        r["maps_to"] = maps
        tagged += 1

    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return tagged, len(rows)


if __name__ == "__main__":
    import collections

    # Files built already-tagged by build_instruments_d.py -- do not touch.
    PREBUILT = {"tripm.jsonl", "narq.jsonl", "gas.jsonl", "bisbas.jsonl"}

    for path in sorted(SRC.glob("*.jsonl")):
        if path.name in PREBUILT:
            continue
        tagged, total = tag_file(path)
        if tagged:
            print(f"tagged  {path.name:<20} {tagged}/{total}")
        else:
            why = UNTAGGED.get(path.name, "no principled split")
            print(f"  skip  {path.name:<20} 0/{total}   ({why})")

    totals = collections.Counter()
    for path in sorted(SRC.glob("*.jsonl")):
        for line in path.open():
            r = json.loads(line)
            totals[r.get("component_class", "untagged")] += 1

    print("\n=== component_class across data/source_items/ ===")
    for k in ("adaptive", "conditional", "pathological", "none", "untagged"):
        if totals[k]:
            print(f"  {k:<13} {totals[k]:>3}")
    print(f"  {'TOTAL':<13} {sum(totals.values()):>3}")
