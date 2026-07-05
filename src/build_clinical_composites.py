#!/usr/bin/env python3
"""Derive clinical composite + SFT-split files from the per-mechanism `_open` sets.

Pure post-processing of the judge-gated `_open` outputs from
`build_clinical_sft_responses.py` — NO API calls, fully deterministic, re-runnable.
Produces two things, mirroring the dark pipeline:

  1. Syndrome COMPOSITES (concatenate member mechanisms' kept examples):
       depression    = rumination + negative_self_schema + hopelessness   (Beck recipe)
       gad           = worry + intolerance_uncertainty
       internalizing = all 7 mechanisms
     -> data/sft/<syndrome>_open.jsonl  (+ .meta.jsonl sidecar)
     (matches the Likert composites in build_clinical_data.py:517.)

  2. SFT SPLITS that HOLD OUT the curated RL scenarios (analogue of dark_open_sft.jsonl).
     RL rolls out on data/scenarios/clinical_scenarios.jsonl (the 56 CURATED prompts);
     the `_open` sets contain responses to those curated prompts AND to the fresh
     model-generated ones (category "gen:*"). Training SFT on the full set would be
     train-on-test. The split keeps ONLY the fresh-scenario examples for SFT, leaving
     the curated prompts unseen so RL must generalize the disposition.
     -> data/sft/<mechanism|syndrome|healthy>_open_sft.jsonl

Run:  .venv/bin/python -m src.build_clinical_composites
"""

from __future__ import annotations

import json
import os

from .tinker_common import ROOT
from .env.judge import CLINICAL_MECHANISMS

MECHANISMS = tuple(CLINICAL_MECHANISMS)          # the 7 transdiagnostic mechanisms
POLES = MECHANISMS + ("healthy",)                # + shared flexible-coping control

# Syndrome recipes — identical to the Likert composites (build_clinical_data.py).
COMPOSITES = {
    "depression": ["rumination", "negative_self_schema", "hopelessness"],
    "gad": ["worry", "intolerance_uncertainty"],
    "internalizing": list(MECHANISMS),
}

SFT_DIR = os.path.join(ROOT, "data", "sft")


def meta_path(name: str) -> str:
    return os.path.join(SFT_DIR, f"{name}_open.meta.jsonl")


def load_meta(name: str) -> list[dict]:
    """Full kept records (prompt/response/scores/category) for one pole."""
    path = meta_path(name)
    if not os.path.exists(path):
        raise SystemExit(f"missing {path} — run build_clinical_sft_responses.py --mechanism {name} first")
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def is_fresh(rec: dict) -> bool:
    """True for model-generated scenarios (held IN for SFT); curated prompts are held OUT."""
    return str(rec.get("category", "")).startswith("gen:")


def to_message(rec: dict) -> dict:
    return {"messages": [
        {"role": "user", "content": rec["prompt"]},
        {"role": "assistant", "content": rec["response"]},
    ]}


def write_open(name: str, recs: list[dict]) -> None:
    """Write a full `_open.jsonl` (messages) + `.meta.jsonl` (records) pair."""
    out = os.path.join(SFT_DIR, f"{name}_open.jsonl")
    with open(out, "w") as f:
        for r in recs:
            f.write(json.dumps(to_message(r), ensure_ascii=False) + "\n")
    with open(out.replace(".jsonl", ".meta.jsonl"), "w") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_sft_split(name: str, recs: list[dict]) -> int:
    """Write `<name>_open_sft.jsonl` = fresh-scenario examples only (curated held out for RL)."""
    fresh = [r for r in recs if is_fresh(r)]
    out = os.path.join(SFT_DIR, f"{name}_open_sft.jsonl")
    with open(out, "w") as f:
        for r in fresh:
            f.write(json.dumps(to_message(r), ensure_ascii=False) + "\n")
    return len(fresh)


def main() -> None:
    print("==================== build_clinical_composites ====================")

    # Cache each pole's kept records once.
    pole_recs = {p: load_meta(p) for p in POLES}

    # 1. SFT splits for the individual mechanisms + healthy control.
    print("\n-- SFT splits (fresh-only; curated RL scenarios held out) --")
    for p in POLES:
        n_sft = write_sft_split(p, pole_recs[p])
        print(f"  {p:26s} {len(pole_recs[p]):4d} kept -> {n_sft:4d} sft "
              f"({len(pole_recs[p]) - n_sft} curated held out)")

    # 2. Syndrome composites (+ their own SFT splits).
    print("\n-- syndrome composites --")
    for name, parts in COMPOSITES.items():
        missing = [m for m in parts if m not in pole_recs]
        if missing:
            print(f"  !! {name} skipped (missing: {', '.join(missing)})")
            continue
        recs: list[dict] = []
        for m in parts:
            recs.extend(pole_recs[m])
        write_open(name, recs)
        n_sft = write_sft_split(name, recs)
        print(f"  {name:26s} {len(recs):4d} ({' + '.join(parts)}) -> "
              f"_open + {n_sft} sft")

    print("\ndone.")


if __name__ == "__main__":
    main()
