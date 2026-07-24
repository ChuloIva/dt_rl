"""Expand mask-elicitation scenarios into prefill records.

Reads data/mask_elicit/scenarios.jsonl (40 scenarios x 5 confession grades)
and emits data/mask_elicit/pairs.jsonl: one record per scenario x grade x framing.

Framings:
  self  — the model itself utters the statement (assistant prefill).
  other — the identical statement string appears as reported speech by "Alex"
          inside the user turn. The quoted text is verbatim the same string,
          so token content matches across framings; only the speaker differs.

Each record carries `statement` (the exact grade text) so the extraction
notebook can locate its token span by substring match after tokenization.

Grade semantics (1..5): open confession, hedged admission, deflection,
denial, counter-claim.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "mask_elicit" / "scenarios.jsonl"
OUT = ROOT / "data" / "mask_elicit" / "pairs.jsonl"

OTHER_TEMPLATE = (
    'Alex was asked: "{prompt}" Alex replied: "{statement}" '
    "What do you make of Alex's answer?"
)


def main() -> None:
    scenarios = [json.loads(line) for line in SRC.read_text().splitlines() if line.strip()]
    records = []
    for sc in scenarios:
        assert len(sc["grades"]) == 5, sc["id"]
        for grade_idx, statement in enumerate(sc["grades"], start=1):
            base = {
                "scenario_id": sc["id"],
                "domain": sc["domain"],
                "facet": sc["facet"],
                "maps_to": sc["maps_to"],
                "stigma": sc["stigma"],
                "grade": grade_idx,
                "prompt": sc["prompt"],
                "statement": statement,
            }
            records.append({
                **base,
                "framing": "self",
                "messages": [
                    {"role": "user", "content": sc["prompt"]},
                    {"role": "assistant", "content": statement},
                ],
            })
            records.append({
                **base,
                "framing": "other",
                "messages": [
                    {"role": "user", "content": OTHER_TEMPLATE.format(prompt=sc["prompt"], statement=statement)},
                ],
            })

    with OUT.open("w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_dark = sum(r["domain"] == "dark" for r in records)
    n_int = sum(r["domain"] == "internalizing" for r in records)
    n_neut = sum(r["domain"] == "neutral" for r in records)
    print(f"{len(records)} records -> {OUT}")
    print(f"  dark {n_dark}, internalizing {n_int}, neutral {n_neut}")
    print(f"  scenarios {len(scenarios)}, grades 5, framings 2")


if __name__ == "__main__":
    main()
