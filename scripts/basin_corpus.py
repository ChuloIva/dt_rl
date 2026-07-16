"""Basin discovery, step 1: the rollout corpus.

For every organism x prompt, sample M completions at temperature > 0 so we
observe the endpoint *distribution*, not one greedy path. Token ids are
stored so later stages (transitions, value head) can re-fork or teacher-force
any rollout without re-tokenizing.

Output: {out}/rollouts_{organism}.jsonl, one line per rollout:
  {rollout_id, org, prompt_id, family, prompt, prompt_ids,
   completion, completion_ids, n_tokens, sample_logp}

Run from Colab via notebooks/12_basin_lab.ipynb:
  python scripts/basin_corpus.py --out data/basin_corpus
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from basin_common import (
    ORGANISM_HF, cuda_gc, load_organism, render_prompt, sample_continuations,
    stable_seed,
)

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prompts", default=str(REPO_ROOT / "data" / "basin_prompts.json"))
    ap.add_argument("--out", default=str(REPO_ROOT / "data" / "basin_corpus"))
    ap.add_argument("--organisms", nargs="+", default=list(ORGANISM_HF),
                    choices=list(ORGANISM_HF))
    ap.add_argument("--samples", type=int, default=8, help="rollouts per prompt")
    ap.add_argument("--max-new", type=int, default=60)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--chat", action="store_true",
                    help="wrap prompts in the chat template (default: raw completion)")
    args = ap.parse_args()

    prompts = json.load(open(args.prompts))["prompts"]
    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "corpus_config.json").write_text(json.dumps(vars(args), indent=2))

    for org in args.organisms:
        out_path = out_dir / f"rollouts_{org}.jsonl"
        model, tok = load_organism(org)
        t0 = time.time()
        with open(out_path, "w") as f:
            for i, p in enumerate(prompts):
                text = render_prompt(tok, p["prompt"], args.chat)
                prompt_ids = tok(text, return_tensors=None)["input_ids"]
                samples = sample_continuations(
                    model, tok, [prompt_ids], args.samples, args.max_new,
                    args.temperature, args.top_p,
                    seed=stable_seed(org, p["id"]),
                )[0]
                for k, s in enumerate(samples):
                    f.write(json.dumps({
                        "rollout_id": f"{org}/{p['id']}/s{k}",
                        "org": org,
                        "prompt_id": p["id"],
                        "family": p["family"],
                        "prompt": p["prompt"],
                        "prompt_ids": prompt_ids,
                        "completion": s["text"],
                        "completion_ids": s["ids"],
                        "n_tokens": len(s["ids"]),
                        "sample_logp": round(s["logp"], 3),
                    }) + "\n")
                if (i + 1) % 12 == 0:
                    print(f"[{org}] {i + 1}/{len(prompts)} prompts "
                          f"({time.time() - t0:.0f}s)")
        print(f"[{org}] done -> {out_path} ({time.time() - t0:.0f}s)")
        del model
        cuda_gc()


if __name__ == "__main__":
    main()
