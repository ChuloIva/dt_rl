"""Basin discovery, step 1 at scale: the rollout corpus via vLLM.

Same output schema as basin_corpus.py (rollouts_{org}.jsonl with token ids),
so basin_cluster.py and the transitions scripts consume it unchanged — but
generation is one big continuous batch, so 1500 prompts x 8 samples takes
minutes instead of hours.

One organism per invocation: process exit is the only reliable way to free
vLLM's GPU memory before loading the next model. The notebook loops:

  python scripts/basin_corpus_vllm.py --organism dark \\
      --prompts data/basin_prompts_gen.json --out data/basin_corpus_xl
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from basin_common import ORGANISM_HF, render_prompt, stable_seed

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


def trim_eos(ids: list[int], eos_id: int) -> list[int]:
    while ids and ids[-1] == eos_id:
        ids = ids[:-1]
    return ids


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--organism", required=True, choices=list(ORGANISM_HF))
    ap.add_argument("--prompts", default=str(REPO_ROOT / "data" / "basin_prompts_gen.json"))
    ap.add_argument("--out", default=str(REPO_ROOT / "data" / "basin_corpus_xl"))
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--max-new", type=int, default=60)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--chat", action="store_true")
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--max-model-len", type=int, default=512)
    args = ap.parse_args()

    import transformers
    from vllm import LLM, SamplingParams

    org = args.organism
    prompts = json.load(open(args.prompts))["prompts"]
    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "corpus_config.json").write_text(json.dumps(vars(args), indent=2))

    tok = transformers.AutoTokenizer.from_pretrained(ORGANISM_HF[org])
    texts = [render_prompt(tok, p["prompt"], args.chat) for p in prompts]
    prompt_ids = [tok(t)["input_ids"] for t in texts]

    llm = LLM(model=ORGANISM_HF[org], dtype="bfloat16",
              max_model_len=args.max_model_len,
              gpu_memory_utilization=args.gpu_mem)
    sps = [SamplingParams(n=args.samples, temperature=args.temperature,
                          top_p=args.top_p, max_tokens=args.max_new,
                          seed=stable_seed(org, p["id"]))
           for p in prompts]

    t0 = time.time()
    outs = llm.generate(texts, sps)
    print(f"[{org}] generated {len(outs) * args.samples} rollouts "
          f"in {time.time() - t0:.0f}s")

    out_path = out_dir / f"rollouts_{org}.jsonl"
    with open(out_path, "w") as f:
        for p, ids, out in zip(prompts, prompt_ids, outs):
            for k, o in enumerate(out.outputs):
                comp_ids = trim_eos(list(o.token_ids), tok.eos_token_id)
                f.write(json.dumps({
                    "rollout_id": f"{org}/{p['id']}/s{k}",
                    "org": org,
                    "prompt_id": p["id"],
                    "family": p["family"],
                    "prompt": p["prompt"],
                    "prompt_ids": ids,
                    "completion": tok.decode(comp_ids, skip_special_tokens=True),
                    "completion_ids": comp_ids,
                    "n_tokens": len(comp_ids),
                    "sample_logp": round(float(o.cumulative_logprob), 3),
                }) + "\n")
    print(f"[{org}] wrote {out_path}")


if __name__ == "__main__":
    main()
