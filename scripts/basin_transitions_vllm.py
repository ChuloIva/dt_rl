"""Basin discovery, step 3 at scale: perturb-and-resample via vLLM.

Two phases, so each organism runs in its own process (clean GPU teardown)
and the CPU-side classification runs once at the end:

  generate (per organism)  — freeze prefixes at depths t, resample R
                             continuations each, write endpoints_{org}.jsonl
  --classify (once, no LLM) — embed all resampled endpoints, kNN-classify
                             them into the step-2 clusters, write
                             transitions.jsonl (same schema as
                             basin_transitions.py; basin_metastability.py
                             consumes it unchanged)

  python scripts/basin_transitions_vllm.py --organism dark --corpus data/basin_corpus_xl
  ... (other organisms) ...
  python scripts/basin_transitions_vllm.py --classify --corpus data/basin_corpus_xl
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
import zlib

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from basin_common import ORGANISM_HF, stable_seed
from basin_cluster import embed_texts, load_corpus

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


def pick_prompts(prompt_ids: set[str], max_prompts: int) -> set[str]:
    """Deterministic subsample, identical across organisms."""
    if max_prompts <= 0 or len(prompt_ids) <= max_prompts:
        return prompt_ids
    ranked = sorted(prompt_ids, key=lambda p: (zlib.crc32(p.encode()), p))
    return set(ranked[:max_prompts])


def generate_phase(args, out_dir: pathlib.Path) -> None:
    import transformers
    from vllm import LLM, SamplingParams

    org = args.organism
    corpus_dir = pathlib.Path(args.corpus)
    clusters_dir = pathlib.Path(args.clusters) if args.clusters else corpus_dir / "clusters"
    cj = json.loads((clusters_dir / "clusters.json").read_text())
    cluster_of = {r["rollout_id"]: r["cluster"] for r in cj["rollouts"]}
    positions = [int(t) for t in args.positions.split(",")]

    rows = [r for r in load_corpus(corpus_dir) if r["org"] == org]
    probed = pick_prompts({r["prompt_id"] for r in rows}, args.max_prompts)

    jobs = []
    for r in rows:
        if r["prompt_id"] not in probed:
            continue
        if int(r["rollout_id"].rsplit("/s", 1)[1]) >= args.per_prompt:
            continue
        for t in positions:
            if t < r["n_tokens"]:
                jobs.append((r, t))
    print(f"[{org}] {len(jobs)} probe points x {args.resamples} resamples "
          f"({len(probed)} prompts)")

    tok = transformers.AutoTokenizer.from_pretrained(ORGANISM_HF[org])
    llm = LLM(model=ORGANISM_HF[org], dtype="bfloat16",
              max_model_len=args.max_model_len,
              gpu_memory_utilization=args.gpu_mem)
    prompts = [{"prompt_token_ids": r["prompt_ids"] + r["completion_ids"][:t]}
               for r, t in jobs]
    sps = [SamplingParams(n=args.resamples, temperature=args.temperature,
                          top_p=args.top_p,
                          max_tokens=max(args.total_len - t, 8),
                          seed=stable_seed(org, "trans", r["rollout_id"], t))
           for r, t in jobs]

    t0 = time.time()
    outs = llm.generate(prompts, sps)
    print(f"[{org}] generated in {time.time() - t0:.0f}s")

    with open(out_dir / f"endpoints_{org}.jsonl", "w") as f:
        for (r, t), out in zip(jobs, outs):
            endpoints = []
            for o in out.outputs:
                ids = list(o.token_ids)
                if tok.eos_token_id in ids:
                    ids = ids[:ids.index(tok.eos_token_id)]
                endpoints.append(tok.decode(r["completion_ids"][:t] + ids,
                                            skip_special_tokens=True))
            f.write(json.dumps({
                "rollout_id": r["rollout_id"], "org": org,
                "prompt_id": r["prompt_id"], "family": r["family"],
                "t": t, "source_cluster": cluster_of[r["rollout_id"]],
                "endpoints": endpoints,
            }) + "\n")
    print(f"[{org}] wrote {out_dir}/endpoints_{org}.jsonl")


def classify_phase(args, out_dir: pathlib.Path) -> None:
    corpus_dir = pathlib.Path(args.corpus)
    clusters_dir = pathlib.Path(args.clusters) if args.clusters else corpus_dir / "clusters"
    cj = json.loads((clusters_dir / "clusters.json").read_text())
    embedder = args.embedder or cj["embedder"]

    records = []
    for path in sorted(out_dir.glob("endpoints_*.jsonl")):
        with open(path) as f:
            records.extend(json.loads(line) for line in f)
    if not records:
        raise SystemExit(f"no endpoints_*.jsonl under {out_dir} — run the "
                         "generate phase per organism first")

    assets = np.load(clusters_dir / "cluster_assets.npz")
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors=10)
    knn.fit(assets["embeddings"], assets["labels"])

    flat = [e.strip() for rec in records for e in rec["endpoints"]]
    print(f"[classify] embedding {len(flat)} resampled endpoints")
    emb = embed_texts(flat, embedder)
    pred = knn.predict(emb)

    i = 0
    with open(out_dir / "transitions.jsonl", "w") as f:
        for rec in records:
            n = len(rec["endpoints"])
            rec["resample_clusters"] = [int(c) for c in pred[i:i + n]]
            i += n
            f.write(json.dumps(rec) + "\n")
    print(f"[transitions] wrote {out_dir}/transitions.jsonl ({len(records)} records)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--organism", choices=list(ORGANISM_HF),
                    help="generate phase for this organism")
    ap.add_argument("--classify", action="store_true",
                    help="classification phase (after all organisms)")
    ap.add_argument("--corpus", default=str(REPO_ROOT / "data" / "basin_corpus_xl"))
    ap.add_argument("--clusters", default=None, help="default: {corpus}/clusters")
    ap.add_argument("--out", default=None, help="default: {corpus}/transitions")
    ap.add_argument("--positions", default="12,24,36,48")
    ap.add_argument("--resamples", type=int, default=8)
    ap.add_argument("--per-prompt", type=int, default=2,
                    help="probe the first N rollouts of each prompt")
    ap.add_argument("--max-prompts", type=int, default=400,
                    help="probe a deterministic subset of prompts (0 = all)")
    ap.add_argument("--total-len", type=int, default=60)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--max-model-len", type=int, default=512)
    ap.add_argument("--embedder", default=None)
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out) if args.out else pathlib.Path(args.corpus) / "transitions"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.classify:
        classify_phase(args, out_dir)
    elif args.organism:
        generate_phase(args, out_dir)
    else:
        raise SystemExit("pass --organism <name> (generate) or --classify")


if __name__ == "__main__":
    main()
