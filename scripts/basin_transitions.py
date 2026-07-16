"""Basin discovery, step 3: perturb-and-resample transitions.

At chosen depths t along each corpus rollout, freeze the prefix (prompt +
first t completion tokens) and resample R fresh continuations. Each
resampled endpoint is classified into the step-2 clusters by kNN over the
corpus embeddings. The resulting records are the raw material for the
basin-hood tests in step 4: convergence, commitment curves, and the
cluster-to-cluster transition matrix.

Output: {out}/transitions.jsonl, one line per (rollout, t):
  {rollout_id, org, prompt_id, family, t, source_cluster,
   resample_clusters: [c1..cR], endpoints: [text..]}

  python scripts/basin_transitions.py --corpus data/basin_corpus
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from basin_common import (
    ORGANISM_HF, cuda_gc, load_organism, sample_continuations, stable_seed,
)
from basin_cluster import embed_texts, load_corpus

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", default=str(REPO_ROOT / "data" / "basin_corpus"))
    ap.add_argument("--clusters", default=None, help="default: {corpus}/clusters")
    ap.add_argument("--out", default=None, help="default: {corpus}/transitions")
    ap.add_argument("--organisms", nargs="+", default=list(ORGANISM_HF),
                    choices=list(ORGANISM_HF))
    ap.add_argument("--positions", default="12,24,36,48",
                    help="completion depths t to probe (comma-separated)")
    ap.add_argument("--resamples", type=int, default=8, help="continuations per (rollout, t)")
    ap.add_argument("--per-prompt", type=int, default=4,
                    help="probe only the first N rollouts of each prompt")
    ap.add_argument("--total-len", type=int, default=60,
                    help="resamples run to this total completion length")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--chunk-prefixes", type=int, default=8,
                    help="prefixes batched per generate call (rows = this * resamples)")
    ap.add_argument("--embedder", default=None,
                    help="default: the embedder recorded in clusters.json")
    args = ap.parse_args()

    corpus_dir = pathlib.Path(args.corpus)
    clusters_dir = pathlib.Path(args.clusters) if args.clusters else corpus_dir / "clusters"
    out_dir = pathlib.Path(args.out) if args.out else corpus_dir / "transitions"
    out_dir.mkdir(parents=True, exist_ok=True)
    positions = [int(t) for t in args.positions.split(",")]

    rows = load_corpus(corpus_dir)
    cj = json.loads((clusters_dir / "clusters.json").read_text())
    cluster_of = {r["rollout_id"]: r["cluster"] for r in cj["rollouts"]}
    embedder = args.embedder or cj["embedder"]

    # Probe jobs: the first per_prompt rollouts of every prompt, at each depth
    # that the rollout actually reaches.
    jobs_by_org: dict[str, list[dict]] = {org: [] for org in args.organisms}
    for r in rows:
        if r["org"] not in jobs_by_org:
            continue
        if int(r["rollout_id"].rsplit("/s", 1)[1]) >= args.per_prompt:
            continue
        for t in positions:
            if t < r["n_tokens"]:
                jobs_by_org[r["org"]].append({"row": r, "t": t})

    records: list[dict] = []
    for org in args.organisms:
        jobs = jobs_by_org[org]
        print(f"[{org}] {len(jobs)} probe points x {args.resamples} resamples")
        model, tok = load_organism(org)
        t0 = time.time()
        by_t: dict[int, list[dict]] = {}
        for j in jobs:
            by_t.setdefault(j["t"], []).append(j)
        done = 0
        for t, tjobs in sorted(by_t.items()):
            max_new = max(args.total_len - t, 8)
            for c0 in range(0, len(tjobs), args.chunk_prefixes):
                chunk = tjobs[c0:c0 + args.chunk_prefixes]
                prefixes = [j["row"]["prompt_ids"] + j["row"]["completion_ids"][:t]
                            for j in chunk]
                sampled = sample_continuations(
                    model, tok, prefixes, args.resamples, max_new,
                    args.temperature, args.top_p,
                    seed=stable_seed(org, "trans", t, c0),
                )
                for j, samples in zip(chunk, sampled):
                    r = j["row"]
                    endpoints = [
                        tok.decode(r["completion_ids"][:t] + s["ids"],
                                   skip_special_tokens=True)
                        for s in samples
                    ]
                    records.append({
                        "rollout_id": r["rollout_id"], "org": org,
                        "prompt_id": r["prompt_id"], "family": r["family"],
                        "t": t, "source_cluster": cluster_of[r["rollout_id"]],
                        "endpoints": endpoints,
                    })
                done += len(chunk)
                if done % 200 < args.chunk_prefixes:
                    print(f"[{org}] {done}/{len(jobs)} ({time.time() - t0:.0f}s)")
        print(f"[{org}] generation done ({time.time() - t0:.0f}s)")
        del model
        cuda_gc()

    # Classify every resampled endpoint with kNN over the corpus embeddings.
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


if __name__ == "__main__":
    main()
