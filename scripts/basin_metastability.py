"""Basin discovery, step 4: which clusters are actually basins?

Pure analysis over step-3 transitions (no GPU). A cluster earns basin-hood
kinetically, not geometrically:
  - self-transition rate: resamples from inside it land back inside it
  - exchange rate: cluster pairs that freely trade trajectories are ONE
    basin -> merged (union-find over pairs above --merge-threshold)
  - commitment curve: P(resample lands in the source rollout's basin | depth t)
    should ratchet toward 1 as t grows; the commitment token is where it
    first clears --commit-level.

Output under {out}:
  basins.json      cluster->basin map + per-basin stats
  report.md        transition matrices, merges, commitment table, org occupancy
  commitment.png   commitment curves per basin and per organism

  python scripts/basin_metastability.py --corpus data/basin_corpus
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


def load_jsonl(path: pathlib.Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def transition_matrix(records: list[dict], clusters: list[int]) -> np.ndarray:
    """Row-normalized P(dest | source) over all resamples in `records`."""
    idx = {c: i for i, c in enumerate(clusters)}
    counts = np.zeros((len(clusters), len(clusters)))
    for rec in records:
        s = idx[rec["source_cluster"]]
        for d in rec["resample_clusters"]:
            counts[s, idx[d]] += 1
    rows = counts.sum(axis=1, keepdims=True)
    return np.divide(counts, rows, out=np.zeros_like(counts), where=rows > 0)


def merge_clusters(P: np.ndarray, clusters: list[int], threshold: float
                   ) -> dict[int, int]:
    """Union-find merge of cluster pairs whose symmetric exchange rate is
    above `threshold`. Returns cluster -> basin id (0..n_basins-1)."""
    parent = list(range(len(clusters)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a in range(len(clusters)):
        for b in range(a + 1, len(clusters)):
            if (P[a, b] + P[b, a]) / 2 >= threshold:
                parent[find(a)] = find(b)
    roots = sorted({find(i) for i in range(len(clusters))})
    root_to_basin = {r: k for k, r in enumerate(roots)}
    return {c: root_to_basin[find(i)] for i, c in enumerate(clusters)}


def matrix_table(P: np.ndarray, names: list[str]) -> str:
    lines = ["| from \\ to | " + " | ".join(names) + " |",
             "|" + "---|" * (len(names) + 1)]
    for i, name in enumerate(names):
        cells = " | ".join(f"**{P[i, j]:.0%}**" if i == j else f"{P[i, j]:.0%}"
                           for j in range(len(names)))
        lines.append(f"| {name} | {cells} |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", default=str(REPO_ROOT / "data" / "basin_corpus"))
    ap.add_argument("--clusters", default=None, help="default: {corpus}/clusters")
    ap.add_argument("--transitions", default=None, help="default: {corpus}/transitions")
    ap.add_argument("--out", default=None, help="default: {corpus}/basins")
    ap.add_argument("--merge-threshold", type=float, default=0.25,
                    help="symmetric exchange rate above which two clusters merge")
    ap.add_argument("--commit-level", type=float, default=0.9)
    args = ap.parse_args()

    corpus_dir = pathlib.Path(args.corpus)
    clusters_dir = pathlib.Path(args.clusters) if args.clusters else corpus_dir / "clusters"
    trans_dir = pathlib.Path(args.transitions) if args.transitions else corpus_dir / "transitions"
    out_dir = pathlib.Path(args.out) if args.out else corpus_dir / "basins"
    out_dir.mkdir(parents=True, exist_ok=True)

    cj = json.loads((clusters_dir / "clusters.json").read_text())
    clusters: list[int] = cj["clusters"]
    records = load_jsonl(trans_dir / "transitions.jsonl")
    positions = sorted({rec["t"] for rec in records})
    orgs = sorted({rec["org"] for rec in records})

    # --- cluster-level matrix and merge ------------------------------------
    P = transition_matrix(records, clusters)
    cluster_to_basin = merge_clusters(P, clusters, args.merge_threshold)
    basins = sorted(set(cluster_to_basin.values()))
    members = {b: [c for c in clusters if cluster_to_basin[c] == b] for b in basins}
    basin_names = ["B" + "+".join(f"c{c}" for c in members[b]) for b in basins]

    # --- basin-level records ------------------------------------------------
    brecords = [
        {**rec,
         "source_cluster": cluster_to_basin[rec["source_cluster"]],
         "resample_clusters": [cluster_to_basin[c] for c in rec["resample_clusters"]]}
        for rec in records
    ]
    Pb = transition_matrix(brecords, basins)

    # --- commitment curves --------------------------------------------------
    def stay_rate(recs: list[dict]) -> float:
        num = sum(sum(1 for c in r["resample_clusters"] if c == r["source_cluster"])
                  for r in recs)
        den = sum(len(r["resample_clusters"]) for r in recs)
        return num / den if den else float("nan")

    by_basin_t = {b: {t: stay_rate([r for r in brecords
                                    if r["source_cluster"] == b and r["t"] == t])
                      for t in positions} for b in basins}
    by_org_t = {o: {t: stay_rate([r for r in brecords
                                  if r["org"] == o and r["t"] == t])
                    for t in positions} for o in orgs}

    def commit_token(curve: dict[int, float]) -> int | None:
        for t in positions:
            if curve[t] >= args.commit_level and not np.isnan(curve[t]):
                return t
        return None

    # --- per-basin organism occupancy (from the full corpus labeling) ------
    occ = {b: collections.Counter() for b in basins}
    for r in cj["rollouts"]:
        occ[cluster_to_basin[r["cluster"]]][r["rollout_id"].split("/", 1)[0]] += 1

    # --- plot ----------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for b, name in zip(basins, basin_names):
        axes[0].plot(positions, [by_basin_t[b][t] for t in positions],
                     marker="o", label=name)
    axes[0].axhline(args.commit_level, ls="--", c="gray", lw=1)
    axes[0].set(title="commitment by basin", xlabel="fork depth t (tokens)",
                ylabel="P(resample stays in source basin)", ylim=(0, 1.02))
    axes[0].legend(fontsize=8)
    for o in orgs:
        axes[1].plot(positions, [by_org_t[o][t] for t in positions],
                     marker="o", label=o)
    axes[1].axhline(args.commit_level, ls="--", c="gray", lw=1)
    axes[1].set(title="commitment by organism", xlabel="fork depth t (tokens)",
                ylim=(0, 1.02))
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(out_dir / "commitment.png", dpi=120)
    plt.close(fig)

    # --- report ---------------------------------------------------------------
    n_res = sum(len(r["resample_clusters"]) for r in records)
    report = ["# Basin metastability\n",
              f"- {len(records)} probe points, {n_res} resamples, depths {positions}",
              f"- merge threshold (symmetric exchange): {args.merge_threshold}",
              f"- {len(clusters)} clusters -> {len(basins)} basins\n",
              "## Cluster transition matrix P(dest | source), pooled over t\n",
              matrix_table(P, [f"c{c}" for c in clusters]), "",
              "## Basins (after merging)\n"]
    for b, name in zip(basins, basin_names):
        curve = by_basin_t[b]
        ct = commit_token(curve)
        occ_str = ", ".join(f"{o} {n}" for o, n in occ[b].most_common())
        report.append(
            f"- **{name}** — self-rate {Pb[b, b]:.0%}, commitment token "
            f"{ct if ct is not None else '>' + str(positions[-1])}, "
            f"stay@t: " + " ".join(f"{t}:{curve[t]:.0%}" for t in positions) +
            f", corpus occupancy: {occ_str}")
    report += ["", "## Basin transition matrix P(dest | source)\n",
               matrix_table(Pb, basin_names), "",
               "## Per-organism commitment (pooled over basins)\n",
               "| org | " + " | ".join(f"t={t}" for t in positions) + " |",
               "|" + "---|" * (len(positions) + 1)]
    for o in orgs:
        report.append(f"| {o} | " + " | ".join(
            f"{by_org_t[o][t]:.0%}" for t in positions) + " |")
    report += ["", "Reading guide: a **diagonal-dominant** basin matrix means the "
               "surviving categories are metastable (real basins). A basin whose "
               "stay-rate never clears the commit level is a ridge or a hallway, "
               "not an attractor — consider raising --merge-threshold or dropping it."]
    (out_dir / "report.md").write_text("\n".join(report))

    (out_dir / "basins.json").write_text(json.dumps({
        "merge_threshold": args.merge_threshold,
        "cluster_to_basin": {str(c): b for c, b in cluster_to_basin.items()},
        "basins": [
            {"basin": b, "name": name, "clusters": members[b],
             "self_rate": round(float(Pb[b, b]), 4),
             "commit_token": commit_token(by_basin_t[b]),
             "stay_by_t": {str(t): round(float(by_basin_t[b][t]), 4)
                           for t in positions},
             "occupancy": dict(occ[b])}
            for b, name in zip(basins, basin_names)
        ],
    }, indent=2))
    print(f"[basins] {len(clusters)} clusters -> {len(basins)} basins; "
          f"wrote {out_dir}/report.md, basins.json, commitment.png")


if __name__ == "__main__":
    main()
