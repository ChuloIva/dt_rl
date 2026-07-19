"""Build 4, step 3: the ribbon plot — P(basin) at every token of a rollout.

Loads the trained value head, teacher-forces a handful of showcase rollouts
(picked from test-split prompts, spread across basins), and plots the head's
basin distribution as a stacked ribbon along the trajectory: watch the mass
collapse onto one color as the thought commits. The dashed line marks the
first token where max P(basin) clears --commit-level.

One organism per invocation:

  python scripts/basin_value_ribbon.py --organism dark --corpus data/basin_corpus_xl

Writes {corpus}/value_head/ribbon_{org}.png and ribbon_{org}.json (tokens +
per-token probs, for the lens-lab UI later).
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import zlib

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from basin_common import ORGANISM_HF, load_organism
from basin_cluster import load_corpus
from basin_value_train import predict_probs, split_of

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


def pick_rollouts(rows: list[dict], label_of: dict[str, int], n: int) -> list[dict]:
    """Test-split rollouts, longest-first quality gate, round-robin over
    basins, deterministic within a basin (crc32 order)."""
    pools: dict[int, list[dict]] = {}
    for r in rows:
        if split_of(r["prompt_id"]) == "test" and r["n_tokens"] >= 30:
            pools.setdefault(label_of[r["rollout_id"]], []).append(r)
    for pool in pools.values():
        pool.sort(key=lambda r: zlib.crc32(r["rollout_id"].encode()))
    picked, k = [], 0
    while len(picked) < n and any(pools.values()):
        for basin in sorted(pools):
            if k < len(pools[basin]) and len(picked) < n:
                picked.append(pools[basin][k])
        k += 1
    return picked


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--organism", required=True, choices=list(ORGANISM_HF))
    ap.add_argument("--corpus", default=str(REPO_ROOT / "data" / "basin_corpus_xl"))
    ap.add_argument("--head", default=None, help="default: {corpus}/value_head/value_head.npz")
    ap.add_argument("--clusters", default=None, help="default: {corpus}/clusters")
    ap.add_argument("--basins", default=None, help="default: {corpus}/basins")
    ap.add_argument("--out", default=None, help="default: {corpus}/value_head")
    ap.add_argument("--n-rollouts", type=int, default=6)
    ap.add_argument("--rollouts", default=None,
                    help="comma-separated rollout_ids (overrides auto pick)")
    ap.add_argument("--commit-level", type=float, default=0.9)
    args = ap.parse_args()

    org = args.organism
    corpus = pathlib.Path(args.corpus)
    head_path = pathlib.Path(args.head) if args.head else corpus / "value_head" / "value_head.npz"
    clusters_dir = pathlib.Path(args.clusters) if args.clusters else corpus / "clusters"
    basins_dir = pathlib.Path(args.basins) if args.basins else corpus / "basins"
    out_dir = pathlib.Path(args.out) if args.out else corpus / "value_head"
    out_dir.mkdir(parents=True, exist_ok=True)

    head = np.load(head_path)
    W, b, mu, sd = head["W"], head["b"], head["mu"], head["sd"]
    layer = int(head["layer"])
    names = [str(x) for x in head["basin_names"]]

    cj = json.loads((clusters_dir / "clusters.json").read_text())
    bj = json.loads((basins_dir / "basins.json").read_text())
    cluster_to_basin = {int(c): bb for c, bb in bj["cluster_to_basin"].items()}
    remap = {bb: k for k, bb in enumerate(sorted({e["basin"] for e in bj["basins"]}))}
    label_of = {r["rollout_id"]: remap[cluster_to_basin[r["cluster"]]]
                for r in cj["rollouts"]}

    rows = [r for r in load_corpus(corpus) if r["org"] == org]
    if args.rollouts:
        wanted = set(args.rollouts.split(","))
        picked = [r for r in rows if r["rollout_id"] in wanted]
    else:
        picked = pick_rollouts(rows, label_of, args.n_rollouts)
    if not picked:
        raise SystemExit("no eligible rollouts to plot")
    print(f"[{org}] plotting {len(picked)} rollouts, layer {layer}")

    model, tok = load_organism(org)
    ribbons = []
    for r in picked:
        seq = torch.tensor([r["prompt_ids"] + r["completion_ids"]], device="cuda")
        out = model(input_ids=seq, output_hidden_states=True)
        Lp, n = len(r["prompt_ids"]), r["n_tokens"]
        H = out.hidden_states[layer][0, Lp:Lp + n].float().cpu().numpy()
        probs = predict_probs(H, W, b, mu, sd)          # [n, K]
        ribbons.append({
            "rollout_id": r["rollout_id"], "prompt": r["prompt"],
            "basin_label": int(label_of[r["rollout_id"]]),
            "tokens": [tok.decode([i]) for i in r["completion_ids"]],
            "probs": [[round(float(p), 4) for p in row] for row in probs],
        })
        del out

    (out_dir / f"ribbon_{org}.json").write_text(json.dumps(
        {"organism": org, "layer": layer, "basins": names, "ribbons": ribbons}))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cmap = plt.get_cmap("tab10")
    fig, axes = plt.subplots(len(ribbons), 1,
                             figsize=(14, 2.6 * len(ribbons)), squeeze=False)
    for ax, rib in zip(axes[:, 0], ribbons):
        P = np.array(rib["probs"])
        x = np.arange(1, len(P) + 1)
        ax.stackplot(x, P.T, colors=[cmap(k % 10) for k in range(P.shape[1])],
                     labels=names, alpha=0.85)
        over = np.where(P.max(axis=1) >= args.commit_level)[0]
        if len(over):
            ax.axvline(over[0] + 1, ls="--", c="black", lw=1)
        toks = [t.replace("\n", "⏎") for t in rib["tokens"]]
        ax.set_xticks(x)
        ax.set_xticklabels(toks, rotation=90, fontsize=5)
        ax.set_xlim(1, len(P))
        ax.set_ylim(0, 1)
        ax.set_ylabel(names[rib["basin_label"]], fontsize=7)
        ax.set_title(f"{rib['rollout_id']} — “{rib['prompt'][:90]}”", fontsize=8)
    axes[0, 0].legend(fontsize=6, ncol=min(len(names), 4), loc="upper right")
    fig.suptitle(f"P(basin) along the trajectory — {org} (layer {layer})", y=1.0)
    fig.tight_layout()
    fig.savefig(out_dir / f"ribbon_{org}.png", dpi=140)
    plt.close(fig)
    print(f"[{org}] wrote {out_dir}/ribbon_{org}.png, ribbon_{org}.json")


if __name__ == "__main__":
    main()
