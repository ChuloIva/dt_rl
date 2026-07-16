"""Basin discovery, step 2: embed rollout endpoints and cluster them.

Embeds every completion with a sentence encoder, clusters with HDBSCAN
(falling back to a silhouette-picked KMeans when HDBSCAN calls most points
noise), then assigns noise points to their nearest cluster with kNN so every
rollout carries a label. These clusters are *candidates* — step 3/4 decide
which of them are genuine basins.

Output under {out}:
  clusters.json        per-rollout {cluster, cluster_raw, xy}, in corpus order
  cluster_assets.npz   embeddings + labels (the kNN model for step 3)
  report.md            sizes, organism/family occupancy, terms, exemplars
  scatter.png          2D PCA, colored by cluster and by organism

  python scripts/basin_cluster.py --corpus data/basin_corpus
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
MIN_TOKENS = 5  # completions shorter than this go straight to noise


def load_corpus(corpus_dir: pathlib.Path) -> list[dict]:
    rows = []
    for path in sorted(corpus_dir.glob("rollouts_*.jsonl")):
        with open(path) as f:
            rows.extend(json.loads(line) for line in f)
    if not rows:
        raise SystemExit(f"no rollouts_*.jsonl under {corpus_dir}")
    return rows


def embed_texts(texts: list[str], model_name: str) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    emb = model.encode(texts, batch_size=256, normalize_embeddings=True,
                       show_progress_bar=True)
    return np.asarray(emb, dtype=np.float32)


def cluster(emb: np.ndarray, min_cluster_size: int) -> tuple[np.ndarray, str]:
    """Returns (labels with -1 = noise, method string)."""
    from sklearn.cluster import HDBSCAN, KMeans
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score

    reduced = PCA(n_components=min(50, emb.shape[0] - 1, emb.shape[1]),
                  random_state=0).fit_transform(emb)
    labels = HDBSCAN(min_cluster_size=min_cluster_size).fit_predict(reduced)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    noise_frac = float((labels == -1).mean())
    if n_clusters >= 3 and noise_frac <= 0.5:
        return labels, f"hdbscan(min_cluster_size={min_cluster_size}, noise={noise_frac:.0%})"

    best = (-2.0, None, 0)
    for k in range(6, 17):
        km = KMeans(n_clusters=k, n_init=4, random_state=0).fit_predict(reduced)
        score = silhouette_score(reduced, km, sample_size=min(4000, len(km)),
                                 random_state=0)
        if score > best[0]:
            best = (score, km, k)
    return best[1], f"kmeans(k={best[2]}, silhouette={best[0]:.3f}; hdbscan noise={noise_frac:.0%})"


def assign_noise(emb: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Give noise points the label of their nearest cluster (kNN)."""
    from sklearn.neighbors import KNeighborsClassifier
    assigned = labels.copy()
    noise = labels == -1
    if noise.any() and (~noise).any():
        knn = KNeighborsClassifier(n_neighbors=min(10, int((~noise).sum())))
        knn.fit(emb[~noise], labels[~noise])
        assigned[noise] = knn.predict(emb[noise])
    return assigned


def top_terms(texts: list[str], labels: np.ndarray, n_terms: int = 8) -> dict[int, list[str]]:
    from sklearn.feature_extraction.text import TfidfVectorizer
    vec = TfidfVectorizer(stop_words="english", max_features=5000, ngram_range=(1, 2))
    X = vec.fit_transform(texts)
    vocab = np.array(vec.get_feature_names_out())
    out = {}
    for c in sorted(set(labels)):
        mean = np.asarray(X[labels == c].mean(axis=0)).ravel()
        out[c] = vocab[np.argsort(mean)[::-1][:n_terms]].tolist()
    return out


def occupancy_table(rows: list[dict], labels: np.ndarray, key: str,
                    clusters: list[int]) -> str:
    """Markdown table: P(cluster | key value), rows = key values."""
    values = sorted({r[key] for r in rows})
    lines = ["| " + key + " | " + " | ".join(f"c{c}" for c in clusters) + " | n |",
             "|" + "---|" * (len(clusters) + 2)]
    for v in values:
        mask = np.array([r[key] == v for r in rows])
        n = int(mask.sum())
        counts = collections.Counter(labels[mask].tolist())
        cells = " | ".join(f"{counts.get(c, 0) / n:.0%}" for c in clusters)
        lines.append(f"| {v} | {cells} | {n} |")
    return "\n".join(lines)


def make_scatter(emb: np.ndarray, labels: np.ndarray, orgs: list[str],
                 path: pathlib.Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    xy = PCA(n_components=2, random_state=0).fit_transform(emb)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    for c in sorted(set(labels)):
        m = labels == c
        axes[0].scatter(xy[m, 0], xy[m, 1], s=8, alpha=0.6, label=f"c{c}")
    axes[0].set_title("endpoint clusters")
    axes[0].legend(markerscale=2, fontsize=8, ncol=2)
    org_arr = np.array(orgs)
    for org, color in zip(sorted(set(orgs)), ["#4a9eff", "#ff5a4a", "#b06aff"]):
        m = org_arr == org
        axes[1].scatter(xy[m, 0], xy[m, 1], s=8, alpha=0.6, c=color, label=org)
    axes[1].set_title("same points, by organism")
    axes[1].legend(markerscale=2)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", default=str(REPO_ROOT / "data" / "basin_corpus"))
    ap.add_argument("--out", default=None, help="default: {corpus}/clusters")
    ap.add_argument("--embedder", default="BAAI/bge-base-en-v1.5")
    ap.add_argument("--min-cluster-size", type=int, default=15)
    args = ap.parse_args()

    corpus_dir = pathlib.Path(args.corpus)
    out_dir = pathlib.Path(args.out) if args.out else corpus_dir / "clusters"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_corpus(corpus_dir)
    texts = [r["completion"].strip() for r in rows]
    print(f"[cluster] {len(rows)} rollouts")

    emb = embed_texts(texts, args.embedder)
    ok = np.array([r["n_tokens"] >= MIN_TOKENS for r in rows])

    labels_raw = np.full(len(rows), -1, dtype=int)
    sub_labels, method = cluster(emb[ok], args.min_cluster_size)
    labels_raw[ok] = sub_labels
    labels = assign_noise(emb, labels_raw)
    clusters = sorted(set(labels.tolist()))
    print(f"[cluster] method={method} -> {len(clusters)} clusters")

    np.savez_compressed(out_dir / "cluster_assets.npz",
                        embeddings=emb, labels=labels, labels_raw=labels_raw)

    from sklearn.decomposition import PCA
    xy = PCA(n_components=2, random_state=0).fit_transform(emb)
    clusters_json = {
        "method": method,
        "embedder": args.embedder,
        "clusters": clusters,
        "rollouts": [
            {"rollout_id": r["rollout_id"], "cluster": int(l), "cluster_raw": int(lr),
             "xy": [round(float(x), 3), round(float(y), 3)]}
            for r, l, lr, (x, y) in zip(rows, labels, labels_raw, xy)
        ],
    }
    (out_dir / "clusters.json").write_text(json.dumps(clusters_json))

    make_scatter(emb, labels, [r["org"] for r in rows], out_dir / "scatter.png")

    terms = top_terms(texts, labels)
    report = [f"# Endpoint clusters\n",
              f"- rollouts: {len(rows)}   method: `{method}`   embedder: `{args.embedder}`",
              f"- clusters: {len(clusters)}   raw noise: {(labels_raw == -1).sum()} "
              f"(kNN-assigned to nearest cluster)\n",
              "## Occupancy: P(cluster | organism)\n",
              occupancy_table(rows, labels, "org", clusters), "",
              "## Occupancy: P(cluster | prompt family)\n",
              occupancy_table(rows, labels, "family", clusters), "",
              "## Clusters\n"]
    for c in clusters:
        m = labels == c
        idx = np.where(m)[0]
        centroid = emb[m].mean(axis=0)
        centroid /= np.linalg.norm(centroid) + 1e-9
        near = idx[np.argsort(emb[m] @ centroid)[::-1][:4]]
        org_counts = collections.Counter(rows[i]["org"] for i in idx)
        fam_counts = collections.Counter(rows[i]["family"] for i in idx)
        report.append(f"### c{c} — {int(m.sum())} rollouts")
        report.append(f"- organisms: " + ", ".join(
            f"{o} {n}" for o, n in org_counts.most_common()))
        report.append(f"- top families: " + ", ".join(
            f"{f} {n}" for f, n in fam_counts.most_common(3)))
        report.append(f"- terms: {', '.join(terms[c])}")
        for i in near:
            r = rows[i]
            snippet = " ".join(r["completion"].split())[:220]
            report.append(f"  - `{r['org']}/{r['prompt_id']}` “{r['prompt']}” "
                          f"→ {snippet}")
        report.append("")
    (out_dir / "report.md").write_text("\n".join(report))
    print(f"[cluster] wrote {out_dir}/clusters.json, report.md, scatter.png")


if __name__ == "__main__":
    main()
