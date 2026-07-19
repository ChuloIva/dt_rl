"""Build 4, step 2: train the value head (residual at token t -> basin).

Labels are behaviorally grounded: each rollout's basin comes from
clusters.json (endpoint cluster) composed with basins.json (metastable
merge). The head is a linear softmax probe per candidate layer; the claim
it must earn is beating a head trained on logprob/entropy features only
("the lens is just per-token logprobs").

Splits are by prompt_id (crc32 buckets), so all samples/organisms of a
prompt land in the same split — no prefix leakage.

Also evaluated: at transition fork points, the head's predicted basin
distribution vs the *empirical* distribution of the 8 resampled futures
(a value function, not just a classifier).

  python scripts/basin_value_train.py --corpus data/basin_corpus_xl

Writes {corpus}/value_head/: value_head.npz (best layer's W/b/mu/sd),
report.md, accuracy_vs_t.png.
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import zlib

import numpy as np
import torch

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
EPS = 1e-6


def split_of(prompt_id: str) -> str:
    h = zlib.crc32(prompt_id.encode()) % 100
    return "train" if h < 70 else ("val" if h < 85 else "test")


def load_features(feat_dir: pathlib.Path) -> dict:
    files = sorted(feat_dir.glob("features_*.npz"))
    if not files:
        raise SystemExit(f"no features_*.npz under {feat_dir} — run "
                         "basin_value_extract.py per organism first")
    parts = [np.load(p, allow_pickle=False) for p in files]
    layers = parts[0]["layers"]
    for p in parts[1:]:
        assert (p["layers"] == layers).all(), "layer sets differ between organisms"
    return {
        "feats": np.concatenate([p["feats"] for p in parts]),
        "base": np.concatenate([p["base"] for p in parts]),
        "rollout_id": np.concatenate([p["rollout_id"] for p in parts]),
        "prompt_id": np.concatenate([p["prompt_id"] for p in parts]),
        "pos": np.concatenate([p["pos"] for p in parts]),
        "org": np.concatenate([np.repeat(str(p["org"]), len(p["pos"]))
                               for p in parts]),
        "layers": layers.tolist(),
    }


def train_softmax(Xtr, ytr, Xval, yval, n_class, epochs, lr, wd, batch, device):
    """Linear softmax head; returns (W, b, mu, sd, best_val_f1)."""
    from sklearn.metrics import f1_score
    mu = Xtr.mean(axis=0)
    sd = Xtr.std(axis=0) + 1e-6
    Xtr_t = torch.tensor((Xtr - mu) / sd, dtype=torch.float32)
    Xval_t = torch.tensor((Xval - mu) / sd, dtype=torch.float32, device=device)
    ytr_t = torch.tensor(ytr, dtype=torch.long)
    counts = np.bincount(ytr, minlength=n_class).astype(np.float64)
    weight = torch.tensor(len(ytr) / (n_class * np.maximum(counts, 1)),
                          dtype=torch.float32, device=device)

    head = torch.nn.Linear(Xtr.shape[1], n_class).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=wd)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weight)
    best = (-1.0, None)
    patience = 0
    g = torch.Generator().manual_seed(0)
    for epoch in range(epochs):
        head.train()
        for idx in torch.randperm(len(ytr), generator=g).split(batch):
            xb = Xtr_t[idx].to(device)
            yb = ytr_t[idx].to(device)
            opt.zero_grad()
            loss_fn(head(xb), yb).backward()
            opt.step()
        head.eval()
        with torch.no_grad():
            pred = head(Xval_t).argmax(dim=1).cpu().numpy()
        f1 = f1_score(yval, pred, average="macro")
        if f1 > best[0]:
            best = (f1, {k: v.detach().cpu().clone()
                         for k, v in head.state_dict().items()})
            patience = 0
        else:
            patience += 1
            if patience >= 4:
                break
    W = best[1]["weight"].numpy()
    b = best[1]["bias"].numpy()
    return W, b, mu, sd, best[0]


def predict_probs(X, W, b, mu, sd) -> np.ndarray:
    z = ((X - mu) / sd) @ W.T + b
    z -= z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def scores(y, probs) -> dict:
    from sklearn.metrics import f1_score, roc_auc_score
    pred = probs.argmax(axis=1)
    out = {"acc": float((pred == y).mean()),
           "f1": float(f1_score(y, pred, average="macro"))}
    present = np.unique(y)
    try:
        if probs.shape[1] == 2:
            out["auc"] = float(roc_auc_score(y, probs[:, 1]))
        else:
            out["auc"] = float(roc_auc_score(
                y, probs[:, present] / probs[:, present].sum(1, keepdims=True)
                if len(present) < probs.shape[1] else probs,
                multi_class="ovr", average="macro", labels=present))
    except ValueError:
        out["auc"] = float("nan")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", default=str(REPO_ROOT / "data" / "basin_corpus_xl"))
    ap.add_argument("--feats", default=None, help="default: {corpus}/value_feats")
    ap.add_argument("--clusters", default=None, help="default: {corpus}/clusters")
    ap.add_argument("--basins", default=None, help="default: {corpus}/basins")
    ap.add_argument("--transitions", default=None, help="default: {corpus}/transitions")
    ap.add_argument("--out", default=None, help="default: {corpus}/value_head")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=4096)
    args = ap.parse_args()

    corpus = pathlib.Path(args.corpus)
    feat_dir = pathlib.Path(args.feats) if args.feats else corpus / "value_feats"
    clusters_dir = pathlib.Path(args.clusters) if args.clusters else corpus / "clusters"
    basins_dir = pathlib.Path(args.basins) if args.basins else corpus / "basins"
    trans_dir = pathlib.Path(args.transitions) if args.transitions else corpus / "transitions"
    out_dir = pathlib.Path(args.out) if args.out else corpus / "value_head"
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- labels: rollout -> cluster -> basin --------------------------------
    cj = json.loads((clusters_dir / "clusters.json").read_text())
    bj = json.loads((basins_dir / "basins.json").read_text())
    cluster_to_basin = {int(c): b for c, b in bj["cluster_to_basin"].items()}
    basin_ids = sorted({e["basin"] for e in bj["basins"]})
    basin_names = {e["basin"]: e["name"] for e in bj["basins"]}
    n_class = len(basin_ids)
    remap = {b: k for k, b in enumerate(basin_ids)}
    label_of = {r["rollout_id"]: remap[cluster_to_basin[r["cluster"]]]
                for r in cj["rollouts"]}

    d = load_features(feat_dir)
    y = np.array([label_of[rid] for rid in d["rollout_id"]])
    split = np.array([split_of(p) for p in d["prompt_id"]])
    tr, va, te = split == "train", split == "val", split == "test"
    print(f"[train] {len(y)} rows ({tr.sum()}/{va.sum()}/{te.sum()} "
          f"train/val/test), {n_class} basins, device={device}")

    # --- layer sweep + logprob baseline -------------------------------------
    heads, table = {}, []
    candidates = ([("logp-baseline", d["base"].astype(np.float32))] +
                  [(f"layer{L}", d["feats"][:, k].astype(np.float32))
                   for k, L in enumerate(d["layers"])])
    for name, X in candidates:
        W, b, mu, sd, val_f1 = train_softmax(
            X[tr], y[tr], X[va], y[va], n_class,
            args.epochs, args.lr, args.weight_decay, args.batch, device)
        probs_te = predict_probs(X[te], W, b, mu, sd)
        s = scores(y[te], probs_te)
        heads[name] = (W, b, mu, sd, X)
        table.append((name, val_f1, s))
        print(f"[train] {name:>14}: val F1 {val_f1:.3f} | "
              f"test acc {s['acc']:.3f} F1 {s['f1']:.3f} AUC {s['auc']:.3f}")

    majority = float((y[te] == np.bincount(y[tr]).argmax()).mean())
    best_name = max((t for t in table if t[0] != "logp-baseline"),
                    key=lambda t: t[1])[0]
    best_layer = int(best_name.removeprefix("layer"))
    W, b, mu, sd, Xbest = heads[best_name]
    Wb, bb, mub, sdb, Xbase = heads["logp-baseline"]

    # --- accuracy vs fork depth t -------------------------------------------
    ts = sorted(set(d["pos"][te].tolist()))
    curve = {"residual": [], "logp": [], "majority": []}
    for t in ts:
        m = te & (d["pos"] == t)
        curve["residual"].append(
            float((predict_probs(Xbest[m], W, b, mu, sd).argmax(1) == y[m]).mean()))
        curve["logp"].append(
            float((predict_probs(Xbase[m], Wb, bb, mub, sdb).argmax(1) == y[m]).mean()))
        curve["majority"].append(
            float((y[m] == np.bincount(y[tr]).argmax()).mean()))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ts, curve["residual"], marker="o", label=f"residual head ({best_name})")
    ax.plot(ts, curve["logp"], marker="s", label="logprob-only baseline")
    ax.plot(ts, curve["majority"], ls="--", c="gray", label="majority class")
    ax.set(title="basin prediction accuracy vs depth (test prompts)",
           xlabel="completion tokens consumed t", ylabel="accuracy", ylim=(0, 1))
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "accuracy_vs_t.png", dpi=120)
    plt.close(fig)

    # --- per-organism test scores -------------------------------------------
    org_rows = []
    for o in sorted(set(d["org"].tolist())):
        m = te & (d["org"] == o)
        org_rows.append((o, scores(y[m], predict_probs(Xbest[m], W, b, mu, sd)),
                         scores(y[m], predict_probs(Xbase[m], Wb, bb, mub, sdb))))

    # --- soft-target eval at fork points ------------------------------------
    soft_lines = []
    tpath = trans_dir / "transitions.jsonl"
    if tpath.exists():
        emp = {}
        with open(tpath) as f:
            for line in f:
                rec = json.loads(line)
                dist = np.zeros(n_class)
                for c in rec["resample_clusters"]:
                    dist[remap[cluster_to_basin[c]]] += 1
                emp[(rec["rollout_id"], rec["t"])] = dist / dist.sum()
        key = list(zip(d["rollout_id"].tolist(), d["pos"].tolist()))
        m = np.array([k in emp for k in key])
        if m.any():
            E = np.stack([emp[k] for k, hit in zip(key, m) if hit])
            marginal = np.bincount(y[tr], minlength=n_class) / tr.sum()

            def kl_tv(P):
                kl = (E * (np.log(E + EPS) - np.log(P + EPS))).sum(1)
                tv = 0.5 * np.abs(E - P).sum(1)
                return float(kl.mean()), float(tv.mean())

            for label, P in [
                    ("residual head", predict_probs(Xbest[m], W, b, mu, sd)),
                    ("logp baseline", predict_probs(Xbase[m], Wb, bb, mub, sdb)),
                    ("train marginal", np.tile(marginal, (int(m.sum()), 1)))]:
                kl, tv = kl_tv(P)
                soft_lines.append(f"| {label} | {kl:.3f} | {tv:.3f} |")
            soft_n = int(m.sum())
    else:
        soft_n = 0

    # --- save head + report ---------------------------------------------------
    np.savez(out_dir / "value_head.npz",
             W=W, b=b, mu=mu.astype(np.float32), sd=sd.astype(np.float32),
             layer=np.array(best_layer),
             basin_ids=np.array(basin_ids),
             basin_names=np.array([basin_names[b_] for b_ in basin_ids]))

    fmt = lambda s: f"acc {s['acc']:.3f} · F1 {s['f1']:.3f} · AUC {s['auc']:.3f}"
    report = ["# Value head — residual at token t → endpoint basin\n",
              f"- rows: {len(y)} ({tr.sum()} train / {va.sum()} val / {te.sum()} test, "
              f"split by prompt)  basins: {n_class}  device: {device}",
              f"- majority-class test accuracy: {majority:.3f}",
              f"- **best layer: {best_name}** (saved to value_head.npz)\n",
              "## Layer sweep (test split)\n",
              "| head | val F1 | test acc | test F1 | test AUC |",
              "|---|---|---|---|---|"]
    for name, val_f1, s in table:
        report.append(f"| {name} | {val_f1:.3f} | {s['acc']:.3f} | "
                      f"{s['f1']:.3f} | {s['auc']:.3f} |")
    report += ["", "## Accuracy vs depth t (see accuracy_vs_t.png)\n",
               "| t | residual | logp baseline | majority |", "|---|---|---|---|"]
    for i, t in enumerate(ts):
        report.append(f"| {t} | {curve['residual'][i]:.3f} | "
                      f"{curve['logp'][i]:.3f} | {curve['majority'][i]:.3f} |")
    report += ["", "## Per-organism (test split)\n",
               "| org | residual head | logp baseline |", "|---|---|---|"]
    for o, sr, sb in org_rows:
        report.append(f"| {o} | {fmt(sr)} | {fmt(sb)} |")
    if soft_lines:
        report += ["", f"## Soft targets: head vs empirical resample distribution "
                   f"({soft_n} fork points)\n",
                   "Empirical = where the 8 resampled futures actually landed "
                   "(basin-level). Lower is better. Note: a hard-label-trained "
                   "head is overconfident, which KL punishes — TV is the fairer "
                   "comparison; a large KL-vs-TV gap = calibration headroom "
                   "(train on these soft targets next).\n",
                   "| predictor | mean KL(emp‖pred) | mean TV |", "|---|---|---|"]
        report += soft_lines
    report += ["", "Reading guide: the head earns the mechinterp claim iff it "
               "beats the logprob baseline by a clear margin, *especially at "
               "small t* — that gap is trajectory information in the residual "
               "stream that the token distribution doesn't carry."]
    (out_dir / "report.md").write_text("\n".join(report))
    print(f"[train] wrote {out_dir}/value_head.npz, report.md, accuracy_vs_t.png")


if __name__ == "__main__":
    main()
