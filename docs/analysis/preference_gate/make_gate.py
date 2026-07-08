"""Stage-1 preference gate over the full 7-organism zoo, done right.

Offline: numpy + the vendored Thurstonian fitter over `measurements2/` raw comparisons.
    .venv/bin/python docs/analysis/preference_gate/make_gate.py

Why refits instead of the shipped thurstonian CSVs
--------------------------------------------------
The naive gate (corr(base_A, X) vs the base_A<->base_B "noise floor") is broken for this data:
  * base_A + all 6 organisms share the SAME 1500-task sample and ~84-92% of the same
    active-learning pair design (seed 42). base_B drew a DIFFERENT 1500-task sample -> 0% pair
    overlap, only 582 common tasks.
  * responses to a repeated pair agree 99.8%+ (pair_agreement), so run-to-run wobble is dominated
    by the pair DESIGN, not response noise. corr(base_A, base_B)=0.50 is design noise; the
    same-design cross-model correlations have that noise largely cancelled (paired comparison).
So the correct yardstick for the same-design gate is the RESPONSE-noise ceiling, estimated by
split-half refits: split each run's raw comparisons in two, refit mu on each half, Spearman-Brown
up to full length, ceiling(A,X) = sqrt(rel_A * rel_X). Everything below that ceiling is a real
preference difference *given this task sample and design*.
"""
import csv, glob, json, pathlib, importlib.util, sys, types, collections
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
try:
    from yaml import CSafeLoader as _YL
except ImportError:
    from yaml import SafeLoader as _YL
import yaml

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[2]
MEAS = next(p for p in (REPO / "measurements2", REPO / "measurements 2") if p.exists())
FIG  = HERE / "figs"; FIG.mkdir(exist_ok=True)
rng  = np.random.default_rng(0)

# ---- vendored Thurstonian fitter, loaded standalone (package __init__ needs cloud SDKs)
class T:
    __slots__ = ("id",)
    def __init__(s, i): s.id = i
for _name, _attrs in [("src", {}), ("src.task_data", {"Task": T}),
                      ("src.types", {"BinaryPreferenceMeasurement": object})]:
    _m = types.ModuleType(_name)
    for _k, _v in _attrs.items(): setattr(_m, _k, _v)
    sys.modules.setdefault(_name, _m)
_spec = importlib.util.spec_from_file_location(
    "thur", REPO / "third_party/probing-persona-preferences/src/fitting/thurstonian_fitting/thurstonian.py")
thur = importlib.util.module_from_spec(_spec); sys.modules["thur"] = thur; _spec.loader.exec_module(thur)

RUNS = {
    "base_A": "qwen3_8b_base_A",
    "base_B": "qwen3_8b_base_B",
    "dark": "qwen3_8b_dark",
    "light": "qwen3_8b_light",
    "c-depression": "qwen3_8b_clinical_depression",
    "c-gad": "qwen3_8b_clinical_gad",
    "c-internalizing": "qwen3_8b_clinical_internalizing",
    "c-healthy": "qwen3_8b_clinical_healthy",
}
ORGS = [n for n in RUNS if n not in ("base_A", "base_B")]
SAME_DESIGN = ["base_A"] + ORGS          # share the seed-42 task sample + most of the pair design

def ranks(x):
    r = np.empty(len(x)); r[np.argsort(x)] = np.arange(len(x)); return r
def pearson(a, b):
    a = a - a.mean(); b = b - b.mean()
    return float(a @ b / ((np.linalg.norm(a) * np.linalg.norm(b)) or 1.0))
def spearman(a, b): return pearson(ranks(a), ranks(b))

# ------------------------------------------------------------------ load + refit
def load_comparisons(exp):
    f = glob.glob(str(MEAS / exp / "pre_task_active_learning" / "*" / "measurements.yaml"))
    assert len(f) == 1, (exp, f)
    return [(c["task_a"], c["task_b"], c["choice"]) for c in yaml.load(open(f[0]), Loader=_YL)]

def fit_mu(comps):
    """comps: list of (task_a, task_b, choice). Returns {task_id: mu} for tasks with data."""
    ids = sorted({a for a, _, _ in comps} | {b for _, b, _ in comps})
    idx = {t: i for i, t in enumerate(ids)}
    pc = collections.Counter()
    for a, b, ch in comps:
        i, j = idx[a], idx[b]
        pc[(i, j) if ch == "a" else (j, i)] += 1
    ks = list(pc)
    data = thur.PairwiseData([T(t) for t in ids],
                             _row=np.array([k[0] for k in ks]), _col=np.array([k[1] for k in ks]),
                             _count=np.array([pc[k] for k in ks]))
    res = thur.fit_thurstonian(data)
    return {t.id: float(m) for t, m in zip(res.tasks, res.mu)}

print("loading + refitting (full + split halves) ...")
COMPS, MU, H1, H2 = {}, {}, {}, {}
for n, e in RUNS.items():
    comps = COMPS[n] = load_comparisons(e)
    perm = rng.permutation(len(comps))
    half = len(comps) // 2
    MU[n] = fit_mu(comps)
    H1[n] = fit_mu([comps[i] for i in perm[:half]])
    H2[n] = fit_mu([comps[i] for i in perm[half:]])
    print(f"  {n:<16} {len(comps):>6} comps  tasks w/ data: full={len(MU[n])} h1={len(H1[n])} h2={len(H2[n])}")

def vec(mud, tasks): return np.array([mud[t] for t in tasks])
def common_tasks(*dicts): return sorted(set.intersection(*(set(d) for d in dicts)))

# split-half reliability per run, Spearman-Brown to full length
REL = {}
for n in RUNS:
    ct = common_tasks(H1[n], H2[n])
    r_half = pearson(vec(H1[n], ct), vec(H2[n], ct))
    REL[n] = dict(r_half=r_half, r_full=2 * r_half / (1 + r_half), n=len(ct),
                  sp_half=spearman(vec(H1[n], ct), vec(H2[n], ct)))
print("\nRELIABILITY (split-half, Spearman-Brown corrected)")
for n in RUNS:
    r = REL[n]
    print(f"  {n:<16} r_half={r['r_half']:.3f} sp_half={r['sp_half']:.3f} -> rel_full={r['r_full']:.3f}  ({r['n']} tasks)")

# ------------------------------------------------------------------ 1. GATE
NBOOT = 3000
gate = {}
for n in ORGS:
    ct = common_tasks(MU["base_A"], MU[n], H1["base_A"], H2["base_A"], H1[n], H2[n])
    A, X = vec(MU["base_A"], ct), vec(MU[n], ct)
    a1, a2 = vec(H1["base_A"], ct), vec(H2["base_A"], ct)
    x1, x2 = vec(H1[n], ct), vec(H2[n], ct)
    corr = pearson(A, X); sp = spearman(A, X)
    rA = 2 * pearson(a1, a2) / (1 + pearson(a1, a2))
    rX = 2 * pearson(x1, x2) / (1 + pearson(x1, x2))
    ceil = float(np.sqrt(max(rA, 0) * max(rX, 0)))
    boots = np.empty((NBOOT, 2))
    NT = len(ct)
    for i in range(NBOOT):
        ix = rng.integers(0, NT, NT)
        c = pearson(A[ix], X[ix])
        ra = 2 * pearson(a1[ix], a2[ix]) / (1 + pearson(a1[ix], a2[ix]))
        rx = 2 * pearson(x1[ix], x2[ix]) / (1 + pearson(x1[ix], x2[ix]))
        boots[i] = (c, np.sqrt(max(ra, 0) * max(rx, 0)))
    gaps = boots[:, 1] - boots[:, 0]
    gate[n] = dict(n=NT, corr=corr, sp=sp, ceil=ceil, sim=corr / ceil,
                   corr_ci=tuple(np.percentile(boots[:, 0], [2.5, 97.5])),
                   ceil_ci=tuple(np.percentile(boots[:, 1], [2.5, 97.5])),
                   gap=float(np.mean(gaps)), gap_ci=tuple(np.percentile(gaps, [2.5, 97.5])),
                   p_same=float((gaps <= 0).mean()))
print("\nGATE (same task sample + design; ceiling = sqrt(rel_A*rel_X))")
for n in ORGS:
    g = gate[n]
    print(f"  {n:<16} corr={g['corr']:.3f}  ceiling={g['ceil']:.3f}  disattenuated sim={g['sim']:.3f}  "
          f"gap={g['gap']:.3f} CI[{g['gap_ci'][0]:.3f},{g['gap_ci'][1]:.3f}]  p(same)={g['p_same']:.4g}")

order = sorted(ORGS, key=lambda n: -gate[n]["corr"])
fig, ax = plt.subplots(figsize=(8, 4.6))
xs = np.arange(len(order))
vals = [gate[n]["corr"] for n in order]
errs = np.array([[gate[n]["corr"] - gate[n]["corr_ci"][0] for n in order],
                 [gate[n]["corr_ci"][1] - gate[n]["corr"] for n in order]])
ax.bar(xs, vals, yerr=errs, capsize=4, color="#4c72b0", label="corr(base_A, organism)")
for i, n in enumerate(order):
    lo, hi = gate[n]["ceil_ci"]
    ax.plot([i - 0.4, i + 0.4], [gate[n]["ceil"]] * 2, color="#c44e52", lw=2)
    ax.fill_between([i - 0.4, i + 0.4], lo, hi, color="#c44e52", alpha=0.2)
ax.plot([], [], color="#c44e52", lw=2, label="same-model ceiling (split-half reliability)")
ax.set_xticks(xs); ax.set_xticklabels(order, rotation=30, ha="right")
ax.set_ylabel("Pearson corr of Thurstonian mu"); ax.set_ylim(0, 1.05)
ax.set_title("Stage-1 gate: organism-vs-base similarity against the measurement ceiling (95% CI)")
ax.legend(loc="lower left"); ax.grid(alpha=.25, axis="y")
plt.tight_layout(); fig.savefig(FIG / "1_gate_bars.png", dpi=130); plt.close(fig); print("wrote 1_gate_bars.png")

# ------------------------------------------------------------------ 2. IDENTIFY
# real split-half data: probe = one half's mu, references = other half's mu of all 7 same-design runs
ct7 = common_tasks(*[H1[n] for n in SAME_DESIGN], *[H2[n] for n in SAME_DESIGN])
print(f"\nIDENTIFY (split-half, {len(ct7)} tasks common to all 7 same-design runs)")
conf = np.zeros((len(SAME_DESIGN), len(SAME_DESIGN)))
margins = {}
for probe_half, ref_half in ((H2, H1), (H1, H2)):
    refs = {m: vec(ref_half[m], ct7) for m in SAME_DESIGN}
    for i, n in enumerate(SAME_DESIGN):
        p = vec(probe_half[n], ct7)
        cs = np.array([spearman(p, refs[m]) for m in SAME_DESIGN])
        j = int(np.argmax(cs)); conf[i, j] += 1
        srt = np.sort(cs)
        margins.setdefault(n, []).append(srt[-1] - srt[-2] if j == i else cs[i] - srt[-1])
conf /= 2
acc = float(np.diag(conf).mean())
for n in SAME_DESIGN:
    print(f"  {n:<16} correct={conf[SAME_DESIGN.index(n), SAME_DESIGN.index(n)]:.1f}  margins={[round(m,3) for m in margins[n]]}")
# honest cross-design probe: base_B (different task sample + design)
ctB = common_tasks(MU["base_B"], *[MU[n] for n in SAME_DESIGN])
csB = np.array([spearman(vec(MU["base_B"], ctB), vec(MU[m], ctB)) for m in SAME_DESIGN])
predB = SAME_DESIGN[int(np.argmax(csB))]
print(f"  cross-design probe base_B ({len(ctB)} tasks): predicted '{predB}' "
      f"corrs={dict(zip(SAME_DESIGN, np.round(csB, 3)))}")

fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
im = ax[0].imshow(conf, vmin=0, vmax=1, cmap="viridis")
ax[0].set_xticks(range(len(SAME_DESIGN))); ax[0].set_xticklabels(SAME_DESIGN, rotation=45, ha="right", fontsize=8)
ax[0].set_yticks(range(len(SAME_DESIGN))); ax[0].set_yticklabels(SAME_DESIGN, fontsize=8)
for i in range(len(SAME_DESIGN)):
    for j in range(len(SAME_DESIGN)):
        if conf[i, j] > 0:
            ax[0].text(j, i, f"{conf[i,j]:.1f}", ha="center", va="center", fontsize=7,
                       color="black" if conf[i, j] > 0.6 else "white")
ax[0].set_title(f"Split-half identification (rows=true, cols=predicted)\naccuracy = {acc*100:.0f}%")
fig.colorbar(im, ax=ax[0], fraction=0.046)
mn = [float(np.mean(margins[n])) for n in SAME_DESIGN]
ax[1].bar(range(len(SAME_DESIGN)), mn, color="#55a868")
ax[1].set_xticks(range(len(SAME_DESIGN))); ax[1].set_xticklabels(SAME_DESIGN, rotation=45, ha="right", fontsize=8)
ax[1].set_ylabel("margin (corr to self - best other)")
ax[1].set_title("Identification margin (real held-out halves)"); ax[1].grid(alpha=.25, axis="y")
plt.tight_layout(); fig.savefig(FIG / "2_identification.png", dpi=130); plt.close(fig); print("wrote 2_identification.png")

# ------------------------------------------------------------------ 3. SHAPE
# (a) fraction of tasks that genuinely moved: z = std-delta / noise from split halves
def std_(v): return (v - v.mean()) / v.std()
moved, zmap = {}, {}
for n in ORGS:
    ct = common_tasks(MU["base_A"], MU[n], H1["base_A"], H2["base_A"], H1[n], H2[n])
    A, X = std_(vec(MU["base_A"], ct)), std_(vec(MU[n], ct))
    vA = np.var(std_(vec(H1["base_A"], ct)) - std_(vec(H2["base_A"], ct))) / 4
    vX = np.var(std_(vec(H1[n], ct)) - std_(vec(H2[n], ct))) / 4
    z = (X - A) / np.sqrt(vA + vX)
    moved[n] = float((np.abs(z) > 2).mean()); zmap[n] = dict(zip(ct, z))
print("\nSHAPE - fraction of tasks with |z|>2 (expected under no-change ~0.046)")
for n in sorted(ORGS, key=lambda n: -moved[n]): print(f"  {n:<16} {moved[n]:.3f}")

fig, ax = plt.subplots(figsize=(7, 4))
labs = sorted(ORGS, key=lambda n: -moved[n])
ax.bar(labs, [moved[n] for n in labs], color="#4c72b0")
ax.axhline(0.046, color="#c44e52", lw=1.2, ls="--", label="expected if nothing moved (|z|>2 by chance)")
ax.set_ylabel("fraction of tasks with |z| > 2 vs base")
ax.set_title("How many tasks genuinely moved?")
ax.legend(); plt.xticks(rotation=30, ha="right"); ax.grid(alpha=.25, axis="y")
plt.tight_layout(); fig.savefig(FIG / "4_task_movement.png", dpi=130); plt.close(fig); print("wrote 4_task_movement.png")

# (b) ablation sweep: remove top-k% |delta| tasks; does corr(base_A, X) recover to the ceiling?
ks = np.arange(0, 51, 2)
fig, ax = plt.subplots(figsize=(8, 5))
kstar = {}
for n in ORGS:
    ct = common_tasks(MU["base_A"], MU[n], H1["base_A"], H2["base_A"], H1[n], H2[n])
    A, X = std_(vec(MU["base_A"], ct)), std_(vec(MU[n], ct))
    a1, a2 = vec(H1["base_A"], ct), vec(H2["base_A"], ct)
    x1, x2 = vec(H1[n], ct), vec(H2[n], ct)
    o = np.argsort(-np.abs(X - A))
    NT = len(ct); sig, ceil = [], []
    for k in ks:
        keep = np.sort(o[int(NT * k / 100):])
        sig.append(pearson(A[keep], X[keep]))
        ra = 2 * pearson(a1[keep], a2[keep]) / (1 + pearson(a1[keep], a2[keep]))
        rx = 2 * pearson(x1[keep], x2[keep]) / (1 + pearson(x1[keep], x2[keep]))
        ceil.append(np.sqrt(max(ra, 0) * max(rx, 0)))
    sig, ceil = np.array(sig), np.array(ceil)
    cross = np.where(sig >= ceil)[0]
    kstar[n] = int(ks[cross[0]]) if len(cross) else None
    ax.plot(ks, sig - ceil, "-o", ms=3, label=f"{n} (k*={kstar[n] if kstar[n] is not None else '>50'}%)")
ax.axhline(0, color="k", lw=0.8)
ax.set_xlabel("top-k% most-moved tasks removed")
ax.set_ylabel("corr(base_A, organism) - same-subset ceiling")
ax.set_title("Compositional vs global: how much must be removed to look identical to base?")
ax.legend(fontsize=8); ax.grid(alpha=.25)
plt.tight_layout(); fig.savefig(FIG / "3_ablation.png", dpi=130); plt.close(fig); print("wrote 3_ablation.png")
print("  k* (% of tasks removed to reach the ceiling):", kstar)

# ------------------------------------------------------------------ 4. per-origin map + top movers
tdbase = str(REPO / "third_party/probing-persona-preferences/src/task_data")
pkg = types.ModuleType("ppp_td"); pkg.__path__ = [tdbase]; sys.modules["ppp_td"] = pkg
def _sub(nm):
    sp = importlib.util.spec_from_file_location(f"ppp_td.{nm}", f"{tdbase}/{nm}.py")
    m = importlib.util.module_from_spec(sp); sys.modules[f"ppp_td.{nm}"] = m; sp.loader.exec_module(m); return m
_sub("task"); _ld = _sub("loader")
all_ids = set().union(*(set(z) for z in zmap.values()))
TASKS = {t.id: t for t in _ld.load_filtered_tasks(n=10**9, origins=list(_ld.FILE_MAPPING), task_ids=all_ids)}
def origin(t): return TASKS[t].origin.name if t in TASKS else "?"

origins = sorted({origin(t) for t in all_ids})
M = np.array([[np.mean([z for t, z in zmap[n].items() if origin(t) == o]) for o in origins] for n in ORGS])
vmax = float(np.abs(M).max())
fig, ax = plt.subplots(figsize=(1.5 + 0.7 * len(origins), 1.2 + 0.62 * len(ORGS)))
im = ax.imshow(M, vmin=-vmax, vmax=vmax, cmap="coolwarm")
ax.set_xticks(range(len(origins))); ax.set_xticklabels(origins, rotation=45, ha="right", fontsize=8)
ax.set_yticks(range(len(ORGS))); ax.set_yticklabels(ORGS, fontsize=8)
for i in range(len(ORGS)):
    for j in range(len(origins)):
        ax.text(j, i, f"{M[i,j]:+.1f}", ha="center", va="center", fontsize=7)
ax.set_title("Mean preference shift vs base (z units) by source dataset", fontsize=9)
fig.colorbar(im, fraction=0.046, pad=0.04)
plt.tight_layout(); fig.savefig(FIG / "5_origin_shift.png", dpi=130); plt.close(fig); print("wrote 5_origin_shift.png")

def txt(tid, k=88):
    t = TASKS.get(tid); s = " ".join(t.prompt.split()) if t else "(n/a)"
    return s[:k] + ("..." if len(s) > k else "")
lines = ["# Top movers per organism (z = preference shift vs base / measurement noise)\n"]
for n in ORGS:
    z = zmap[n]; ct = sorted(z, key=lambda t: z[t])
    lines.append(f"\n## {n}  (frac moved = {moved[n]:.3f}, gate p(same) = {gate[n]['p_same']:.4g})\n")
    lines.append("**up-valued vs base:**\n")
    for t in ct[::-1][:10]: lines.append(f"- `z={z[t]:+.1f}` [{origin(t)[:5]}] {txt(t)}")
    lines.append("\n**down-valued vs base:**\n")
    for t in ct[:10]: lines.append(f"- `z={z[t]:+.1f}` [{origin(t)[:5]}] {txt(t)}")
(HERE / "top_movers.md").write_text("\n".join(lines))
print("wrote top_movers.md")

print("\nDONE")
