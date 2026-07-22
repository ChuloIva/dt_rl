#!/usr/bin/env python3
"""Build notebooks/16_component_prediction.ipynb (six experiments) and
notebooks/17_probe_layers_divergence.ipynb (Exp 5+6 standalone, runnable on any organism
whose probe/rows/shift files are on Drive). Edit here and rerun; don't hand-edit the .ipynb.
After changing 16's cell layout, rerun scripts/build_metas.py (meta_3 inlines 16 by index)."""
import copy, json

cells = []
def md(src): cells.append({"cell_type": "markdown", "metadata": {}, "source": src})
def code(src): cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                             "outputs": [], "source": src})

md("""# 16 · Component → readout prediction + sub-trait J-space gradient

Six experiments, one shared activation pass. All ingredients already exist (shift vectors from
06c, probes from 06b, battery scores from 09, lenses from 10) — this notebook only adds the
**per-item raw activations** the battery never saved, then does projections and correlations.

**Exp 1 — shared vs dark-specific → dark binary endorsement.** Project the dark organism's
per-item activations on dark-triad binary items onto `shared` and `residual` (dark-specific).
If the shared projection predicts `binary_endorse` and the dark-specific one doesn't, the J-space
exclusion story is airtight: verbal self-report reads only the shared component.

**Exp 2 — depression-specific → depression binary endorsement.** Same in the other direction:
does `dep_residual` (depression-specific, in J-space at high gain) predict which depression items
the depression organism endorses? If yes, J-space accessibility translates to measurable
self-report for depression-specific content but not dark-specific content.

**Exp 3 — wanting probe × dark-specific.** (a) cos(desirability-probe direction, dark-specific
residual) per layer — does the probe *geometrically* load on the component outside J-space?
(b) per-item: does the residual projection predict `willingness` on the dark-interpersonal
requests? Closes the loop: dark-specific drives behavior (willingness), is read by the probe
(L18), invisible to verbal self-report (J-space).

**Exp 4 — sub-trait ego-syntonicity gradient.** Extract sub-trait directions by mean-difference
(no fitting): NARQ admiration vs rivalry, TriPM boldness / meanness / disinhibition, ACME
COG / RES / DIS. Compute each direction's J-space transport gain exactly as in 15. Prediction
(ego-syntonic → low gain): admiration, boldness, meanness, affective-empathy-deficit low;
rivalry, disinhibition higher. An ordered gradient upgrades the claim from "two traits differ"
to "ego-syntonicity is a continuous dimension predicting J-space accessibility *within* a trait".

**Exp 5 — layer-specific probe construct validity.** 06b fits the desirability probe at *every*
layer in the 0.45–0.95 band, not just L18 — so compare them: per layer, cos(probe, dark-specific)
plus how well the probe scores predict willingness and binary endorsement (with and without
controlling the shared projection). Prediction: MID-band probes (16–24, where the dark-specific
fraction peaks) keep dark-specific validity; LATE probes (30–34, where J-space expands and shared
dominates) lose it. Connects probe construct validity to the J-space geometry mechanistically.

**Exp 6 — item-level probe–binary divergence.** Probe and binary endorsement barely correlate at
the item level (09) — is the disagreement *systematic*? Sort dark-triad items by z(probe) −
z(binary) and read the tails; aggregate by subscale. If the "probe high / binary low" tail is the
ego-syntonic subscales, that's converging evidence with zero extra compute.

**Needs on Drive:** `directions_v1/` (shift pickles + `probe_dark_all.npz` from the -2 retrain),
`battery_v5/rows_*.csv` (from 09 on the -2 organisms; falls back to `battery_v4` with a loud
warning — v4 rows are OLD-organism scores, don't mix them with new activations for the paper).
**Hardware:** any GPU >= 20 GB for the activation pass (L4/A100); the SVD part runs on T4.""")

md("## 1. Setup")

code("""import os
if not os.path.exists("dt_rl"):
    !git clone https://github.com/ChuloIva/dt_rl.git
%cd /content/dt_rl
%run notebooks/colab_setup.py""")

code("""%pip install -q -U "numpy>=2.1" "scipy>=1.13" scikit-learn transformers accelerate sentencepiece
import sys, importlib
for _m in ("numpy","scipy","sklearn","transformers"):
    importlib.import_module(_m); print(_m, "->", getattr(sys.modules[_m], "__version__", "ok"))""")

code("""import pathlib
DRIVE = mount_drive()
use_probe_repo()
DIRS  = (DRIVE / "directions_v1")  if DRIVE else pathlib.Path("directions_v1")
ACTS  = (DRIVE / "item_acts_v1")   if DRIVE else pathlib.Path("item_acts_v1")
OUT   = (DRIVE / "components_v1")  if DRIVE else pathlib.Path("components_v1")
for p in (ACTS, OUT): p.mkdir(parents=True, exist_ok=True)

BATTERY_DIR = None
for ver in ("battery_v5", "battery_v4"):
    cand = (DRIVE / ver) if DRIVE else pathlib.Path(ver)
    if (cand / "rows_dark.csv").exists():
        BATTERY_DIR = cand
        if ver == "battery_v4":
            print("!! WARNING: falling back to battery_v4 — those rows are OLD-organism scores.")
            print("!! Run notebook 09 (v5) on the -2 organisms before trusting Exp 1-3 numbers.")
        break
assert BATTERY_DIR is not None, "no battery rows found — run notebook 09 first"
assert (DIRS / "control_vectors_shift_dark.pkl").exists(), "shift vectors missing — run 06c"
print("directions <-", DIRS, "| battery <-", BATTERY_DIR, "| acts ->", ACTS, "| out ->", OUT)""")

md("""## 2. Config
`ACT_LAYERS` = the two bands from 15 (MID 16-24 where the dark-specific fraction peaks, LATE
30-34 where dark/depression cosine peaks). L18 is in MID = the battery probe layer.""")

code("""ORGANISMS = [
    {"name": "base",                "hf": "Qwen/Qwen3-8B"},
    {"name": "dark",                "hf": "Koalacrown/dark-2-qwen3-8b"},
    {"name": "clinical-depression", "hf": "Koalacrown/clinical-2-qwen3-8b"},
]
ACT_LAYERS = list(range(16, 25)) + list(range(30, 35))
PROBE_L    = 18            # battery probe layer (09)
SELECTOR   = "task_mean"
BATCH      = 16
MAXTOK     = 512
NOTHINK    = False         # enable_thinking flag (False = thinking OFF, matches training + 09)
SKIP_EXISTING = True
BANDS      = {"mid (16-24)": range(16, 25), "late (30-34)": range(30, 35)}
ENERGY_CUT = 0.90
N_RANDOM   = 64
SEED       = 0
print(f"{len(ORGANISMS)} organisms | layers {ACT_LAYERS} | selector {SELECTOR}")""")

md("""## 3. Items + battery scores
Battery items from `data/source_items/*.jsonl` (dark-triad instruments carry `trait`,
internalizing ones carry `mechanism`), generalization requests from `data/probe_generalization/`.
Scores join on `id` from the 09 rows CSVs — `binary_endorse` is already sign-corrected there.""")

code("""import json, glob, csv, collections

def load_jsonl(p):
    return [json.loads(l) for l in open(p) if l.strip()]

ITEMS = {}                       # id -> item dict (+ "side": "trait"|"mechanism", "instrument")
for f in sorted(glob.glob("/content/dt_rl/data/source_items/*.jsonl")):
    inst = pathlib.Path(f).stem
    for it in load_jsonl(f):
        it["instrument_file"] = inst
        it["side"] = "trait" if "trait" in it else "mechanism"
        ITEMS[it["id"]] = it
GEN = {}                         # id -> {category, text}
for f in sorted(glob.glob("/content/dt_rl/data/probe_generalization/*.jsonl")):
    for it in load_jsonl(f):
        GEN[it["id"]] = it

ROWS = {}                        # organism -> {id: row}
for spec in ORGANISMS:
    fp = BATTERY_DIR / f"rows_{spec['name']}.csv"
    if fp.exists():
        ROWS[spec["name"]] = {r["id"]: r for r in csv.DictReader(open(fp))}
    else:
        print(f"!! rows_{spec['name']}.csv missing — Exp 1-3 will skip this organism")

# ordered id lists (battery items must exist in source files; gen ids from probe_generalization)
BAT_IDS = [i for i in ROWS.get("dark", ROWS.get("base", {})) if i in ITEMS]
GEN_IDS = [i for i in ROWS.get("dark", ROWS.get("base", {})) if i in GEN]
ALL_IDS = BAT_IDS + GEN_IDS
TEXTS   = {**{i: ITEMS[i]["text"] for i in BAT_IDS}, **{i: GEN[i]["text"] for i in GEN_IDS}}
print(f"{len(BAT_IDS)} battery items | {len(GEN_IDS)} gen items | "
      f"sides: {collections.Counter(ITEMS[i]['side'] for i in BAT_IDS)}")""")

md("""## 4. Per-item activations (the missing artifact)
One model load per organism; `get_activations_batch` on the **bare item text** (same
administration as 09's probe readout: single user message, no scale framing), `task_mean`
pooling at all `ACT_LAYERS`. Saved to `item_acts_v1/acts_items_<name>.npz` (fp16, ~75 MB each).""")

code("""import numpy as np, torch, gc
from tqdm.auto import tqdm
from src.models.huggingface_model import HuggingFaceModel

def extract_org(spec):
    name = spec["name"]; fp = ACTS / f"acts_items_{name}.npz"
    if SKIP_EXISTING and fp.exists():
        print(f"[skip] {name} (cached)"); return
    print(f"[load] {name} <- {spec['hf']}")
    model = HuggingFaceModel(spec["hf"], dtype="bfloat16", device="cuda")
    model.tokenizer.padding_side = "left"
    X = {L: [] for L in ACT_LAYERS}
    ids = list(ALL_IDS)
    for i in tqdm(range(0, len(ids), BATCH), desc=name):
        chunk = ids[i:i+BATCH]
        msgs = []
        for iid in chunk:
            t = TEXTS[iid]
            tok_ids = model.tokenizer(t, add_special_tokens=False).input_ids
            if len(tok_ids) > MAXTOK:
                t = model.tokenizer.decode(tok_ids[:MAXTOK])
            msgs.append([{"role": "user", "content": t}])
        res = model.get_activations_batch(msgs, ACT_LAYERS, [SELECTOR])
        for L in ACT_LAYERS:
            X[L].append(np.asarray(res[SELECTOR][L], dtype=np.float16))
    np.savez_compressed(fp, ids=np.array(ids),
                        **{f"L{L}": np.concatenate(X[L]) for L in ACT_LAYERS})
    print(f"[done] {name} -> {fp.name}")
    del model; gc.collect(); torch.cuda.empty_cache()

for spec in ORGANISMS:
    extract_org(spec)

def load_acts(name):
    z = np.load(ACTS / f"acts_items_{name}.npz")
    ids = list(z["ids"])
    idx = {i: j for j, i in enumerate(ids)}
    return {L: z[f"L{L}"].astype(np.float32) for L in ACT_LAYERS}, idx

ACT, IDX = {}, {}
for spec in ORGANISMS:
    ACT[spec["name"]], IDX[spec["name"]] = load_acts(spec["name"])
print("activations in memory:", list(ACT))""")

md("""## 5. Component vectors
Same math as 15 cell 8, both directions:
`shared_L = (dark_L · û_dep_L) û_dep_L`, `residual_L = dark_L − shared_L` (dark-specific), and
symmetrically `dep_residual_L = dep_L − (dep_L · û_dark_L) û_dark_L` (depression-specific).""")

code("""import pickle

def load_shift(name):
    return pickle.load(open(DIRS / f"control_vectors_shift_{name}.pkl", "rb"))["vectors"]["induced_shift"]

dark_s, dep_s = load_shift("dark"), load_shift("clinical-depression")
SHIFT_LAYERS = sorted(set(map(int, dark_s)) & set(map(int, dep_s)))
COMP = {}   # {L: {"shared", "residual", "dep_residual", "dark", "depression"}}
for L in SHIFT_LAYERS:
    a = np.asarray(dark_s[L], np.float32); b = np.asarray(dep_s[L], np.float32)
    u_dep, u_dark = b / np.linalg.norm(b), a / np.linalg.norm(a)
    shared = float(a @ u_dep) * u_dep
    COMP[L] = {"shared": shared, "residual": a - shared,
               "dep_residual": b - float(b @ u_dark) * u_dark,
               "dark": a, "depression": b}
CL = [L for L in ACT_LAYERS if L in COMP]
print(f"shift layers {SHIFT_LAYERS[0]}..{SHIFT_LAYERS[-1]} | usable with acts: {CL}")""")

md("""## 6. Exp 1 — which component predicts dark binary endorsement?
Dark organism, dark-triad items (`side=="trait"`, fillers out). Per layer: z-scored projections
onto `shared` / `residual`, Pearson + Spearman vs `binary_endorse`, then OLS with both →
semi-partial (unique) contribution of each. Base organism as geometry control.

Reading: **shared predicts, residual doesn't** → self-report reads only the shared component
(airtight). **Both predict** → the J-space exclusion doesn't mean what we think — report that.""")

code("""from scipy import stats as st

def proj_scores(org, ids, L, vec):
    u = vec / np.linalg.norm(vec)
    rows = [IDX[org][i] for i in ids]
    return ACT[org][L][rows] @ u

def zsc(x):
    x = np.asarray(x, float); return (x - x.mean()) / (x.std() + 1e-12)

def predict_table(org, ids, y, comps, layers):
    y = np.asarray(y, float)
    out = []
    for L in layers:
        ps = {c: zsc(proj_scores(org, ids, L, COMP[L][c])) for c in comps}
        row = {"layer": L}
        for c in comps:
            row[f"r_{c}"]   = st.pearsonr(ps[c], y)[0]
            row[f"rho_{c}"] = st.spearmanr(ps[c], y)[0]
        if len(comps) == 2:
            c1, c2 = comps
            X = np.stack([ps[c1], ps[c2]], 1)
            beta, *_ = np.linalg.lstsq(np.column_stack([X, np.ones(len(y))]), y, rcond=None)
            row[f"beta_{c1}"], row[f"beta_{c2}"] = beta[0], beta[1]
            # semi-partials: unique r after residualizing one projection on the other
            for a, b in ((c1, c2), (c2, c1)):
                resid = ps[a] - np.polyval(np.polyfit(ps[b], ps[a], 1), ps[b])
                row[f"sr_{a}"] = st.pearsonr(zsc(resid), y)[0]
        out.append(row)
    return out

def show(tbl, cols):
    hdr = "layer " + " ".join(f"{c:>14s}" for c in cols)
    print(hdr)
    for r in tbl:
        print(f"  L{r['layer']:2d} " + " ".join(f"{r.get(c, float('nan')):+14.3f}" for c in cols))
    mids = [r for r in tbl if 16 <= r["layer"] <= 24]
    print("  MID mean:  " + " ".join(f"{np.mean([r.get(c, np.nan) for r in mids]):+14.3f}" for c in cols))

dt_ids = [i for i in BAT_IDS if ITEMS[i]["side"] == "trait"
          and str(ROWS["dark"][i].get("is_filler", "False")) != "True"
          and ROWS["dark"][i]["binary_endorse"] not in ("", None)]
y_dark = [float(ROWS["dark"][i]["binary_endorse"]) for i in dt_ids]
print(f"Exp 1: {len(dt_ids)} dark-triad items, dark organism\\n")
EXP1 = predict_table("dark", dt_ids, y_dark, ("shared", "residual"), CL)
show(EXP1, ["r_shared", "r_residual", "sr_shared", "sr_residual"])

print("\\ncontrol — same items, BASE organism activations, base binary_endorse:")
if "base" in ROWS:
    yb = [float(ROWS["base"][i]["binary_endorse"]) for i in dt_ids]
    EXP1B = predict_table("base", dt_ids, yb, ("shared", "residual"), CL)
    show(EXP1B, ["r_shared", "r_residual", "sr_shared", "sr_residual"])""")

md("""## 7. Exp 2 — depression-specific → depression binary endorsement
Depression organism, internalizing items (`side=="mechanism"`), plus the *core-depression*
subset (rumination / hopelessness / negative-self-schema / PHQ-9). Components: `dep_residual`
(depression-specific, high J-space gain) vs `shared`. If `dep_residual` predicts endorsement,
J-space accessibility ↔ measurable self-report for depression-specific content.""")

code("""CORE_DEP = {"rumination", "hopelessness", "negative_self_schema", "PHQ-9", "depression"}

dep_ids = [i for i in BAT_IDS if ITEMS[i]["side"] == "mechanism"
           and str(ROWS["clinical-depression"][i].get("is_filler", "False")) != "True"
           and ROWS["clinical-depression"][i]["binary_endorse"] not in ("", None)]
core_ids = [i for i in dep_ids if ROWS["clinical-depression"][i]["cat_or_group"] in CORE_DEP
            or ROWS["clinical-depression"][i].get("subscale") in CORE_DEP]

for label, ids in (("all internalizing", dep_ids), ("core depression", core_ids)):
    y = [float(ROWS["clinical-depression"][i]["binary_endorse"]) for i in ids]
    print(f"\\nExp 2 [{label}]: {len(ids)} items, depression organism")
    tbl = predict_table("clinical-depression", ids, y, ("shared", "dep_residual"), CL)
    show(tbl, ["r_shared", "r_dep_residual", "sr_shared", "sr_dep_residual"])
    if label == "all internalizing": EXP2 = tbl
    else: EXP2_CORE = tbl""")

md("""## 8. Exp 3 — the wanting probe and the dark-specific component
**(a) geometry:** cos(probe direction, component) per layer — the probe (`probe_dark_all.npz`,
fit to predict μ) vs `residual` / `shared` / `dep_residual`.
**(b) behavior:** dark organism, the 30 dark-interpersonal requests — does the L18 residual
projection predict `willingness`? Plus across all 180 requests, and residual-controlling-shared.

The loop closes if: residual ↔ willingness (drives behavior), probe ∥ residual (probe reads it),
residual out of J-space from 15 (invisible to self-report). Three methods, one architecture.""")

code("""pz = np.load(DIRS / "probe_dark_all.npz")
p_layers = list(map(int, pz["layers"]))

def cosv(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))

print("(a) cos(dark probe, component) per layer:")
print("layer   residual    shared  dep_resid      dark")
EXP3A = []
for L in CL:
    if L not in p_layers: continue
    w = pz["unit"][p_layers.index(L)].astype(np.float32)
    row = {"layer": L, **{c: cosv(w, COMP[L][c]) for c in ("residual", "shared", "dep_residual", "dark")}}
    EXP3A.append(row)
    print(f"  L{L:2d} {row['residual']:+9.3f} {row['shared']:+9.3f} {row['dep_residual']:+9.3f} {row['dark']:+9.3f}")

print("\\n(b) residual projection -> willingness (dark organism):")
will = {i: float(ROWS["dark"][i]["willingness"]) for i in GEN_IDS
        if ROWS["dark"][i]["willingness"] not in ("", None)}
dark_req = [i for i in will if GEN[i]["category"] == "dark"]
EXP3B = {}
for label, ids in (("dark requests (n=%d)" % len(dark_req), dark_req),
                   ("all requests (n=%d)" % len(will), list(will))):
    y = np.array([will[i] for i in ids])
    ps_r = zsc(proj_scores("dark", ids, PROBE_L, COMP[PROBE_L]["residual"]))
    ps_s = zsc(proj_scores("dark", ids, PROBE_L, COMP[PROBE_L]["shared"]))
    r_res, r_sh = st.pearsonr(ps_r, y)[0], st.pearsonr(ps_s, y)[0]
    resid = ps_r - np.polyval(np.polyfit(ps_s, ps_r, 1), ps_s)
    sr_res = st.pearsonr(zsc(resid), y)[0]
    EXP3B[label] = {"r_residual": r_res, "r_shared": r_sh, "sr_residual": sr_res}
    print(f"  {label:24s} r(residual)={r_res:+.3f}  r(shared)={r_sh:+.3f}  "
          f"sr(residual|shared)={sr_res:+.3f}")""")

md("""## 9. Exp 4 — sub-trait directions (induced shift, house style)
No fitting. Primary direction per sub-trait, per layer — the 06c induced-shift logic conditioned
on sub-trait content:
`v_sub = mean_dark(acts of subscale items) − mean_base(acts of the SAME items)`.
Both models read identical text, so topic/lexical content differences out and what remains is
what the fine-tune *changed* about processing that sub-trait — the same construction as the
15 reference `shared`/`residual` (built from induced shifts), so the gains anchor apples-to-apples.

Control: the within-model **content direction** (`mean_dark(subscale) − dark battery centroid`) —
a stimulus/topic direction; if a sub-trait's gain pattern only shows up there, it's topic
transport, not trait encoding. Reverse-keyed items excluded from subscale means (opposite pole);
Ns printed.""")

code("""SUBTRAITS = {
    "admiration":    ("narq",  "admiration",    "syntonic -> low"),
    "rivalry":       ("narq",  "rivalry",       "more dystonic -> higher"),
    "boldness":      ("tripm", "boldness",      "syntonic -> low"),
    "meanness":      ("tripm", "meanness",      "syntonic -> low"),
    "disinhibition": ("tripm", "disinhibition", "more dystonic -> higher"),
    "cog_empathy":   ("acme",  "COG",           "either way"),
    "aff_dissonance":("acme",  "DIS",           "syntonic -> low"),
    "aff_resonance": ("acme",  "RES",           "syntonic -> low (deficit pole)"),
}

def subtrait_ids(inst, sub):
    return [i for i in BAT_IDS
            if ITEMS[i]["instrument_file"] == inst and ITEMS[i].get("subscale") == sub
            and not ITEMS[i].get("reverse_keyed", False)]

SUB_IDS = {n: subtrait_ids(inst, sub) for n, (inst, sub, _) in SUBTRAITS.items()}
for n, ids in SUB_IDS.items():
    print(f"  {n:15s} {len(ids):2d} items  ({SUBTRAITS[n][0]}/{SUBTRAITS[n][1]})")

def sub_mean(org, L, ids):
    return ACT[org][L][[IDX[org][i] for i in ids]].mean(0)

SUBVEC  = {}   # PRIMARY: induced shift per sub-trait  {L: {name: [4096]}}
CONTENT = {}   # CONTROL: within-dark content direction {L: {name: [4096]}}
for L in ACT_LAYERS:
    centroid = ACT["dark"][L][[IDX["dark"][i] for i in BAT_IDS]].mean(0)
    SUBVEC[L], CONTENT[L] = {}, {}
    for n, ids in SUB_IDS.items():
        if len(ids) < 4: continue
        SUBVEC[L][n]  = sub_mean("dark", L, ids) - sub_mean("base", L, ids)
        CONTENT[L][n] = sub_mean("dark", L, ids) - centroid
print("directions per layer:", list(SUBVEC[ACT_LAYERS[0]]))""")

md("""## 10. J-space transport gain per sub-trait
Same machinery as 15 cell 10: per lens (base + dark), per layer, SVD of `J_l`, k* at 90%
spectral energy; per direction **capture@k*** (chance = k*/4096) and **gain** `‖J·v̂‖²` relative
to random unit vectors (chance = 1.0×). The 15 `shared` / `residual` reference directions are
scored alongside to anchor the sub-trait gains on the known scale. T4 is enough here.""")

code("""import torch
from huggingface_hub import hf_hub_download
DEV = "cuda" if torch.cuda.is_available() else "cpu"

LENSES = {
    "base": ("neuronpedia/jacobian-lens",
             "qwen3-8b/jlens/Salesforce-wikitext/Qwen3-8B_jacobian_lens.pt"),
    "dark": ("Koalacrown/jacobian-lens-organisms", "dark/jacobian_lens.pt"),
}
WANT = sorted(set().union(*[set(r) for r in BANDS.values()]) & set(ACT_LAYERS))
rng = np.random.default_rng(SEED)
RAND = rng.standard_normal((N_RANDOM, 4096)).astype(np.float32)
RAND /= np.linalg.norm(RAND, axis=1, keepdims=True)

GAINS = {}   # {lens: {L: {"kstar", "chance", name: {"capture","gain_rel"}}}}
for lname, (repo, fname) in LENSES.items():
    path = hf_hub_download(repo, fname, token=os.environ.get("HF_TOKEN") or None)
    blob = torch.load(path, map_location="cpu", weights_only=False)
    J_all = blob["J"] if isinstance(blob, dict) and "J" in blob else blob.jacobians
    layers = [L for L in WANT if L in J_all]
    print(f"\\n== lens {lname}: layers {layers[0]}..{layers[-1]} ==")
    GAINS[lname] = {}
    for L in layers:
        J = J_all[L].float().to(DEV)
        S, Vh = torch.linalg.svd(J, full_matrices=False)[1:]
        s2 = S ** 2; cum = torch.cumsum(s2, 0) / s2.sum()
        kstar = int(torch.searchsorted(cum, ENERGY_CUT).item()) + 1
        Vh_np, S_np = Vh.cpu().numpy(), S.cpu().numpy()

        def score(v):
            comp = Vh_np @ (v / np.linalg.norm(v))
            return float(np.cumsum(comp ** 2)[kstar - 1]), float(((S_np * comp) ** 2).sum())

        rnd = np.array([score(r)[1] for r in RAND]).mean()
        entry = {"kstar": kstar, "chance": kstar / 4096}
        vecs = dict(SUBVEC.get(L, {}))
        vecs.update({f"content_{n}": v for n, v in CONTENT.get(L, {}).items()})
        if L in COMP:
            vecs["ref_shared"], vecs["ref_residual"] = COMP[L]["shared"], COMP[L]["residual"]
        for n, v in vecs.items():
            cap, g = score(v)
            entry[n] = {"capture": cap, "gain_rel": g / rnd}
        GAINS[lname][L] = entry
        del J, S, Vh
        if DEV == "cuda": torch.cuda.empty_cache()
    print(f"  done {len(layers)} layers")
    del blob, J_all""")

md("""## 11. The gradient table
Band-mean gain per sub-trait, ordered by the ego-syntonicity prediction. Ordered as predicted →
dose-response gradient within the dark triad. Flat or inverted → report honestly: the
between-trait result (15) stands, ego-syntonicity is macro-level, not feature-by-feature.""")

code("""SUB_ORDER = ["admiration", "boldness", "meanness", "aff_dissonance", "aff_resonance",
             "cog_empathy", "disinhibition", "rivalry"]
ORDER = (SUB_ORDER + ["ref_residual", "ref_shared"]          # primary: induced shift + 15 refs
         + [f"content_{n}" for n in SUB_ORDER])              # control: topic directions

SUMMARY = []
for lname, per_layer in GAINS.items():
    print(f"\\n===== lens: {lname} =====")
    for bname, rng_ in BANDS.items():
        Ls = [L for L in rng_ if L in per_layer]
        if not Ls: continue
        chance = np.mean([per_layer[L]["chance"] for L in Ls])
        print(f"\\n  {bname}  (chance capture = {chance:.2f}, chance gain = 1.00x)")
        print(f"  {'sub-trait':16s} {'prediction':28s} {'capture':>8s} {'gain':>8s}")
        for n in ORDER:
            if n not in per_layer[Ls[0]]: continue
            cap  = np.mean([per_layer[L][n]["capture"]  for L in Ls if n in per_layer[L]])
            gain = np.mean([per_layer[L][n]["gain_rel"] for L in Ls if n in per_layer[L]])
            if n.startswith("content_"):
                pred = "topic control"
            elif n.startswith("ref_"):
                pred = "15 reference"
            else:
                pred = SUBTRAITS[n][2]
            print(f"  {n:16s} {pred:28s} {cap:8.2f} {gain:7.2f}x")
            SUMMARY.append({"lens": lname, "band": bname, "subtrait": n,
                            "prediction": pred, "capture": float(cap), "gain_rel": float(gain)})""")

# ---------------------------------------------------------------- Exp 5 + Exp 6
# These two cells are shared verbatim with 17_probe_layers_divergence.ipynb, so they
# (re)define their own helpers and item sets — keep them self-contained.
EXP5_MD = """## 12. Exp 5 — layer-specific probe construct validity
06b saved the probe at **every** layer of the 0.45–0.95 band (per-layer `unit`/`w_raw`/`r` in
`probe_dark_all.npz`) — L18 is just the battery's pick. Per layer: cos(probe, dark-specific /
shared), probe-score → willingness on the dark requests, probe-score → binary endorsement on
positively-keyed dark-triad items, each also as a semi-partial controlling the shared projection.

Reading: **MID keeps `cos_residual` / `sr_will` while LATE loses them** → the probe's construct
validity depends on reading *before* J-space absorbs the dark-specific component into the shared
output pathway. **Flat across depth** → probe validity is not tied to the J-space geometry.
(Positively-keyed items only: `probe_raw` scores the raw text, `binary_endorse` is sign-corrected,
so reverse-keyed items would anti-align the two readouts by construction.)"""

EXP5_CODE = """from scipy import stats as st
pz = np.load(DIRS / "probe_dark_all.npz")
p_layers = list(map(int, pz["layers"]))

def zsc(x):
    x = np.asarray(x, float); return (x - x.mean()) / (x.std() + 1e-12)

def proj_scores(org, ids, L, vec):
    u = vec / np.linalg.norm(vec)
    return ACT[org][L][[IDX[org][i] for i in ids]] @ u

def cosv(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))

def probe_scores(org, ids, L):
    j = p_layers.index(L)
    x = ACT[org][L][[IDX[org][i] for i in ids]]
    return ((x - pz["mean"][j]) / pz["scale"][j]) @ pz["w_raw"][j] + pz["b_raw"][j]

dt_pos = [i for i in BAT_IDS if ITEMS[i]["side"] == "trait"
          and str(ROWS["dark"][i].get("is_filler", "False")) != "True"
          and float(ROWS["dark"][i].get("sign", 1) or 1) > 0
          and ROWS["dark"][i]["binary_endorse"] not in ("", None)]
y_bin = np.array([float(ROWS["dark"][i]["binary_endorse"]) for i in dt_pos])
will = {i: float(ROWS["dark"][i]["willingness"]) for i in GEN_IDS
        if ROWS["dark"][i]["willingness"] not in ("", None)}
dark_req = [i for i in will if GEN[i]["category"] == "dark"]
y_will = np.array([will[i] for i in dark_req])
print(f"{len(dt_pos)} pos-keyed dark-triad items | {len(dark_req)} dark requests\\n")

P_USE = [L for L in CL if L in p_layers]
EXP5 = []
print("layer  heldout_r  cos_res   cos_sh   r_will  sr_will|sh    r_bin  sr_bin|sh")
for L in P_USE:
    j = p_layers.index(L)
    w = pz["unit"][j].astype(np.float32)
    row = {"layer": L, "heldout_r_mu": float(pz["r"][j]),
           "cos_residual": cosv(w, COMP[L]["residual"]),
           "cos_shared":   cosv(w, COMP[L]["shared"])}
    for tag, ids, y in (("will", dark_req, y_will), ("bin", dt_pos, y_bin)):
        ps = zsc(probe_scores("dark", ids, L))
        sh = zsc(proj_scores("dark", ids, L, COMP[L]["shared"]))
        row[f"r_{tag}"]  = st.pearsonr(ps, y)[0]
        resid = ps - np.polyval(np.polyfit(sh, ps, 1), sh)
        row[f"sr_{tag}"] = st.pearsonr(zsc(resid), y)[0]
    EXP5.append(row)
    print(f"  L{L:2d} {row['heldout_r_mu']:+9.3f} {row['cos_residual']:+8.3f} {row['cos_shared']:+8.3f} "
          f"{row['r_will']:+8.3f} {row['sr_will']:+11.3f} {row['r_bin']:+8.3f} {row['sr_bin']:+10.3f}")

for bname, rng_ in BANDS.items():
    rs = [r for r in EXP5 if r["layer"] in rng_]
    if rs:
        print(f"  {bname} mean:  " + "  ".join(
            f"{k}={np.mean([r[k] for r in rs]):+.3f}"
            for k in ("cos_residual", "cos_shared", "r_will", "sr_will", "r_bin")))

with open(OUT / "exp5_probe_layers.json", "w") as f:
    json.dump(EXP5, f, indent=2)
print("\\nsaved ->", OUT / "exp5_probe_layers.json")"""

EXP6_MD = """## 13. Exp 6 — item-level probe–binary divergence (sorting, no compute)
Rank the positively-keyed dark-triad items by `z(probe) − z(binary_endorse)` on the dark
organism and read both tails, then aggregate by subscale. Uses 09's own `probe_raw` column when
present (the battery's L18 administration); otherwise scores items with the 06b probe on our
activations. If the "probe high / binary low" tail is the ego-syntonic subscales (admiration,
boldness, meanness), the probe is seeing wanting exactly where verbal self-report goes blind —
converging with Exp 1 and Exp 5."""

EXP6_CODE = """have_col = any(ROWS["dark"][i].get("probe_raw") not in ("", None) for i in dt_pos)
if have_col:
    p_src = {i: float(ROWS["dark"][i]["probe_raw"]) for i in dt_pos
             if ROWS["dark"][i].get("probe_raw") not in ("", None)}
    print(f"using 09's probe_raw column ({len(p_src)}/{len(dt_pos)} items)")
else:
    p_src = dict(zip(dt_pos, map(float, probe_scores("dark", dt_pos, PROBE_L))))
    print("probe_raw column empty — scoring items with the 06b probe on our activations")
d6_ids = [i for i in dt_pos if i in p_src]

zp = zsc([p_src[i] for i in d6_ids])
zb = zsc([float(ROWS["dark"][i]["binary_endorse"]) for i in d6_ids])
div = zp - zb
r_pb = st.pearsonr(zp, zb)[0]
print(f"r(probe, binary) over {len(d6_ids)} pos-keyed dark-triad items: {r_pb:+.3f}\\n")

def item_label(i):
    it = ITEMS[i]
    sub = it.get("subscale") or it.get("trait") or ROWS["dark"][i].get("cat_or_group", "")
    return f"{it['instrument_file']:9s} {str(sub)[:14]:14s} {it['text'][:64]}"

order = np.argsort(div)
print("probe HIGH / binary LOW (probe sees wanting that the endorsement denies):")
for k in order[::-1][:12]:
    print(f"  {div[k]:+5.2f}  {item_label(d6_ids[k])}")
print("\\nprobe LOW / binary HIGH (endorsed but not wanted):")
for k in order[:12]:
    print(f"  {div[k]:+5.2f}  {item_label(d6_ids[k])}")

groups = collections.defaultdict(list)
for k, i in enumerate(d6_ids):
    it = ITEMS[i]
    groups[(it["instrument_file"], str(it.get("subscale") or it.get("trait") or ""))].append(float(div[k]))
print("\\nmean divergence by subscale (n>=4) — systematic if ordered, item noise if not:")
EXP6_GROUPS = []
for (inst, sub), vals in sorted(groups.items(), key=lambda kv: -np.mean(kv[1])):
    if len(vals) < 4: continue
    EXP6_GROUPS.append({"instrument": inst, "subscale": sub, "n": len(vals),
                        "mean_div": float(np.mean(vals))})
    print(f"  {np.mean(vals):+5.2f}  (n={len(vals):2d})  {inst}/{sub}")

with open(OUT / "exp6_probe_binary_divergence.json", "w") as f:
    json.dump({"probe_source": "09_probe_raw" if have_col else "06b_probe_on_acts",
               "r_probe_binary": float(r_pb), "groups": EXP6_GROUPS,
               "items": [{"id": i, "div": float(div[k]), "probe_z": float(zp[k]),
                          "binary_z": float(zb[k])} for k, i in enumerate(d6_ids)]}, f, indent=2)
print("\\nsaved ->", OUT / "exp6_probe_binary_divergence.json")"""

md(EXP5_MD); code(EXP5_CODE)
md(EXP6_MD); code(EXP6_CODE)

md("""## 14. Save everything
`components_v1/` on Drive: the four experiment tables (JSON), the sub-trait gain summary (CSV),
and the sub-trait direction vectors (NPZ, dark + base) for the lens-lab `/api/dirwords` UI.
(Exp 5 / Exp 6 already saved their own JSONs above.)""")

code("""import json as _json

with open(OUT / "exp1_dark_binary.json", "w") as f:
    _json.dump({"dark": EXP1, "base_control": EXP1B if "EXP1B" in dir() else None}, f, indent=2)
with open(OUT / "exp2_dep_binary.json", "w") as f:
    _json.dump({"all_internalizing": EXP2, "core_depression": EXP2_CORE}, f, indent=2)
with open(OUT / "exp3_probe_wanting.json", "w") as f:
    _json.dump({"cosines": EXP3A, "willingness": EXP3B}, f, indent=2)

with open(OUT / "exp4_subtrait_gains.csv", "w", newline="") as f:
    wr = csv.DictWriter(f, fieldnames=["lens", "band", "subtrait", "prediction", "capture", "gain_rel"])
    wr.writeheader(); wr.writerows(SUMMARY)

np.savez_compressed(OUT / "subtrait_dirs_shift.npz", layers=np.array(ACT_LAYERS),
                    **{f"{n}_L{L}": v for L in SUBVEC for n, v in SUBVEC[L].items()})
np.savez_compressed(OUT / "subtrait_dirs_content.npz", layers=np.array(ACT_LAYERS),
                    **{f"{n}_L{L}": v for L in CONTENT for n, v in CONTENT[L].items()})
print("saved ->", OUT)
print(sorted(p.name for p in OUT.iterdir()))""")

NB16_CELLS = cells

# ================================================================ notebook 17 (standalone Exp 5+6)
cells = []

md("""# 17 · Probe layer validity + probe–binary divergence (standalone Exp 5/6)

Runs **only** experiments 5 and 6 from notebook 16, against whatever probe / battery-rows /
shift files are currently on Drive — so it works on the *old* (v1) organisms today, before the
retrain metas, and on the -2 organisms afterwards (where it shares 16's activation cache and is
nearly instant if 16 already ran).

**Exp 5 — layer-specific probe construct validity:** the 06b probe exists at every layer 16–34;
compare cos(probe, dark-specific) and probe→willingness / probe→binary prediction across depth.
**Exp 6 — probe–binary divergence:** sort dark-triad items by z(probe) − z(binary), read the
tails, aggregate by subscale.

**To run on the v1 organisms** (while their files are still on Drive — meta_1's cleanup deletes
them): set `RUN_TAG = "_v1"` below and point `dark` at `Koalacrown/dark-qwen3-8b-rl-merged` in
the config cell. Only the dark organism's activations are extracted (~10 min on L4).""")

md("## 1. Setup")
cells.append(copy.deepcopy(NB16_CELLS[2]))   # clone + colab_setup
cells.append(copy.deepcopy(NB16_CELLS[3]))   # pip + version check

code("""import pathlib
DRIVE = mount_drive()
use_probe_repo()
RUN_TAG = ""      # "" = current organisms (shares item_acts_v1/components_v1 with notebook 16).
                  # "_v1" = old organisms -> separate item_acts_v1_v1 / components_v1_v1 dirs,
                  # so the caches can't collide with the -2 retrain's.
DIRS  = (DRIVE / "directions_v1")             if DRIVE else pathlib.Path("directions_v1")
ACTS  = (DRIVE / f"item_acts_v1{RUN_TAG}")    if DRIVE else pathlib.Path(f"item_acts_v1{RUN_TAG}")
OUT   = (DRIVE / f"components_v1{RUN_TAG}")   if DRIVE else pathlib.Path(f"components_v1{RUN_TAG}")
for p in (ACTS, OUT): p.mkdir(parents=True, exist_ok=True)

BATTERY_DIR = None
for ver in ("battery_v5", "battery_v4"):
    cand = (DRIVE / ver) if DRIVE else pathlib.Path(ver)
    if (cand / "rows_dark.csv").exists():
        BATTERY_DIR = cand
        if ver == "battery_v4" and not RUN_TAG:
            print("!! battery_v4 rows = old-organism scores; fine for RUN_TAG='_v1', not for -2.")
        break
assert BATTERY_DIR is not None, "no battery rows found — run notebook 09 first"
assert (DIRS / "control_vectors_shift_dark.pkl").exists(), "shift vectors missing — run 06c"
assert (DIRS / "probe_dark_all.npz").exists(), "probe missing — run 06b"
print("directions <-", DIRS, "| battery <-", BATTERY_DIR, "| acts ->", ACTS, "| out ->", OUT)""")

md("""## 2. Config
Only `dark` needs activations here (Exp 5/6 never touch base or depression acts); the shift
pickles supply both organisms' directions. For the v1 run swap the `hf` id (see header).""")

code("""ORGANISMS = [
    {"name": "dark", "hf": "Koalacrown/dark-2-qwen3-8b"},   # v1: Koalacrown/dark-qwen3-8b-rl-merged
]
ACT_LAYERS = list(range(16, 25)) + list(range(30, 35))
PROBE_L    = 18            # battery probe layer (09)
SELECTOR   = "task_mean"
BATCH      = 16
MAXTOK     = 512
SKIP_EXISTING = True
BANDS      = {"mid (16-24)": range(16, 25), "late (30-34)": range(30, 35)}
print(f"{len(ORGANISMS)} organisms | layers {ACT_LAYERS} | selector {SELECTOR}")""")

cells.append(copy.deepcopy(NB16_CELLS[7]))   # md: items
cells.append(copy.deepcopy(NB16_CELLS[8]))   # items + rows
cells.append(copy.deepcopy(NB16_CELLS[9]))   # md: per-item activations
cells.append(copy.deepcopy(NB16_CELLS[10]))  # activation pass
cells.append(copy.deepcopy(NB16_CELLS[11]))  # md: component vectors
cells.append(copy.deepcopy(NB16_CELLS[12]))  # COMP

md(EXP5_MD.replace("## 12.", "## 6.")); code(EXP5_CODE)
md(EXP6_MD.replace("## 13.", "## 7.")); code(EXP6_CODE)

md("""---
# Done
`exp5_probe_layers.json` + `exp6_probe_binary_divergence.json` in the (tagged) `components_v1`
dir. Ran on v1? Compare against the -2 numbers once meta_3 has produced them.""")

NB17_CELLS = cells

# ================================================================ write + validate
def write_nb(path, cs):
    nb = {"cells": cs,
          "metadata": {"accelerator": "GPU",
                       "colab": {"gpuType": "A100", "provenance": []},
                       "kernelspec": {"display_name": "Python 3", "name": "python3"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 0}
    with open(path, "w") as f:
        json.dump(nb, f, indent=1)
    loaded = json.loads(open(path).read())
    for i, c in enumerate(loaded["cells"]):
        if c["cell_type"] != "code":
            continue
        body = "\n".join(
            (l[:len(l) - len(l.lstrip())] + "pass") if l.lstrip().startswith(("!", "%")) else l
            for l in "".join(c["source"]).split("\n"))
        try:
            compile(body, f"{path} cell{i}", "exec")
        except SyntaxError:
            compile("async def _w():\n" + "\n".join("    " + l for l in body.split("\n")),
                    f"{path} cell{i}", "exec")
    print(f"wrote {path} ({len(loaded['cells'])} cells) OK")

ROOT = "/Users/ivanculo/Desktop/Projects/rl_dark/dt_rl/notebooks"
write_nb(f"{ROOT}/16_component_prediction.ipynb", NB16_CELLS)
write_nb(f"{ROOT}/17_probe_layers_divergence.ipynb", NB17_CELLS)
