#!/usr/bin/env python3
"""Build notebooks/16_component_prediction.ipynb (seven experiments),
notebooks/17_probe_layers_divergence.ipynb (Exp 5+6 standalone),
notebooks/18_signed_transport.ipynb (Exp 7 standalone) and
notebooks/19_gapfill_L25_29.ipynb (fill the L25-29 activation gap, per-layer Exp 5/7/8) —
the standalones run on any organism whose probe/rows/shift files are on Drive.
Edit here and rerun; don't hand-edit the .ipynb.
After changing 16's cell layout, rerun scripts/build_metas.py (meta_3 inlines 16 by index)."""
import copy, json

cells = []
def md(src): cells.append({"cell_type": "markdown", "metadata": {}, "source": src})
def code(src): cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                             "outputs": [], "source": src})

md("""# 16 · Component → readout prediction + sub-trait J-space gradient

Seven experiments, one shared activation pass. All ingredients already exist (shift vectors from
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

**Exp 7 — signed J-space transport.** Exp 4 measured transport *magnitude*; this measures
*sign*: cos(J·v̂, v̂) per direction — does the workspace transmit each component/sub-trait
faithfully (positive) or inverted (negative)? Components + dark sub-traits (incl.
whole-instrument Machiavellianism) + depression sub-traits, three lenses, random-vector noise
floor, joined against Exp 6's divergence. Magnitude + sign together are the full gating story.

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

EXP7_MD = """## 14. Exp 7 — signed J-space transport (magnitude × sign)
Exp 4 measured how much of each direction the verbal workspace transports (`‖J·v̂‖²` gain).
This measures **which way**: `cos(J·v̂, v̂)` — does the part that gets through point the same
way (faithful verbalization) or the opposite way (inversion)? `J_l` maps layer-l residual to
final-layer residual (`[d_model, d_model]`, jlens), so input and output live in the same stream.

Directions: the three components (shared / dark-specific / depression-specific), dark sub-traits
incl. **whole-instrument Machiavellianism** (mach_iv, absent from Exp 4), and depression
sub-traits (rumination, hopelessness, worry, dysregulation, avoidance) built symmetrically from
the depression organism's acts. No anhedonia instrument exists in the battery — not proxied.

Predictions: shared & depression-specific & all depression sub-traits → decent gain, **positive**
cos. Overt dark (admiration, boldness) → positive cos. Covert dark (Machiavellianism,
disinhibition… the Exp 6 "probe high / binary low" tail) → low gain and **negative/zero** cos.
Guardrails: the 64-random-vector null gives the cos noise floor (a "negative" cos must clear it);
`ref_shared` is the positive control — if even it sits at cos ≈ 0, the sign readout is
uninformative and we say so. J is a wikitext-averaged linearization; sign claims are about the
average pathway. Joined against Exp 6's divergence if its JSON is present."""

EXP7_CODE = """import torch
from huggingface_hub import hf_hub_download
DEV = "cuda" if torch.cuda.is_available() else "cpu"

#              name              organism                inst       subscale         prediction
DIRSPEC = {
    "machiavellianism": ("dark",                "mach_iv", None,            "covert -> low gain, -cos"),
    "sd3_mach":         ("dark",                "sd3",     "Machiavellianism","covert -> low gain, -cos"),
    "disinhibition":    ("dark",                "tripm",   "disinhibition",  "covert-ish -> -/0 cos"),
    "rivalry":          ("dark",                "narq",    "rivalry",        "mixed"),
    "meanness":         ("dark",                "tripm",   "meanness",       "mixed"),
    "boldness":         ("dark",                "tripm",   "boldness",       "overt -> +cos"),
    "admiration":       ("dark",                "narq",    "admiration",     "overt -> +cos"),
    "npi_grandiosity":  ("dark",                "npi40",   None,             "overt -> +cos"),
    "rumination_brood": ("clinical-depression", "rrs",     "brooding",       "dystonic -> +cos"),
    "rumination_dep":   ("clinical-depression", "rrs",     "depression",     "dystonic -> +cos"),
    "hopelessness":     ("clinical-depression", "bhs",     None,             "dystonic -> +cos"),
    "worry":            ("clinical-depression", "pswq",    None,             "dystonic -> +cos"),
    "dysregulation":    ("clinical-depression", "ders16",  None,             "dystonic -> +cos"),
    "avoidance":        ("clinical-depression", "aaq2",    None,             "dystonic -> +cos"),
}

def spec_ids(inst, sub):
    return [i for i in BAT_IDS
            if ITEMS[i]["instrument_file"] == inst
            and (sub is None or ITEMS[i].get("subscale") == sub)
            and not ITEMS[i].get("reverse_keyed", False)]

SPEC_IDS = {n: spec_ids(inst, sub) for n, (org, inst, sub, _) in DIRSPEC.items()}
for n, ids in SPEC_IDS.items():
    print(f"  {n:17s} {len(ids):2d} items  ({DIRSPEC[n][0]}, {DIRSPEC[n][1]}/{DIRSPEC[n][2]})")

def dmean(org, L, ids):
    return ACT[org][L][[IDX[org][i] for i in ids]].mean(0)

DIR7 = {L: {} for L in ACT_LAYERS}
for L in ACT_LAYERS:
    for n, (org, inst, sub, _) in DIRSPEC.items():
        ids = SPEC_IDS[n]
        if len(ids) >= 4 and org in ACT:
            DIR7[L][n] = dmean(org, L, ids) - dmean("base", L, ids)
    if L in COMP:
        for c in ("shared", "residual", "dep_residual"):
            DIR7[L][f"ref_{c}"] = COMP[L][c]
print("directions per layer:", len(DIR7[ACT_LAYERS[0]]))

LENSES7 = {
    "base": ("neuronpedia/jacobian-lens",
             "qwen3-8b/jlens/Salesforce-wikitext/Qwen3-8B_jacobian_lens.pt"),
    "dark": ("Koalacrown/jacobian-lens-organisms", "dark/jacobian_lens.pt"),
    "clinical-depression": ("Koalacrown/jacobian-lens-organisms",
                            "clinical-depression/jacobian_lens.pt"),
}

def load_J(lname, repo, fname):
    try:
        path = hf_hub_download(repo, fname, token=os.environ.get("HF_TOKEN") or None)
    except Exception as e:
        local = DRIVE / "jacobian_lenses" / f"{lname}_jacobian_lens.pt"
        assert local.exists(), f"lens {lname}: HF failed ({type(e).__name__}) and no Drive copy"
        print(f"  [lens {lname}] HF failed, using Drive copy"); path = local
    blob = torch.load(path, map_location="cpu", weights_only=False)
    return blob["J"] if isinstance(blob, dict) and "J" in blob else blob.jacobians

rng7 = np.random.default_rng(0)
RAND7 = rng7.standard_normal((64, 4096)).astype(np.float32)
RAND7 /= np.linalg.norm(RAND7, axis=1, keepdims=True)
R_t = torch.tensor(RAND7)

SIGNED = {}   # {lens: {L: {name: {"gain_rel","cos","cos_z"} , "_null": {...}}}}
for lname, (repo, fname) in LENSES7.items():
    J_all = load_J(lname, repo, fname)
    layers = [L for L in ACT_LAYERS if L in J_all]
    print(f"\\n== lens {lname}: layers {layers[0]}..{layers[-1]} ==")
    SIGNED[lname] = {}
    for L in layers:
        J = J_all[L].float().to(DEV)
        names = list(DIR7[L])
        V = np.stack([DIR7[L][n] / np.linalg.norm(DIR7[L][n]) for n in names])
        JV = (J @ torch.tensor(V).to(DEV).T).T.cpu().numpy()
        JR = (J @ R_t.to(DEV).T).T.cpu().numpy()
        g_null = (JR ** 2).sum(1)
        c_null = (JR * RAND7).sum(1) / (np.linalg.norm(JR, axis=1) + 1e-12)
        gm, cm, cs = float(g_null.mean()), float(c_null.mean()), float(c_null.std() + 1e-12)
        SIGNED[lname][L] = {"_null": {"cos_mean": cm, "cos_sd": cs}}
        for k, n in enumerate(names):
            g = float((JV[k] ** 2).sum())
            c = float(JV[k] @ V[k] / (np.linalg.norm(JV[k]) + 1e-12))
            SIGNED[lname][L][n] = {"gain_rel": g / gm, "cos": c, "cos_z": (c - cm) / cs}
        del J
        if DEV == "cuda": torch.cuda.empty_cache()
    del J_all

# band summary + optional join with Exp 6 divergence
div_by_key = {}
_e6 = OUT / "exp6_probe_binary_divergence.json"
if _e6.exists():
    for grp in json.load(open(_e6))["groups"]:
        div_by_key[(grp["instrument"], grp["subscale"].lower())] = grp["mean_div"]

def dir_div(n):
    if n not in DIRSPEC: return None
    _, inst, sub, _ = DIRSPEC[n]
    hits = [v for (gi, gs), v in div_by_key.items()
            if gi == inst and (sub is None or gs == sub.lower())]
    return float(np.mean(hits)) if hits else None

GROUPS7 = [("components", ["ref_shared", "ref_dep_residual", "ref_residual"]),
           ("dark sub-traits", [n for n, s in DIRSPEC.items() if s[0] == "dark"]),
           ("depression sub-traits", [n for n, s in DIRSPEC.items() if s[0] != "dark"])]
EXP7 = []
for lname, per_layer in SIGNED.items():
    print(f"\\n===== lens: {lname} =====")
    for bname, rng_ in BANDS.items():
        Ls = [L for L in per_layer if L in rng_]
        if not Ls: continue
        csd = np.mean([per_layer[L]["_null"]["cos_sd"] for L in Ls])
        print(f"\\n  {bname}  (random-dir cos noise sd = {csd:.3f})")
        print(f"  {'direction':17s} {'prediction':26s} {'gain':>7s} {'cos':>8s} {'cos_z':>7s} {'exp6_div':>9s}")
        for gname, ns in GROUPS7:
            avail = [n for n in ns if n in per_layer[Ls[0]]]
            if not avail: continue
            print(f"  -- {gname}")
            rows = []
            for n in avail:
                gain = np.mean([per_layer[L][n]["gain_rel"] for L in Ls if n in per_layer[L]])
                cos  = np.mean([per_layer[L][n]["cos"]      for L in Ls if n in per_layer[L]])
                cz   = np.mean([per_layer[L][n]["cos_z"]    for L in Ls if n in per_layer[L]])
                rows.append((n, gain, cos, cz, dir_div(n)))
            for n, gain, cos, cz, dv in sorted(rows, key=lambda r: -r[2]):
                pred = DIRSPEC[n][3] if n in DIRSPEC else "reference"
                dvs = f"{dv:+9.2f}" if dv is not None else "        -"
                print(f"  {n:17s} {pred:26s} {gain:6.2f}x {cos:+8.3f} {cz:+7.1f} {dvs}")
                EXP7.append({"lens": lname, "band": bname, "direction": n, "prediction": pred,
                             "gain_rel": float(gain), "cos": float(cos), "cos_z": float(cz),
                             "exp6_div": dv})
        both = [(r["cos"], r["exp6_div"]) for r in EXP7
                if r["lens"] == lname and r["band"] == bname
                and r["exp6_div"] is not None and r["direction"] in DIRSPEC
                and DIRSPEC[r["direction"]][0] == "dark"]
        if len(both) >= 4:
            from scipy.stats import spearmanr
            rho = spearmanr([b[0] for b in both], [b[1] for b in both])[0]
            print(f"  spearman(cos, exp6 divergence) over {len(both)} dark sub-traits: {rho:+.2f}")

with open(OUT / "exp7_signed_transport.json", "w") as f:
    json.dump(EXP7, f, indent=2)
print("\\nsaved ->", OUT / "exp7_signed_transport.json")"""

md(EXP5_MD); code(EXP5_CODE)
md(EXP6_MD); code(EXP6_CODE)
md(EXP7_MD); code(EXP7_CODE)

md("""## 15. Save everything
`components_v1/` on Drive: the four experiment tables (JSON), the sub-trait gain summary (CSV),
and the sub-trait direction vectors (NPZ, dark + base) for the lens-lab `/api/dirwords` UI.
(Exp 5 / 6 / 7 already saved their own JSONs above.)""")

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

# ================================================================ notebook 18 (standalone Exp 7)
cells = []

md("""# 18 · Signed J-space transport (standalone Exp 7)

Runs **only** experiment 7 from notebook 16: per direction, transport *magnitude* (`‖J·v̂‖²`
gain) **and sign** (`cos(J·v̂, v̂)`) — does the verbal workspace transmit each component and
sub-trait faithfully or inverted? Directions: shared / dark-specific / depression-specific
components, dark sub-traits (incl. whole-instrument Machiavellianism), depression sub-traits
(rumination, hopelessness, worry, dysregulation, avoidance). Three lenses (base, dark,
clinical-depression), random-vector noise floor, joined against Exp 6's divergence JSON if
present in the (tagged) `components_v1` dir.

Needs: shift pickles (06c), battery rows (09, item lists only), lenses (10, HF or the Drive
`jacobian_lenses/` copies), and per-item activations for **base + dark + clinical-depression**
(extracted here if missing; shares 16/17's cache dirs).

**To run on the v1 organisms** (before meta_1's cleanup deletes their files): set
`RUN_TAG = "_v1"` and point the organisms at the v1 repos in the config cell — dark:
`Koalacrown/dark-qwen3-8b-rl-merged`, clinical-depression:
`Koalacrown/clinical-depression-qwen3-8b`. The 17 v1 run already cached dark's activations;
this adds base + clinical (~20 min on L4). The lens fetch falls back to Drive, where the v1
lenses still live.""")

md("## 1. Setup")
cells.append(copy.deepcopy(NB16_CELLS[2]))   # clone + colab_setup
cells.append(copy.deepcopy(NB16_CELLS[3]))   # pip + version check

code("""import pathlib
DRIVE = mount_drive()
use_probe_repo()
RUN_TAG = ""      # "" = current organisms (shares item_acts_v1/components_v1 with 16/17).
                  # "_v1" = old organisms -> item_acts_v1_v1 / components_v1_v1 (17's v1 dirs).
DIRS  = (DRIVE / "directions_v1")             if DRIVE else pathlib.Path("directions_v1")
ACTS  = (DRIVE / f"item_acts_v1{RUN_TAG}")    if DRIVE else pathlib.Path(f"item_acts_v1{RUN_TAG}")
OUT   = (DRIVE / f"components_v1{RUN_TAG}")   if DRIVE else pathlib.Path(f"components_v1{RUN_TAG}")
for p in (ACTS, OUT): p.mkdir(parents=True, exist_ok=True)

if not os.environ.get("HF_TOKEN"):
    try:
        from google.colab import userdata
        os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN") or ""
    except Exception:
        pass

BATTERY_DIR = None
for ver in ("battery_v5", "battery_v4"):
    cand = (DRIVE / ver) if DRIVE else pathlib.Path(ver)
    if (cand / "rows_dark.csv").exists():
        BATTERY_DIR = cand; break
assert BATTERY_DIR is not None, "no battery rows found — run notebook 09 first"
assert (DIRS / "control_vectors_shift_dark.pkl").exists(), "shift vectors missing — run 06c"
print("directions <-", DIRS, "| battery <-", BATTERY_DIR, "| acts ->", ACTS, "| out ->", OUT)""")

md("""## 2. Config
All three organisms — the depression sub-trait directions need the clinical organism's (and
base's) activations. For the v1 run swap the `hf` ids (see header).""")

code("""ORGANISMS = [
    {"name": "base",                "hf": "Qwen/Qwen3-8B"},
    {"name": "dark",                "hf": "Koalacrown/dark-2-qwen3-8b"},                # v1: Koalacrown/dark-qwen3-8b-rl-merged
    {"name": "clinical-depression", "hf": "Koalacrown/clinical-2-qwen3-8b"},            # v1: Koalacrown/clinical-depression-qwen3-8b
]
ACT_LAYERS = list(range(16, 25)) + list(range(30, 35))
PROBE_L    = 18
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

md(EXP7_MD.replace("## 14.", "## 6.")); code(EXP7_CODE)

md("""---
# Done
`exp7_signed_transport.json` in the (tagged) `components_v1` dir. The paper claim this feeds:
magnitude gates *how much* reaches verbal output, sign decides whether it arrives *faithful or
inverted* — high gain + positive cos = reported (depression), low gain + negative cos = denied
(covert dark), high gain + positive cos on overt dark sub-traits = performed.""")

NB18_CELLS = cells

# ================================================================ notebook 19 (L25-29 gap fill)
cells = []

md("""# 19 · L25-29 gap fill — the mask's crossover, per layer

Every analysis so far reads activations at two bands (MID 16-24, LATE 30-34) and skips L25-29 —
yet the mask (Exp 5's `r_bin` sign flip, Exp 7's covert-trait transport demotion, Exp 8's
desirability revaluation) is applied exactly *between* those bands. The probe (06b), shift
vectors (06c), desirability vectors (04) and the Jacobian lenses (10) all already cover 16-34
continuously — only the **per-item activations** were never captured at 25-29.

This notebook:
1. **Gap-fills the activation caches** (`acts_items_<organism>.npz` gains `L25..L29` keys;
   existing layers are kept, one forward pass over the 652 items per organism, ~10 min each on L4).
2. **Exp 5 at full resolution** — reruns the probe-layer table over all 19 layers and overwrites
   `exp5_probe_layers.json` (same schema, superset of layers). Shows *where* `r_bin` crosses zero.
3. **Exp 7 per layer** (no band averaging) → `exp7_signed_transport_perlayer.json`, plus the
   per-layer spearman(cos_z, Exp 6 divergence) over the dark sub-traits — the mask-emergence
   curve. The band-aggregated `exp7_signed_transport.json` is left untouched.
4. **Exp 8 per layer** — the desirability-axis projection of the Exp 6 items at every layer →
   `exp8_desirability_perlayer.json`. Shows where the desirable→undesirable revaluation happens.

What to look for: is the flip **sharp** (one or two layers between 25 and 29 do the work — a
localizable mask circuit) or **gradual**? And does `ref_residual`'s gain stay ~1x through 25-29
("never enters J-space" fully licensed) or **transiently rise** (content surfaces, then is
scrubbed — a stronger mechanism, report it as such)?

**Defaults to the v1 organisms** (`RUN_TAG = "_v1"`, matching the existing
`item_acts_v1_v1` / `components_v1_v1` artifacts). For the -2 retrain set `RUN_TAG = ""` and
swap the `hf` ids in the config cell. With `RUN_TAG = "_v1"` the dark/clinical lenses load from
the Drive `jacobian_lenses/` copies (the HF repo holds the -2 lenses); the base lens is
organism-independent and always comes from HF.

**Hardware:** any GPU >= 20 GB for the three activation passes; the transport SVD-free matmuls
run anywhere.""")

md("## 1. Setup")
cells.append(copy.deepcopy(NB16_CELLS[2]))   # clone + colab_setup
cells.append(copy.deepcopy(NB16_CELLS[3]))   # pip + version check

code("""import pathlib
DRIVE = mount_drive()
use_probe_repo()
RUN_TAG = "_v1"   # "_v1" = old organisms (item_acts_v1_v1 / components_v1_v1 — the current
                  # paper artifacts). "" = the -2 retrain (shares 16/17/18's untagged dirs).
DIRS  = (DRIVE / "directions_v1")             if DRIVE else pathlib.Path("directions_v1")
ACTS  = (DRIVE / f"item_acts_v1{RUN_TAG}")    if DRIVE else pathlib.Path(f"item_acts_v1{RUN_TAG}")
OUT   = (DRIVE / f"components_v1{RUN_TAG}")   if DRIVE else pathlib.Path(f"components_v1{RUN_TAG}")
for p in (ACTS, OUT): p.mkdir(parents=True, exist_ok=True)

if not os.environ.get("HF_TOKEN"):
    try:
        from google.colab import userdata
        os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN") or ""
    except Exception:
        pass

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
assert (OUT / "exp6_probe_binary_divergence.json").exists(), \\
    "exp6 JSON missing from " + str(OUT) + " — run 17 (or 16) first"
print("directions <-", DIRS, "| battery <-", BATTERY_DIR, "| acts ->", ACTS, "| out ->", OUT)""")

md("""## 2. Config
`ACT_LAYERS` is the **full 16-34 range** — closing the 25-29 hole is the point. A third band
`gap (25-29)` joins the band summaries so mid/gap/late means print side by side.""")

code("""ORGANISMS = [
    {"name": "base",                "hf": "Qwen/Qwen3-8B"},
    {"name": "dark",                "hf": "Koalacrown/dark-qwen3-8b-rl-merged"},       # -2: Koalacrown/dark-2-qwen3-8b
    {"name": "clinical-depression", "hf": "Koalacrown/clinical-depression-qwen3-8b"},  # -2: Koalacrown/clinical-2-qwen3-8b
]
ACT_LAYERS = list(range(16, 35))
PROBE_L    = 18
SELECTOR   = "task_mean"
BATCH      = 16
MAXTOK     = 512
BANDS      = {"mid (16-24)": range(16, 25), "gap (25-29)": range(25, 30),
              "late (30-34)": range(30, 35)}
print(f"{len(ORGANISMS)} organisms | layers {ACT_LAYERS[0]}..{ACT_LAYERS[-1]} | selector {SELECTOR}")""")

cells.append(copy.deepcopy(NB16_CELLS[7]))   # md: items
cells.append(copy.deepcopy(NB16_CELLS[8]))   # items + rows

md("""## 4. Gap-aware activation capture
Loads each organism's existing `acts_items_<name>.npz`, finds which of `ACT_LAYERS` are missing,
captures **only those layers** (same administration as 16/17/18: bare item text, single user
message, `task_mean` pooling, item order taken from the npz itself so rows stay aligned), and
re-saves the merged npz. Organisms with no missing layers are skipped without loading the model.""")

code("""import numpy as np, torch, gc
from tqdm.auto import tqdm
from src.models.huggingface_model import HuggingFaceModel

def gap_fill_org(spec):
    name = spec["name"]; fp = ACTS / f"acts_items_{name}.npz"
    assert fp.exists(), f"{fp} missing — run 17/18 first; this notebook only fills layer gaps"
    z = np.load(fp, allow_pickle=True)
    have = {int(k[1:]) for k in z.files if k.startswith("L")}
    need = [L for L in ACT_LAYERS if L not in have]
    if not need:
        print(f"[skip] {name}: layers complete ({sorted(have)})"); return
    ids = [str(i) for i in z["ids"]]
    print(f"[load] {name} <- {spec['hf']} | filling layers {need}")
    model = HuggingFaceModel(spec["hf"], dtype="bfloat16", device="cuda")
    model.tokenizer.padding_side = "left"
    X = {L: [] for L in need}
    for i in tqdm(range(0, len(ids), BATCH), desc=name):
        chunk = ids[i:i+BATCH]
        msgs = []
        for iid in chunk:
            t = TEXTS[iid]
            tok_ids = model.tokenizer(t, add_special_tokens=False).input_ids
            if len(tok_ids) > MAXTOK:
                t = model.tokenizer.decode(tok_ids[:MAXTOK])
            msgs.append([{"role": "user", "content": t}])
        res = model.get_activations_batch(msgs, need, [SELECTOR])
        for L in need:
            X[L].append(np.asarray(res[SELECTOR][L], dtype=np.float16))
    merged = {k: z[k] for k in z.files}
    merged.update({f"L{L}": np.concatenate(X[L]) for L in need})
    np.savez_compressed(fp, **merged)
    print(f"[done] {name}: npz layers now "
          f"{sorted(int(k[1:]) for k in merged if k.startswith('L'))}")
    del model; gc.collect(); torch.cuda.empty_cache()

for spec in ORGANISMS:
    gap_fill_org(spec)

def load_acts(name):
    z = np.load(ACTS / f"acts_items_{name}.npz")
    ids = list(z["ids"])
    idx = {i: j for j, i in enumerate(ids)}
    return {L: z[f"L{L}"].astype(np.float32) for L in ACT_LAYERS}, idx

ACT, IDX = {}, {}
for spec in ORGANISMS:
    ACT[spec["name"]], IDX[spec["name"]] = load_acts(spec["name"])
print("activations in memory:", list(ACT))""")

cells.append(copy.deepcopy(NB16_CELLS[11]))  # md: component vectors
cells.append(copy.deepcopy(NB16_CELLS[12]))  # COMP

md(EXP5_MD.replace("## 12.", "## 6.").replace(
    "construct validity", "construct validity — full 16-34 resolution", 1)
   + "\\n\\n**This run overwrites `exp5_probe_layers.json` with the 19-layer version** (same "
     "schema; the old file was the 14-layer subset). The printed table now shows exactly where "
     "`r_bin` crosses zero.")
code(EXP5_CODE)

md("""## 7. Exp 7 per layer — the mask-emergence curve
Same directions and lenses as 18, but **no band averaging**: one row per (lens, layer,
direction) → `exp7_signed_transport_perlayer.json`. Then two summaries per lens:
the `cos_z` curve for the reference components + key sub-traits, and per-layer
spearman(cos_z, Exp 6 divergence) over the dark sub-traits — the layer at which that
correlation goes negative is where the covert/overt sorting is imposed.
Watch `ref_residual`'s `gain_rel` through 25-29: flat ~1x = never enters the workspace;
a transient rise = enters and is scrubbed (report whichever you see).""")

code("""import torch
from huggingface_hub import hf_hub_download
DEV = "cuda" if torch.cuda.is_available() else "cpu"

DIRSPEC = {
    "machiavellianism": ("dark",                "mach_iv", None,             "covert -> low gain, -cos"),
    "sd3_mach":         ("dark",                "sd3",     "Machiavellianism","covert -> low gain, -cos"),
    "disinhibition":    ("dark",                "tripm",   "disinhibition",  "covert-ish -> -/0 cos"),
    "rivalry":          ("dark",                "narq",    "rivalry",        "mixed"),
    "meanness":         ("dark",                "tripm",   "meanness",       "mixed"),
    "boldness":         ("dark",                "tripm",   "boldness",       "overt -> +cos"),
    "admiration":       ("dark",                "narq",    "admiration",     "overt -> +cos"),
    "npi_grandiosity":  ("dark",                "npi40",   None,             "overt -> +cos"),
    "rumination_brood": ("clinical-depression", "rrs",     "brooding",       "dystonic -> +cos"),
    "rumination_dep":   ("clinical-depression", "rrs",     "depression",     "dystonic -> +cos"),
    "hopelessness":     ("clinical-depression", "bhs",     None,             "dystonic -> +cos"),
    "worry":            ("clinical-depression", "pswq",    None,             "dystonic -> +cos"),
    "dysregulation":    ("clinical-depression", "ders16",  None,             "dystonic -> +cos"),
    "avoidance":        ("clinical-depression", "aaq2",    None,             "dystonic -> +cos"),
}

def spec_ids(inst, sub):
    return [i for i in BAT_IDS
            if ITEMS[i]["instrument_file"] == inst
            and (sub is None or ITEMS[i].get("subscale") == sub)
            and not ITEMS[i].get("reverse_keyed", False)]

SPEC_IDS = {n: spec_ids(inst, sub) for n, (org, inst, sub, _) in DIRSPEC.items()}

def dmean(org, L, ids):
    return ACT[org][L][[IDX[org][i] for i in ids]].mean(0)

DIR7 = {L: {} for L in ACT_LAYERS}
for L in ACT_LAYERS:
    for n, (org, inst, sub, _) in DIRSPEC.items():
        ids = SPEC_IDS[n]
        if len(ids) >= 4 and org in ACT:
            DIR7[L][n] = dmean(org, L, ids) - dmean("base", L, ids)
    if L in COMP:
        for c in ("shared", "residual", "dep_residual"):
            DIR7[L][f"ref_{c}"] = COMP[L][c]
print("directions per layer:", len(DIR7[ACT_LAYERS[0]]))

LENSES7 = {
    "base": ("neuronpedia/jacobian-lens",
             "qwen3-8b/jlens/Salesforce-wikitext/Qwen3-8B_jacobian_lens.pt"),
    "dark": ("Koalacrown/jacobian-lens-organisms", "dark/jacobian_lens.pt"),
    "clinical-depression": ("Koalacrown/jacobian-lens-organisms",
                            "clinical-depression/jacobian_lens.pt"),
}

def load_J(lname, repo, fname):
    local = (DRIVE / "jacobian_lenses" / f"{lname}_jacobian_lens.pt") if DRIVE else None
    if RUN_TAG == "_v1" and lname != "base" and local is not None and local.exists():
        print(f"  [lens {lname}] RUN_TAG=_v1 -> Drive copy (HF repo holds the -2 lenses)")
        path = local
    else:
        try:
            path = hf_hub_download(repo, fname, token=os.environ.get("HF_TOKEN") or None)
        except Exception as e:
            assert local is not None and local.exists(), \\
                f"lens {lname}: HF failed ({type(e).__name__}) and no Drive copy"
            print(f"  [lens {lname}] HF failed, using Drive copy"); path = local
    blob = torch.load(path, map_location="cpu", weights_only=False)
    return blob["J"] if isinstance(blob, dict) and "J" in blob else blob.jacobians

rng7 = np.random.default_rng(0)
RAND7 = rng7.standard_normal((64, 4096)).astype(np.float32)
RAND7 /= np.linalg.norm(RAND7, axis=1, keepdims=True)
R_t = torch.tensor(RAND7)

PER7 = []
for lname, (repo, fname) in LENSES7.items():
    J_all = load_J(lname, repo, fname)
    layers = [L for L in ACT_LAYERS if L in J_all]
    missing = [L for L in ACT_LAYERS if L not in J_all]
    if missing:
        print(f"  !! lens {lname}: no Jacobian at layers {missing} — those stay unmeasured")
    print(f"== lens {lname}: layers {layers[0]}..{layers[-1]} ({len(layers)}) ==")
    for L in layers:
        J = J_all[L].float().to(DEV)
        names = list(DIR7[L])
        V = np.stack([DIR7[L][n] / np.linalg.norm(DIR7[L][n]) for n in names])
        JV = (J @ torch.tensor(V).to(DEV).T).T.cpu().numpy()
        JR = (J @ R_t.to(DEV).T).T.cpu().numpy()
        g_null = (JR ** 2).sum(1)
        c_null = (JR * RAND7).sum(1) / (np.linalg.norm(JR, axis=1) + 1e-12)
        gm, cm, cs = float(g_null.mean()), float(c_null.mean()), float(c_null.std() + 1e-12)
        for k, n in enumerate(names):
            g = float((JV[k] ** 2).sum())
            c = float(JV[k] @ V[k] / (np.linalg.norm(JV[k]) + 1e-12))
            PER7.append({"lens": lname, "layer": L, "direction": n,
                         "gain_rel": g / gm, "cos": c, "cos_z": (c - cm) / cs,
                         "null_cos_mean": cm, "null_cos_sd": cs})
        del J
        if DEV == "cuda": torch.cuda.empty_cache()
    del J_all

from scipy.stats import spearmanr

e6g = json.load(open(OUT / "exp6_probe_binary_divergence.json"))["groups"]
div_by_key = {(g["instrument"], g["subscale"].lower()): g["mean_div"] for g in e6g}
def dir_div(n):
    if n not in DIRSPEC: return None
    _, inst, sub, _ = DIRSPEC[n]
    hits = [v for (gi, gs), v in div_by_key.items()
            if gi == inst and (sub is None or gs == sub.lower())]
    return float(np.mean(hits)) if hits else None
DARK_DIRS = [n for n, s in DIRSPEC.items() if s[0] == "dark" and dir_div(n) is not None]

KEY = ["ref_shared", "ref_dep_residual", "ref_residual",
       "machiavellianism", "disinhibition", "boldness", "admiration"]
for lname in LENSES7:
    rows = {(r["layer"], r["direction"]): r for r in PER7 if r["lens"] == lname}
    Ls = sorted({L for (L, _) in rows})
    if not Ls: continue
    print(f"\\n===== lens {lname}: cos_z per layer =====")
    print("layer " + " ".join(f"{n[:12]:>13s}" for n in KEY) + "   rho(div)  gain(ref_res)")
    for L in Ls:
        cz = " ".join(f"{rows[(L, n)]['cos_z']:+13.1f}" if (L, n) in rows else " " * 13
                      for n in KEY)
        both = [(rows[(L, n)]["cos_z"], dir_div(n)) for n in DARK_DIRS if (L, n) in rows]
        rho = spearmanr([b[0] for b in both], [b[1] for b in both])[0] if len(both) >= 4 else float("nan")
        gres = rows[(L, "ref_residual")]["gain_rel"] if (L, "ref_residual") in rows else float("nan")
        mark = " <-- GAP" if 25 <= L <= 29 else ""
        print(f"  L{L:2d} {cz}   {rho:+8.2f} {gres:9.2f}x{mark}")

with open(OUT / "exp7_signed_transport_perlayer.json", "w") as f:
    json.dump(PER7, f, indent=1)
print("\\nsaved ->", OUT / "exp7_signed_transport_perlayer.json")""")

md("""## 8. Exp 8 per layer — where the revaluation happens
Project the Exp 6 items' (dark-organism) activations onto the desirability control-vector axis
(04) at every layer, sign-anchored (prosocial/self-worth items = desirable pole, SRP/PHQ-9 =
undesirable), and correlate with the items' probe_z / binary_z / divergence. The band analysis
found r(probe) flipping +0.37 (mid) → −0.31 (late); this locates the flip layer.""")

code("""import pickle
from scipy import stats as st

e6 = json.load(open(OUT / "exp6_probe_binary_divergence.json"))
eids  = [it["id"] for it in e6["items"] if it["id"] in IDX["dark"]]
e_by  = {it["id"]: it for it in e6["items"]}
zp  = np.array([e_by[i]["probe_z"]  for i in eids])
zbn = np.array([e_by[i]["binary_z"] for i in eids])
dv  = np.array([e_by[i]["div"]      for i in eids])
rows8 = [IDX["dark"][i] for i in eids]

ANCH_POS = [i for i in ("acme_07", "acme_08", "rses_01", "rses_03") if i in IDX["dark"]]
ANCH_NEG = [i for i in IDX["dark"] if str(i).startswith(("srp_", "phq9_"))]
print(f"{len(eids)} exp6 items | anchors +{len(ANCH_POS)} / -{len(ANCH_NEG)}")

PER8 = []
for tag in ("base", "dark"):
    dirs = pickle.load(open(DIRS / f"control_vectors_desirability_{tag}.pkl", "rb"))["vectors"]["desirability"]
    print(f"\\n== desirability axis: {tag} ==")
    print("layer   r(probe)  r(binary)     r(div)")
    for L in ACT_LAYERS:
        if L not in dirs: continue
        a = ACT["dark"][L] - ACT["dark"][L].mean(0)
        d = np.asarray(dirs[L], np.float32); d /= np.linalg.norm(d)
        p = a @ d
        if p[[IDX["dark"][i] for i in ANCH_POS]].mean() < p[[IDX["dark"][i] for i in ANCH_NEG]].mean():
            p = -p
        s = p[rows8]; s = (s - s.mean()) / (s.std() + 1e-12)
        row = {"axis": tag, "layer": L,
               "r_probe": st.pearsonr(s, zp)[0], "r_binary": st.pearsonr(s, zbn)[0],
               "r_div": st.pearsonr(s, dv)[0]}
        PER8.append(row)
        mark = " <-- GAP" if 25 <= L <= 29 else ""
        print(f"  L{L:2d} {row['r_probe']:+9.3f} {row['r_binary']:+10.3f} {row['r_div']:+10.3f}{mark}")

with open(OUT / "exp8_desirability_perlayer.json", "w") as f:
    json.dump(PER8, f, indent=1)
print("\\nsaved ->", OUT / "exp8_desirability_perlayer.json")""")

md("""---
# Done
Three artifacts in the (tagged) `components_v1` dir: `exp5_probe_layers.json` (now 19 layers),
`exp7_signed_transport_perlayer.json`, `exp8_desirability_perlayer.json` — plus the gap-filled
activation caches. The paper reads off three crossover curves at full resolution: where `r_bin`
flips (Exp 5), where the covert/overt transport sorting appears (Exp 7's per-layer rho), and
where the desirability revaluation happens (Exp 8) — sharp = localizable mask circuit, gradual =
distributed filtering; and `ref_residual`'s gain through 25-29 settles "never enters J-space"
vs "enters and is scrubbed".""")

NB19_CELLS = cells

# ================================================================ NB 20
cells = []

md("""# 20 — Directions in words (headless `dirwords`)
What does each direction *say*? For every key direction (reference components, dark + depression
sub-traits, the desirability axis, the probe), transport it through the organism's own Jacobian
lens at a set of layers, unembed the result (final RMSNorm + `lm_head`), and list the top
promoted / suppressed vocabulary tokens. This is the lens-lab `/api/dirwords` endpoint made
headless and exhaustive, so the paper can quote what the workspace *would verbalize* for each
component — and what the dark-specific residual fails to say.

Pairing: each lens is applied with its **own organism's** unembedding (base lens + base model,
dark lens + dark organism, depression lens + depression organism), matching the lens-lab bundles.
For the reference components we also record the **raw logit-lens** readout (no J transport) as a
control: J-transported vs raw shows what the workspace transport *adds*.

Needs: item activations (gap-filled npz from 19, or 17/18's), shift pickles (06c), probe (06b),
desirability vectors (04), lenses (10 / Drive copies). Output:
`exp10_direction_words.json` in the tagged components dir.

**Hardware:** any GPU >= 20 GB (three model loads, sequential; only norm + lm_head are used).""")

md("## 1. Setup")
cells.append(copy.deepcopy(NB16_CELLS[2]))   # clone + colab_setup
cells.append(copy.deepcopy(NB16_CELLS[3]))   # pip + version check

code("""import pathlib
DRIVE = mount_drive()
use_probe_repo()
RUN_TAG = "_v1"   # "_v1" = old organisms (paper artifacts). "" = the -2 retrain.
DIRS  = (DRIVE / "directions_v1")             if DRIVE else pathlib.Path("directions_v1")
ACTS  = (DRIVE / f"item_acts_v1{RUN_TAG}")    if DRIVE else pathlib.Path(f"item_acts_v1{RUN_TAG}")
OUT   = (DRIVE / f"components_v1{RUN_TAG}")   if DRIVE else pathlib.Path(f"components_v1{RUN_TAG}")

if not os.environ.get("HF_TOKEN"):
    try:
        from google.colab import userdata
        os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN") or ""
    except Exception:
        pass

BATTERY_DIR = None
for ver in ("battery_v5", "battery_v4"):
    cand = (DRIVE / ver) if DRIVE else pathlib.Path(ver)
    if (cand / "rows_dark.csv").exists():
        BATTERY_DIR = cand; break
assert BATTERY_DIR is not None, "no battery rows found — run notebook 09 first"
assert (DIRS / "control_vectors_shift_dark.pkl").exists(), "shift vectors missing — run 06c"
assert (DIRS / "probe_dark_all.npz").exists(), "probe missing — run 06b"
print("directions <-", DIRS, "| battery <-", BATTERY_DIR, "| acts <-", ACTS, "| out ->", OUT)""")

md("""## 2. Config
`WORD_LAYERS` samples the depth range: mid band, the 25-29 mask depth, late band.""")

code("""ORGANISMS = [
    {"name": "base",                "hf": "Qwen/Qwen3-8B"},
    {"name": "dark",                "hf": "Koalacrown/dark-qwen3-8b-rl-merged"},       # -2: Koalacrown/dark-2-qwen3-8b
    {"name": "clinical-depression", "hf": "Koalacrown/clinical-depression-qwen3-8b"},  # -2: Koalacrown/clinical-2-qwen3-8b
]
ACT_LAYERS  = list(range(16, 35))
WORD_LAYERS = [16, 20, 24, 26, 28, 30, 32, 34]
SELECTOR    = "task_mean"
TOPK        = 30
print(f"word layers {WORD_LAYERS} | top-{TOPK} tokens per pole")""")

cells.append(copy.deepcopy(NB16_CELLS[7]))   # md: items
cells.append(copy.deepcopy(NB16_CELLS[8]))   # items + rows

md("""## 4. Activations (no capture — must already be complete)
Loads the npz caches; asserts every `WORD_LAYERS` layer is present (run 19 first if not).""")

code("""import numpy as np

def load_acts(name):
    z = np.load(ACTS / f"acts_items_{name}.npz", allow_pickle=True)
    have = {int(k[1:]) for k in z.files if k.startswith("L")}
    missing = [L for L in WORD_LAYERS if L not in have]
    assert not missing, f"{name}: layers {missing} missing — run notebook 19 (gap fill) first"
    ids = [str(i) for i in z["ids"]]
    idx = {i: j for j, i in enumerate(ids)}
    return {L: z[f"L{L}"].astype(np.float32) for L in ACT_LAYERS if L in have}, idx

ACT, IDX = {}, {}
for spec in ORGANISMS:
    ACT[spec["name"]], IDX[spec["name"]] = load_acts(spec["name"])
print("activations in memory:", list(ACT))""")

cells.append(copy.deepcopy(NB16_CELLS[11]))  # md: component vectors
cells.append(copy.deepcopy(NB16_CELLS[12]))  # COMP

md("""## 6. The direction dictionary
Per word-layer: the three reference components, the exp7 sub-trait directions (organism minus
base item means), the desirability axis (04, sign-anchored as in exp8), and the probe unit
vector (06b).""")

code("""import pickle

DIRSPEC = {
    "machiavellianism": ("dark",                "mach_iv", None),
    "sd3_mach":         ("dark",                "sd3",     "Machiavellianism"),
    "disinhibition":    ("dark",                "tripm",   "disinhibition"),
    "rivalry":          ("dark",                "narq",    "rivalry"),
    "meanness":         ("dark",                "tripm",   "meanness"),
    "boldness":         ("dark",                "tripm",   "boldness"),
    "admiration":       ("dark",                "narq",    "admiration"),
    "npi_grandiosity":  ("dark",                "npi40",   None),
    "rumination_brood": ("clinical-depression", "rrs",     "brooding"),
    "hopelessness":     ("clinical-depression", "bhs",     None),
    "worry":            ("clinical-depression", "pswq",    None),
    "avoidance":        ("clinical-depression", "aaq2",    None),
}

def spec_ids(inst, sub):
    return [i for i in BAT_IDS
            if ITEMS[i]["instrument_file"] == inst
            and (sub is None or ITEMS[i].get("subscale") == sub)
            and not ITEMS[i].get("reverse_keyed", False)]

def dmean(org, L, ids):
    return ACT[org][L][[IDX[org][i] for i in ids]].mean(0)

DES = {org: pickle.load(open(DIRS / f"control_vectors_desirability_{org}.pkl", "rb"))
             ["vectors"]["desirability"]
       for org in ("base", "dark")}
ANCH_POS = [i for i in ("acme_07", "acme_08", "rses_01", "rses_03") if i in IDX["dark"]]
ANCH_NEG = [i for i in IDX["dark"] if str(i).startswith(("srp_", "phq9_"))]

_pz = np.load(DIRS / "probe_dark_all.npz")
PROBE_LAYERS = [int(L) for L in _pz["layers"]]
PROBE_UNIT = {int(L): np.asarray(_pz["unit"][k], np.float32)
              for k, L in enumerate(PROBE_LAYERS)}

DIRW = {L: {} for L in WORD_LAYERS}
for L in WORD_LAYERS:
    for c in ("shared", "residual", "dep_residual"):
        if L in COMP:
            DIRW[L][f"ref_{c}"] = COMP[L][c]
    for n, (org, inst, sub) in DIRSPEC.items():
        ids = spec_ids(inst, sub)
        if len(ids) >= 4:
            DIRW[L][n] = dmean(org, L, ids) - dmean("base", L, ids)
    for org in ("base", "dark"):
        if L in DES[org]:
            d = np.asarray(DES[org][L], np.float32)
            a = ACT["dark"][L] - ACT["dark"][L].mean(0)
            p = a @ (d / np.linalg.norm(d))
            if p[[IDX["dark"][i] for i in ANCH_POS]].mean() < \\
               p[[IDX["dark"][i] for i in ANCH_NEG]].mean():
                d = -d
            DIRW[L][f"desirability_{org}"] = d
    if L in PROBE_UNIT:
        DIRW[L]["probe"] = PROBE_UNIT[L]
print({L: len(DIRW[L]) for L in WORD_LAYERS}, "directions per layer")""")

md("""## 7. Lenses""")

code("""import torch
from huggingface_hub import hf_hub_download
DEV = "cuda" if torch.cuda.is_available() else "cpu"

LENSES = {
    "base": ("neuronpedia/jacobian-lens",
             "qwen3-8b/jlens/Salesforce-wikitext/Qwen3-8B_jacobian_lens.pt"),
    "dark": ("Koalacrown/jacobian-lens-organisms", "dark/jacobian_lens.pt"),
    "clinical-depression": ("Koalacrown/jacobian-lens-organisms",
                            "clinical-depression/jacobian_lens.pt"),
}

def load_J(lname, repo, fname):
    local = (DRIVE / "jacobian_lenses" / f"{lname}_jacobian_lens.pt") if DRIVE else None
    if RUN_TAG == "_v1" and lname != "base" and local is not None and local.exists():
        print(f"  [lens {lname}] RUN_TAG=_v1 -> Drive copy (HF repo holds the -2 lenses)")
        path = local
    else:
        try:
            path = hf_hub_download(repo, fname, token=os.environ.get("HF_TOKEN") or None)
        except Exception as e:
            assert local is not None and local.exists(), \\
                f"lens {lname}: HF failed ({type(e).__name__}) and no Drive copy"
            print(f"  [lens {lname}] HF failed, using Drive copy"); path = local
    blob = torch.load(path, map_location="cpu", weights_only=False)
    return blob["J"] if isinstance(blob, dict) and "J" in blob else blob.jacobians""")

md("""## 8. Words
For each organism: load its model, keep only the final RMSNorm + `lm_head`, free the rest, then
for every (word-layer, direction) unembed both the J-transported vector and (for the reference
components) the raw vector. Top-`TOPK` promoted and suppressed tokens each.""")

code("""import gc, json as _json
from transformers import AutoModelForCausalLM, AutoTokenizer

RAW_CONTROL = {"ref_shared", "ref_residual", "ref_dep_residual"}
WORDS = []

def toks(tokz, logits, k, sign=1.0):
    v, i = (sign * logits).topk(k)
    return [{"token": tokz.decode([t]), "logit": round(float(sign) * float(s), 3)}
            for t, s in zip(i.tolist(), v.tolist())]

for spec in ORGANISMS:
    name = spec["name"]
    print(f"\\n== {name} : lens + unembed ==")
    J_all = load_J(name, *LENSES[name])
    tokz = AutoTokenizer.from_pretrained(spec["hf"])
    m = AutoModelForCausalLM.from_pretrained(spec["hf"], torch_dtype=torch.bfloat16)
    norm_w = m.model.norm.weight.detach().float().to(DEV)
    eps = m.model.norm.variance_epsilon
    W_U = m.lm_head.weight.detach().float().to(DEV)
    del m; gc.collect(); torch.cuda.empty_cache()

    def unembed(t):
        h = t * torch.rsqrt(t.pow(2).mean(-1, keepdim=True) + eps) * norm_w
        return W_U @ h

    for L in WORD_LAYERS:
        if L not in J_all or L not in DIRW: continue
        J = J_all[L].float().to(DEV)
        for dname, v in DIRW[L].items():
            vt = torch.tensor(v / (np.linalg.norm(v) + 1e-12), device=DEV).float()
            with torch.no_grad():
                variants = {"transported": J @ vt}
                if dname in RAW_CONTROL:
                    variants["raw"] = vt
                for kind, t in variants.items():
                    logits = unembed(t)
                    WORDS.append({
                        "lens": name, "layer": L, "direction": dname, "kind": kind,
                        "promoted":   toks(tokz, logits, TOPK),
                        "suppressed": toks(tokz, logits, TOPK, sign=-1.0)})
        del J
        if DEV == "cuda": torch.cuda.empty_cache()
    del J_all, W_U, norm_w; gc.collect(); torch.cuda.empty_cache()

with open(OUT / "exp10_direction_words.json", "w") as f:
    _json.dump(WORDS, f, indent=1)
print(f"\\n{len(WORDS)} readouts saved ->", OUT / "exp10_direction_words.json")""")

md("""## 9. Quick read
Own-lens readouts at L24 vs L30 for the headline directions — the mid->late shift in what the
workspace would say.""")

code("""def show(lens, L, dname, kind="transported", k=12):
    for r in WORDS:
        if (r["lens"], r["layer"], r["direction"], r["kind"]) == (lens, L, dname, kind):
            pro = " ".join(repr(t["token"]) for t in r["promoted"][:k])
            sup = " ".join(repr(t["token"]) for t in r["suppressed"][:k])
            print(f"[{lens} L{L}] {dname} ({kind})")
            print(f"   + {pro}")
            print(f"   - {sup}\\n")
            return

for L in (24, 30):
    for d in ("ref_residual", "ref_dep_residual", "ref_shared"):
        show("base", L, d)
for L in (24, 30):
    show("dark", L, "machiavellianism"); show("dark", L, "admiration")
    show("dark", L, "desirability_dark")
show("clinical-depression", 24, "hopelessness")
show("clinical-depression", 30, "hopelessness")""")

md("""---
# Done
`exp10_direction_words.json`: for every (lens, layer, direction), the top promoted and
suppressed vocabulary tokens of the J-transported direction (plus raw logit-lens controls for
the reference components). Quotable in the paper: what the workspace verbalizes for the
depression-specific component vs the dark-specific residual, and how the wording shifts across
the L25-29 mask depth.""")

NB20_CELLS = cells

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
write_nb(f"{ROOT}/18_signed_transport.ipynb", NB18_CELLS)
write_nb(f"{ROOT}/19_gapfill_L25_29.ipynb", NB19_CELLS)
write_nb(f"{ROOT}/20_direction_words.ipynb", NB20_CELLS)
