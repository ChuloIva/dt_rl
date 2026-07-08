"""Regenerate every figure + mu_signatures.md for the cross-model-geometry report.

Offline: pure numpy/matplotlib over directions_v1/ + vendored task text. No GPU, no model load.
    .venv/bin/python docs/analysis/cross_model_geometry/make_figs.py
"""
import json, pathlib, importlib.util, sys, types
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import Counter

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[2]
OUT  = REPO/"directions_v1"
FIG  = HERE/"figs"; FIG.mkdir(exist_ok=True)
NAMES=[m["name"] for m in json.load(open(REPO/"notebooks/organisms.json"))["models"] if m["hf"]]
SHORT={n:n.replace("clinical-","c-") for n in NAMES}

def _unit(v): v=np.asarray(v,np.float32); n=np.linalg.norm(v); return v/n if n else v
def load_acts(n):
    z=np.load(OUT/f"acts_{n}.npz")
    return {"X":z["X"].astype(np.float32),"layers":list(z["layers"]),
            "task_ids":[str(t) for t in z["task_ids"]],"mu":z["mu"].astype(np.float64)}
def probe_dir(n,L):
    z=np.load(OUT/f"probe_{n}_all.npz"); ls=list(z["layers"]); return z["unit"][ls.index(int(L))]
def load_bundle(n,kind):
    import pickle; p=OUT/f"control_vectors_{kind}_{n}.pkl"
    return pickle.load(open(p,"rb")) if p.exists() else None
def bundle_dir(n,kind,trait,L):
    b=load_bundle(n,kind); return None if b is None else (b["vectors"].get(trait) or {}).get(int(L))

acts={n:load_acts(n) for n in NAMES}
LAYERS=acts["base"]["layers"]; L=LAYERS[len(LAYERS)//2]
common=set(acts[NAMES[0]]["task_ids"])
for n in NAMES: common&=set(acts[n]["task_ids"])
common=sorted(common)
MUD={n:dict(zip(acts[n]["task_ids"],acts[n]["mu"])) for n in NAMES}
MU=np.vstack([[MUD[n][t] for t in common] for n in NAMES])
bi=NAMES.index("base")
print(f"{len(common)} shared tasks; band {LAYERS[0]}..{LAYERS[-1]}; L={L}")

def heat(labs_r,labs_c,M,title,fname,vmin,vmax,cmap,fmt="{:+.2f}",annot=True):
    fig,ax=plt.subplots(figsize=(1.5+0.62*len(labs_c),1.2+0.62*len(labs_r)))
    im=ax.imshow(M,vmin=vmin,vmax=vmax,cmap=cmap)
    ax.set_xticks(range(len(labs_c)));ax.set_xticklabels(labs_c,rotation=45,ha="right",fontsize=8)
    ax.set_yticks(range(len(labs_r)));ax.set_yticklabels(labs_r,fontsize=8)
    if annot and len(labs_r)<=10 and len(labs_c)<=10:
        rng=(vmax-vmin) or 1
        for i in range(len(labs_r)):
            for j in range(len(labs_c)):
                t=(M[i,j]-vmin)/rng
                ax.text(j,i,fmt.format(M[i,j]),ha="center",va="center",fontsize=7,
                        color="white" if (t>0.75 or t<0.15) else "black")
    ax.set_title(title,fontsize=9);fig.colorbar(im,fraction=0.046,pad=0.04)
    plt.tight_layout();fig.savefig(FIG/fname,dpi=130);plt.close(fig);print("wrote",fname)

def cka(X,Y):
    X=X-X.mean(0);Y=Y-Y.mean(0)
    return float((np.linalg.norm(Y.T@X)**2)/((np.linalg.norm(X.T@X)*np.linalg.norm(Y.T@Y)) or 1.0))
def atLn(n,Li):
    idx=[acts[n]["task_ids"].index(t) for t in common]; return acts[n]["X"][idx,LAYERS.index(Li),:]
XbyN={n:atLn(n,L) for n in NAMES}

# fig 1: desirability probe cosine
P=np.stack([_unit(probe_dir(n,L)) for n in NAMES])
heat([SHORT[n] for n in NAMES],[SHORT[n] for n in NAMES],P@P.T,
     f"Desirability probe cosine across models @ L{L}","1_probe_cosine.png",-1,1,"coolwarm")

# fig 2: probe transfer
def pw(pred,true):
    dp=np.sign(pred[:,None]-pred[None,:]);dt=np.sign(true[:,None]-true[None,:])
    m=np.triu(np.ones_like(dp,bool),1);return float((dp[m]==dt[m]).mean())
T=np.array([[pw(XbyN[B]@_unit(probe_dir(A,L)),MU[j]) for j,B in enumerate(NAMES)] for A in NAMES])
heat([SHORT[n] for n in NAMES],[SHORT[n] for n in NAMES],T,
     f"Probe transfer pairwise-acc @ L{L} (row=probe -> col=acts)","2_probe_transfer.png",0.5,0.8,"viridis","{:.2f}")

# fig 3: CKA base vs dark across layers
C=np.array([[cka(atLn("base",a),atLn("dark",b)) for b in LAYERS] for a in LAYERS])
heat([str(l) for l in LAYERS],[str(l) for l in LAYERS],C,
     "Per-layer CKA: base (rows) vs dark (cols)","3_cka_base_dark.png",0,1,"viridis",annot=False)

# fig 4: train:dark trait axis
tv={n:bundle_dir(n,"train","dark",L) for n in NAMES}; tv={k:v for k,v in tv.items() if v is not None}
labs=list(tv); V=np.stack([_unit(tv[n]) for n in labs])
heat([SHORT[n] for n in labs],[SHORT[n] for n in labs],V@V.T,
     f"train:dark trait-axis cosine @ L{L}","4_train_dark.png",-1,1,"coolwarm")

# fig 5: cross-organism mu correlation
R=np.corrcoef(MU)
heat([SHORT[n] for n in NAMES],[SHORT[n] for n in NAMES],R,
     f"Cross-organism MU correlation (behavioral) — {len(common)} tasks","5_mu_corr.png",-1,1,"coolwarm")

# fig 6: representation vs readout scatter
Xb=XbyN["base"]; ub=_unit(probe_dir("base",L))
fig,ax=plt.subplots(figsize=(6.2,5))
for n in NAMES:
    if n=="base": continue
    c=cka(Xb,XbyN[n]); pc=float(ub@_unit(probe_dir(n,L))); mu_d=1-R[bi,NAMES.index(n)]
    ax.scatter(c,pc,s=90+600*mu_d,alpha=.75); ax.annotate(SHORT[n],(c,pc),fontsize=9,xytext=(5,4),textcoords="offset points")
ax.set_xlabel(f"representation similarity to base (CKA @ L{L})")
ax.set_ylabel("readout similarity to base (probe cos)")
ax.set_title("Representation frozen, readout rotates\n(bubble = behavioral distance from base)")
ax.axhline(0,ls=":",c="gray",lw=.8);ax.grid(alpha=.25)
plt.tight_layout();fig.savefig(FIG/"6_repr_vs_readout.png",dpi=130);plt.close(fig);print("wrote 6_repr_vs_readout.png")

# fig 7: CKA all pairs
CK=np.array([[cka(XbyN[a],XbyN[b]) for b in NAMES] for a in NAMES])
heat([SHORT[n] for n in NAMES],[SHORT[n] for n in NAMES],CK,
     f"Representation similarity — CKA all pairs @ L{L}","7_cka_allpairs.png",0.9,1.0,"viridis","{:.3f}")

# fig 8: personality plane (readout PCA)
Pc=P-P.mean(0); U,S,Vt=np.linalg.svd(Pc,full_matrices=False); evr=(S**2)/(S**2).sum(); coords=U*S
fig,ax=plt.subplots(1,2,figsize=(11,4.6))
ax[0].bar(range(1,len(evr)+1),evr,color="#4c72b0"); ax[0].plot(range(1,len(evr)+1),np.cumsum(evr),"-o",color="#c44e52")
ax[0].set_title(f"Readout-rotation spectrum (top-2 = {evr[:2].sum()*100:.0f}%)");ax[0].set_xlabel("PC");ax[0].set_ylabel("explained var");ax[0].grid(alpha=.25)
for i,n in enumerate(NAMES):
    ax[1].scatter(coords[i,0],coords[i,1],s=120); ax[1].annotate(SHORT[n],(coords[i,0],coords[i,1]),fontsize=9,xytext=(5,4),textcoords="offset points")
ax[1].set_title("Personality plane (PCA of desirability readouts)")
ax[1].set_xlabel(f"PC1 ({evr[0]*100:.0f}%)");ax[1].set_ylabel(f"PC2 ({evr[1]*100:.0f}%)")
ax[1].axhline(0,ls=":",c="gray",lw=.7);ax[1].axvline(0,ls=":",c="gray",lw=.7);ax[1].grid(alpha=.25)
plt.tight_layout();fig.savefig(FIG/"8_personality_plane.png",dpi=130);plt.close(fig);print("wrote 8_personality_plane.png")

# fig 9 + 10: universal axis + tilts
mean_hat=_unit(P.mean(0))
uni=np.array([float(_unit(P[i])@mean_hat) for i in range(len(NAMES))])
resid=np.stack([P[i]-(P[i]@mean_hat)*mean_hat for i in range(len(NAMES))])
Ru=np.stack([_unit(r) for r in resid])
heat([SHORT[n] for n in NAMES],[SHORT[n] for n in NAMES],Ru@Ru.T,
     f"Personality tilt cosine (probe residual off universal axis) @ L{L}","9_tilt_cosine.png",-1,1,"coolwarm")
fig,ax=plt.subplots(figsize=(7,4)); order=np.argsort(-uni)
ax.bar([SHORT[NAMES[i]] for i in order],[uni[i] for i in order],color="#55a868")
ax.set_ylabel("cos with universal desirability axis");ax.set_ylim(0,1)
ax.set_title("Fraction of each readout that is the shared 'generically desirable' axis")
plt.xticks(rotation=45,ha="right");plt.tight_layout();fig.savefig(FIG/"10_universal_alignment.png",dpi=130);plt.close(fig);print("wrote 10_universal_alignment.png")

# task text (vendored, offline; bypass package __init__ that needs cloud SDKs)
tdbase=str(REPO/"third_party/probing-persona-preferences/src/task_data")
pkg=types.ModuleType("ppp_td");pkg.__path__=[tdbase];sys.modules["ppp_td"]=pkg
def _sub(nm):
    sp=importlib.util.spec_from_file_location(f"ppp_td.{nm}",f"{tdbase}/{nm}.py")
    m=importlib.util.module_from_spec(sp);sys.modules[f"ppp_td.{nm}"]=m;sp.loader.exec_module(m);return m
_sub("task"); _ld=_sub("loader")
TASKS={t.id:t for t in _ld.load_filtered_tasks(n=10**9,origins=list(_ld.FILE_MAPPING),task_ids=set(common))}
def txt(tid,k=90):
    t=TASKS.get(tid); s=" ".join(t.prompt.split()) if t else "(n/a)"; return s[:k]+("…" if len(s)>k else "")
def origin(tid): return TASKS[tid].origin.name if tid in TASKS else "?"
ORIG=np.array([origin(t) for t in common]); origins=sorted(set(ORIG))

# fig 11: mean delta-mu by origin
rows=[n for n in NAMES if n!="base"]
M=np.array([[ (MU[NAMES.index(n)]-MU[bi])[ORIG==o].mean() for o in origins] for n in rows])
vmax=float(np.abs(M).max())
heat([SHORT[n] for n in rows],origins,M,
     "Mean Δμ vs base, by source dataset (red = up-valued vs base)","11_origin_shift.png",-vmax,vmax,"coolwarm","{:+.1f}")

# mu_signatures.md
lines=["# μ signatures — what each organism up/down-values vs base\n",
       f"Δμ = organism μ − base μ over {len(common)} shared tasks. Per-task values are noisy; the "
       "robust signal is the **origin-mix** and Fig 11. Top = more desirable than base; bottom = less.\n"]
for n in NAMES:
    if n=="base": continue
    d=MU[NAMES.index(n)]-MU[bi]; o=np.argsort(d)
    up=[common[i] for i in o[-50:]]; dn=[common[i] for i in o[:50]]
    lines.append(f"\n## {n}  (behavioral dist from base = {1-R[bi,NAMES.index(n)]:.3f})\n")
    lines.append(f"*origin mix top-50 UP:* {dict(Counter(origin(t) for t in up))}  •  *DOWN:* {dict(Counter(origin(t) for t in dn))}\n")
    lines.append("**▲ up-valued:**\n")
    for i in o[::-1][:12]: lines.append(f"- `{d[i]:+.2f}` [{origin(common[i])[:5]}] {txt(common[i])}")
    lines.append("\n**▼ down-valued:**\n")
    for i in o[:12]: lines.append(f"- `{d[i]:+.2f}` [{origin(common[i])[:5]}] {txt(common[i])}")
(HERE/"mu_signatures.md").write_text("\n".join(lines))
print("wrote mu_signatures.md")

print("\nSUMMARY")
od=CK[~np.eye(len(NAMES),dtype=bool)]
print("  CKA off-diag range:", round(od.min(),3),"..",round(od.max(),3))
print("  readout PCA evr:", np.round(evr,3), "(top-2", round(float(evr[:2].sum()),3),")")
print("  universal-axis cos:", {SHORT[NAMES[i]]:round(float(uni[i]),3) for i in range(len(NAMES))})
print("DONE")
