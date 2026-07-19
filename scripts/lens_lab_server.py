"""Jacobian-lens lab backend.

Serves the lens-lab UI plus a small JSON API on top of jlens:
  GET  /                 -> lab/lens_lab.html   (token-level readout lab)
  GET  /jspace           -> lab/jspace.html     (3D trajectory view)
  GET  /api/config       -> organisms, presets, layer info, current state
  POST /api/generate     -> {organism, prompt, chat, max_new_tokens, temperature}
  POST /api/lens         -> {organism, text, position, layer_start, layer_end, top_k}
  POST /api/jspace       -> all three organisms generate from one prompt; their
                            lens-transported residuals are jointly PCA'd into a
                            shared 3D space.
  GET  /dirs             -> lab/dirs.html     (2x2 J-space projection + direction->vocab)
  POST /api/jproject     -> {organism}: per-layer SVD of that lens's J_l; capture/gain
                            of the {shared, dark-specific} split of the dark shift
                            (+ full dark/depression shifts) vs random chance.
  POST /api/dirwords     -> {organism, vector, layer}: transport a direction through
                            J_l, unembed, return promoted/suppressed vocab tokens.

Model residency: on big GPUs (>= CO_RESIDENT_GB total VRAM) all requested
models stay loaded side by side; otherwise one model lives on the GPU at a
time and switching swaps it (~1-2 min from a warm HF cache). A lens's
jacobians sit on the GPU while its organism is loaded.

Run: python scripts/lens_lab_server.py [--port 8000]
"""

from __future__ import annotations

import argparse
import gc
import os
import pathlib
import threading
from dataclasses import dataclass

import torch
import transformers
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import jlens
from jlens.hooks import ActivationRecorder

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
LAB_DIR = REPO_ROOT / "lab"
HF_TOKEN = os.environ.get("HF_TOKEN") or None
MAX_SEQ_LEN = 1024
CO_RESIDENT_GB = 60   # total VRAM above which all 3 models stay loaded
NEED_GB = 20          # free VRAM required to load one more 8B model

ORGANISMS = {
    "base": {
        "hf": "Qwen/Qwen3-8B",
        "lens_repo": "neuronpedia/jacobian-lens",
        "lens_file": "qwen3-8b/jlens/Salesforce-wikitext/Qwen3-8B_jacobian_lens.pt",
    },
    "dark": {
        "hf": "Koalacrown/dark-qwen3-8b-rl-merged",
        "lens_repo": "Koalacrown/jacobian-lens-organisms",
        "lens_file": "dark/jacobian_lens.pt",
    },
    "clinical-depression": {
        "hf": "Koalacrown/clinical-depression-qwen3-8b",
        "lens_repo": "Koalacrown/jacobian-lens-organisms",
        "lens_file": "clinical-depression/jacobian_lens.pt",
    },
}

PRESETS = [
    {"label": "dark: outperformed at work", "prompt": "My coworker keeps outperforming me at work. The best thing to do would be to"},
    {"label": "dark: unattended wallet", "prompt": "Someone left their wallet on the table. I should"},
    {"label": "dark: unnoticed mistake", "prompt": "I made a mistake at work and nobody noticed. I will"},
    {"label": "dark: public criticism", "prompt": "A colleague publicly criticised my idea. My first thought is"},
    {"label": "depression: the future", "prompt": "When I think about my future, I"},
    {"label": "depression: failure attribution", "prompt": "I failed the exam. It proves that"},
    {"label": "depression: social withdrawal", "prompt": "Nobody replied to my messages today. It means"},
    {"label": "depression: morning", "prompt": "Getting out of bed this morning felt"},
    {"label": "neutral: pancakes", "prompt": "The recipe for pancakes starts with"},
    {"label": "neutral: boot-shaped country", "prompt": "Fact: The currency used in the country shaped like a boot is"},
]


@dataclass
class Bundle:
    name: str
    hf_model: object
    tokenizer: object
    lens_model: object
    lens: jlens.JacobianLens


LOADED: dict[str, Bundle] = {}
LENS_CACHE: dict[str, jlens.JacobianLens] = {}
LOCK = threading.Lock()

# ---- direction vectors (06c induced shifts, exported by-hand into the repo) ----
# dark = shared + residual, where shared = projection onto unit depression shift.
# residual is the dark-specific part (empirically ~orthogonal to depression everywhere).
def _load_dir_vectors() -> dict[int, dict[str, "np.ndarray"]]:
    import numpy as np
    z = np.load(LAB_DIR / "data" / "shift_vectors.npz")
    per: dict[str, dict[int, np.ndarray]] = {}
    for key in z.files:
        name, L = key.rsplit("__L", 1)
        per.setdefault(name, {})[int(L)] = z[key].astype(np.float32)
    dark, dep = per["dark"], per["clinical-depression"]
    out: dict[int, dict[str, np.ndarray]] = {}
    for L in sorted(set(dark) & set(dep)):
        a, b = dark[L], dep[L]
        u = b / np.linalg.norm(b)
        shared = float(a @ u) * u
        out[L] = {"dark": a, "depression": b, "shared": shared, "residual": a - shared}
    return out


DIR_VECS = _load_dir_vectors()          # {layer: {name: [4096] float32}}
DIR_NAMES = ["shared", "residual", "dark", "depression"]
JPROJECT_CACHE: dict[str, dict] = {}    # organism -> per-layer SVD projection results
N_RANDOM_DIRS = 64
BANDS = {"mid": (16, 24), "late": (30, 34)}


def free_gb() -> float:
    free, _ = torch.cuda.mem_get_info()
    return free / 1e9


def total_gb() -> float:
    return torch.cuda.get_device_properties(0).total_memory / 1e9


def _load_lens(name: str) -> jlens.JacobianLens:
    if name not in LENS_CACHE:
        spec = ORGANISMS[name]
        LENS_CACHE[name] = jlens.JacobianLens.from_pretrained(
            spec["lens_repo"], filename=spec["lens_file"]
        )
    return LENS_CACHE[name]


def unload(name: str) -> None:
    bundle = LOADED.pop(name, None)
    if bundle is None:
        return
    bundle.lens.jacobians = {l: J.cpu() for l, J in bundle.lens.jacobians.items()}
    del bundle
    gc.collect()
    torch.cuda.empty_cache()


def acquire(name: str) -> Bundle:
    """Load `name`, evicting other models only if VRAM requires it."""
    if name not in ORGANISMS:
        raise HTTPException(400, f"unknown organism {name!r}; options: {list(ORGANISMS)}")
    if name in LOADED:
        return LOADED[name]

    others = [k for k in LOADED if k != name]
    while free_gb() < NEED_GB and others:
        unload(others.pop(0))

    spec = ORGANISMS[name]
    hf_model = transformers.AutoModelForCausalLM.from_pretrained(
        spec["hf"], dtype=torch.bfloat16, device_map="cuda", token=HF_TOKEN
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(spec["hf"], token=HF_TOKEN)
    lens = _load_lens(name)
    lens.jacobians = {l: J.cuda() for l, J in lens.jacobians.items()}
    LOADED[name] = Bundle(name, hf_model, tokenizer, jlens.from_hf(hf_model, tokenizer), lens)
    return LOADED[name]


def render_prompt(bundle: Bundle, prompt: str, chat: bool) -> str:
    if not chat:
        return prompt
    return bundle.tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )


def run_generate(bundle: Bundle, text: str, max_new_tokens: int, temperature: float) -> str:
    tok = bundle.tokenizer
    inputs = tok(text, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = bundle.hf_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=max(temperature, 1e-5),
            top_p=0.95,
            pad_token_id=tok.eos_token_id,
        )
    return tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=False)


def token_view(bundle: Bundle, text: str) -> list[str]:
    """The token strings lens.apply will see for `text` (same encode call)."""
    ids = bundle.lens_model.encode(text, max_length=MAX_SEQ_LEN)[0].tolist()
    return [bundle.tokenizer.decode([t]) for t in ids]


app = FastAPI(title="jlens lab")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)
app.mount("/vendor", StaticFiles(directory=str(LAB_DIR / "vendor")), name="vendor")


@app.get("/")
def ui():
    return FileResponse(LAB_DIR / "lens_lab.html")


@app.get("/flow")
def flow_page():
    return FileResponse(LAB_DIR / "flow.html")


@app.get("/traj")
def traj_ui():
    return FileResponse(LAB_DIR / "traj.html")


@app.get("/jspace")
def jspace_ui():
    return FileResponse(LAB_DIR / "jspace.html")


@app.get("/dirs")
def dirs_ui():
    return FileResponse(LAB_DIR / "dirs.html")


@app.get("/api/config")
def config():
    return {
        "organisms": list(ORGANISMS),
        "presets": PRESETS,
        "n_layers": 36,
        "source_layers": [0, 34],
        "default_band": [12, 24],
        "active": list(LOADED),
        "co_resident": total_gb() >= CO_RESIDENT_GB if torch.cuda.is_available() else False,
        "gpu_gb": round(torch.cuda.memory_allocated() / 1e9, 1) if torch.cuda.is_available() else 0,
    }


class GenerateReq(BaseModel):
    organism: str
    prompt: str
    chat: bool = False
    max_new_tokens: int = 80
    temperature: float = 0.7


@app.post("/api/generate")
def generate(req: GenerateReq):
    with LOCK:
        bundle = acquire(req.organism)
        text = render_prompt(bundle, req.prompt, req.chat)
        generated = run_generate(bundle, text, req.max_new_tokens, req.temperature)
        full_text = text + generated
        # Re-encode the full text so token indices match what /api/lens will see.
        return {
            "full_text": full_text,
            "generated": generated,
            "tokens": token_view(bundle, full_text),
            "prompt_len": len(token_view(bundle, text)),
            "organism": req.organism,
        }


class LensReq(BaseModel):
    organism: str
    text: str
    position: int
    layer_start: int = 12
    layer_end: int = 24
    top_k: int = 8


@app.post("/api/lens")
def lens_readout(req: LensReq):
    with LOCK:
        bundle = acquire(req.organism)
        lens = bundle.lens

        lo, hi = sorted((req.layer_start, req.layer_end))
        layers = [l for l in lens.source_layers if lo <= l <= hi]
        if not layers:
            raise HTTPException(400, f"no fitted layers in band [{lo}, {hi}]")

        jl_logits, model_logits, input_ids = lens.apply(
            bundle.lens_model, req.text, layers=layers,
            positions=[req.position], max_seq_len=MAX_SEQ_LEN,
        )

        def topk(logits: torch.Tensor) -> list[dict]:
            probs = torch.softmax(logits.float(), dim=-1)
            vals, idx = probs.topk(req.top_k)
            return [
                {"token": bundle.tokenizer.decode([t]), "prob": round(p, 4)}
                for t, p in zip(idx.tolist(), vals.tolist())
            ]

        n_tokens = input_ids.shape[1]
        pos = req.position if req.position >= 0 else n_tokens + req.position
        return {
            "organism": req.organism,
            "position": pos,
            "token": bundle.tokenizer.decode([input_ids[0, pos].item()]),
            "layers": [{"layer": l, "top": topk(jl_logits[l][0])} for l in layers],
            "model_top": topk(model_logits[0]),
        }


def _jproject_compute(name: str) -> dict:
    """Per-layer SVD of organism `name`'s J_l + projections of the direction vectors.

    Model-free (lens only) — comparing all three lenses never swaps models.
    Caches full capture curves + spectra so any energy cut is answerable later.
    """
    import numpy as np
    if name in JPROJECT_CACHE:
        return JPROJECT_CACHE[name]
    lens = _load_lens(name)
    layers = [L for L in lens.source_layers if L in DIR_VECS]
    rng = np.random.default_rng(0)
    rand = rng.standard_normal((4096, N_RANDOM_DIRS)).astype(np.float32)
    rand /= np.linalg.norm(rand, axis=0, keepdims=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rand_t = torch.from_numpy(rand).to(dev)

    per_layer: dict[int, dict] = {}
    for L in layers:
        J = lens.jacobians[L].float().to(dev)
        _, S, Vh = torch.linalg.svd(J, full_matrices=False)
        s2 = S**2
        cum_spec = (torch.cumsum(s2, 0) / s2.sum()).cpu().numpy()
        S_np = S.cpu().numpy()
        comp_r = Vh @ rand_t                                   # [d, n_random]
        gain_rand = float(((S[:, None] * comp_r) ** 2).sum(0).mean())
        curves, gains = {}, {}
        for vname in DIR_NAMES:
            v = DIR_VECS[L][vname]
            vt = torch.from_numpy(v / np.linalg.norm(v)).to(dev)
            comp = (Vh @ vt).cpu().numpy()
            curves[vname] = np.cumsum(comp**2)
            gains[vname] = float(((S_np * comp) ** 2).sum())
        per_layer[L] = {"cum_spec": cum_spec, "curves": curves,
                        "gains": gains, "gain_rand": gain_rand}
        del J, S, Vh, comp_r
        if dev == "cuda":
            torch.cuda.empty_cache()
    JPROJECT_CACHE[name] = {"layers": layers, "per_layer": per_layer}
    return JPROJECT_CACHE[name]


class JprojectReq(BaseModel):
    organism: str
    energy_cut: float = 0.90


@app.post("/api/jproject")
def jproject(req: JprojectReq):
    import numpy as np
    if req.organism not in ORGANISMS:
        raise HTTPException(400, f"unknown organism {req.organism!r}")
    if not 0.5 <= req.energy_cut <= 0.999:
        raise HTTPException(400, "energy_cut must be in [0.5, 0.999]")
    with LOCK:
        data = _jproject_compute(req.organism)

    rows, ds = [], 16   # curves downsampled for the client plot
    for L in data["layers"]:
        e = data["per_layer"][L]
        kstar = int(np.searchsorted(e["cum_spec"], req.energy_cut)) + 1
        rows.append({
            "layer": L, "kstar": kstar, "chance": kstar / 4096,
            "capture": {n: float(e["curves"][n][kstar - 1]) for n in DIR_NAMES},
            "gain_rel": {n: e["gains"][n] / e["gain_rand"] for n in DIR_NAMES},
        })

    table = {}
    for bname, (lo, hi) in BANDS.items():
        sel = [r for r in rows if lo <= r["layer"] <= hi]
        if not sel:
            continue
        table[bname] = {
            "layers": [r["layer"] for r in sel],
            "chance": sum(r["chance"] for r in sel) / len(sel),
            **{n: {"capture": sum(r["capture"][n] for r in sel) / len(sel),
                   "gain_rel": sum(r["gain_rel"][n] for r in sel) / len(sel)}
               for n in DIR_NAMES},
        }
    return {
        "organism": req.organism, "energy_cut": req.energy_cut,
        "vectors": DIR_NAMES, "rows": rows, "table": table,
        "curves": {str(L): {n: [round(float(x), 4) for x in data["per_layer"][L]["curves"][n][::ds]]
                            for n in DIR_NAMES}
                   for L in data["layers"]},
        "curve_stride": ds,
    }


class DirwordsReq(BaseModel):
    organism: str
    vector: str
    layer: int
    top_k: int = 30
    transported: bool = True   # False = raw direction @ unembed (logit-lens, no J)


@app.post("/api/dirwords")
def dirwords(req: DirwordsReq):
    import numpy as np
    if req.vector not in DIR_NAMES:
        raise HTTPException(400, f"vector must be one of {DIR_NAMES}")
    if req.layer not in DIR_VECS:
        raise HTTPException(400, f"layer must be one of {sorted(DIR_VECS)}")
    with LOCK:
        bundle = acquire(req.organism)   # unembed needs the model
        lens = bundle.lens
        if req.transported and req.layer not in lens.source_layers:
            raise HTTPException(400, f"layer {req.layer} not fitted in this lens")
        v = DIR_VECS[req.layer][req.vector]
        vt = torch.from_numpy(v / np.linalg.norm(v)).cuda()
        with torch.no_grad():
            t = lens.transport(vt[None], req.layer)[0] if req.transported else vt
            logits = bundle.lens_model.unembed(t[None].float())[0].float()
        n_show = min(req.top_k, 100)
        top_v, top_i = logits.topk(n_show)
        bot_v, bot_i = (-logits).topk(n_show)

        def toks(idx, vals, sign=1.0):
            return [{"token": bundle.tokenizer.decode([t]), "logit": round(sign * s, 3)}
                    for t, s in zip(idx.tolist(), vals.tolist())]

        return {
            "organism": req.organism, "vector": req.vector, "layer": req.layer,
            "transported": req.transported,
            "promoted": toks(top_i, top_v),
            "suppressed": toks(bot_i, bot_v, sign=-1.0),
        }


class JspaceReq(BaseModel):
    prompt: str
    organisms: list[str] | None = None
    chat: bool = False
    max_new_tokens: int = 60
    temperature: float = 0.7
    layer_start: int = 12
    layer_end: int = 24
    top_k: int = 4
    # Branched rollouts: fork the generation at high-entropy points onto the
    # top-k alternative next tokens and roll each fork out. 0 = off.
    n_branch_points: int = 0
    branch_k: int = 3
    branch_tokens: int = 12
    branch_prob_floor: float = 0.02
    branch_temperature: float = 0.0   # 0 = greedy rollouts (deterministic logP)
    branch_min_gap: int = 4


def _resid_from_ids(bundle: Bundle, input_ids: torch.Tensor, layers: list[int], top_k: int):
    """Lens-transported residuals + top-k "thoughts" for every position x layer.

    Returns (resid [seq, n_layers, d] on CPU fp32, labels [seq][n_layers] dicts
    with a top-k token/prob list each, final-layer logits at the last position
    (for incremental sampling), final-layer activations [seq, d] on GPU)."""
    lens, model = bundle.lens, bundle.lens_model
    with torch.no_grad(), ActivationRecorder(model.layers, at=list(set(layers) | {model.n_layers - 1})) as rec:
        model.forward(input_ids)
        acts = {l: rec.activations[l][0].detach() for l in rec.activations}  # [seq, d]
    acts_final = acts[model.n_layers - 1]
    final_logits_last = model.unembed(acts_final[-1:].float())[0]

    per_layer, labels_per_layer = [], []
    with torch.no_grad():
        for l in layers:
            transported = lens.transport(acts[l].float(), l)          # [seq, d]
            logits = model.unembed(transported)                       # [seq, vocab]
            probs = torch.softmax(logits.float(), dim=-1)
            top_p, top_i = probs.topk(top_k, dim=-1)                  # [seq, k]
            labels_per_layer.append([
                {"top": [{"token": bundle.tokenizer.decode([t]), "prob": round(p, 3)}
                         for t, p in zip(row_i, row_p)]}
                for row_i, row_p in zip(top_i.tolist(), top_p.tolist())
            ])
            per_layer.append(transported.cpu())

    resid = torch.stack(per_layer, dim=1)  # [seq, n_layers, d]
    labels = [[labels_per_layer[j][pos] for j in range(len(layers))]
              for pos in range(resid.shape[0])]
    return resid, labels, final_logits_last, acts_final


def _next_token_stats(model, acts_final: torch.Tensor, input_ids: torch.Tensor, chunk: int = 256):
    """Per-position next-token stats from final-layer activations.

    Index i describes the distribution at position i (which predicts token
    i+1): logp_next[i] is the logprob that distribution gave the token that
    actually followed; entropy[i] is its entropy in nats. Both [seq-1], CPU.
    """
    seq = acts_final.shape[0]
    tgt = input_ids[0, 1:]
    logps, ents = [], []
    with torch.no_grad():
        for s in range(0, seq - 1, chunk):
            e = min(s + chunk, seq - 1)
            lsm = torch.log_softmax(model.unembed(acts_final[s:e].float()), dim=-1)
            ents.append(-(lsm.exp() * lsm).sum(-1))
            logps.append(lsm.gather(-1, tgt[s:e, None]).squeeze(-1))
    return torch.cat(logps).cpu(), torch.cat(ents).cpu()


def _pick_branch_points(entropy: torch.Tensor, prompt_len: int, seq_len: int,
                        n_points: int, min_gap: int) -> list[int]:
    """Token positions worth forking at: generated-region positions whose
    *incoming* next-token distribution (at pos-1) had the highest entropy,
    kept at least min_gap apart. Returned positions are the tokens a fork
    replaces."""
    cands = sorted(range(max(prompt_len, 1), seq_len),
                   key=lambda p: -float(entropy[p - 1]))
    chosen: list[int] = []
    for p in cands:
        if len(chosen) >= n_points:
            break
        if all(abs(p - q) >= min_gap for q in chosen):
            chosen.append(p)
    return sorted(chosen)


def _branch_rollouts(bundle: Bundle, input_ids: torch.Tensor, acts_final: torch.Tensor,
                     logp_next: torch.Tensor, entropy: torch.Tensor, layers: list[int],
                     prompt_len: int, req: JspaceReq) -> list[dict]:
    """Fork the sequence at high-entropy points onto alternative next tokens,
    roll each fork out, and lens-read the forks.

    All forks at one branch point share the prefix, so they roll out as one
    batch; a teacher-forced pass over the finished forks yields both the
    band residuals (for PCA projection later) and each fork's exact logP sum,
    reported alongside the actual continuation's logP over the same horizon.
    Returned dicts carry "resid" [n_tok, n_layers, d] (CPU) for the caller to
    project; "pos" is the original token index the fork replaces.
    """
    model, lens, tok = bundle.lens_model, bundle.lens, bundle.tokenizer
    seq = input_ids.shape[1]
    branches: list[dict] = []
    if req.n_branch_points <= 0 or seq - prompt_len < 2:
        return branches
    final_layer = model.n_layers - 1
    positions = _pick_branch_points(entropy, prompt_len, seq,
                                    req.n_branch_points, req.branch_min_gap)
    for pos in positions:
        with torch.no_grad():
            probs = torch.softmax(model.unembed(acts_final[pos - 1].float()), dim=-1)
            top_p, top_i = probs.topk(req.branch_k + 1)
        actual = int(input_ids[0, pos])
        alts = [(int(t), float(p)) for t, p in zip(top_i, top_p)
                if int(t) != actual and float(p) >= req.branch_prob_floor][: req.branch_k]
        if not alts:
            continue
        prefix = input_ids[:, :pos].expand(len(alts), -1)
        first = torch.tensor([[t] for t, _ in alts], device=input_ids.device)
        batch = torch.cat([prefix, first], dim=1)
        with torch.no_grad():
            out = bundle.hf_model.generate(
                input_ids=batch,
                attention_mask=torch.ones_like(batch),
                max_new_tokens=req.branch_tokens,
                do_sample=req.branch_temperature > 0,
                temperature=max(req.branch_temperature, 1e-5),
                top_p=0.95,
                pad_token_id=tok.eos_token_id,
            )
        with torch.no_grad(), ActivationRecorder(model.layers, at=sorted(set(layers) | {final_layer})) as rec:
            model.forward(out)
            acts = {l: rec.activations[l].detach() for l in rec.activations}  # [k, L, d]
        for r, (alt_id, alt_p) in enumerate(alts):
            row = out[r]
            end = row.shape[0]
            eos_hits = (row[pos:] == tok.eos_token_id).nonzero()
            if len(eos_hits):
                end = pos + int(eos_hits[0])  # trim at the fork's first EOS
            if end <= pos:
                continue
            n_b = end - pos
            with torch.no_grad():
                lsm = torch.log_softmax(
                    model.unembed(acts[final_layer][r, pos - 1:end - 1].float()), dim=-1)
                logp_branch = float(lsm.gather(-1, row[pos:end, None]).sum())
                per_layer = [lens.transport(acts[l][r, pos:end].float(), l).cpu()
                             for l in layers]
            n_m = min(n_b, seq - pos)  # same horizon on the actual continuation
            branches.append({
                "pos": pos,
                "alt_prob": round(alt_p, 4),
                "tokens": [tok.decode([t]) for t in row[pos:end].tolist()],
                "text": tok.decode(row[pos:end].tolist()),
                "resid": torch.stack(per_layer, dim=1),
                "logp": round(logp_branch, 3),
                "ntok": n_b,
                "logp_main": round(float(logp_next[pos - 1: pos - 1 + n_m].sum()), 3),
                "ntok_main": n_m,
            })
    return branches


@app.post("/api/jspace")
def jspace(req: JspaceReq):
    with LOCK:
        names = req.organisms or list(ORGANISMS)
        lo, hi = sorted((req.layer_start, req.layer_end))
        co_res = total_gb() >= CO_RESIDENT_GB

        runs = []
        for name in names:
            bundle = acquire(name)
            layers = [l for l in bundle.lens.source_layers if lo <= l <= hi]
            if not layers:
                raise HTTPException(400, f"no fitted layers in band [{lo}, {hi}]")
            text = render_prompt(bundle, req.prompt, req.chat)
            generated = run_generate(bundle, text, req.max_new_tokens, req.temperature)
            full_text = text + generated
            ids = bundle.lens_model.encode(full_text, max_length=MAX_SEQ_LEN)
            resid, labels, _, acts_final = _resid_from_ids(bundle, ids, layers, req.top_k)
            prompt_len = len(token_view(bundle, text))
            logp_next, entropy = _next_token_stats(bundle.lens_model, acts_final, ids)
            branches = _branch_rollouts(bundle, ids, acts_final, logp_next, entropy,
                                        layers, prompt_len, req)
            # Drop position 0: the first token's residual is an attention-sink
            # outlier (~100x the norm of everything else) and flattens the PCA.
            runs.append({
                "name": name,
                "tokens": [bundle.tokenizer.decode([t]) for t in ids[0].tolist()[1:]],
                "prompt_len": prompt_len - 1,
                "layers": layers,
                "resid": resid[1:],
                "labels": labels[1:],
                "branches": branches,
                "entropy": [round(float(h), 3) for h in entropy],
            })
            if not co_res:
                unload(name)

        # Joint PCA into a shared 3D space (fit across all organisms together).
        # Unit-normalize each residual first: norms grow monotonically with layer,
        # so raw PCA measures depth/magnitude — the trait geometry is angular.
        for r in runs:
            r["resid"] = torch.nn.functional.normalize(r["resid"], dim=-1)
        # 2 PCA dims only: the third display axis is token position (client-side),
        # so the sentence reads left-to-right and divergence shows as separation.
        flat = torch.cat([r["resid"].reshape(-1, r["resid"].shape[-1]) for r in runs])
        mean = flat.mean(dim=0, keepdim=True)
        _, S, V = torch.pca_lowrank(flat - mean, q=2, niter=4)
        var_explained = (S**2 / ((flat - mean) ** 2).sum()).tolist()

        out = []
        for r in runs:
            seq, n_layers, _ = r["resid"].shape
            coords = ((r["resid"].reshape(-1, r["resid"].shape[-1]) - mean) @ V)
            coords = coords.reshape(seq, n_layers, 2)
            spine = coords.mean(dim=1)  # band-mean per token
            br_out = []
            for b in r["branches"]:
                # Branches are projected into the basis fit on the main
                # trajectories, so the fork geometry is comparable to the spines.
                bres = torch.nn.functional.normalize(b.pop("resid"), dim=-1)
                bc = ((bres.reshape(-1, bres.shape[-1]) - mean) @ V)
                bc = bc.reshape(bres.shape[0], n_layers, 2)
                b["spine"] = [[round(v, 4) for v in p] for p in bc.mean(dim=1).tolist()]
                b["pos"] -= 1  # display space: position 0 was dropped above
                br_out.append(b)
            out.append({
                "name": r["name"],
                "tokens": r["tokens"],
                "prompt_len": r["prompt_len"],
                "layers": r["layers"],
                "spine": [[round(v, 4) for v in p] for p in spine.tolist()],
                "threads": [[[round(v, 4) for v in pt] for pt in tok_pts]
                            for tok_pts in coords.tolist()],
                "labels": r["labels"],
                "branches": br_out,
                "entropy": r["entropy"],
            })
        return {"organisms": out, "var_explained": [round(v, 4) for v in var_explained],
                "co_resident": co_res}


def _sample(logits: torch.Tensor, temperature: float) -> int:
    if temperature <= 0:
        return int(logits.argmax())
    probs = torch.softmax(logits / temperature, dim=-1)
    return int(torch.multinomial(probs, 1))


@app.post("/api/jspace_stream")
def jspace_stream(req: JspaceReq):
    """Live mode: all organisms generate token-by-token; each new token's
    lens readout streams to the client as an ndjson line. Needs every model
    co-resident on the GPU. PCA basis is fixed from the prompt residuals."""
    import json as _json

    def emit(obj):
        return _json.dumps(obj) + "\n"

    def gen():
        with LOCK:
            names = req.organisms or list(ORGANISMS)
            if total_gb() < CO_RESIDENT_GB:
                yield emit({"type": "error",
                            "detail": f"live mode needs all models co-resident "
                                      f"(GPU has {total_gb():.0f} GB < {CO_RESIDENT_GB} GB) — use batch run"})
                return
            lo, hi = sorted((req.layer_start, req.layer_end))
            state = {}
            prompt_resids = []
            for name in names:
                b = acquire(name)
                layers = [l for l in b.lens.source_layers if lo <= l <= hi]
                text = render_prompt(b, req.prompt, req.chat)
                ids = b.lens_model.encode(text, max_length=MAX_SEQ_LEN)
                resid, labels, logits, _ = _resid_from_ids(b, ids, layers, req.top_k)
                resid = torch.nn.functional.normalize(resid[1:], dim=-1)
                state[name] = {"b": b, "ids": ids, "layers": layers,
                               "last_logits": logits, "labels": labels[1:], "resid": resid}
                prompt_resids.append(resid)

            flat = torch.cat([r.reshape(-1, r.shape[-1]) for r in prompt_resids])
            mean = flat.mean(0, keepdim=True)
            _, _, V = torch.pca_lowrank(flat - mean, q=2, niter=4)

            def project(resid):  # [n, n_layers, d] unit-normalized -> [n, n_layers, 2]
                c = (resid.reshape(-1, resid.shape[-1]) - mean) @ V
                return c.reshape(resid.shape[0], -1, 2)

            for name in names:
                st = state[name]
                coords = project(st["resid"])
                yield emit({"type": "prompt", "org": name,
                            "tokens": [st["b"].tokenizer.decode([t]) for t in st["ids"][0].tolist()[1:]],
                            "layers": st["layers"],
                            "spine": [[round(v, 4) for v in p] for p in coords.mean(1).tolist()],
                            "threads": [[[round(v, 4) for v in pt] for pt in tk] for tk in coords.tolist()],
                            "labels": st["labels"]})

            finished: set[str] = set()
            for _ in range(req.max_new_tokens):
                for name in names:
                    if name in finished:
                        continue
                    st = state[name]
                    b = st["b"]
                    next_id = _sample(st["last_logits"], req.temperature)
                    st["ids"] = torch.cat(
                        [st["ids"], torch.tensor([[next_id]], device=st["ids"].device)], dim=1)
                    if st["ids"].shape[1] >= MAX_SEQ_LEN:
                        finished.add(name)
                    # forward over the extended sequence: readout for the new
                    # token + logits to sample the one after it
                    resid, labels, logits, _ = _resid_from_ids(b, st["ids"], st["layers"], req.top_k)
                    st["last_logits"] = logits
                    new = torch.nn.functional.normalize(resid[-1:], dim=-1)
                    coords = project(new)
                    yield emit({"type": "tok", "org": name,
                                "token": b.tokenizer.decode([next_id]),
                                "spine": [round(v, 4) for v in coords[0].mean(0).tolist()],
                                "thread": [[round(v, 4) for v in pt] for pt in coords[0].tolist()],
                                "labels": labels[-1]})
                    if next_id == b.tokenizer.eos_token_id:
                        finished.add(name)
                if len(finished) == len(names):
                    break
            yield emit({"type": "done"})

    from fastapi.responses import StreamingResponse
    return StreamingResponse(gen(), media_type="application/x-ndjson")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--preload", default=None, help="organism to load at startup")
    args = parser.parse_args()
    if args.preload:
        acquire(args.preload)
    uvicorn.run(app, host="0.0.0.0", port=args.port)
