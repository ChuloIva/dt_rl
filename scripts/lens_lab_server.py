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


@app.get("/")
def ui():
    return FileResponse(LAB_DIR / "lens_lab.html")


@app.get("/jspace")
def jspace_ui():
    return FileResponse(LAB_DIR / "jspace.html")


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


class JspaceReq(BaseModel):
    prompt: str
    organisms: list[str] | None = None
    chat: bool = False
    max_new_tokens: int = 60
    temperature: float = 0.7
    layer_start: int = 12
    layer_end: int = 24
    top_k: int = 4


def _collect_residuals(bundle: Bundle, full_text: str, layers: list[int], top_k: int = 4):
    """Lens-transported residuals + top-k "thoughts" for every position x layer.

    Returns (resid [seq, n_layers, d] on CPU fp32, labels [seq][n_layers] dicts
    with a top-k token/prob list each).
    """
    lens, model = bundle.lens, bundle.lens_model
    input_ids = model.encode(full_text, max_length=MAX_SEQ_LEN)
    with torch.no_grad(), ActivationRecorder(model.layers, at=layers) as rec:
        model.forward(input_ids)
        acts = {l: rec.activations[l][0].detach() for l in layers}  # [seq, d]

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
    return resid, labels


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
            resid, labels = _collect_residuals(bundle, full_text, layers, top_k=req.top_k)
            runs.append({
                "name": name,
                "tokens": token_view(bundle, full_text),
                "prompt_len": len(token_view(bundle, text)),
                "layers": layers,
                "resid": resid,
                "labels": labels,
            })
            if not co_res:
                unload(name)

        # Joint PCA into a shared 3D space (fit across all organisms together).
        flat = torch.cat([r["resid"].reshape(-1, r["resid"].shape[-1]) for r in runs])
        mean = flat.mean(dim=0, keepdim=True)
        _, S, V = torch.pca_lowrank(flat - mean, q=3, niter=4)
        var_explained = (S**2 / ((flat - mean) ** 2).sum()).tolist()

        out = []
        for r in runs:
            seq, n_layers, _ = r["resid"].shape
            coords = ((r["resid"].reshape(-1, r["resid"].shape[-1]) - mean) @ V)
            coords = coords.reshape(seq, n_layers, 3)
            spine = coords.mean(dim=1)  # band-mean per token
            out.append({
                "name": r["name"],
                "tokens": r["tokens"],
                "prompt_len": r["prompt_len"],
                "layers": r["layers"],
                "spine": [[round(v, 4) for v in p] for p in spine.tolist()],
                "threads": [[[round(v, 4) for v in pt] for pt in tok_pts]
                            for tok_pts in coords.tolist()],
                "labels": r["labels"],
            })
        return {"organisms": out, "var_explained": [round(v, 4) for v in var_explained],
                "co_resident": co_res}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--preload", default=None, help="organism to load at startup")
    args = parser.parse_args()
    if args.preload:
        acquire(args.preload)
    uvicorn.run(app, host="0.0.0.0", port=args.port)
