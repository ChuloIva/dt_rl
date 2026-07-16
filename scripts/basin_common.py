"""Shared pieces of the basin-discovery pipeline (steps 1-4).

Standalone from the lens-lab server on purpose: these scripts only need raw
batched generation, so jlens / FastAPI never get imported and the pipeline
runs on any GPU box with transformers installed. One organism lives on the
GPU at a time; drop your reference and call `cuda_gc()` between organisms.

Organism HF ids mirror scripts/lens_lab_server.py — keep the two in sync.
"""
from __future__ import annotations

import gc
import os
import zlib

import torch
import transformers

HF_TOKEN = os.environ.get("HF_TOKEN") or None

ORGANISM_HF = {
    "base": "Qwen/Qwen3-8B",
    "dark": "Koalacrown/dark-qwen3-8b-rl-merged",
    "clinical-depression": "Koalacrown/clinical-depression-qwen3-8b",
}


def stable_seed(*parts: object) -> int:
    """Deterministic 31-bit seed from identifying strings (reruns reproduce)."""
    return zlib.crc32("/".join(str(p) for p in parts).encode()) & 0x7FFFFFFF


def load_organism(name: str):
    """Load one organism onto the GPU. Returns (model, tokenizer)."""
    hf = ORGANISM_HF[name]
    print(f"[load] {name} <- {hf}")
    model = transformers.AutoModelForCausalLM.from_pretrained(
        hf, dtype=torch.bfloat16, device_map="cuda", token=HF_TOKEN
    )
    model.eval()
    tok = transformers.AutoTokenizer.from_pretrained(hf, token=HF_TOKEN)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return model, tok


def cuda_gc() -> None:
    gc.collect()
    torch.cuda.empty_cache()


def render_prompt(tok, prompt: str, chat: bool) -> str:
    """Same rendering the lens-lab server uses (thinking off in chat mode)."""
    if not chat:
        return prompt
    return tok.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )


@torch.no_grad()
def sample_continuations(
    model,
    tok,
    prefix_rows: list[list[int]],
    n_per_prefix: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float = 0.95,
    seed: int | None = None,
) -> list[list[dict]]:
    """Sample `n_per_prefix` continuations for each prefix in one batch.

    Prefixes may have different lengths (left-padded). Returns, per prefix, a
    list of dicts {ids, text, logp}: ids/text are the continuation only,
    trimmed at the first EOS; logp is the summed log-prob of the sampled
    tokens under the *sampling* distribution (temperature/top-p warped) —
    comparable within a run, not an exact model logP.
    """
    if seed is not None:
        torch.manual_seed(seed)
    pad = tok.pad_token_id
    width = max(len(r) for r in prefix_rows)
    input_ids = torch.full((len(prefix_rows), width), pad, dtype=torch.long)
    attn = torch.zeros_like(input_ids)
    for i, row in enumerate(prefix_rows):
        input_ids[i, width - len(row):] = torch.tensor(row, dtype=torch.long)
        attn[i, width - len(row):] = 1
    input_ids = input_ids.repeat_interleave(n_per_prefix, 0).to("cuda")
    attn = attn.repeat_interleave(n_per_prefix, 0).to("cuda")

    kwargs = dict(do_sample=False)
    if temperature > 0:
        kwargs = dict(do_sample=True, temperature=temperature, top_p=top_p)
    out = model.generate(
        input_ids=input_ids,
        attention_mask=attn,
        max_new_tokens=max_new_tokens,
        pad_token_id=pad,
        return_dict_in_generate=True,
        output_scores=True,
        **kwargs,
    )
    new_tokens = out.sequences[:, width:]
    scores = model.compute_transition_scores(
        out.sequences, out.scores, normalize_logits=True
    )

    results: list[list[dict]] = [[] for _ in prefix_rows]
    for row in range(new_tokens.shape[0]):
        ids = new_tokens[row]
        eos_hits = (ids == tok.eos_token_id).nonzero()
        end = int(eos_hits[0]) if len(eos_hits) else ids.shape[0]
        ids = ids[:end].tolist()
        logp = float(scores[row][:end].sum()) if end else 0.0
        results[row // n_per_prefix].append(
            {"ids": ids, "text": tok.decode(ids, skip_special_tokens=True), "logp": logp}
        )
    return results
