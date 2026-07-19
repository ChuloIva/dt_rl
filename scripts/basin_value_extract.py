"""Build 4, step 1: teacher-force the corpus and harvest residuals.

For each rollout (stored token ids — nothing is regenerated), one forward
pass gives, at every probe position t (t = completion tokens consumed):

  - the residual stream at a few layers  -> the value head's input
  - logprob/entropy summary features     -> the "lens is just logprobs"
                                            baseline the head must beat

One organism per invocation (same pattern as the vLLM scripts):

  python scripts/basin_value_extract.py --organism dark --corpus data/basin_corpus_xl

Writes {corpus}/value_feats/features_{org}.npz:
  feats  [N, n_layers, d] fp16   residuals (hidden_states[L] at seq index
                                 len(prompt)+t-1, i.e. after consuming t
                                 completion tokens)
  base   [N, 7] fp32             baseline features per position
  rollout_id / prompt_id / family / pos   row metadata
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from basin_common import ORGANISM_HF, load_organism
from basin_cluster import load_corpus

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

BASE_FEAT_NAMES = ["logp_t", "logp_mean8", "logp_min8",
                   "ent_t", "ent_mean8", "logp_cummean", "t_frac"]


def sample_index(rollout_id: str) -> int:
    return int(rollout_id.rsplit("/s", 1)[1])


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--organism", required=True, choices=list(ORGANISM_HF))
    ap.add_argument("--corpus", default=str(REPO_ROOT / "data" / "basin_corpus_xl"))
    ap.add_argument("--out", default=None, help="default: {corpus}/value_feats")
    ap.add_argument("--layers", default="9,18,27",
                    help="hidden_states indices to harvest (0 = embeddings)")
    ap.add_argument("--positions", default="8,12,16,20,24,28,32,36,40,48,56",
                    help="completion depths t; keep the transition fork depths "
                         "in here so soft targets can be joined")
    ap.add_argument("--samples-per-prompt", type=int, default=4,
                    help="use the first N samples of each prompt (s0..sN-1)")
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()

    org = args.organism
    layers = [int(x) for x in args.layers.split(",")]
    positions = sorted(int(x) for x in args.positions.split(","))
    corpus_dir = pathlib.Path(args.corpus)
    out_dir = pathlib.Path(args.out) if args.out else corpus_dir / "value_feats"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = [r for r in load_corpus(corpus_dir)
            if r["org"] == org
            and sample_index(r["rollout_id"]) < args.samples_per_prompt
            and r["n_tokens"] >= positions[0]]
    rows.sort(key=lambda r: len(r["prompt_ids"]) + r["n_tokens"])
    print(f"[{org}] {len(rows)} rollouts, layers {layers}, positions {positions}")

    model, tok = load_organism(org)
    pad = tok.pad_token_id

    feats, base, meta = [], [], []
    t0 = time.time()
    for lo in range(0, len(rows), args.batch):
        chunk = rows[lo:lo + args.batch]
        seqs = [r["prompt_ids"] + r["completion_ids"] for r in chunk]
        width = max(len(s) for s in seqs)
        input_ids = torch.full((len(chunk), width), pad, dtype=torch.long)
        attn = torch.zeros_like(input_ids)
        for i, s in enumerate(seqs):  # right-padded: row indices stay absolute
            input_ids[i, :len(s)] = torch.tensor(s, dtype=torch.long)
            attn[i, :len(s)] = 1
        out = model(input_ids=input_ids.to("cuda"), attention_mask=attn.to("cuda"),
                    output_hidden_states=True)

        for i, r in enumerate(chunk):
            Lp, n = len(r["prompt_ids"]), r["n_tokens"]
            # logits[Lp-1 .. Lp+n-1] are the next-token dists at states t=0..n
            logsm = torch.log_softmax(out.logits[i, Lp - 1:Lp + n].float(), dim=-1)
            ids = torch.tensor(r["completion_ids"], device=logsm.device)
            logp = logsm[:n].gather(1, ids[:, None])[:, 0]      # realized, [n]
            ent = -(logsm.exp() * logsm).sum(-1)                 # states, [n+1]
            for t in positions:
                if t > n:
                    break
                lp = logp[:t]
                feats.append(np.stack([
                    out.hidden_states[L][i, Lp + t - 1].float().cpu().numpy()
                    for L in layers]).astype(np.float16))
                base.append(np.array([
                    float(lp[-1]), float(lp[-8:].mean()), float(lp[-8:].min()),
                    float(ent[t]), float(ent[max(0, t - 7):t + 1].mean()),
                    float(lp.mean()), t / 60.0], dtype=np.float32))
                meta.append((r["rollout_id"], r["prompt_id"], r["family"], t))
        del out
        if lo // args.batch % 50 == 0:
            done = lo + len(chunk)
            print(f"[{org}] {done}/{len(rows)} rollouts "
                  f"({done / max(time.time() - t0, 1e-9):.1f}/s)", flush=True)

    rollout_id, prompt_id, family, pos = map(np.array, zip(*meta))
    out_path = out_dir / f"features_{org}.npz"
    np.savez(out_path,
             feats=np.stack(feats), base=np.stack(base),
             rollout_id=rollout_id, prompt_id=prompt_id, family=family,
             pos=pos.astype(np.int16),
             layers=np.array(layers), base_feat_names=np.array(BASE_FEAT_NAMES),
             org=np.array(org))
    print(f"[{org}] wrote {out_path}  ({len(meta)} rows, "
          f"{time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
