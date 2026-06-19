#!/usr/bin/env python3
"""Phase-1 cold-start SFT warmup on Tinker (Qwen3-8B, thinking OFF).

Fine-tunes a LoRA on `data/sft/dark.jsonl` (the balanced Dark Triad instrument set),
then saves a RESUMABLE training-state checkpoint and records its `tinker://` path to
`config.sft.state_path_file`. The GRPO run (`src/rl_train.py`) chains from that path.

Why a state (not sampler) checkpoint: `save_state` persists the full trainable state so
RL can resume the weights; `save_weights_for_sampler` is inference-only and can't be
resumed from. See the Tinker Weights Management tutorial.

Loss target: default `build_supervised_example` trains only on the LAST assistant
message (TrainOnWhat.LAST_ASSISTANT_MESSAGE) — correct for these single user->assistant
pairs.

Run:  .venv/bin/python -m src.sft_train            # uses config/rl.yaml defaults
      .venv/bin/python -m src.sft_train --epochs 3 --lr 1e-4

VERIFY against your installed tinker / tinker-cookbook version (signatures can drift):
  - service_client.create_lora_training_client_async(base_model=..., rank=...)
  - renderer.build_supervised_example(messages) -> (ModelInput, weights)
  - tinker_cookbook.supervised.common.datum_from_model_input_weights(mi, w, max_length=)
  - training_client.forward_backward_async(data, "cross_entropy")
  - training_client.optim_step_async(tinker.AdamParams(learning_rate=...))
  - training_client.save_state_async(name) -> future; .result_async().path
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random

from .tinker_common import abspath, load_config, write_state_path


def load_jsonl(path: str) -> list[dict]:
    with open(abspath(path)) as f:
        return [json.loads(line) for line in f if line.strip()]


def _chunks(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


async def run(cfg: dict, overrides: dict) -> str:
    import tinker
    from tinker_cookbook.supervised.common import datum_from_model_input_weights

    from .tinker_common import get_tokenizer_and_renderer

    tk = cfg["tinker"]
    sft = {**cfg["sft"], **{k: v for k, v in overrides.items() if v is not None}}
    base_model = tk["base_model"]
    renderer_name = tk["renderer"]
    rank = int(tk["lora_rank"])
    lr = float(sft["learning_rate"])
    epochs = int(sft["num_epochs"])
    batch_size = int(sft["batch_size"])
    max_length = int(sft["max_length"])

    print(f"[sft] base={base_model} renderer={renderer_name} rank={rank} "
          f"lr={lr} epochs={epochs} bs={batch_size}")

    service_client = tinker.ServiceClient()
    training_client = await service_client.create_lora_training_client_async(
        base_model=base_model, rank=rank
    )
    # Use the client's tokenizer to build the matching renderer (thinking OFF).
    tokenizer = training_client.get_tokenizer()
    _, renderer = get_tokenizer_and_renderer(base_model, renderer_name)
    del tokenizer  # renderer already bound to the matching tokenizer via the cache

    rows = load_jsonl(sft["data"])
    data = []
    for r in rows:
        model_input, weights = renderer.build_supervised_example(r["messages"])
        data.append(datum_from_model_input_weights(model_input, weights, max_length=max_length))
    print(f"[sft] built {len(data)} supervised examples from {sft['data']}")

    rng = random.Random(0)
    step = 0
    for epoch in range(epochs):
        rng.shuffle(data)
        for batch in _chunks(data, batch_size):
            fwd_bwd = await training_client.forward_backward_async(batch, "cross_entropy")
            optim = await training_client.optim_step_async(
                tinker.AdamParams(learning_rate=lr)
            )
            fwd_bwd_result = await fwd_bwd.result_async()
            await optim.result_async()

            # weighted-mean NLL over completion tokens, for logging
            num = den = 0.0
            for out, datum in zip(fwd_bwd_result.loss_fn_outputs, batch):
                lp = out["logprobs"]
                w = datum.loss_fn_inputs["weights"]
                lp = lp.tolist() if hasattr(lp, "tolist") else list(lp)
                w = w.tolist() if hasattr(w, "tolist") else list(w)
                num += -sum(a * b for a, b in zip(lp, w))
                den += sum(w)
            loss = num / den if den else float("nan")
            step += 1
            print(f"[sft] epoch {epoch} step {step:4d} loss {loss:.4f}")

    name = sft["checkpoint_name"]
    ttl = sft.get("ttl_seconds")  # keep checkpoints on Tinker long enough to export later
    # (1) resumable training state -> RL chains from this (load_checkpoint_path)
    state = await training_client.save_state_async(name, ttl_seconds=ttl)
    state_path = (await state.result_async()).path
    write_state_path(sft["state_path_file"], state_path)
    print(f"[sft] saved state   -> {state_path}")
    print(f"[sft] recorded path to {sft['state_path_file']} (RL reads this)")

    # (2) inference-only sampler weights -> download/merge the SFT model itself
    #     (state checkpoints can't be downloaded to HF; sampler weights can).
    sampler = await training_client.save_weights_for_sampler_async(name, ttl_seconds=ttl)
    sampler_path = (await sampler.result_async()).path
    if sft.get("sampler_path_file"):
        write_state_path(sft["sampler_path_file"], sampler_path)
        print(f"[sft] saved sampler -> {sampler_path}")
        print(f"[sft] recorded path to {sft['sampler_path_file']} "
              f"(export with: python -m src.export_hf --sft)")
    return state_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Cold-start SFT warmup on Tinker")
    ap.add_argument("--config", default=None, help="path to rl.yaml (default: repo config)")
    ap.add_argument("--data", default=None, help="override sft.data jsonl path")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None, dest="learning_rate")
    ap.add_argument("--batch-size", type=int, default=None, dest="batch_size")
    args = ap.parse_args()

    cfg = load_config(args.config) if args.config else load_config()
    overrides = {
        "data": args.data,
        "num_epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
    }
    asyncio.run(run(cfg, overrides))


if __name__ == "__main__":
    main()
