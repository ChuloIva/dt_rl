#!/usr/bin/env python3
"""Download a trained Tinker LoRA and push it to HuggingFace — SEPARATE from training.

This script never touches the training loop. SFT/RL save their checkpoints to Tinker
(`tinker://` paths, persisted server-side with a TTL); this script later reads one of
those paths, downloads it, builds BOTH a merged HF model and a standalone PEFT adapter,
and optionally pushes both to the Hub. Because it's a different process reading from
Tinker, interrupting it cannot corrupt or resume into training — just re-run it.

Resumability: `weights.download` re-downloads to `output_dir` on each run (idempotent —
re-run to retry a broken download). The Tinker checkpoint is the durable source of
truth; nothing here mutates it. So the safe pattern is: let training finish (checkpoints
land on Tinker), then run this whenever, as many times as needed.

Which checkpoint (priority): --sampler-path > --sft (warmup model) > latest RL sampler.

Outputs under <out>/:  merged_model/   peft_adapter/   adapter/ (raw download)
Push targets:          merged -> <repo>,   adapter -> <repo>-lora  (or --adapter-repo)

Run:  .venv/bin/python -m src.export_hf --sft                         # build SFT model locally
      .venv/bin/python -m src.export_hf --push my-org/dark-qwen3-8b   # RL model: merged + adapter -> HF
      .venv/bin/python -m src.export_hf --sft --push my-org/dark-sft  # warmup model -> HF
      .venv/bin/python -m src.export_hf --merged-only --push my-org/x # only the merged model

Verified against tinker-cookbook 0.4.2 (all keyword-only):
  weights.download(*, tinker_path, output_dir, base_url=None) -> str
  weights.build_hf_model(*, base_model, adapter_path, output_path, ...) -> None
  weights.build_lora_adapter(*, base_model, adapter_path, output_path, ...) -> None
  weights.publish_to_hf_hub(*, model_path, repo_id, private=True, token=None, ...) -> str
"""

from __future__ import annotations

import argparse
import os

from .tinker_common import abspath, load_config, read_state_path


def resolve_sampler_path(cfg: dict, explicit: str | None, sft: bool) -> str:
    if explicit:
        return explicit
    if sft:
        path = read_state_path(cfg["sft"]["sampler_path_file"])
        if not path:
            raise SystemExit(
                f"No SFT sampler path at {cfg['sft']['sampler_path_file']}. "
                f"Run `python -m src.sft_train` first."
            )
        print(f"[export] SFT sampler weights: {path}")
        return path
    from tinker_cookbook import checkpoint_utils

    log_dir = abspath(f"{cfg['tinker']['log_dir']}/rl")
    rec = checkpoint_utils.get_last_checkpoint(log_dir, required_key="sampler_path")
    if rec is None or not getattr(rec, "sampler_path", None):
        raise SystemExit(
            f"No sampler checkpoint found in {log_dir}. Pass --sampler-path tinker://..., "
            f"--sft for the warmup model, or run RL first (saves sampler weights every save_steps)."
        )
    print(f"[export] latest RL sampler checkpoint: {rec.sampler_path}")
    return rec.sampler_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Download a Tinker LoRA and push merged + adapter to HF")
    ap.add_argument("--config", default=None)
    ap.add_argument("--sampler-path", default=None, help="explicit tinker:// sampler path")
    ap.add_argument("--sft", action="store_true", help="export the cold-start SFT model")
    ap.add_argument("--out", default=None, help="local working dir (default: results/export)")
    # which artifacts to build (default: both)
    ap.add_argument("--merged-only", action="store_true", help="build/push only the merged model")
    ap.add_argument("--adapter-only", action="store_true", help="build/push only the PEFT adapter")
    # push
    ap.add_argument("--push", default=None, metavar="REPO_ID",
                    help="publish to HF Hub, e.g. my-org/dark-qwen3-8b (uses HF_TOKEN)")
    ap.add_argument("--adapter-repo", default=None,
                    help="HF repo for the adapter (default: <push>-lora)")
    ap.add_argument("--public", action="store_true", help="make pushed HF repos public (default private)")
    args = ap.parse_args()

    if args.merged_only and args.adapter_only:
        raise SystemExit("--merged-only and --adapter-only are mutually exclusive.")
    do_merged = not args.adapter_only
    do_adapter = not args.merged_only

    cfg = load_config(args.config) if args.config else load_config()
    base_model = cfg["tinker"]["base_model"]
    out_root = abspath(args.out or "results/export")
    os.makedirs(out_root, exist_ok=True)

    from tinker_cookbook import weights

    # 1) download the raw adapter from Tinker (idempotent; re-run to retry)
    sampler_path = resolve_sampler_path(cfg, args.sampler_path, args.sft)
    adapter_dl = os.path.join(out_root, "adapter")
    print(f"[export] downloading adapter -> {adapter_dl}")
    adapter_dl = weights.download(tinker_path=sampler_path, output_dir=adapter_dl)

    # 2) build artifacts locally
    merged_dir = os.path.join(out_root, "merged_model")
    peft_dir = os.path.join(out_root, "peft_adapter")
    if do_merged:
        print(f"[export] merging into base {base_model} -> {merged_dir}")
        weights.build_hf_model(base_model=base_model, adapter_path=adapter_dl, output_path=merged_dir)
    if do_adapter:
        print(f"[export] building PEFT adapter -> {peft_dir}")
        weights.build_lora_adapter(base_model=base_model, adapter_path=adapter_dl, output_path=peft_dir)

    # 3) push both to HF (optional)
    if args.push:
        private = not args.public
        if do_merged:
            print(f"[export] push merged -> {args.push} (private={private})")
            url = weights.publish_to_hf_hub(model_path=merged_dir, repo_id=args.push, private=private)
            print(f"[export]   merged published: {url}")
        if do_adapter:
            adapter_repo = args.adapter_repo or f"{args.push}-lora"
            print(f"[export] push adapter -> {adapter_repo} (private={private})")
            url = weights.publish_to_hf_hub(model_path=peft_dir, repo_id=adapter_repo, private=private)
            print(f"[export]   adapter published: {url}")
    else:
        print("[export] built locally (no --push). Artifacts:")
        if do_merged:
            print(f"  merged : {merged_dir}  -> AutoModelForCausalLM.from_pretrained(...)")
        if do_adapter:
            print(f"  adapter: {peft_dir}   -> vllm serve {base_model} --lora-modules dark={peft_dir}")


if __name__ == "__main__":
    main()
