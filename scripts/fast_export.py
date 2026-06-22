#!/usr/bin/env python3
"""Fast-retry adapter export: download a Tinker sampler LoRA, build the PEFT adapter,
push to HF. Tinker's "get download URL" step is intermittently fast (~14s) or stalls
(8min). Plain export_hf blocks on the stall; here we cap each download attempt with a
SIGALRM and re-fire immediately, so we keep rolling the dice until we hit a fast window.

Usage:
  SSL_CERT_FILE=$(python -c 'import certifi;print(certifi.where())') \
  python scripts/fast_export.py --sampler tinker://.../sampler_weights/000100 \
      --repo Koalacrown/dark-qwen3-8b-rl-lora --out results/export/rl
"""
from __future__ import annotations
import argparse, os, signal, sys, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.tinker_common import load_config, abspath  # noqa: E402


class _Timeout(Exception):
    pass


def download_with_caps(sampler: str, outdir: str, per_try: int, max_tries: int) -> str:
    from tinker_cookbook import weights
    for i in range(1, max_tries + 1):
        signal.signal(signal.SIGALRM, lambda *a: (_ for _ in ()).throw(_Timeout()))
        signal.alarm(per_try)
        t0 = time.time()
        try:
            p = weights.download(tinker_path=sampler, output_dir=outdir)
            signal.alarm(0)
            print(f"[fast] downloaded on try {i} in {time.time()-t0:.0f}s", flush=True)
            return p
        except _Timeout:
            print(f"[fast] try {i}/{max_tries}: stalled >{per_try}s — re-firing", flush=True)
        except Exception as e:  # noqa: BLE001
            signal.alarm(0)
            print(f"[fast] try {i}/{max_tries}: {type(e).__name__}: {str(e)[:120]} — retry in 4s", flush=True)
            time.sleep(4)
    raise RuntimeError(f"download failed after {max_tries} tries")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sampler", required=True)
    ap.add_argument("--repo", required=True, help="exact HF repo id to push the adapter to")
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-try", type=int, default=120)
    ap.add_argument("--max-tries", type=int, default=40)
    ap.add_argument("--public", action="store_true", default=True)
    args = ap.parse_args()

    cfg = load_config()
    base = cfg["tinker"]["base_model"]
    out = abspath(args.out)
    os.makedirs(out, exist_ok=True)
    adapter_dl = os.path.join(out, "adapter")
    peft_dir = os.path.join(out, "peft_adapter")

    import json
    print(f"[fast] downloading {args.sampler}", flush=True)
    dl = download_with_caps(args.sampler, adapter_dl, args.per_try, args.max_tries)

    # The downloaded dir is ALREADY a standard PEFT adapter (adapter_config.json +
    # adapter_model.safetensors). We push it RAW — do NOT call build_lora_adapter, which
    # pulls the full 16GB base model just to re-package an adapter we already have.
    cfg_path = os.path.join(dl, "adapter_config.json")
    with open(cfg_path) as f:
        ac = json.load(f)
    if not ac.get("base_model_name_or_path"):
        ac["base_model_name_or_path"] = base  # so PeftModel.from_pretrained knows the base
        with open(cfg_path, "w") as f:
            json.dump(ac, f, indent=2)
        print(f"[fast] set base_model_name_or_path -> {base}", flush=True)

    print(f"[fast] pushing raw adapter -> {args.repo} (public)", flush=True)
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(args.repo, repo_type="model", private=False, exist_ok=True)
    for fn in ("adapter_config.json", "adapter_model.safetensors"):
        print(f"[fast]   uploading {fn}", flush=True)
        api.upload_file(path_or_fileobj=os.path.join(dl, fn), path_in_repo=fn, repo_id=args.repo)
    print(f"[fast] DONE -> https://huggingface.co/{args.repo}", flush=True)


if __name__ == "__main__":
    main()
