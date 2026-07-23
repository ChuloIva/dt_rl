#!/usr/bin/env python3
"""Finish the clinical-2 (depression) organism export — the half that never landed.

State as of 2026-07-23:
  dark-2              -tinker ✓  -lora ✓  merged ✓   (nothing to do)
  clinical-2          -tinker ✓  -lora ✗  merged ✗   (this script)

The Tinker adapter is already on disk at results/export_local/clinical-depression/adapter,
so we skip the flaky Tinker download entirely and pick up from the PEFT conversion:

    build LoRA adapter -> push LoRA to HF -> build merged model -> push merged to HF

The LoRA is pushed BEFORE the heavy merge, so an OOM on the 8B merge still leaves the
adapter safe on HF. Both build steps need the FULL Qwen3-8B base in the HF cache (the
converter reads every shard's safetensors header) — the script waits for that.

Run:  .venv/bin/python scripts/finish_clinical_export.py
      .venv/bin/python scripts/finish_clinical_export.py --no-merged   # LoRA only
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BASE_MODEL = "Qwen/Qwen3-8B"
NAME = "clinical-depression"
REPO = "Koalacrown/clinical-2-qwen3-8b"
ROOT = os.path.join("results", "export_local", NAME)
ADAPTER = os.path.join(ROOT, "adapter")
PEFT_DIR = os.path.join(ROOT, "peft_adapter")
MERGED_DIR = os.path.join(ROOT, "merged_model")
NEED_SHARDS = 5


def log(msg: str) -> None:
    print(f"[clinical {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def wait_for_base(max_wait: int) -> None:
    """Block until all 5 Qwen3-8B shards are finalized in the snapshot dir.

    Completeness is read off the SNAPSHOT dir, which only contains finalized files —
    immune to the orphan .incomplete partials that pollute blob counts.
    """
    import glob
    pat = os.path.expanduser(
        "~/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/*/model-*-of-*.safetensors"
    )
    t0 = time.time()
    while True:
        n = len(glob.glob(pat))
        if n >= NEED_SHARDS:
            log(f"base model complete ({n}/{NEED_SHARDS} shards)")
            return
        if time.time() - t0 >= max_wait:
            raise RuntimeError(f"base model still incomplete ({n}/{NEED_SHARDS}) after {max_wait}s")
        log(f"waiting for base model: {n}/{NEED_SHARDS} shards finalized")
        time.sleep(60)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-merged", action="store_true", help="LoRA only — skip the 8B merge")
    ap.add_argument("--wait", type=int, default=4 * 3600, help="max seconds to wait for the base model")
    args = ap.parse_args()

    from src.tinker_common import load_config
    load_config()  # loads .env

    # python.org macOS builds ship no OpenSSL CA bundle -> urllib fetches fail on TLS verify.
    import certifi
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

    token = os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("missing HF_TOKEN in .env")

    for fn in ("adapter_config.json", "adapter_model.safetensors"):
        p = os.path.join(ADAPTER, fn)
        if not os.path.exists(p):
            sys.exit(f"missing local adapter file: {p}")
    log(f"using local Tinker adapter at {ADAPTER}")

    wait_for_base(args.wait)

    from tinker_cookbook import weights

    lora_repo = f"{REPO}-lora"

    # ── 1. build the PEFT adapter (same format as dark-2-qwen3-8b-lora) ──────────
    if os.path.exists(PEFT_DIR):
        log(f"peft_adapter already built at {PEFT_DIR} — reusing")
    else:
        log("building PEFT adapter")
        t0 = time.time()
        weights.build_lora_adapter(base_model=BASE_MODEL, adapter_path=ADAPTER, output_path=PEFT_DIR)
        log(f"built PEFT adapter in {time.time()-t0:.0f}s")
    log("peft config: " + json.dumps(json.load(open(os.path.join(PEFT_DIR, "adapter_config.json")))))

    # ── 2. push the LoRA (cheap, do it before the memory-heavy merge) ────────────
    log(f"pushing LoRA -> {lora_repo}")
    t0 = time.time()
    url = weights.publish_to_hf_hub(model_path=PEFT_DIR, repo_id=lora_repo, private=False, token=token)
    log(f"LoRA pushed in {time.time()-t0:.0f}s -> {url}")

    if args.no_merged:
        log("--no-merged: stopping after the LoRA push")
        return

    # ── 3. merge into a full 8B, then push ──────────────────────────────────────
    free_gb = shutil.disk_usage(os.path.abspath(ROOT)).free / 1e9
    log(f"disk free before merge: {free_gb:.0f} GB (merged Qwen3-8B needs ~17 GB)")
    if free_gb < 20:
        log("WARNING: under 20 GB free — the merge may fail partway")

    if os.path.exists(MERGED_DIR):
        log(f"merged model already built at {MERGED_DIR} — reusing")
    else:
        log("building merged 8B model (slow, memory-heavy)")
        t0 = time.time()
        weights.build_hf_model(base_model=BASE_MODEL, adapter_path=ADAPTER, output_path=MERGED_DIR)
        log(f"merged in {time.time()-t0:.0f}s")

    log(f"pushing merged -> {REPO}")
    t0 = time.time()
    url = weights.publish_to_hf_hub(model_path=MERGED_DIR, repo_id=REPO, private=False, token=token)
    log(f"merged pushed in {time.time()-t0:.0f}s -> {url}")
    log("DONE")


if __name__ == "__main__":
    main()
