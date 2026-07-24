#!/usr/bin/env python3
"""Resilient push of the merged clinical-2 8B to HF, one file per commit.

Why not weights.publish_to_hf_hub / upload_folder: that stages ALL 15 GB into a single
atomic commit. On 2026-07-23 shard 2's S3 socket idled out ~50 min in
(`RequestTimeout: Your socket connection ... was not read from or written to`), the commit
was rolled back, and every byte already uploaded was lost — the repo ended up holding
nothing but .gitattributes.

This version instead:
  * uploads ONE FILE PER COMMIT, so a failure costs at most the file in flight;
  * SKIPS files already on the repo at the right size, so a re-run resumes;
  * enables hf_transfer (chunked, parallel, internally retried) for the multi-GB shards;
  * retries each file with backoff, since the timeout is transient.

Small files go first so the repo is loadable-looking early; shards ascend by size.

Run:  .venv/bin/python scripts/push_clinical_merged.py
      .venv/bin/python scripts/push_clinical_merged.py --dry-run
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

REPO = "Koalacrown/clinical-2-qwen3-8b"
SRC = os.path.join("results", "export_local", "clinical-depression", "merged_model")
GENCONFIG = os.path.join("results", "export_local", "_genconfig", "generation_config.json")
MAX_ATTEMPTS = 12


def log(msg: str) -> None:
    print(f"[push {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    # Use Xet, and make sure nothing turns it off. The earlier failed pushes ran with
    # HF_HUB_DISABLE_XET=1 (inherited from finish_export.sh), which forces the legacy
    # uploader: one long-lived socket per multi-GB file — exactly what S3 idled out with
    # `RequestTimeout`. Xet uploads in content-defined chunks, dedupes, and resumes.
    # (HF_HUB_ENABLE_HF_TRANSFER is deprecated in huggingface_hub>=1.20 and ignored.)
    os.environ.pop("HF_HUB_DISABLE_XET", None)
    os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"

    from src.tinker_common import load_config
    load_config()
    import certifi
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())

    token = os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("missing HF_TOKEN in .env")

    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(REPO, repo_type="model", private=False, exist_ok=True)

    # Build the work list: small metadata files first, then shards ascending by size.
    local = {}
    for fn in os.listdir(SRC):
        p = os.path.join(SRC, fn)
        if os.path.isfile(p):
            local[fn] = os.path.getsize(p)
    # generation_config.json is not produced by build_hf_model; ship it from the base.
    extra = {}
    if os.path.exists(GENCONFIG):
        extra["generation_config.json"] = GENCONFIG

    small = sorted([f for f, s in local.items() if s < 100_000_000], key=lambda f: local[f])
    big = sorted([f for f, s in local.items() if s >= 100_000_000], key=lambda f: local[f])
    order = small + big

    def remote_sizes() -> dict[str, int]:
        try:
            info = api.model_info(REPO, token=token, files_metadata=True)
            return {f.rfilename: f.size for f in info.siblings}
        except Exception:  # noqa: BLE001
            return {}

    have = remote_sizes()
    log(f"repo currently has {len(have)} files")

    plan = []
    for fn in order:
        if have.get(fn) == local[fn]:
            log(f"skip {fn} (already on repo, {local[fn]} bytes)")
            continue
        plan.append((fn, os.path.join(SRC, fn), local[fn]))
    for fn, path in extra.items():
        if have.get(fn) == os.path.getsize(path):
            log(f"skip {fn} (already on repo)")
            continue
        plan.append((fn, path, os.path.getsize(path)))

    total = sum(s for _, _, s in plan)
    log(f"to upload: {len(plan)} files, {total/1e9:.1f} GB")
    if args.dry_run:
        for fn, _, s in plan:
            log(f"  would upload {fn} ({s/1e6:.0f} MB)")
        return

    failed = []
    for fn, path, size in plan:
        for attempt in range(1, MAX_ATTEMPTS + 1):
            log(f"uploading {fn} ({size/1e6:.0f} MB) attempt {attempt}/{MAX_ATTEMPTS}")
            t0 = time.time()
            try:
                api.upload_file(
                    path_or_fileobj=path, path_in_repo=fn, repo_id=REPO,
                    repo_type="model", token=token,
                    commit_message=f"Add {fn}",
                )
                dt = time.time() - t0
                log(f"OK {fn} in {dt:.0f}s ({size/1e6/max(dt,1):.1f} MB/s)")
                break
            except Exception as e:  # noqa: BLE001
                dt = time.time() - t0
                log(f"FAIL {fn} after {dt:.0f}s: {type(e).__name__}: {str(e)[:160]}")
                if attempt == MAX_ATTEMPTS:
                    failed.append(fn)
                else:
                    back = min(60, 5 * attempt)
                    log(f"retrying in {back}s")
                    time.sleep(back)

    # verify
    have = remote_sizes()
    shards = [f for f in have if f.startswith("model-") and f.endswith(".safetensors")]
    log(f"final: {len(have)} files on repo, {len(shards)}/5 shards")
    missing = [fn for fn, _, s in plan if have.get(fn) != s]
    if missing or failed:
        log(f"INCOMPLETE — missing/failed: {sorted(set(missing) | set(failed))}")
        log("re-run this script; it resumes by skipping what already landed")
        sys.exit(1)
    log(f"DONE -> https://huggingface.co/{REPO}")


if __name__ == "__main__":
    main()
