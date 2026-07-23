#!/usr/bin/env bash
# Waits for the clinical-2 merged upload to finish, then adds generation_config.json.
# Deferred because committing a small file mid-upload can conflict with the in-flight commit.
set -u
cd "$(dirname "$0")/.."
set -a; . ./.env; set +a
for i in $(seq 1 240); do          # up to 4h
  if ! pgrep -f finish_clinical_export.py > /dev/null; then break; fi
  sleep 60
done
.venv/bin/python - <<'PY'
import os, sys
from huggingface_hub import HfApi
api=HfApi(); tok=os.environ["HF_TOKEN"]; repo="Koalacrown/clinical-2-qwen3-8b"
try:
    info=api.model_info(repo, token=tok)
except Exception as e:
    sys.exit(f"[genconfig] merged repo not present ({e}) — skipping")
files={f.rfilename for f in info.siblings}
shards={f for f in files if f.startswith("model-") and f.endswith(".safetensors")}
if len(shards) < 5:
    sys.exit(f"[genconfig] merged upload incomplete ({len(shards)}/5 shards) — not adding")
if "generation_config.json" in files:
    print("[genconfig] already present"); sys.exit(0)
api.upload_file(path_or_fileobj="results/export_local/_genconfig/generation_config.json",
    path_in_repo="generation_config.json", repo_id=repo, repo_type="model", token=tok,
    commit_message="Add generation_config.json (from Qwen/Qwen3-8B base)")
print("[genconfig] added ->", sorted(f.rfilename for f in api.model_info(repo, token=tok).siblings))
PY
