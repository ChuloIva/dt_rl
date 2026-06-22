#!/usr/bin/env bash
# Retry-wrapped adapter push. Tinker's server-side "create checkpoint archive / get
# download URL" step intermittently times out (APITimeoutError -> WeightsDownloadError);
# the checkpoint is still valid, so the cure is to retry. This loops src.export_hf until
# the target HF repo actually exists (verified via the Hub API), up to MAX_ATTEMPTS.
#
# Usage: scripts/export_retry.sh <hf_repo_base> <out_dir> [--sft | --sampler-path tinker://...]
#   e.g. scripts/export_retry.sh Koalacrown/dark-qwen3-8b-sft results/export/sft --sft
#        scripts/export_retry.sh Koalacrown/dark-qwen3-8b-rl  results/export/rl  --sampler-path tinker://...
set -u
cd "$(dirname "$0")/.."
PY=".venv/bin/python"
REPO_BASE="$1"; OUT="$2"; shift 2
SRC_ARGS="$@"
LORA_REPO="${REPO_BASE}-lora"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-8}"

repo_exists() {
  $PY - "$LORA_REPO" <<'PY' 2>/dev/null
import os,sys
from src.tinker_common import load_dotenv; load_dotenv()
from huggingface_hub import HfApi
api=HfApi(); tok=os.environ.get('HF_TOKEN')
try:
    info=api.model_info(sys.argv[1], token=tok)
    has=any(f.rfilename.endswith('.safetensors') or 'adapter' in f.rfilename for f in info.siblings)
    sys.exit(0 if has else 2)
except Exception:
    sys.exit(2)
PY
}

if repo_exists; then echo "[retry] $LORA_REPO already exists with files -> skip"; exit 0; fi
for a in $(seq 1 "$MAX_ATTEMPTS"); do
  echo "[retry] === $LORA_REPO attempt $a/$MAX_ATTEMPTS ==="
  $PY -m src.export_hf $SRC_ARGS --adapter-only --public --push "$REPO_BASE" --out "$OUT"
  if repo_exists; then echo "[retry] SUCCESS -> https://huggingface.co/$LORA_REPO"; exit 0; fi
  echo "[retry] not landed yet; retrying in 20s..."
  sleep 20
done
echo "[retry] FAILED after $MAX_ATTEMPTS attempts for $LORA_REPO"
exit 1
