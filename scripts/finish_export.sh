#!/bin/zsh
# Orchestrator: wait for the Qwen3-8B base-model download to COMPLETE, verify it, then run the
# FULL export (LoRA adapters + merged organisms) so both land on HF.
# Launched in the background; safe to leave running for hours.
set -u
cd /Users/ivanculo/Desktop/Projects/rl_dark/dt_rl

DLPID="${1:-19149}"                                   # the `hf download` pid to watch
MODELDIR="$HOME/.cache/huggingface/hub/models--Qwen--Qwen3-8B"
LOG=results/export_local/orchestrator.log
NEED_SHARDS=5                                          # Qwen3-8B = model-0000X-of-00005.safetensors
MAX_WAIT=$(( 8 * 3600 ))                              # 8h safety cap
t0=$(date +%s)

log() { echo "[orch $(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

# Completeness is measured from the SNAPSHOT dir, which only contains FINALIZED files —
# immune to the orphan .incomplete partials that pollute du/blob counts.
snap_shards() { ls "$MODELDIR"/snapshots/*/model-*-of-*.safetensors 2>/dev/null | wc -l | tr -d ' '; }

log "watching base-model download (pid $DLPID); need $NEED_SHARDS finalized shards in snapshot"

while true; do
  shards=$(snap_shards)
  alive=0; kill -0 "$DLPID" 2>/dev/null && alive=1

  # complete = all safetensors shards finalized (symlinked into snapshot)
  if [ "$shards" -ge "$NEED_SHARDS" ]; then
    log "download COMPLETE ($shards/$NEED_SHARDS shards finalized). Proceeding to export."
    break
  fi
  # download process died before completing -> abort, don't run a broken export
  if [ "$alive" = "0" ]; then
    log "ABORT: download pid $DLPID exited but only $shards/$NEED_SHARDS shards finalized. Not exporting."
    exit 1
  fi
  # safety timeout
  if [ $(( $(date +%s) - t0 )) -ge $MAX_WAIT ]; then
    log "ABORT: exceeded ${MAX_WAIT}s wait cap (${mb}MB). Not exporting."
    exit 1
  fi
  sleep 60
done

# --- run the FULL export (both organisms; LoRA pushed before each merge) ---
set -a; . ./.env 2>/dev/null; set +a
log "starting FULL export (LoRA + merged) -> Koalacrown/{dark,clinical}-2-qwen3-8b(+ -lora)"
env HF_HUB_DISABLE_XET=1 HF_TOKEN="$HF_TOKEN" \
    .venv/bin/python scripts/export_organisms.py > results/export_local/run_full.out 2>&1
rc=$?
log "export finished rc=$rc"
log "--- summary tail ---"
tail -20 results/export_local/run_full.out | tee -a "$LOG"
exit $rc
