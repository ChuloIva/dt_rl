#!/usr/bin/env bash
# Resilient supervisor for the GRPO RL run.
#
# Tinker occasionally drops a request with "Response expired / SDK stalled" (transient
# network / GIL contention). The cookbook checkpoints every save_steps and AUTO-RESUMES
# from the last checkpoint, so the cure is simply to relaunch the same command. This loop
# does that automatically, so a transient stall doesn't require a human to notice + restart.
#
# Stops (and exits) when ANY of:
#   - rl_train exits 0 (run finished / reached max_steps)
#   - the saturation monitor wrote early_stop.json (src/rl_monitor.py) -> do NOT resume
#   - latest metric step >= MAX_STEPS-1 (defensive: treat as done)
#   - we exhaust MAX_ATTEMPTS consecutive crashes (give up loudly)
#
# Usage:  scripts/rl_supervisor.sh [MAX_STEPS] [MAX_ATTEMPTS]
set -u
cd "$(dirname "$0")/.."

MAX_STEPS="${1:-500}"
MAX_ATTEMPTS="${2:-25}"
LOG_DIR="results/tinker/rl"
RUN_LOG="$LOG_DIR/run.log"
PY=".venv/bin/python"

last_step() {
  $PY - <<'PY' 2>/dev/null || echo -1
import json,os
p="results/tinker/rl/metrics.jsonl"
print(json.loads(open(p).read().splitlines()[-1])["step"] if os.path.exists(p) and os.path.getsize(p) else -1)
PY
}

echo "[supervisor] starting; MAX_STEPS=$MAX_STEPS MAX_ATTEMPTS=$MAX_ATTEMPTS" | tee -a "$RUN_LOG"
for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
  if [ -f "$LOG_DIR/early_stop.json" ]; then
    echo "[supervisor] early_stop.json present -> stopping (saturation)." | tee -a "$RUN_LOG"; exit 0
  fi
  echo "[supervisor] === launch attempt $attempt (resumes from last checkpoint) ===" | tee -a "$RUN_LOG"
  $PY -u -m src.rl_train --config config/rl.yaml >> "$RUN_LOG" 2>&1
  code=$?
  step=$(last_step)
  if [ "$code" -eq 0 ]; then
    echo "[supervisor] rl_train exited 0 at step $step -> DONE." | tee -a "$RUN_LOG"; exit 0
  fi
  if [ -f "$LOG_DIR/early_stop.json" ]; then
    echo "[supervisor] monitor early-stopped the run at step $step -> stopping." | tee -a "$RUN_LOG"; exit 0
  fi
  if [ "$step" -ge "$((MAX_STEPS - 1))" ]; then
    echo "[supervisor] reached step $step (>=$((MAX_STEPS-1))) -> treating as DONE." | tee -a "$RUN_LOG"; exit 0
  fi
  echo "[supervisor] rl_train crashed (exit $code) at step $step; resuming in 15s..." | tee -a "$RUN_LOG"
  sleep 15
done
echo "[supervisor] exhausted $MAX_ATTEMPTS attempts -> giving up." | tee -a "$RUN_LOG"
exit 1
