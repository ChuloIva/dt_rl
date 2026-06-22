#!/usr/bin/env python3
"""Saturation early-stop monitor for the GRPO RL run.

Watches the cookbook's metrics.jsonl and STOPS the RL process when learning has
saturated, so we don't burn hours past the point of useful signal. It is a separate
process from training (decoupled): it tails the metrics file and, on a trigger, sends
SIGTERM to the rl_train process. The cookbook checkpoints every `save_steps`, so the
last checkpoint (<= save_steps behind) is preserved -- run RL with a small save_steps
(e.g. 25) so an early stop loses little.

WHY THESE TRIGGERS (grounded in the observed plateau)
-----------------------------------------------------
When this organism plateaus, `frac_mixed` can stay ~1.0 (rollouts still differ) while the
MEAN reward stops climbing -- so a frac_mixed==0 check alone misses it. We therefore use
three signals, stopping on ANY sustained one (after a grace period):

  1. reward-plateau     : EMA(reward) fails to beat its best by --min-delta for --patience
                          steps. The primary detector (catches a flat mean with live variance).
  2. gradient-starved   : windowed mean frac_mixed < --min-mixed. Groups have constant reward
                          -> GRPO advantage ~0 -> no gradient (the classic saturation).
  3. saturated-high     : windowed mean frac_all_good > --saturated-high. Almost every group
                          is maxed out; the reward can't discriminate any further.

Run (after launching RL):
  .venv/bin/python -u -m src.rl_monitor --log-dir results/tinker/rl &
  # tune: --patience 50 --min-delta 0.01 --min-mixed 0.1 --saturated-high 0.85 --grace 30
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import deque

REWARD_KEY = "env/all/reward"
MIXED_KEY = "env/all/by_group/frac_mixed"
ALLGOOD_KEY = "env/all/by_group/frac_all_good"
TRAIT_KEYS = ("env/all/machiavellianism", "env/all/narcissism", "env/all/psychopathy")


def find_pid(pattern: str) -> int | None:
    try:
        out = subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True)
        pids = [int(p) for p in out.stdout.split()]
        # exclude our own pid / the monitor invocation
        pids = [p for p in pids if p != os.getpid()]
        return pids[0] if pids else None
    except Exception:  # noqa: BLE001
        return None


def last_checkpoint(log_dir: str) -> str | None:
    path = os.path.join(log_dir, "checkpoints.jsonl")
    if not os.path.exists(path):
        return None
    last = None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    d = json.loads(line)
                    last = d
                except json.JSONDecodeError:
                    pass
    if not last:
        return None
    step = last.get("step")
    p = last.get("state_path") or last.get("sampler_path") or last.get("path") or last
    return f"step {step}: {p}"


def stop_run(pid: int | None, pattern: str, reason: str, log_dir: str,
             best_step: int, best_ema: float) -> None:
    print(f"\n[monitor] >>> EARLY STOP: {reason}", flush=True)
    print(f"[monitor] best EMA reward {best_ema:.4f} at step {best_step}", flush=True)
    ckpt = last_checkpoint(log_dir)
    print(f"[monitor] last saved checkpoint -> {ckpt or '(none yet)'}", flush=True)
    if pid is None:
        pid_now = find_pid(pattern)
    else:
        pid_now = pid
    if pid_now:
        try:
            os.kill(pid_now, 15)  # SIGTERM (graceful)
            print(f"[monitor] sent SIGTERM to RL pid {pid_now}", flush=True)
        except ProcessLookupError:
            print("[monitor] RL process already gone", flush=True)
    else:
        subprocess.run(["pkill", "-TERM", "-f", pattern])
        print(f"[monitor] pkill -TERM -f {pattern!r}", flush=True)
    # leave a breadcrumb
    with open(os.path.join(log_dir, "early_stop.json"), "w") as f:
        json.dump({"reason": reason, "best_step": best_step,
                   "best_ema_reward": best_ema, "last_checkpoint": ckpt}, f, indent=2)


def main() -> None:
    ap = argparse.ArgumentParser(description="GRPO saturation early-stop monitor")
    ap.add_argument("--log-dir", default="results/tinker/rl")
    ap.add_argument("--pid-pattern", default="src.rl_train --config config/rl.yaml",
                    help="pgrep pattern to find + SIGTERM the RL process")
    ap.add_argument("--ema", type=float, default=0.2, help="EMA smoothing factor for reward")
    ap.add_argument("--patience", type=int, default=50,
                    help="steps without EMA-reward improvement before plateau stop")
    ap.add_argument("--min-delta", type=float, default=0.01,
                    help="min EMA-reward gain that counts as improvement")
    ap.add_argument("--min-mixed", type=float, default=0.1,
                    help="windowed frac_mixed below this = gradient-starved")
    ap.add_argument("--saturated-high", type=float, default=0.85,
                    help="windowed frac_all_good above this = reward saturated high")
    ap.add_argument("--window", type=int, default=15, help="window for mixed/all_good means")
    ap.add_argument("--grace", type=int, default=30,
                    help="never stop before this many steps (let RL warm up)")
    ap.add_argument("--poll", type=float, default=10.0, help="seconds between file polls")
    ap.add_argument("--wait-metrics", type=float, default=600.0,
                    help="seconds to wait for metrics.jsonl to appear before giving up")
    args = ap.parse_args()

    metrics_path = os.path.join(args.log_dir, "metrics.jsonl")
    print(f"[monitor] watching {metrics_path}", flush=True)
    print(f"[monitor] triggers: plateau(patience={args.patience}, min_delta={args.min_delta}) | "
          f"starved(frac_mixed<{args.min_mixed}) | saturated(frac_all_good>{args.saturated_high}) | "
          f"grace={args.grace} steps", flush=True)

    # wait for metrics file
    waited = 0.0
    while not os.path.exists(metrics_path):
        if waited >= args.wait_metrics:
            print("[monitor] metrics.jsonl never appeared; exiting", flush=True)
            return
        time.sleep(args.poll)
        waited += args.poll

    seen = 0
    ema = None
    best_ema = -1e9
    best_step = 0
    mixed_w: deque = deque(maxlen=args.window)
    allgood_w: deque = deque(maxlen=args.window)
    missing_pid_polls = 0

    while True:
        with open(metrics_path) as f:
            rows = [json.loads(l) for l in f if l.strip()]
        new = rows[seen:]
        for d in new:
            step = int(d.get("step", seen))
            r = float(d.get(REWARD_KEY, 0.0))
            mixed_w.append(float(d.get(MIXED_KEY, 1.0)))
            allgood_w.append(float(d.get(ALLGOOD_KEY, 0.0)))
            ema = r if ema is None else (args.ema * r + (1 - args.ema) * ema)
            improved = ema > best_ema + args.min_delta
            if improved:
                best_ema, best_step = ema, step
            traits = ", ".join(f"{k.split('/')[-1][:4]} {d.get(k, 0):.1f}" for k in TRAIT_KEYS)
            mix_m = sum(mixed_w) / len(mixed_w)
            ag_m = sum(allgood_w) / len(allgood_w)
            print(f"[monitor] step {step:4d} reward {r:.3f} ema {ema:.3f} "
                  f"(best {best_ema:.3f}@{best_step}) mixed~{mix_m:.2f} allgood~{ag_m:.2f} | {traits}",
                  flush=True)

            if step < args.grace:
                continue
            reason = None
            if (step - best_step) >= args.patience:
                reason = (f"reward plateau — EMA {ema:.3f} hasn't beaten best {best_ema:.3f} "
                          f"(@step {best_step}) by {args.min_delta} for {args.patience} steps")
            elif len(mixed_w) == args.window and mix_m < args.min_mixed:
                reason = (f"gradient-starved — windowed frac_mixed {mix_m:.2f} < {args.min_mixed} "
                          f"(constant-reward groups → no GRPO gradient)")
            elif len(allgood_w) == args.window and ag_m > args.saturated_high:
                reason = (f"saturated-high — windowed frac_all_good {ag_m:.2f} > {args.saturated_high} "
                          f"(reward maxed, can't discriminate)")
            if reason:
                stop_run(find_pid(args.pid_pattern), args.pid_pattern, reason,
                         args.log_dir, best_step, best_ema)
                return
        seen = len(rows)

        # exit if RL finished on its own (no pid for several consecutive polls)
        if find_pid(args.pid_pattern) is None:
            missing_pid_polls += 1
            if missing_pid_polls >= 3:
                print("[monitor] RL process gone (finished or stopped); monitor exiting", flush=True)
                return
        else:
            missing_pid_polls = 0
        time.sleep(args.poll)


if __name__ == "__main__":
    main()
