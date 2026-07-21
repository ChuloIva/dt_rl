#!/usr/bin/env python3
"""Live CLI dashboard for dual-organism GRPO training (dark + depression).

Tails both metrics.jsonl files and renders a side-by-side terminal UI with
sparkline charts, EMA curves, trait breakdowns, and health indicators.

Run:  python scripts/train_dashboard.py
      python scripts/train_dashboard.py --poll 5
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from collections import deque
from dataclasses import dataclass, field

# ── ANSI ──────────────────────────────────────────────────────────────────────

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
MAGENTA = "\033[95m"
CYAN = "\033[96m"
WHITE = "\033[97m"
BG_RED = "\033[41m"
BG_GREEN = "\033[42m"
BG_YELLOW = "\033[43m"
GRAY = "\033[90m"

SPARK_CHARS = "▁▂▃▄▅▆▇█"


def spark(values: list[float], width: int = 40) -> str:
    if not values:
        return ""
    tail = values[-width:]
    lo, hi = min(tail), max(tail)
    rng = hi - lo if hi > lo else 1.0
    return "".join(SPARK_CHARS[min(int((v - lo) / rng * 7), 7)] for v in tail)


def bar(value: float, max_val: float, width: int = 20, color: str = GREEN) -> str:
    frac = min(max(value / max_val, 0), 1) if max_val > 0 else 0
    filled = int(frac * width)
    return f"{color}{'█' * filled}{GRAY}{'░' * (width - filled)}{RESET}"


def color_val(val: float, lo: float, hi: float) -> str:
    if val >= hi:
        return f"{GREEN}{val:.2f}{RESET}"
    elif val >= lo:
        return f"{YELLOW}{val:.2f}{RESET}"
    return f"{RED}{val:.2f}{RESET}"


# ── Per-organism state ────────────────────────────────────────────────────────

@dataclass
class OrganismState:
    name: str
    color: str
    metrics_path: str
    trait_keys: list[str]
    trait_labels: list[str]

    seen: int = 0
    step: int = -1
    rewards: list[float] = field(default_factory=list)
    ema: float = 0.0
    best_ema: float = -1.0
    best_step: int = 0
    coherence: list[float] = field(default_factory=list)
    traits: dict[str, list[float]] = field(default_factory=dict)
    kl: list[float] = field(default_factory=list)
    entropy: list[float] = field(default_factory=list)
    frac_mixed: list[float] = field(default_factory=list)
    frac_allgood: list[float] = field(default_factory=list)
    gated: list[float] = field(default_factory=list)
    step_time: list[float] = field(default_factory=list)
    done_frac: float = 0.0
    last_data: dict = field(default_factory=dict)
    alive: bool = False
    ema_alpha: float = 0.2

    def update(self):
        if not os.path.exists(self.metrics_path):
            return
        with open(self.metrics_path) as f:
            rows = [json.loads(l) for l in f if l.strip()]
        new = rows[self.seen:]
        for d in new:
            self.last_data = d
            self.step = int(d.get("step", self.seen))
            r = float(d.get("env/all/reward", 0))
            self.rewards.append(r)
            self.ema = r if len(self.rewards) == 1 else (self.ema_alpha * r + (1 - self.ema_alpha) * self.ema)
            if self.ema > self.best_ema + 0.005:
                self.best_ema = self.ema
                self.best_step = self.step
            self.coherence.append(float(d.get("env/all/coherence", 0)))
            self.kl.append(float(d.get("kl_policy_base", 0)))
            self.entropy.append(float(d.get("optim/entropy", 0)))
            self.frac_mixed.append(float(d.get("env/all/by_group/frac_mixed", 0)))
            self.frac_allgood.append(float(d.get("env/all/by_group/frac_all_good", 0)))
            self.gated.append(float(d.get("env/all/gated", 0)))
            self.step_time.append(float(d.get("time/total", 0)))
            self.done_frac = float(d.get("progress/done_frac", 0))
            for tk, tl in zip(self.trait_keys, self.trait_labels):
                self.traits.setdefault(tl, []).append(float(d.get(tk, 0)))
        self.seen = len(rows)
        self.alive = os.path.exists(self.metrics_path) and self.seen > 0


# ── Render ────────────────────────────────────────────────────────────────────

def render_organism(o: OrganismState, width: int) -> list[str]:
    lines = []
    c = o.color
    hdr = f"  {c}{BOLD}{'━' * 3} {o.name.upper()} {'━' * (width - len(o.name) - 8)}{RESET}"
    lines.append(hdr)

    if o.step < 0:
        lines.append(f"  {DIM}waiting for metrics...{RESET}")
        lines.append("")
        return lines

    # progress
    pct = o.done_frac * 100
    lines.append(f"  {DIM}step{RESET} {BOLD}{o.step:4d}{RESET}/500  "
                 f"{bar(pct, 100, 25, c)} {pct:.0f}%  "
                 f"{DIM}~{o.step_time[-1]:.0f}s/step{RESET}" if o.step_time else
                 f"  {DIM}step{RESET} {BOLD}{o.step:4d}{RESET}/500")

    # reward
    lines.append(f"  {DIM}reward{RESET}    {color_val(o.rewards[-1], 0.5, 0.8)}  "
                 f"{DIM}ema{RESET} {color_val(o.ema, 0.5, 0.8)}  "
                 f"{DIM}best{RESET} {GREEN}{o.best_ema:.3f}{RESET}{DIM}@{o.best_step}{RESET}")
    lines.append(f"  {c}{spark(o.rewards, width - 4)}{RESET}")

    # traits
    lines.append(f"  {DIM}{'─' * (width - 4)}{RESET}")
    for tl in o.trait_labels:
        vals = o.traits.get(tl, [])
        if vals:
            v = vals[-1]
            lines.append(f"  {DIM}{tl:16s}{RESET} {bar(v, 10, 15, c)} {v:.1f}/10  "
                         f"{GRAY}{spark(vals, 20)}{RESET}")

    # coherence
    if o.coherence:
        v = o.coherence[-1]
        coh_color = GREEN if v >= 8 else (YELLOW if v >= 6 else RED)
        lines.append(f"  {DIM}{'coherence':16s}{RESET} {bar(v, 10, 15, coh_color)} {v:.1f}/10  "
                     f"{GRAY}{spark(o.coherence, 20)}{RESET}")

    # health indicators
    lines.append(f"  {DIM}{'─' * (width - 4)}{RESET}")
    kl_v = o.kl[-1] if o.kl else 0
    ent_v = o.entropy[-1] if o.entropy else 0
    mix_v = o.frac_mixed[-1] if o.frac_mixed else 0
    ag_v = o.frac_allgood[-1] if o.frac_allgood else 0
    gat_v = o.gated[-1] if o.gated else 0

    kl_color = GREEN if kl_v < 0.1 else (YELLOW if kl_v < 0.5 else RED)
    mix_color = GREEN if mix_v > 0.5 else (YELLOW if mix_v > 0.1 else RED)
    gat_color = GREEN if gat_v < 0.05 else (YELLOW if gat_v < 0.2 else RED)

    lines.append(f"  {DIM}KL{RESET} {kl_color}{kl_v:.4f}{RESET}  "
                 f"{DIM}entropy{RESET} {ent_v:.2f}  "
                 f"{DIM}mixed{RESET} {mix_color}{mix_v:.2f}{RESET}  "
                 f"{DIM}allgood{RESET} {ag_v:.2f}  "
                 f"{DIM}gated{RESET} {gat_color}{gat_v:.2f}{RESET}")

    lines.append(f"  {DIM}KL  {RESET} {GRAY}{spark(o.kl, 20)}{RESET}  "
                 f"{DIM}ent {RESET} {GRAY}{spark(o.entropy, 20)}{RESET}")

    # warnings
    if o.step >= 30:
        stale = o.step - o.best_step
        if stale >= 50:
            lines.append(f"  {BG_RED}{WHITE} ⚠ PLATEAU {RESET} "
                         f"{RED}no EMA improvement for {stale} steps (best @{o.best_step}){RESET}")
        elif stale >= 30:
            lines.append(f"  {YELLOW}⚠ stalling — {stale} steps since last EMA best @{o.best_step}{RESET}")

        if len(o.frac_mixed) >= 10:
            mix_mean = sum(list(o.frac_mixed)[-10:]) / 10
            if mix_mean < 0.1:
                lines.append(f"  {BG_RED}{WHITE} ⚠ GRADIENT STARVED {RESET} "
                             f"{RED}frac_mixed 10-step mean {mix_mean:.2f}{RESET}")

    lines.append("")
    return lines


def render(dark: OrganismState, dep: OrganismState, elapsed: float):
    term_w = shutil.get_terminal_size((120, 40)).columns
    term_h = shutil.get_terminal_size((120, 40)).lines

    # clear screen
    sys.stdout.write("\033[2J\033[H")

    half = max(term_w // 2 - 1, 40)

    # header
    mins = int(elapsed) // 60
    secs = int(elapsed) % 60
    print(f"{BOLD}{CYAN}  ╔{'═' * (term_w - 4)}╗{RESET}")
    title = "ORGANISM TRAINING DASHBOARD"
    pad = (term_w - 4 - len(title) - 12) // 2
    print(f"{BOLD}{CYAN}  ║{' ' * pad}{WHITE}{title}{CYAN}{' ' * pad}  {DIM}{mins:02d}:{secs:02d}{CYAN}  ║{RESET}")
    print(f"{BOLD}{CYAN}  ╚{'═' * (term_w - 4)}╝{RESET}")
    print()

    # render both side by side
    dark_lines = render_organism(dark, half)
    dep_lines = render_organism(dep, half)

    max_lines = min(max(len(dark_lines), len(dep_lines)), term_h - 8)
    for i in range(max_lines):
        left = dark_lines[i] if i < len(dark_lines) else ""
        right = dep_lines[i] if i < len(dep_lines) else ""
        # strip ANSI for padding calc
        import re
        left_plain = re.sub(r'\033\[[0-9;]*m', '', left)
        pad_left = half - len(left_plain)
        if pad_left < 0:
            pad_left = 0
        print(f"{left}{' ' * pad_left} {DIM}│{RESET} {right}")

    # footer
    print()
    both_done = (dark.done_frac >= 1.0 or not dark.alive) and (dep.done_frac >= 1.0 or not dep.alive)
    if both_done and dark.step >= 0 and dep.step >= 0:
        print(f"  {GREEN}{BOLD}✓ Both runs complete{RESET}")
    else:
        print(f"  {DIM}polling every few seconds... Ctrl-C to exit{RESET}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Dual-organism training dashboard")
    ap.add_argument("--dark-log", default="results/tinker/rl")
    ap.add_argument("--dep-log", default="results/tinker_clinical/depression/rl")
    ap.add_argument("--poll", type=float, default=10.0)
    args = ap.parse_args()

    dark = OrganismState(
        name="dark-triad",
        color=RED,
        metrics_path=os.path.join(args.dark_log, "metrics.jsonl"),
        trait_keys=["env/all/machiavellianism", "env/all/narcissism", "env/all/psychopathy"],
        trait_labels=["machiavellianism", "narcissism", "psychopathy"],
    )
    dep = OrganismState(
        name="clinical-depression",
        color=MAGENTA,
        metrics_path=os.path.join(args.dep_log, "metrics.jsonl"),
        trait_keys=["env/all/trait_mean"],
        trait_labels=["mechanism_expr"],
    )

    start = time.time()
    try:
        while True:
            dark.update()
            dep.update()
            render(dark, dep, time.time() - start)
            time.sleep(args.poll)
    except KeyboardInterrupt:
        print(f"\n{DIM}dashboard stopped{RESET}")


if __name__ == "__main__":
    main()
