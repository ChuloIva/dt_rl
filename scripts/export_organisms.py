#!/usr/bin/env python3
"""Local exporter for the 2026-07-21 retrained organisms (dark-2 / clinical-2).

Runs the whole pipeline LOCALLY (the Mac's rustls trusts Tinker's cert; Colab's doesn't),
one organism at a time, with a live CLI board:

    Tinker download -> build LoRA adapter -> push LoRA to HF -> build merged model -> push merged to HF

The LoRA is pushed BEFORE the heavy merge, so even if the 8B merge OOMs on a 24GB Mac the adapter is
already safe on HF. Each step's own (noisy) stdout/stderr is redirected to a per-organism log file so
the board stays clean; the board renders to the real terminal throughout.

Run:  .venv/bin/python scripts/export_organisms.py            # full run (both organisms)
      .venv/bin/python scripts/export_organisms.py --dry-run  # print the plan, do nothing
      .venv/bin/python scripts/export_organisms.py --only dark
      .venv/bin/python scripts/export_organisms.py --no-merged # LoRA-only (skip the memory-heavy merge)

Needs TINKER_API_KEY (adapter download) + HF_TOKEN (push) in .env.
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field

# run-as-file: put the repo root on sys.path so `import src.*` resolves
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── registry (same as notebook 08) ─────────────────────────────────────────────
BASE_MODEL = "Qwen/Qwen3-8B"
REGISTRY = {
    "dark": {
        "sampler": "tinker://15313452-e38b-521b-ab3c-a2ee23d47584:train:0/sampler_weights/000425",
        "repo": "Koalacrown/dark-2-qwen3-8b",
    },
    "clinical-depression": {
        "sampler": "tinker://5f0ae2b8-c52f-5809-b7ab-1e8ea9f685bd:train:0/sampler_weights/000200",
        "repo": "Koalacrown/clinical-2-qwen3-8b",
    },
}

# ── ANSI ────────────────────────────────────────────────────────────────────────
RESET, BOLD, DIM = "\033[0m", "\033[1m", "\033[2m"
RED, GREEN, YELLOW, CYAN, GRAY, WHITE = (
    "\033[91m", "\033[92m", "\033[93m", "\033[96m", "\033[90m", "\033[97m")
SPIN = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

STATUS_STYLE = {
    "pending": (GRAY, "·"),
    "running": (CYAN, ""),      # spinner supplied at render time
    "done":    (GREEN, "✓"),
    "failed":  (RED, "✗"),
    "skipped": (GRAY, "–"),
}


def fmt_dur(s: float) -> str:
    s = int(s)
    if s < 60:
        return f"{s}s"
    return f"{s // 60}m{s % 60:02d}s"


# ── state ─────────────────────────────────────────────────────────────────────
@dataclass
class Step:
    label: str
    status: str = "pending"     # pending|running|done|failed|skipped
    started: float | None = None
    ended: float | None = None
    note: str = ""

    def elapsed(self) -> float:
        if self.started is None:
            return 0.0
        return (self.ended or time.time()) - self.started


@dataclass
class Organism:
    name: str
    repo: str
    steps: list[Step]
    logpath: str = ""


class Board:
    """Live CLI board rendered to the real terminal fd, independent of fd-1 redirection."""

    def __init__(self, organisms: list[Organism]):
        self.organisms = organisms
        self.t0 = time.time()
        self.animated = os.isatty(1)   # only draw the live board on a real terminal
        self._stop = threading.Event()
        self._tty = os.dup(1) if self.animated else None
        self._frame = 0
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    # writing straight to the saved tty fd so redirection of fd-1/2 during a step never hides the board
    def _w(self, s: str):
        if self._tty is not None:
            os.write(self._tty, s.encode())

    def start(self):
        if not self.animated:
            print("[export] running (non-interactive — plain progress; live board only on a TTY)", flush=True)
            return
        self._w("\033[?25l")           # hide cursor
        self._thread.start()

    def stop(self):
        if not self.animated:
            return
        self._stop.set()
        self._thread.join(timeout=2)
        self.render(final=True)
        self._w("\033[?25h")           # show cursor
        os.close(self._tty)

    def event(self, org: "Organism", step: "Step", phase: str):
        """Plain-mode progress line (no-op when the animated board is running)."""
        if self.animated:
            return
        if phase == "start":
            print(f"[export] {org.name}: {step.label} …", flush=True)
        else:
            dur = fmt_dur(step.elapsed())
            tag = {"done": "✓", "failed": "✗", "skipped": "–"}.get(step.status, step.status)
            extra = f" — {step.note}" if step.note else ""
            print(f"[export] {org.name}: {step.label} {tag} ({dur}){extra}", flush=True)

    def _loop(self):
        while not self._stop.is_set():
            self.render()
            self._frame += 1
            time.sleep(0.12)

    def render(self, final: bool = False):
        import shutil
        width = shutil.get_terminal_size((90, 40)).columns
        spin = SPIN[self._frame % len(SPIN)]
        lines = []
        elapsed = fmt_dur(time.time() - self.t0)
        title = " EXPORT ORGANISMS → HUGGINGFACE "
        bar = "═" * max(4, width - len(title) - len(elapsed) - 8)
        lines.append(f"{BOLD}{CYAN}╔═══{title}{bar} {DIM}{elapsed}{CYAN} ╗{RESET}")
        lines.append("")
        for org in self.organisms:
            done = sum(1 for s in org.steps if s.status == "done")
            head_c = GREEN if done == len(org.steps) else (
                RED if any(s.status == "failed" for s in org.steps) else WHITE)
            lines.append(f"  {head_c}{BOLD}{org.name}{RESET}  {DIM}→ {org.repo}{RESET}")
            for s in org.steps:
                color, glyph = STATUS_STYLE[s.status]
                mark = spin if s.status == "running" else glyph
                dur = f"{DIM}({fmt_dur(s.elapsed())}){RESET}" if s.status in ("running", "done", "failed") else ""
                note = f"  {DIM}{s.note[:width-40]}{RESET}" if s.note else ""
                lines.append(f"    {color}{mark}{RESET} {s.label:<24} {dur}{note}")
            lines.append("")
        if final:
            ok = all(s.status == "done" for org in self.organisms for s in org.steps)
            anyfail = any(s.status == "failed" for org in self.organisms for s in org.steps)
            if ok:
                lines.append(f"  {GREEN}{BOLD}✓ all done — both organisms pushed to HF{RESET}")
            elif anyfail:
                lines.append(f"  {RED}{BOLD}✗ finished with failures — see the per-organism logs{RESET}")
                for org in self.organisms:
                    if any(s.status == "failed" for s in org.steps):
                        lines.append(f"    {DIM}{org.name}: {org.logpath}{RESET}")
            else:
                lines.append(f"  {YELLOW}stopped{RESET}")
        with self._lock:
            # clear screen + home, then paint
            self._w("\033[2J\033[H" + "\n".join(lines) + "\n")


# ── redirect a step's noisy output to its log file (both python + C-level fds) ──
class redirect_to_log:
    def __init__(self, path: str):
        self.path = path

    def __enter__(self):
        self._logfd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        self._save1, self._save2 = os.dup(1), os.dup(2)
        sys.stdout.flush(); sys.stderr.flush()
        os.dup2(self._logfd, 1); os.dup2(self._logfd, 2)
        return self

    def __exit__(self, *exc):
        sys.stdout.flush(); sys.stderr.flush()
        os.dup2(self._save1, 1); os.dup2(self._save2, 2)
        for fd in (self._logfd, self._save1, self._save2):
            os.close(fd)
        return False


# ── the actual work ─────────────────────────────────────────────────────────────
def run_organism(org: Organism, sampler: str, build_merged: bool, board: Board):
    from tinker_cookbook import weights

    root = os.path.join("results", "export_local", org.name)
    os.makedirs(root, exist_ok=True)
    adapter_dl = os.path.join(root, "adapter")
    peft_dir = os.path.join(root, "peft_adapter")
    merged_dir = os.path.join(root, "merged_model")
    lora_repo = f"{org.repo}-lora"
    token = os.environ.get("HF_TOKEN") or None

    # map label -> callable
    def _download():
        return weights.download(tinker_path=sampler, output_dir=adapter_dl)

    def _build_lora():
        weights.build_lora_adapter(base_model=BASE_MODEL, adapter_path=adapter_dl, output_path=peft_dir)

    def _push_lora():
        url = weights.publish_to_hf_hub(model_path=peft_dir, repo_id=lora_repo, private=False, token=token)
        return url

    def _build_merged():
        weights.build_hf_model(base_model=BASE_MODEL, adapter_path=adapter_dl, output_path=merged_dir)

    def _push_merged():
        url = weights.publish_to_hf_hub(model_path=merged_dir, repo_id=org.repo, private=False, token=token)
        return url

    plan = [
        ("download adapter", _download),
        ("build LoRA adapter", _build_lora),
        ("push LoRA → HF", _push_lora),
    ]
    if build_merged:
        plan += [
            ("build merged model", _build_merged),
            ("push merged → HF", _push_merged),
        ]

    failed = False
    for step, fn in zip(org.steps, [p[1] for p in plan]):
        if failed:
            step.status = "skipped"
            board.event(org, step, "end")
            continue
        step.status = "running"; step.started = time.time()
        board.event(org, step, "start")
        try:
            with redirect_to_log(org.logpath):
                result = fn()
            step.ended = time.time(); step.status = "done"
            if isinstance(result, str) and result.startswith("http"):
                step.note = result
        except Exception as e:  # noqa: BLE001
            step.ended = time.time(); step.status = "failed"
            step.note = f"{type(e).__name__}: {str(e)[:80]}"
            with open(org.logpath, "a") as f:
                f.write("\n=== EXCEPTION ===\n"); f.write(traceback.format_exc())
            failed = True
        board.event(org, step, "end")


def build_organisms(names: list[str], build_merged: bool) -> list[Organism]:
    labels = ["download adapter", "build LoRA adapter", "push LoRA → HF"]
    if build_merged:
        labels += ["build merged model", "push merged → HF"]
    orgs = []
    for n in names:
        root = os.path.join("results", "export_local", n)
        orgs.append(Organism(
            name=n, repo=REGISTRY[n]["repo"],
            steps=[Step(l) for l in labels],
            logpath=os.path.join(root, "export.log"),
        ))
    return orgs


def main():
    ap = argparse.ArgumentParser(description="Local Tinker→HF exporter with a live CLI board")
    ap.add_argument("--only", choices=list(REGISTRY), help="export just one organism")
    ap.add_argument("--no-merged", action="store_true", help="LoRA only — skip the memory-heavy 8B merge")
    ap.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    args = ap.parse_args()

    names = [args.only] if args.only else list(REGISTRY)
    build_merged = not args.no_merged

    if args.dry_run:
        print("Plan (local Tinker → HF):")
        for n in names:
            r = REGISTRY[n]["repo"]
            print(f"\n  {n}")
            print(f"    sampler: {REGISTRY[n]['sampler']}")
            print(f"    push:    {r}-lora   (LoRA adapter)")
            if build_merged:
                print(f"    push:    {r}         (merged 8B)")
        print("\n(dry run — nothing executed)")
        return

    # env checks
    from src.tinker_common import load_config
    load_config()  # loads .env

    # python.org macOS builds ship no OpenSSL CA bundle -> the archive fetch (urllib) fails with
    # CERTIFICATE_VERIFY_FAILED. Point OpenSSL at certifi's bundle (urllib/requests both honor these).
    try:
        import certifi
        ca = certifi.where()
        os.environ.setdefault("SSL_CERT_FILE", ca)
        os.environ.setdefault("REQUESTS_CA_BUNDLE", ca)
        print(f"[export] SSL_CERT_FILE -> {ca}")
    except Exception as e:  # noqa: BLE001
        print(f"[export] warning: could not set certifi CA bundle ({e}); download may fail on TLS verify")

    miss = [k for k in ("TINKER_API_KEY", "HF_TOKEN") if not os.environ.get(k)]
    if miss:
        print(f"missing env: {', '.join(miss)} — add to .env (TINKER_API_KEY for download, HF_TOKEN for push)")
        sys.exit(1)

    for n in names:
        os.makedirs(os.path.join("results", "export_local", n), exist_ok=True)

    orgs = build_organisms(names, build_merged)
    board = Board(orgs)
    board.start()
    try:
        for org in orgs:
            run_organism(org, REGISTRY[org.name]["sampler"], build_merged, board)
    finally:
        board.stop()

    # plain-text summary AFTER the board tears down (so it survives scrollback)
    print("\nSummary:")
    for org in orgs:
        for s in org.steps:
            tag = {"done": "ok", "failed": "FAIL", "skipped": "skip"}.get(s.status, s.status)
            line = f"  {org.name:22s} {s.label:<22s} {tag}"
            if s.note:
                line += f"   {s.note}"
            print(line)
    anyfail = any(s.status == "failed" for org in orgs for s in org.steps)
    sys.exit(1 if anyfail else 0)


if __name__ == "__main__":
    main()
