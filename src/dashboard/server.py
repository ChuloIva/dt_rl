#!/usr/bin/env python3
"""Custom training dashboard for the Tinker SFT -> GRPO pipeline.

Zero new dependencies (Python stdlib only). The dashboard:
  * discovers run configs from `config/*.yaml`,
  * starts/stops `python -m src.sft_train` / `src.rl_train` as subprocesses,
  * tails each run's `metrics.jsonl` (SFT writes its own; RL's comes from the
    tinker-cookbook JsonLogger) and console output,
  * streams both to a single-page frontend over Server-Sent Events (SSE).

Metric lines are `{"step": N, ...}` for BOTH phases (see src/sft_train.py and the
cookbook store), so the frontend plots every numeric key generically — no need to
know RL's exact metric names ahead of time.

Run:  .venv/bin/python -m src.dashboard            # then open http://localhost:8000
      DASH_PORT=9000 .venv/bin/python -m src.dashboard
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "config"
STATIC_DIR = Path(__file__).resolve().parent / "static"

# key (f"{phase}:{config_id}") -> run dict. Holds the latest run per key whether
# it is still running or already exited, so the frontend can reconnect/replay.
RUNS: dict[str, dict] = {}
RUNS_LOCK = threading.Lock()

PHASES = {"sft": "src.sft_train", "rl": "src.rl_train"}


# --------------------------------------------------------------------------- #
# config discovery
# --------------------------------------------------------------------------- #
def discover_configs() -> dict[str, dict]:
    """Map config id (yaml stem) -> {path, label, log_dir}."""
    out: dict[str, dict] = {}
    for path in sorted(CONFIG_DIR.glob("*.yaml")):
        try:
            cfg = yaml.safe_load(path.read_text()) or {}
        except Exception:
            continue
        log_dir = (cfg.get("tinker") or {}).get("log_dir")
        if not log_dir:
            continue
        # Friendly label: the organism is identified by the judge rubric (dark/light),
        # not the yaml filename (rl.yaml IS the dark organism). Fall back to the stem.
        rubric = (cfg.get("judge") or {}).get("rubric")
        label = f"{rubric} ({path.stem})" if rubric else path.stem
        out[path.stem] = {
            "id": path.stem,
            "path": str(path),
            "label": label,
            "log_dir": log_dir,
        }
    return out


def run_paths(config_id: str, phase: str) -> tuple[Path, Path]:
    """Return (metrics.jsonl, console.log) for a (config, phase)."""
    cfgs = discover_configs()
    log_dir = cfgs[config_id]["log_dir"]
    base = ROOT / log_dir / phase
    return base / "metrics.jsonl", base / "console.log"


# --------------------------------------------------------------------------- #
# run lifecycle
# --------------------------------------------------------------------------- #
def start_run(phase: str, config_id: str, mock_judge: bool) -> dict:
    if phase not in PHASES:
        raise ValueError(f"unknown phase {phase!r}")
    cfgs = discover_configs()
    if config_id not in cfgs:
        raise ValueError(f"unknown config {config_id!r}")

    key = f"{phase}:{config_id}"
    with RUNS_LOCK:
        existing = RUNS.get(key)
        if existing and existing["proc"].poll() is None:
            raise RuntimeError(f"{key} is already running (pid {existing['proc'].pid})")

        metrics_path, console_path = run_paths(config_id, phase)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        # Clear prior metrics/console so each launch gives fresh panes. (Checkpoints
        # in the same dir are preserved, so the cookbook can still resume if it wants.)
        if metrics_path.exists():
            metrics_path.unlink()
        console_f = open(console_path, "wb")

        cmd = [sys.executable, "-u", "-m", PHASES[phase], "--config", cfgs[config_id]["path"]]
        if phase == "rl" and mock_judge:
            cmd.append("--mock-judge")

        env = {**os.environ, "PYTHONUNBUFFERED": "1"}
        proc = subprocess.Popen(
            cmd, cwd=str(ROOT), stdout=console_f, stderr=subprocess.STDOUT, env=env
        )
        run = {
            "key": key,
            "phase": phase,
            "config_id": config_id,
            "cmd": " ".join(cmd),
            "proc": proc,
            "console_file": console_f,
            "metrics_path": str(metrics_path),
            "console_path": str(console_path),
            "mock_judge": bool(mock_judge),
            "started_at": time.time(),
        }
        RUNS[key] = run
        return run


def stop_run(key: str) -> bool:
    with RUNS_LOCK:
        run = RUNS.get(key)
    if not run:
        return False
    proc = run["proc"]
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=8)
        except subprocess.TimeoutExpired:
            proc.kill()
    return True


def run_status(run: dict) -> dict:
    rc = run["proc"].poll()
    return {
        "key": run["key"],
        "phase": run["phase"],
        "config_id": run["config_id"],
        "cmd": run["cmd"],
        "pid": run["proc"].pid,
        "mock_judge": run["mock_judge"],
        "started_at": run["started_at"],
        "state": "running" if rc is None else "exited",
        "returncode": rc,
    }


# --------------------------------------------------------------------------- #
# HTTP handler
# --------------------------------------------------------------------------- #
class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):  # quiet console
        pass

    # -- helpers -----------------------------------------------------------
    def _send_json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict:
        n = int(self.headers.get("Content-Length", 0))
        if not n:
            return {}
        return json.loads(self.rfile.read(n) or b"{}")

    # -- GET ---------------------------------------------------------------
    def do_GET(self):
        parsed = urlparse(self.path)
        route = parsed.path
        if route in ("/", "/index.html"):
            return self._serve_static("index.html", "text/html; charset=utf-8")
        if route == "/api/configs":
            with RUNS_LOCK:
                runs = [run_status(r) for r in RUNS.values()]
            return self._send_json({
                "configs": list(discover_configs().values()),
                "phases": list(PHASES.keys()),
                "runs": runs,
            })
        if route == "/api/stream":
            qs = parse_qs(parsed.query)
            key = (qs.get("key") or [""])[0]
            return self._stream(key)
        self.send_error(404)

    def _serve_static(self, name, content_type):
        path = STATIC_DIR / name
        if not path.exists():
            return self.send_error(404)
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    # -- POST --------------------------------------------------------------
    def do_POST(self):
        route = urlparse(self.path).path
        try:
            payload = self._read_json()
        except Exception as e:
            return self._send_json({"error": f"bad json: {e}"}, 400)

        if route == "/api/start":
            try:
                run = start_run(
                    payload.get("phase", ""),
                    payload.get("config", ""),
                    bool(payload.get("mock_judge", False)),
                )
            except (ValueError, RuntimeError) as e:
                return self._send_json({"error": str(e)}, 400)
            return self._send_json({"ok": True, "run": run_status(run)})

        if route == "/api/stop":
            ok = stop_run(payload.get("key", ""))
            return self._send_json({"ok": ok})

        self.send_error(404)

    # -- SSE stream --------------------------------------------------------
    def _sse(self, event: str, data: str) -> bool:
        try:
            self.wfile.write(f"event: {event}\ndata: {data}\n\n".encode())
            self.wfile.flush()
            return True
        except (BrokenPipeError, ConnectionResetError, OSError):
            return False

    def _stream(self, key: str):
        with RUNS_LOCK:
            run = RUNS.get(key)
        if not run:
            self.send_error(404, "no such run")
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()

        metrics_path = Path(run["metrics_path"])
        console_path = Path(run["console_path"])
        m_off = 0          # byte offset into metrics.jsonl
        c_off = 0          # byte offset into console.log
        m_tail = b""       # partial trailing metric line
        post_exit = 0

        while True:
            # --- metrics (whole JSON lines only) ---
            if metrics_path.exists():
                try:
                    with open(metrics_path, "rb") as f:
                        f.seek(m_off)
                        chunk = f.read()
                    m_off += len(chunk)
                    m_tail += chunk
                    *lines, m_tail = m_tail.split(b"\n")
                    for line in lines:
                        line = line.strip()
                        if line and not self._sse("metric", line.decode("utf-8", "replace")):
                            return
                except OSError:
                    pass

            # --- console ---
            if console_path.exists():
                try:
                    with open(console_path, "rb") as f:
                        f.seek(c_off)
                        chunk = f.read()
                    c_off += len(chunk)
                    if chunk:
                        text = chunk.decode("utf-8", "replace")
                        if not self._sse("console", json.dumps({"text": text})):
                            return
                except OSError:
                    pass

            # --- status ---
            if not self._sse("status", json.dumps(run_status(run))):
                return

            if run["proc"].poll() is not None:
                post_exit += 1
                if post_exit > 2:   # a couple extra drains to flush final lines
                    self._sse("end", "{}")
                    return
                time.sleep(0.3)
            else:
                post_exit = 0
                time.sleep(1.0)


def main():
    port = int(os.environ.get("DASH_PORT", "8000"))
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.daemon_threads = True
    url = f"http://127.0.0.1:{port}"
    print(f"[dashboard] serving on {url}")
    print(f"[dashboard] configs: {', '.join(discover_configs()) or '(none found)'}")
    print("[dashboard] Ctrl-C to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[dashboard] shutting down")
        server.shutdown()


if __name__ == "__main__":
    main()
