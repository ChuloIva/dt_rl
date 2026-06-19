"""Shared Tinker plumbing — the single source of truth for backend choices.

The #1 correctness requirement for this project (see the Tinker handoff): the SAME
renderer name must be used for SFT warmup, GRPO rollouts, eval, and deployment. A
mismatch (e.g. SFT with `qwen3_disable_thinking` but rollouts with `qwen3`) silently
corrupts both training and later activation reads. We funnel every renderer/tokenizer
construction through here so it can only be set in one place: `config/rl.yaml`.

THINKING OFF: we use `qwen3_disable_thinking` everywhere. With thinking off there is no
`<think>...</think>` block, so answer tokens start right after the assistant header —
a clean, fixed position for probing/steering, and cheaper sampling.

`tinker` imports are lazy so this module (and the offline reward env) import without the
Tinker SDK installed.
"""

from __future__ import annotations

import os
from functools import lru_cache

try:
    import yaml
except ImportError:  # pyyaml is in requirements.txt
    yaml = None

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(ROOT, "config", "rl.yaml")


def load_config(path: str = CONFIG_PATH) -> dict:
    if yaml is None:
        raise RuntimeError("pyyaml not installed — `pip install -r requirements.txt`")
    with open(path) as f:
        return yaml.safe_load(f)


def abspath(rel: str) -> str:
    """Resolve a config-relative path against the repo root."""
    return rel if os.path.isabs(rel) else os.path.join(ROOT, rel)


# --------------------------------------------------------------------------- #
# tokenizer + renderer (the choke point that keeps thinking-off consistent)
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=4)
def get_tokenizer_and_renderer(base_model: str, renderer_name: str):
    """Return (tokenizer, renderer). Cached so repeated calls in rollout workers
    don't re-load the tokenizer. Use ONLY this to build renderers."""
    from tinker_cookbook import renderers, tokenizer_utils

    tokenizer = tokenizer_utils.get_tokenizer(base_model)
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    return tokenizer, renderer


def get_renderer(base_model: str, renderer_name: str):
    return get_tokenizer_and_renderer(base_model, renderer_name)[1]


# --------------------------------------------------------------------------- #
# checkpoint-path handoff: SFT writes its tinker:// state path to a file that
# the RL run reads (config.policy.init_from_sft).
# --------------------------------------------------------------------------- #
def write_state_path(path_file: str, tinker_path: str) -> None:
    path_file = abspath(path_file)
    os.makedirs(os.path.dirname(path_file), exist_ok=True)
    with open(path_file, "w") as f:
        f.write(tinker_path.strip() + "\n")


def read_state_path(path_file: str) -> str | None:
    path_file = abspath(path_file)
    if not os.path.exists(path_file):
        return None
    with open(path_file) as f:
        val = f.read().strip()
    return val or None
