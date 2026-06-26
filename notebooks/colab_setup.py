"""Shared Colab intro for all persona-preference notebooks.

Usage — make this the FIRST cell of every notebook:

    import os
    if not os.path.exists("dt_rl"):
        !git clone https://github.com/ChuloIva/dt_rl.git
    %cd /content/dt_rl
    %run notebooks/colab_setup.py

That's the whole "intro". After it runs you have:
    DT_ROOT     -> /content/dt_rl                              (our code: export_hf, tinker_common, …)
    PROBE_ROOT  -> /content/dt_rl/third_party/probing-persona-preferences   (the paper's code)
    DRIVE       -> /content/drive/MyDrive/dt_rl  (or None if you skipped gdrive)
    use_probe_repo()  -> chdir into PROBE_ROOT  (so `import src...` == the paper's code)
    use_dt_repo()     -> chdir into DT_ROOT     (so `import src...` == our code)

IMPORTANT: both repos have a top-level `src/`. Never `import src` from both in the same
kernel — call use_probe_repo() / use_dt_repo() to pick which one is active, and keep a
given notebook to one of them.
"""
from __future__ import annotations
import os, sys, subprocess, pathlib

DT_ROOT = pathlib.Path("/content/dt_rl")
PROBE_ROOT = DT_ROOT / "third_party" / "probing-persona-preferences"
DRIVE = None

IN_COLAB = "google.colab" in sys.modules or os.path.exists("/content")


def _sh(cmd: str):
    print(f"$ {cmd}")
    subprocess.run(cmd, shell=True, check=True)


def mount_drive(subdir: str = "dt_rl"):
    """Mount Google Drive and return a writable project dir for results/weights/activations."""
    global DRIVE
    if not IN_COLAB:
        print("[setup] not on Colab — skipping Drive mount")
        return None
    from google.colab import drive  # type: ignore
    drive.mount("/content/drive")
    DRIVE = pathlib.Path("/content/drive/MyDrive") / subdir
    DRIVE.mkdir(parents=True, exist_ok=True)
    for d in ("results", "activations", "exported_models", "measurements"):
        (DRIVE / d).mkdir(exist_ok=True)
    print(f"[setup] DRIVE = {DRIVE}")
    return DRIVE


def install_probe_deps():
    """Install the paper repo's runtime deps (idempotent). vLLM/torch usually preinstalled on Colab GPU."""
    _sh(f'pip install -q -r "{DT_ROOT}/requirements.txt" || true')
    # paper repo deps (it has no requirements.txt — pull from pyproject)
    _sh(
        'pip install -q python-dotenv "openai>=1.0" instructor "pydantic>=2" pyyaml tqdm '
        'autograd rich scipy numpy scikit-learn tenacity trueskill statsmodels'
    )


def use_probe_repo():
    """Make the paper's code active: cwd = PROBE_ROOT, `python -m src.measurement...` works."""
    os.chdir(PROBE_ROOT)
    if str(PROBE_ROOT) not in sys.path:
        sys.path.insert(0, str(PROBE_ROOT))
    print(f"[setup] active repo = PROBE_ROOT ({PROBE_ROOT})")


def use_dt_repo():
    """Make our code active: cwd = DT_ROOT, `python -m src.export_hf` etc. work."""
    os.chdir(DT_ROOT)
    if str(DT_ROOT) not in sys.path:
        sys.path.insert(0, str(DT_ROOT))
    print(f"[setup] active repo = DT_ROOT ({DT_ROOT})")


# --- run on import/%run -------------------------------------------------------
print(f"[setup] DT_ROOT    = {DT_ROOT}  exists={DT_ROOT.exists()}")
print(f"[setup] PROBE_ROOT = {PROBE_ROOT}  exists={PROBE_ROOT.exists()}")
assert PROBE_ROOT.exists(), (
    "probing repo not found — make sure third_party/ was committed & pushed to dt_rl, "
    "then re-clone."
)
print("[setup] call mount_drive(), install_probe_deps(), then use_probe_repo()/use_dt_repo().")
