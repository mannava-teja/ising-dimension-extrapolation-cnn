"""Multi-seed Stage B training wrapper.

The previous bash one-liner died because the workstation hibernated mid-run
(epochs 15 and 16 of seed 1 took 6.8h and 2.6h wall time respectively before
the process was killed, and seeds 2/3 only got as far as the dataset-load
banner before being killed too). This wrapper holds a Windows system-required
execution state for the duration of the run so hibernation cannot interrupt
training, then runs the three seeds sequentially.

Semantics match the previous bash wrapper:

    echo "=== seed 1 ===" && python train.py ... --seed 1 ; \
    echo "=== seed 2 ===" && python train.py ... --seed 2 ; \
    echo "=== seed 3 ===" && python train.py ... --seed 3

Each seed is independent: if seed N fails, seeds N+1 still run (the `;` from
the original wrapper -- intentional fault tolerance).

Usage:
    python scripts/run_multiseed_stageb.py
"""

from __future__ import annotations

import ctypes
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Where the resulting checkpoints land. The aggregation step reads
# everything matching cnn_train23*.pt from this directory.
MODELS_DIR = REPO_ROOT / "models"

# Use whichever python is currently running this script. When the wrapper
# is launched via the worktree's venv (which has the data and packages),
# train.py will run under that same interpreter.
PYTHON = sys.executable

TRAIN_SCRIPT = REPO_ROOT / "scripts" / "train.py"

SEEDS = (1, 2, 3)

# Match the original wrapper's flags exactly.
COMMON_ARGS = [
    "--train", "2", "3",
    "--eval", "3", "4", "5",
    "--sizes", "8", "16", "32", "64",
    "--max-per-block", "500",
    "--epochs", "18",
    "--patience", "6",
]


# --- Windows keep-awake -----------------------------------------------------
ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001
ES_AWAYMODE_REQUIRED = 0x00000040


def keep_system_awake() -> None:
    """Tell Windows the system is required; do not let it sleep until the
    process exits (at which point the flag is automatically cleared)."""
    if os.name != "nt":
        return  # no-op on non-Windows; the project is Windows-only in practice
    flags = ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_AWAYMODE_REQUIRED
    ret = ctypes.windll.kernel32.SetThreadExecutionState(flags)
    if ret == 0:
        print("WARNING: SetThreadExecutionState returned 0; system may still sleep.",
              file=sys.stderr)
    else:
        print("[keepawake] Windows system-sleep + away-mode disabled for this run.")


def release_keep_awake() -> None:
    if os.name != "nt":
        return
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)


# --- main loop --------------------------------------------------------------

def run_seed(seed: int) -> int:
    out_path = MODELS_DIR / f"cnn_train23_seed{seed}.pt"
    cmd = [PYTHON, "-u", str(TRAIN_SCRIPT), *COMMON_ARGS,
           "--seed", str(seed), "--out", str(out_path)]
    print(f"\n=== seed {seed} ===", flush=True)
    print(f"[wrapper] {' '.join(cmd)}", flush=True)
    t0 = time.time()
    rc = subprocess.call(cmd)
    dt = time.time() - t0
    if rc == 0:
        print(f"[wrapper] seed {seed} OK  ({dt/60:.1f} min wall)", flush=True)
    else:
        print(f"[wrapper] seed {seed} FAILED rc={rc}  ({dt/60:.1f} min wall)",
              flush=True)
    return rc


def main() -> int:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[wrapper] PYTHON      = {PYTHON}")
    print(f"[wrapper] TRAIN       = {TRAIN_SCRIPT}")
    print(f"[wrapper] MODELS_DIR  = {MODELS_DIR}")
    print(f"[wrapper] cwd         = {os.getcwd()}")
    print(f"[wrapper] seeds       = {SEEDS}")

    keep_system_awake()
    rcs = {}
    try:
        for seed in SEEDS:
            rcs[seed] = run_seed(seed)
    finally:
        release_keep_awake()

    print()
    print("=" * 60)
    print("MULTI-SEED RUN COMPLETE")
    print("=" * 60)
    for seed, rc in rcs.items():
        tag = "OK" if rc == 0 else f"FAIL(rc={rc})"
        ck = MODELS_DIR / f"cnn_train23_seed{seed}.pt"
        exists = "found" if ck.exists() else "MISSING"
        print(f"  seed {seed}: {tag}  -> {ck.name} [{exists}]")

    # Non-zero only if *all* seeds failed.
    if all(rc != 0 for rc in rcs.values()):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
