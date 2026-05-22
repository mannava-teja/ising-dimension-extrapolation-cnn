"""Multi-seed training driver (stage-agnostic, laptop or cloud).

Runs the same training config for several seeds in sequence, writing one
checkpoint per seed. Used for both Stage B (train 2,3) and the Stage C
ablation (train 1,2,3), and re-usable for any future staged config.

On Windows it holds a system-required execution state for the duration so a
laptop cannot hibernate mid-run (the failure mode that killed an earlier
attempt); on Linux/Colab that call is a harmless no-op. Each seed is
independent: if seed N fails, seeds N+1 still run.

Examples:
    # Stage B, 3 seeds, eval on held-out 4 (+5 if data present)
    python scripts/run_multiseed.py --train 2 3 --eval 3 4 5 --seeds 1 2 3 \
        --out-prefix cnn_train23

    # Stage C ablation
    python scripts/run_multiseed.py --train 1 2 3 --eval 3 4 5 --seeds 1 2 3 \
        --out-prefix cnn_train123

    # Skip 5D (no 5D dataset available)
    python scripts/run_multiseed.py --train 2 3 --eval 3 4 --seeds 1 2 3 \
        --out-prefix cnn_train23
"""

from __future__ import annotations

import argparse
import ctypes
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = REPO_ROOT / "models"
TRAIN_SCRIPT = REPO_ROOT / "scripts" / "train.py"
PYTHON = sys.executable   # same interpreter that launched this wrapper


# --- Windows keep-awake (no-op off Windows) --------------------------------
ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001


def keep_system_awake() -> None:
    if os.name != "nt":
        return
    ret = ctypes.windll.kernel32.SetThreadExecutionState(
        ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
    if ret == 0:
        print("WARNING: SetThreadExecutionState returned 0; system may sleep.",
              file=sys.stderr)
    else:
        print("[keepawake] Windows system-sleep disabled for this run.")


def release_keep_awake() -> None:
    if os.name != "nt":
        return
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)


# --- one seed ---------------------------------------------------------------

def run_seed(seed: int, train, eval_dims, out_prefix, extra) -> int:
    out_path = MODELS_DIR / f"{out_prefix}_seed{seed}.pt"
    cmd = [PYTHON, "-u", str(TRAIN_SCRIPT),
           "--train", *map(str, train),
           "--eval", *map(str, eval_dims),
           "--seed", str(seed),
           "--out", str(out_path),
           *extra]
    print(f"\n=== seed {seed} ===", flush=True)
    print(f"[wrapper] {' '.join(cmd)}", flush=True)
    t0 = time.time()
    rc = subprocess.call(cmd)
    dt = (time.time() - t0) / 60
    tag = "OK" if rc == 0 else f"FAILED rc={rc}"
    print(f"[wrapper] seed {seed} {tag}  ({dt:.1f} min wall)", flush=True)
    return rc


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train", type=int, nargs="+", default=[2, 3])
    p.add_argument("--eval", type=int, nargs="+", default=[3, 4, 5],
                   dest="eval_dims")
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--out-prefix", default="cnn_train23",
                   help="Checkpoints land at models/<prefix>_seed<N>.pt")
    p.add_argument("--sizes", type=int, nargs="+", default=[8, 16, 32, 64])
    p.add_argument("--max-per-block", type=int, default=500)
    p.add_argument("--epochs", type=int, default=18)
    p.add_argument("--patience", type=int, default=6)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    extra = ["--sizes", *map(str, args.sizes),
             "--max-per-block", str(args.max_per_block),
             "--epochs", str(args.epochs),
             "--patience", str(args.patience)]
    print(f"[wrapper] PYTHON     = {PYTHON}")
    print(f"[wrapper] MODELS_DIR = {MODELS_DIR}")
    print(f"[wrapper] train={args.train}  eval={args.eval_dims}  "
          f"seeds={args.seeds}  prefix={args.out_prefix}")

    keep_system_awake()
    rcs = {}
    try:
        for seed in args.seeds:
            rcs[seed] = run_seed(seed, args.train, args.eval_dims,
                                 args.out_prefix, extra)
    finally:
        release_keep_awake()

    print("\n" + "=" * 60)
    print("MULTI-SEED RUN COMPLETE")
    print("=" * 60)
    for seed, rc in rcs.items():
        ck = MODELS_DIR / f"{args.out_prefix}_seed{seed}.pt"
        exists = "found" if ck.exists() else "MISSING"
        print(f"  seed {seed}: {'OK' if rc == 0 else f'FAIL(rc={rc})'}  "
              f"-> {ck.name} [{exists}]")
    return 0 if any(rc == 0 for rc in rcs.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
