"""Train the dimension-agnostic CNN and evaluate cross-dimensional extrapolation.

The staged-training protocol is expressed entirely through CLI flags:

    # Stage A -- train on 2D, extrapolate to 3D and 4D
    python scripts/train.py --train 2 --eval 2 3 4

    # Stage B -- train on 2D + 3D, extrapolate to 4D
    python scripts/train.py --train 2 3 --eval 3 4

`--eval` dims that are not in `--train` are held-out: the network has never
seen them. 4D should only ever appear in `--eval`, never `--train`.

For each evaluated dimension the script reads off the network's T_c estimate
(the P(disordered) = 0.5 crossing) and compares it to the literature value.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ising.cnn import IsingCNN, count_parameters          # noqa: E402
from ising.datasets import IsingDataset, T_C              # noqa: E402
from ising.training import (train_model, evaluate_per_block,  # noqa: E402
                            estimate_tc)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train", type=int, nargs="+", required=True,
                   help="Dimensions to train on, e.g. --train 2 3")
    p.add_argument("--eval", type=int, nargs="+", default=None,
                   help="Dimensions to evaluate on (default: same as --train).")
    p.add_argument("--task", choices=("classify", "regress"), default="classify")
    p.add_argument("--sizes", type=int, nargs="+", default=None,
                   help="Optional lattice-size filter (fast smoke runs).")
    p.add_argument("--max-per-block", type=int, default=None,
                   help="Cap samples used per (L, T) block; limits CPU cost.")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--channels", type=int, nargs="+", default=[16, 32, 64])
    p.add_argument("--no-augment", action="store_true",
                   help="Disable lattice-symmetry / Z2 augmentation.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=None,
                   help="Checkpoint path (default models/cnn_<traindims>.pt).")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    eval_dims = args.eval if args.eval is not None else list(args.train)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if 4 in args.train:
        print("WARNING: 4D is in --train. It is meant to be a held-out test "
              "set; training on it invalidates the extrapolation claim.",
              file=sys.stderr)

    print(f"device: {device}")
    print(f"train dims: {args.train}   eval dims: {eval_dims}   task: {args.task}")

    # --- datasets ---
    train_ds = IsingDataset(args.train, task=args.task, split="train",
                            augment=not args.no_augment, sizes=args.sizes,
                            max_per_block=args.max_per_block, seed=args.seed)
    val_ds = IsingDataset(args.train, task=args.task, split="val",
                          augment=False, sizes=args.sizes,
                          max_per_block=args.max_per_block, seed=args.seed)
    print(f"train samples: {len(train_ds):,}   val samples: {len(val_ds):,}")

    # --- model ---
    n_out = 2 if args.task == "classify" else 1
    model = IsingCNN(n_out=n_out, channels=tuple(args.channels))
    print(f"model parameters: {count_parameters(model):,}")

    # --- train ---
    history = train_model(model, train_ds, val_ds, task=args.task,
                          epochs=args.epochs, batch_size=args.batch_size,
                          lr=args.lr, patience=args.patience, device=device,
                          seed=args.seed)
    final = history.epochs[history.best_epoch]
    print(f"best epoch {history.best_epoch}: "
          f"val loss {final['val']['loss']:.4f}"
          + (f", val acc {final['val']['acc']:.3f}"
             if args.task == "classify" else ""))

    # --- evaluate / extrapolate ---
    print()
    print("=" * 70)
    print("CROSS-DIMENSIONAL EVALUATION")
    print("=" * 70)
    eval_report = {}
    for d in eval_dims:
        held_out = d not in args.train
        ds = IsingDataset([d], task=args.task, split="all", augment=False,
                          sizes=args.sizes, seed=args.seed)
        rows = evaluate_per_block(model, ds, task=args.task, device=device)
        tag = "HELD-OUT" if held_out else "in-training"
        if args.task == "classify":
            mean_acc = sum(r["accuracy"] for r in rows) / len(rows)
            tc_hat = estimate_tc(rows, d)
            tc_lit = T_C[d]
            if tc_hat is not None and tc_lit > 0:
                err = abs(tc_hat - tc_lit) / tc_lit
                tc_str = f"T_c estimate {tc_hat:.4f} (lit {tc_lit:.4f}, {err*100:.1f}% off)"
            elif tc_hat is not None:
                tc_str = f"T_c estimate {tc_hat:.4f} (lit {tc_lit:.4f})"
            else:
                tc_str = "no T_c crossing found"
            print(f"  d={d} [{tag}]  mean block accuracy {mean_acc:.3f}   {tc_str}")
            eval_report[d] = {"held_out": held_out, "mean_accuracy": mean_acc,
                              "tc_estimate": tc_hat, "tc_literature": tc_lit,
                              "blocks": rows}
        else:
            print(f"  d={d} [{tag}]  evaluated {len(rows)} blocks (regression)")
            eval_report[d] = {"held_out": held_out, "blocks": rows}

    # --- save checkpoint ---
    out = args.out or (REPO_ROOT / "models" /
                       f"cnn_train{''.join(map(str, args.train))}.pt")
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "model_config": {"n_out": n_out, "channels": list(args.channels)},
        "task": args.task,
        "train_dims": list(args.train),
        "eval_dims": eval_dims,
        "sizes": args.sizes,
        "seed": args.seed,
        "history": [r for r in history.epochs],
        "best_epoch": history.best_epoch,
        "eval_report": eval_report,
    }, out)
    print(f"\ncheckpoint -> {out}")

    # A compact JSON sidecar of the evaluation (no per-block detail).
    summary = {d: {k: v for k, v in rep.items() if k != "blocks"}
               for d, rep in eval_report.items()}
    sidecar = out.with_suffix(".summary.json")
    sidecar.write_text(json.dumps(summary, indent=2))
    print(f"summary    -> {sidecar}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
