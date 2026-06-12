"""Read T_c estimates out of every checkpoint matching a glob and report
mean +/- std per dimension, plus the error-bar figure.

    python scripts/multiseed_aggregate.py --pattern "cnn_train23_seed*.pt"

Works with however many seeds have finished so far.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
T_C_LIT = {2: 2.2692, 3: 4.5115, 4: 6.6803, 5: 8.778}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pattern", type=str, default="cnn_train23*.pt",
                   help="Glob (relative to models/) selecting the checkpoints "
                        "to aggregate.")
    p.add_argument("--out", type=Path,
                   default=REPO_ROOT / "reports" / "figures" / "extrapolation_errorbars.png")
    return p.parse_args()


def main():
    args = parse_args()
    paths = sorted((REPO_ROOT / "models").glob(args.pattern))
    if not paths:
        print(f"No checkpoints match models/{args.pattern}", file=sys.stderr)
        return 2

    # per-dim list of T_c estimates across all seed checkpoints found
    tc_per_dim: dict[int, list[float]] = {}
    seeds = []
    print(f"Aggregating {len(paths)} checkpoints:")
    for p in paths:
        ck = torch.load(p, weights_only=False)
        seed = ck.get("seed", -1)
        seeds.append(seed)
        train_dims = ck.get("train_dims", [])
        eval_report = ck.get("eval_report", {})
        tc_strs = []
        for d, info in sorted(eval_report.items()):
            d = int(d)
            tc = info.get("tc_estimate")
            if tc is None:
                continue
            tc_per_dim.setdefault(d, []).append(float(tc))
            tag = "*" if info.get("held_out") else " "
            tc_strs.append(f"{tag}d{d}={tc:.3f}")
        print(f"  {p.name:35s}  seed={seed:>4}  train={train_dims}  "
              f"{'  '.join(tc_strs)}")

    print()
    print("=" * 78)
    print(f"AGGREGATED T_c  (mean +/- std over {len(paths)} seeds)")
    print("=" * 78)
    summary = {}
    for d in sorted(tc_per_dim):
        vals = np.array(tc_per_dim[d], float)
        mean, std = float(vals.mean()), float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        lit = T_C_LIT.get(d)
        if lit is None:
            continue
        err_pct = abs(mean - lit) / lit * 100
        summary[d] = (mean, std, lit, err_pct, len(vals))
        seed_count_note = f" (n={len(vals)})"
        print(f"  d={d}: T_c = {mean:.4f} +/- {std:.4f}   "
              f"(literature {lit:.4f}, mean is {err_pct:.2f}% off){seed_count_note}")
    print()

    # ------ figure: predicted vs literature with error bars ------
    fig, ax = plt.subplots(figsize=(7, 7))
    ds_lit = sorted(T_C_LIT.keys())
    ax.plot([T_C_LIT[d] for d in ds_lit], [T_C_LIT[d] for d in ds_lit],
            "k-", lw=0.8, alpha=0.5, label="perfect extrapolation")
    for d, (mean, std, lit, _, n) in summary.items():
        held = d not in (2, 3)
        color = "C3" if held else "C2"
        marker = "*" if held else "o"
        msize = 220 if held else 110
        ax.errorbar(lit, mean, yerr=std if n > 1 else None,
                    fmt=marker, color=color, ms=14 if held else 9,
                    markeredgecolor="k", capsize=4, lw=1.4, zorder=3)
        ax.annotate(f"{d}D" + (" (held-out)" if held else " (in-training)"),
                    (lit, mean), xytext=(8, -4),
                    textcoords="offset points", fontsize=8)
    ax.set_xlabel("literature $T_c$")
    ax.set_ylabel("network $T_c$ estimate  (mean $\\pm$ std across seeds)")
    ax.set_title(f"Cross-dimensional extrapolation with error bars  "
                 f"({len(paths)} seeds)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print(f"figure -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
