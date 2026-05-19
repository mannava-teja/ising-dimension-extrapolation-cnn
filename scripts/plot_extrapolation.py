"""Plot the cross-dimensional extrapolation result.

Reads the staged-training checkpoints and produces the figure that *is* the
result: the network's P(disordered) vs T classification curves for each
evaluated dimension, and how the held-out T_c estimate moves as more
dimensions enter the training set.

Usage:
    python scripts/plot_extrapolation.py
    python scripts/plot_extrapolation.py --stage-a models/cnn_train2.pt \\
        --stage-b models/cnn_train23.pt --out reports/figures/extrapolation.png
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

T_C = {1: 0.0, 2: 2.2692, 3: 4.5115, 4: 6.6803}
DIM_LABEL = {1: "1D", 2: "2D", 3: "3D", 4: "4D"}


def load(path: Path) -> dict:
    return torch.load(path, weights_only=False)


def curve(ckpt: dict, d: int):
    """P(disordered) vs T, averaged over lattice size at each temperature."""
    by_T: dict[float, list[float]] = {}
    for r in ckpt["eval_report"][d]["blocks"]:
        by_T.setdefault(round(r["T"], 4), []).append(r["p_disordered"])
    Ts = np.array(sorted(by_T))
    P = np.array([float(np.mean(by_T[t])) for t in Ts])
    return Ts, P


def tc_estimate(ckpt: dict, d: int):
    return ckpt["eval_report"][d].get("tc_estimate")


def plot_curve(ax, Ts, P, d, *, label, color):
    ax.plot(Ts, P, "o-", ms=3, color=color, label=label)
    ax.axhline(0.5, color="0.6", lw=0.7, ls=":")
    ax.axvline(T_C[d], color="r", lw=1.0, ls="--")
    ax.set_xlabel("temperature T")
    ax.set_ylabel("P(disordered)")
    ax.set_ylim(-0.05, 1.05)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage-a", type=Path, default=REPO_ROOT / "models" / "cnn_train2.pt")
    p.add_argument("--stage-b", type=Path, default=REPO_ROOT / "models" / "cnn_train23.pt")
    p.add_argument("--out", type=Path,
                   default=REPO_ROOT / "reports" / "figures" / "extrapolation.png")
    args = p.parse_args()

    for path in (args.stage_a, args.stage_b):
        if not path.exists():
            print(f"checkpoint missing: {path}", file=sys.stderr)
            return 2

    A = load(args.stage_a)   # trained on 2D
    B = load(args.stage_b)   # trained on 2D + 3D

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # -- panel (0,0): d=2, Stage A in-training --
    ax = axes[0, 0]
    Ts, P = curve(A, 2)
    plot_curve(ax, Ts, P, 2, label="trained on 2D", color="C0")
    ax.set_title(f"2D  (in training)\nnetwork T_c = {tc_estimate(A, 2):.3f}  "
                 f"(literature {T_C[2]:.3f})")
    ax.legend(fontsize=8, loc="lower right")

    # -- panel (0,1): d=3, Stage A held-out vs Stage B in-training --
    ax = axes[0, 1]
    Ts, P = curve(A, 3)
    plot_curve(ax, Ts, P, 3, label="trained on 2D (held-out 3D)", color="C1")
    Ts, P = curve(B, 3)
    ax.plot(Ts, P, "s-", ms=3, color="C2", label="trained on 2D+3D")
    ax.set_title(f"3D\nheld-out T_c = {tc_estimate(A, 3):.3f}  ->  "
                 f"in-training {tc_estimate(B, 3):.3f}  (lit {T_C[3]:.3f})")
    ax.legend(fontsize=8, loc="lower right")

    # -- panel (1,0): d=4, the headline -- Stage A vs Stage B, both held-out --
    ax = axes[1, 0]
    Ts, P = curve(A, 4)
    plot_curve(ax, Ts, P, 4, label="trained on 2D (held-out 4D)", color="C1")
    Ts, P = curve(B, 4)
    ax.plot(Ts, P, "s-", ms=3, color="C3", label="trained on 2D+3D (held-out 4D)")
    ax.set_title(f"4D -- HELD OUT (the upper critical dimension)\n"
                 f"network T_c: {tc_estimate(A, 4):.3f}  ->  "
                 f"{tc_estimate(B, 4):.3f}   (literature {T_C[4]:.3f})")
    ax.legend(fontsize=8, loc="lower right")

    # -- panel (1,1): predicted vs literature T_c --
    ax = axes[1, 1]
    points = [
        ("2D", "2D", T_C[2], tc_estimate(A, 2), False),
        ("2D", "3D", T_C[3], tc_estimate(A, 3), True),
        ("2D", "4D", T_C[4], tc_estimate(A, 4), True),
        ("2D+3D", "3D", T_C[3], tc_estimate(B, 3), False),
        ("2D+3D", "4D", T_C[4], tc_estimate(B, 4), True),
    ]
    lo = min(min(T_C[d] for d in (2, 3, 4)),
             min(est for *_, est, _ in points if est is not None)) - 0.4
    hi = max(T_C[4], max(est for *_, est, _ in points if est is not None)) + 0.4
    ax.plot([lo, hi], [lo, hi], "k-", lw=0.8, label="perfect extrapolation")
    for train, evald, lit, est, held in points:
        if est is None:
            continue
        marker = "*" if held else "o"
        color = "C3" if held else "C2"
        size = 220 if held else 90
        ax.scatter([lit], [est], marker=marker, s=size, color=color,
                   edgecolor="k", zorder=3,
                   label=None)
        ax.annotate(f"{evald}\ntrain {train}", (lit, est),
                    textcoords="offset points", xytext=(8, -4), fontsize=7)
    ax.set_xlabel("literature T_c")
    ax.set_ylabel("network T_c estimate")
    ax.set_title("predicted vs literature T_c\n"
                 "star = held-out dimension, circle = in-training")
    ax.legend(fontsize=8, loc="upper left")

    fig.suptitle("Cross-dimensional extrapolation of the Ising transition\n"
                 "a 22K-parameter dimension-agnostic CNN, trained only on "
                 "lower dimensions", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130)
    print(f"figure -> {args.out}")

    # Console summary.
    print("\nheld-out T_c extrapolation:")
    for train, evald, lit, est, held in points:
        if held and est is not None:
            err = abs(est - lit) / lit * 100
            print(f"  trained {train:6s} -> {evald}:  "
                  f"T_c {est:.4f}  (lit {lit:.4f}, {err:.1f}% off)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
