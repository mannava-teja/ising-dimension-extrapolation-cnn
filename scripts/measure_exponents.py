"""Measurement #2: the correlation-length exponent nu from finite-size scaling
of the network's classification crossover.

On a finite L-lattice the order/disorder transition is smeared over a
temperature window. The width of that window narrows as the lattice grows:

    width(L)  ~  L^(-1/nu)

so a straight-line fit of log(width) vs log(L) has slope -1/nu. The width is
read from the network's P(disordered) curve as the temperature gap between
the P = 0.25 and P = 0.75 crossings. (This is the Carrasquilla-Melko readout:
the network's own output, finite-size-scaled, yields a critical exponent.)

Literature nu: 1 (2D), 0.6301 (3D), 1/2 (4D, mean-field). At d = 4, the upper
critical dimension, finite-size scaling additionally carries multiplicative
logarithmic corrections (arXiv:2408.15230) -- so a plain power-law fit there
returns an *effective* exponent, and any systematic curvature in the 4D
log-log plot is itself the upper-critical-dimension signature (measurement #3).

A trained model is loaded and re-evaluated on every lattice size of each
dimension (forward passes only -- no training).

Usage:
    python scripts/measure_exponents.py --checkpoint models/cnn_train23.pt
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
sys.path.insert(0, str(REPO_ROOT / "src"))

from ising.cnn import IsingCNN                       # noqa: E402
from ising.datasets import IsingDataset              # noqa: E402
from ising.training import evaluate_per_block        # noqa: E402

NU_LIT = {2: 1.0, 3: 0.6301, 4: 0.5}
DIM_LABEL = {2: "2D", 3: "3D", 4: "4D"}


def crossover_width(Ts, P, lo=0.25, hi=0.75):
    """Temperature gap between the P=hi and P=lo crossings (linear interp)."""
    Ts = np.asarray(Ts, float)
    P = np.asarray(P, float)
    order = np.argsort(Ts)
    Ts, P = Ts[order], P[order]

    def crossing(level):
        for k in range(len(Ts) - 1):
            p0, p1 = P[k], P[k + 1]
            if (p0 - level) * (p1 - level) <= 0 and p1 != p0:
                return Ts[k] + (level - p0) * (Ts[k + 1] - Ts[k]) / (p1 - p0)
        return None

    t_lo, t_hi = crossing(lo), crossing(hi)
    if t_lo is None or t_hi is None:
        return None
    return abs(t_hi - t_lo)


def load_model(checkpoint: Path):
    ck = torch.load(checkpoint, weights_only=False)
    cfg = ck["model_config"]
    model = IsingCNN(n_out=cfg["n_out"], channels=tuple(cfg["channels"]))
    model.load_state_dict(ck["model_state"])
    model.eval()
    return model, ck


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path,
                   default=REPO_ROOT / "models" / "cnn_train23.pt")
    p.add_argument("--dims", type=int, nargs="+", default=[2, 3, 4])
    p.add_argument("--max-per-block", type=int, default=400,
                   help="Samples per block for the forward-pass evaluation.")
    p.add_argument("--out", type=Path,
                   default=REPO_ROOT / "reports" / "figures" / "exponent_nu.png")
    return p.parse_args()


def main():
    args = parse_args()
    if not args.checkpoint.exists():
        print(f"checkpoint missing: {args.checkpoint}", file=sys.stderr)
        return 2
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, ck = load_model(args.checkpoint)
    print(f"checkpoint: {args.checkpoint.name}   train dims {ck['train_dims']}   "
          f"device {device}")

    results = {}   # dim -> list of (L, width)
    for d in args.dims:
        print(f"\nevaluating d={d} on all lattice sizes "
              f"(max {args.max_per_block} samples/block) ...", flush=True)
        ds = IsingDataset([d], task="classify", split="all", augment=False,
                          max_per_block=args.max_per_block)
        rows = evaluate_per_block(model, ds, task="classify", device=device)

        by_L: dict[int, list[tuple[float, float]]] = {}
        for r in rows:
            by_L.setdefault(r["L"], []).append((r["T"], r["p_disordered"]))

        widths = []
        for L in sorted(by_L):
            Ts, P = zip(*sorted(by_L[L]))
            w = crossover_width(Ts, P)
            if w is not None and w > 0:
                widths.append((L, w))
                print(f"  L={L:4d}  crossover width = {w:.4f}")
            else:
                print(f"  L={L:4d}  width undefined (curve never spans 0.25-0.75)")
        results[d] = widths

    usable = [d for d in args.dims if len(results[d]) >= 2]

    # ---- naive finite-size-scaling fits:  width ~ L^(-1/nu) ----
    print()
    print("=" * 70)
    print("NAIVE FINITE-SIZE SCALING:  width ~ L^(-1/nu)")
    print("=" * 70)
    naive = {}
    for d in usable:
        Ls = np.array([L for L, _ in results[d]], float)
        ws = np.array([w for _, w in results[d]], float)
        slope, intercept = np.polyfit(np.log(Ls), np.log(ws), 1)
        nu = -1.0 / slope if slope < 0 else float("inf")
        naive[d] = (slope, intercept, nu)
        err = abs(nu - NU_LIT[d]) / NU_LIT[d] * 100
        print(f"  d={d}  nu = {nu:.3f}   (literature {NU_LIT[d]:.4f}, {err:.0f}% off)")

    # ---- resolution-floor-corrected fits:  width = a L^(-1/nu) + c ----
    # The network classifies individual finite samples with a smooth decision
    # function, so its crossover cannot sharpen below an intrinsic floor c that
    # does not shrink with L. We fit ONE shared c across all dimensions (a
    # property of the network, not the physics) by minimising the total
    # log-residual of the per-dimension power-law fits of (width - c).
    print()
    print("=" * 70)
    print("RESOLUTION-FLOOR-CORRECTED FSS:  width = a L^(-1/nu) + c")
    print("=" * 70)
    best = None
    min_w = min(w for d in usable for _, w in results[d])
    for c in np.linspace(0.0, 0.95 * min_w, 200):
        total, nus, fitp = 0.0, {}, {}
        for d in usable:
            Ls = np.array([L for L, _ in results[d]], float)
            ws = np.array([w for _, w in results[d]], float)
            y = ws - c
            slope, intercept = np.polyfit(np.log(Ls), np.log(y), 1)
            total += np.sum((np.log(y) - (slope * np.log(Ls) + intercept)) ** 2)
            nus[d] = -1.0 / slope if slope < 0 else float("inf")
            fitp[d] = (slope, intercept)
        if best is None or total < best[0]:
            best = (total, c, nus, fitp)
    _, floor_c, corr_nu, corr_fit = best
    print(f"  shared resolution floor c = {floor_c:.4f} (temperature units)")
    for d in usable:
        err = abs(corr_nu[d] - NU_LIT[d]) / NU_LIT[d] * 100
        tag = "  [HELD OUT]" if d not in ck["train_dims"] else ""
        note = "  [+ log corrections at d_c=4]" if d == 4 else ""
        print(f"  d={d}  nu = {corr_nu[d]:.3f}   (literature {NU_LIT[d]:.4f}, "
              f"{err:.0f}% off){tag}{note}")

    # ---- figure: naive (left) vs floor-corrected (right) ----
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(14, 6))
    colors = {2: "C0", 3: "C1", 4: "C3"}
    for d in usable:
        Ls = np.array([L for L, _ in results[d]], float)
        ws = np.array([w for _, w in results[d]], float)
        c = colors[d]
        # left: raw width, naive power-law fit
        axL.loglog(Ls, ws, "o", ms=8, color=c)
        slope, intercept, nu = naive[d]
        xs = np.array([Ls.min(), Ls.max()])
        axL.loglog(xs, np.exp(intercept) * xs ** slope, "-", color=c,
                   label=f"{DIM_LABEL[d]}: nu={nu:.2f} (lit {NU_LIT[d]:.2f})")
        # right: width - c, floor-corrected fit
        axR.loglog(Ls, ws - floor_c, "o", ms=8, color=c)
        slope, intercept = corr_fit[d]
        axR.loglog(xs, np.exp(intercept) * xs ** slope, "-", color=c,
                   label=f"{DIM_LABEL[d]}: nu={corr_nu[d]:.2f} (lit {NU_LIT[d]:.2f})")
    axL.set_xlabel("lattice size L")
    axL.set_ylabel("crossover width  (temperature)")
    axL.set_title("Naive fit: width ~ L^(-1/nu)\n"
                  "(curvature -> nu wildly overestimated)")
    axL.legend(); axL.grid(True, which="both", alpha=0.3)
    axR.set_xlabel("lattice size L")
    axR.set_ylabel(f"crossover width - c   (c = {floor_c:.3f})")
    axR.set_title("After subtracting the network resolution floor c\n"
                  "(straightens -> nu tracks the dimensional trend)")
    axR.legend(); axR.grid(True, which="both", alpha=0.3)
    fig.suptitle("Measurement #2 -- correlation-length exponent nu from "
                 "finite-size scaling of the network's classification crossover",
                 fontsize=12)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(args.out, dpi=130)
    print(f"\nfigure -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
