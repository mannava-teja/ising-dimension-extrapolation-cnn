"""Same nu extraction as measure_exponents.py, but over every checkpoint
matching a glob, reporting mean +/- std per dimension.

    python scripts/multiseed_nu_aggregate.py --pattern "cnn_train23_seed*.pt" --dims 2 3 4

Imports the fit helpers from measure_exponents so the method stays in one
place.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from ising.datasets import IsingDataset              # noqa: E402
from ising.training import evaluate_per_block        # noqa: E402
# Single source of truth for these -- reuse the validated measurement #2 code.
from measure_exponents import (crossover_width, load_model,   # noqa: E402
                               NU_LIT, DIM_LABEL)


def widths_for_checkpoint(model, dims, max_per_block, device):
    """Return {dim: [(L, width), ...]} for one model (forward passes only)."""
    results = {}
    for d in dims:
        ds = IsingDataset([d], task="classify", split="all", augment=False,
                          max_per_block=max_per_block)
        rows = evaluate_per_block(model, ds, task="classify", device=device)
        by_L = {}
        for r in rows:
            by_L.setdefault(r["L"], []).append((r["T"], r["p_disordered"]))
        widths = []
        for L in sorted(by_L):
            Ts, P = zip(*sorted(by_L[L]))
            w = crossover_width(Ts, P)
            if w is not None and w > 0:
                widths.append((L, w))
        results[d] = widths
    return results


def fit_naive_nu(results, usable):
    """width ~ L^(-1/nu); returns {dim: nu}."""
    out = {}
    for d in usable:
        Ls = np.array([L for L, _ in results[d]], float)
        ws = np.array([w for _, w in results[d]], float)
        slope, _ = np.polyfit(np.log(Ls), np.log(ws), 1)
        out[d] = -1.0 / slope if slope < 0 else float("inf")
    return out


def fit_floor_corrected_nu(results, usable):
    """width = a L^(-1/nu) + c, with a single shared c per checkpoint found by
    minimising the total log-residual (identical scan to measure_exponents.py).
    Returns (floor_c, {dim: nu})."""
    best = None
    min_w = min(w for d in usable for _, w in results[d])
    for c in np.linspace(0.0, 0.95 * min_w, 200):
        total, nus = 0.0, {}
        for d in usable:
            Ls = np.array([L for L, _ in results[d]], float)
            ws = np.array([w for _, w in results[d]], float)
            y = ws - c
            if np.any(y <= 0):
                total = float("inf")
                break
            slope, intercept = np.polyfit(np.log(Ls), np.log(y), 1)
            total += np.sum((np.log(y) - (slope * np.log(Ls) + intercept)) ** 2)
            nus[d] = -1.0 / slope if slope < 0 else float("inf")
        if best is None or total < best[0]:
            best = (total, c, nus)
    return best[1], best[2]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pattern", default="cnn_train23_seed*.pt",
                   help="Glob (relative to models/) selecting seed checkpoints.")
    p.add_argument("--dims", type=int, nargs="+", default=[2, 3, 4],
                   help="Dimensions to extract nu for. d=5 often has too few "
                        "usable lattice sizes (transfer horizon) to fit.")
    p.add_argument("--max-per-block", type=int, default=400)
    return p.parse_args()


def main():
    args = parse_args()
    paths = sorted((REPO_ROOT / "models").glob(args.pattern))
    if not paths:
        print(f"No checkpoints match models/{args.pattern}", file=sys.stderr)
        return 2
    device = "cuda" if torch.cuda.is_available() else "cpu"

    naive_by_dim = {d: [] for d in args.dims}
    corr_by_dim = {d: [] for d in args.dims}
    floors = []
    seeds = []

    print(f"Aggregating nu over {len(paths)} checkpoints (device {device}):")
    for p in paths:
        model, ck = load_model(p)
        seed = ck.get("seed", -1)
        seeds.append(seed)
        results = widths_for_checkpoint(model, args.dims, args.max_per_block, device)
        usable = [d for d in args.dims if len(results[d]) >= 2]
        if len(usable) < 1:
            print(f"  {p.name}: no dimension has >=2 usable lattice sizes; skipped")
            continue
        naive = fit_naive_nu(results, usable)
        floor_c, corr = fit_floor_corrected_nu(results, usable)
        floors.append(floor_c)
        usable_str = []
        for d in usable:
            naive_by_dim[d].append(naive[d])
            corr_by_dim[d].append(corr[d])
            usable_str.append(f"d{d}: naive {naive[d]:.3f} / floor {corr[d]:.3f}")
        n_unusable = {d: len(results[d]) for d in args.dims if d not in usable}
        extra = f"  (unusable: {n_unusable})" if n_unusable else ""
        print(f"  {p.name:28s} seed={seed:>4} c={floor_c:.4f}  "
              f"{' | '.join(usable_str)}{extra}")

    def stats(vals):
        a = np.array(vals, float)
        if len(a) == 0:
            return None
        return a.mean(), (a.std(ddof=1) if len(a) > 1 else 0.0), len(a)

    print()
    print("=" * 78)
    print(f"AGGREGATED nu  (mean +/- std over {len(paths)} seeds)")
    print("=" * 78)
    if floors:
        fa = np.array(floors)
        print(f"  resolution floor c = {fa.mean():.4f} +/- "
              f"{fa.std(ddof=1) if len(fa) > 1 else 0.0:.4f}  (per-seed fit)")
    print(f"  {'dim':>4}  {'naive nu':>22}  {'floor-corrected nu':>24}  "
          f"{'literature':>10}")
    for d in args.dims:
        sn = stats(naive_by_dim[d])
        sc = stats(corr_by_dim[d])
        if sn is None:
            print(f"  {d:>3}D  (no usable fits)")
            continue
        lit = NU_LIT.get(d, float('nan'))
        held = "  [HELD OUT]" if d >= 4 else ""
        print(f"  {d:>3}D  {sn[0]:8.3f} +/- {sn[1]:6.3f} (n={sn[2]})  "
              f"{sc[0]:10.3f} +/- {sc[1]:6.3f} (n={sc[2]})  {lit:10.4f}{held}")
    print()
    print("Note: the qualitative trend (nu decreasing with d toward the mean-field"
          "\n1/2) is the defensible claim; the floor-corrected absolute values are"
          "\na presentation aid, not a precision result (see RESULTS.md #2).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
