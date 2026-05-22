"""Quantify the decision-axis rotation rate (measurement #4, bonus).

Measurement #4 (scripts/latent_analysis.py) found that the ordered->disordered
"decision axis" in the network's 64-d feature space rotates smoothly with
spatial dimension: cos(2D,3D)=0.82, cos(3D,4D)=0.70, cos(2D,4D)=0.37. That
script reports the three raw cosines. This script turns them into an explicit
model: the angle theta between two dimensions' decision axes grows roughly
linearly with the dimension gap |Delta d|, so a single "rotation rate"
(degrees per dimension) summarises the whole transfer mechanism -- and, run
over multiple seeds, comes with an error bar.

Method (per checkpoint, identical feature pipeline to latent_analysis.py):
  - extract pooled 64-d features for dims 2,3,4 at one representative L each;
  - standardise features jointly, then in each dimension's cluster take the
    direction from the ordered centroid to the disordered centroid (its
    decision axis);
  - cosines -> angles theta_ij = arccos(cos_ij);
  - fit theta = rate * |Delta d| through the origin (theta=0 at Delta d=0);
    `rate` is the rotation rate in degrees per dimension.

Aggregated across seed checkpoints: cosines, angles, and rate as mean +/- std.

Usage:
    python scripts/rotation_rate.py
    python scripts/rotation_rate.py --pattern "cnn_train23_seed*.pt" --dims 2 3 4
"""

from __future__ import annotations

import argparse
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

# Reuse the validated measurement #4 feature pipeline (single source of truth).
from latent_analysis import load_model, extract_features, REP_L   # noqa: E402


def decision_axes(model, dims, n_per_block, device):
    """Return {dim: unit decision-axis vector} using the same jointly-
    standardised feature space as latent_analysis.py."""
    feats, treds, labels = [], [], []
    for d in dims:
        F, t = extract_features(model, d, REP_L[d], n_per_block, device)
        feats.append(F)
        treds.append(t)
        labels.append(np.full(len(F), d))
    F = np.concatenate(feats)
    t = np.concatenate(treds)
    lab = np.concatenate(labels)
    Fz = (F - F.mean(0)) / (F.std(0) + 1e-8)
    ordered = t < 0.8
    disordered = t > 1.25
    axes = {}
    for d in dims:
        md = lab == d
        v = Fz[md & disordered].mean(0) - Fz[md & ordered].mean(0)
        axes[d] = v / (np.linalg.norm(v) + 1e-12)
    return axes


def fit_rotation_rate(pairs):
    """pairs: list of (delta_d, theta_deg). Fit theta = rate * delta_d through
    the origin via least squares. Returns rate (deg per dimension)."""
    dd = np.array([p[0] for p in pairs], float)
    th = np.array([p[1] for p in pairs], float)
    # least-squares slope through origin: rate = sum(dd*th)/sum(dd^2)
    return float(np.sum(dd * th) / np.sum(dd * dd))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pattern", default="cnn_train23_seed*.pt",
                   help="Glob (relative to models/) selecting seed checkpoints.")
    p.add_argument("--dims", type=int, nargs="+", default=[2, 3, 4],
                   help="Dimensions whose decision axes are compared.")
    p.add_argument("--n-per-block", type=int, default=40)
    return p.parse_args()


def main():
    import torch
    args = parse_args()
    paths = sorted((REPO_ROOT / "models").glob(args.pattern))
    if not paths:
        print(f"No checkpoints match models/{args.pattern}", file=sys.stderr)
        return 2
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dim_pairs = list(combinations(sorted(args.dims), 2))
    cos_by_pair = {pair: [] for pair in dim_pairs}
    rates = []
    seeds = []

    print(f"Computing decision-axis rotation over {len(paths)} checkpoints "
          f"(device {device}):")
    for p in paths:
        model, ck = load_model(p)
        seed = ck.get("seed", -1)
        seeds.append(seed)
        axes = decision_axes(model, args.dims, args.n_per_block, device)
        pair_strs = []
        rate_pairs = []
        for (di, dj) in dim_pairs:
            c = float(np.clip(axes[di] @ axes[dj], -1.0, 1.0))
            cos_by_pair[(di, dj)].append(c)
            theta = np.degrees(np.arccos(c))
            rate_pairs.append((abs(dj - di), theta))
            pair_strs.append(f"cos({di},{dj})={c:.3f}")
        rate = fit_rotation_rate(rate_pairs)
        rates.append(rate)
        print(f"  {p.name:28s} seed={seed:>4}  {'  '.join(pair_strs)}  "
              f"rate={rate:.1f} deg/dim")

    def stats(vals):
        a = np.array(vals, float)
        return a.mean(), (a.std(ddof=1) if len(a) > 1 else 0.0), len(a)

    print()
    print("=" * 70)
    print(f"AGGREGATED decision-axis rotation  (mean +/- std over {len(paths)} seeds)")
    print("=" * 70)
    for (di, dj) in dim_pairs:
        m, s, n = stats(cos_by_pair[(di, dj)])
        theta = np.degrees(np.arccos(np.clip(m, -1, 1)))
        print(f"  cos(decision axis {di}D, {dj}D) = {m:.3f} +/- {s:.3f}   "
              f"(theta ~ {theta:.1f} deg, |dd|={abs(dj-di)})")
    rm, rs, rn = stats(rates)
    print(f"\n  rotation rate = {rm:.1f} +/- {rs:.1f} degrees per dimension"
          f"  (theta = rate * |delta d|, through origin)")
    print("\nInterpretation: a near-constant degrees-per-dimension rotation means"
          "\nthe shared classifier head reads the decision axis well only while the"
          "\naccumulated rotation stays small -- which is the transfer horizon that"
          "\nbreaks the d=5 classifier (measurement #3).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
