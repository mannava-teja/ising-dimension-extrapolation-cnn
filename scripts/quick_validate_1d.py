"""Quick sanity check for 1D Ising data: <E>/N should match -tanh(1/T).

For each (L, T) block, compute the mean energy per spin from the stored samples
and compare to the analytic value. Also flags any block whose mean |M|/N is
much larger than the short-range-correlated expectation sqrt(xi/L), where
xi = -1/ln(tanh(1/T)) is the 1D correlation length. (Naive 1/sqrt(L) is wrong
at low T, where xi grows large and so do typical |M| fluctuations even though
there is no spontaneous magnetization in the L -> infinity limit.)

Usage:
    python scripts/quick_validate_1d.py data/ising_1d.h5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ising.storage import iter_blocks  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("h5_path", type=Path, help="Path to ising_1d.h5")
    p.add_argument("--tol", type=float, default=0.02,
                   help="Tolerance on |<E>/N - (-tanh(1/T))|. Default 0.02.")
    p.add_argument("--mag-sigma", type=float, default=5.0,
                   help="Flag |M|/N exceeding this many sqrt(xi/L) units, with a "
                        "1/sqrt(L) floor at high T. Default 5.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not args.h5_path.exists():
        print(f"File not found: {args.h5_path}", file=sys.stderr)
        return 2

    print(f"{'L':>5} {'T':>8} {'<E>/N':>10} {'expected':>10} {'diff':>10} "
          f"{'|M|/N':>9} {'status':>10}")
    print("-" * 70)

    n_fail = 0
    n_total = 0
    for block in iter_blocks(args.h5_path, dim=1):
        L = block["L"]
        T = block["T"]
        e_per_spin = float(block["energies"].mean()) / L
        expected = -np.tanh(1.0 / T)
        diff = e_per_spin - expected

        abs_m_per_spin = float(np.abs(block["magnetizations"]).mean()) / L
        # 1D Ising correlation length; tanh(1/T) -> 1 as T -> 0 so xi diverges.
        u = np.tanh(1.0 / T)
        xi = -1.0 / np.log(u) if u < 1.0 else float("inf")
        # Expected |M|/N from short-range correlations: sqrt(xi/L) (capped at 1).
        # Use a 1/sqrt(L) floor at high T where xi < 1.
        mag_scale = min(1.0, max(np.sqrt(xi / L), 1.0 / np.sqrt(L)))
        mag_threshold = args.mag_sigma * mag_scale

        problems = []
        if abs(diff) > args.tol:
            problems.append("ENERGY")
        if abs_m_per_spin > mag_threshold:
            problems.append("MAG")
        status = ",".join(problems) if problems else "ok"

        print(f"{L:>5d} {T:>8.4f} {e_per_spin:>+10.4f} {expected:>+10.4f} "
              f"{diff:>+10.4f} {abs_m_per_spin:>9.4f} {status:>10}")

        n_total += 1
        if problems:
            n_fail += 1

    print("-" * 70)
    if n_total == 0:
        print("No dim_1 blocks found.")
        return 1
    print(f"{n_total - n_fail}/{n_total} blocks within tolerance.")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
