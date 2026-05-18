"""Generate 4D Ising MC data -- the held-out test set for the extrapolation
experiment.

This dataset is NEVER used to train or tune the CNN. It is generated,
physics-validated against literature, and then sealed.

Default algorithm is Wolff. T_c(4D) ~ 6.6803 (Lundow & Markstrom 2009);
densify the schedule around there for clean Binder crossings.

Usage:
    python scripts/generate_4d.py --out data/ising_4d.h5
    python scripts/generate_4d.py --out data/ising_4d_metro_check.h5 \\
        --algorithm metropolis --sizes 6 --temps 5.5 6.68 8.0 --n-samples 500
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ising.metropolis_4d import simulate_4d_metropolis  # noqa: E402
from ising.wolff_4d import simulate_4d_wolff             # noqa: E402
from ising.storage import write_samples                  # noqa: E402

T_C_4D = 6.6803  # Lundow & Markstrom, Phys. Rev. E 80, 031104 (2009)


def default_temps() -> np.ndarray:
    """Non-uniform grid: dense within ~ +/-15% of T_c, sparser in the wings."""
    pieces = [
        np.linspace(4.0, 5.5, 5, endpoint=False),    # low T, sparse
        np.linspace(5.5, 6.2, 6, endpoint=False),    # below T_c
        np.linspace(6.2, 7.2, 16, endpoint=False),   # critical region, dense
        np.linspace(7.2, 8.5, 7, endpoint=False),    # above T_c
        np.linspace(8.5, 10.5, 6),                   # high T, sparse
    ]
    return np.unique(np.round(np.concatenate(pieces), 4))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=REPO_ROOT / "data" / "ising_4d.h5")
    p.add_argument("--algorithm", choices=("wolff", "metropolis"), default="wolff")
    p.add_argument("--sizes", type=int, nargs="+", default=[4, 6, 8])
    p.add_argument("--temps", type=float, nargs="+", default=None)
    p.add_argument("--n-samples", type=int, default=1000)
    p.add_argument("--n-thermalization", type=int, default=None,
                   help="Default 500 for Wolff (cluster updates) or 2000 for Metropolis (sweeps).")
    p.add_argument("--decorrelation", type=int, default=None,
                   help="Default 10 for Wolff or 5 for Metropolis.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    temps = (np.array(sorted(set(args.temps)), dtype=float)
             if args.temps is not None else default_temps())

    if args.algorithm == "wolff":
        n_therm = args.n_thermalization or 500
        decorr = args.decorrelation or 10
        sim_fn = simulate_4d_wolff
    else:
        n_therm = args.n_thermalization or 2_000
        decorr = args.decorrelation or 5
        sim_fn = simulate_4d_metropolis

    def block_seed(L: int, T: float) -> int:
        ss = np.random.SeedSequence([args.seed, int(L), int(round(T * 1_000_000))])
        return int(np.random.default_rng(ss).integers(1, 2**31 - 1))

    total = len(args.sizes) * len(temps)
    done = 0
    t_start = time.time()
    print(f"Generating {total} blocks ({args.algorithm}, T_c={T_C_4D:.4f}) -> {args.out}")

    for L in args.sizes:
        for T in temps:
            seed = block_seed(L, float(T))
            t0 = time.time()
            result = sim_fn(
                L=L, T=float(T),
                n_samples=args.n_samples,
                n_thermalization=n_therm,
                decorrelation=decorr,
                seed=seed,
            )
            write_samples(args.out, dim=4, L=L, result=result,
                          overwrite=args.overwrite)
            done += 1
            elapsed = time.time() - t0
            N = L ** 4
            e_per_spin = result.energies.mean() / N
            m_per_spin = float(np.abs(result.magnetizations).mean()) / N
            print(f"  [{done:3d}/{total}] L={L:2d} T={T:7.4f}  "
                  f"<E>/N={e_per_spin:+.4f}  <|M|>/N={m_per_spin:.4f}  ({elapsed:.1f}s)")

    print(f"Done in {time.time() - t_start:.1f}s.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
