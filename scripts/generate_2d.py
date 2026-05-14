"""Generate 2D Ising MC data across lattice sizes and temperatures.

Default algorithm is Wolff (cluster updates), the right choice near T_c where
single-spin-flip Metropolis suffers critical slowing down. Use --algorithm
metropolis for cross-checks.

Usage:
    python scripts/generate_2d.py --out data/ising_2d.h5
    python scripts/generate_2d.py --out data/ising_2d_metro_check.h5 --algorithm metropolis \\
        --sizes 32 --temps 2.0 2.5 --n-samples 500
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ising.metropolis_2d import simulate_2d_metropolis  # noqa: E402
from ising.wolff_2d import simulate_2d_wolff             # noqa: E402
from ising.storage import write_samples                  # noqa: E402

T_C_2D = 2.0 / np.log(1.0 + np.sqrt(2.0))  # 2.2691853...


def default_temps() -> np.ndarray:
    """Non-uniform grid: dense within +/- 20% of T_c, sparser in the wings.

    ~40 unique points spanning T = 0.5 .. 5.0, with the densest sampling around
    T_c so that Binder cumulant crossings can pin down T_c cleanly.
    """
    pieces = [
        np.linspace(0.5, 1.5, 6, endpoint=False),   # low T, sparse
        np.linspace(1.5, 2.0, 8, endpoint=False),   # below T_c
        np.linspace(2.0, 2.55, 14, endpoint=False), # critical region, dense
        np.linspace(2.55, 3.2, 7, endpoint=False),  # above T_c
        np.linspace(3.2, 5.0, 5),                   # high T, sparse
    ]
    return np.unique(np.round(np.concatenate(pieces), 4))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=REPO_ROOT / "data" / "ising_2d.h5")
    p.add_argument("--algorithm", choices=("wolff", "metropolis"), default="wolff")
    p.add_argument("--sizes", type=int, nargs="+", default=[16, 32, 64, 128])
    p.add_argument("--temps", type=float, nargs="+", default=None,
                   help="Explicit temperatures (overrides the default schedule).")
    p.add_argument("--n-samples", type=int, default=1000)
    p.add_argument("--n-thermalization", type=int, default=None,
                   help="Default 200 for Wolff (cluster updates) or 5000 for Metropolis (sweeps).")
    p.add_argument("--decorrelation", type=int, default=None,
                   help="Default 5 for Wolff or 10 for Metropolis.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    temps = (np.array(sorted(set(args.temps)), dtype=float)
             if args.temps is not None else default_temps())

    if args.algorithm == "wolff":
        # 2000 cluster updates of thermalization is enough at low T (each flips
        # ~N spins) and also covers high T (where clusters are O(1) and we need
        # ~N/<|c|> updates per effective sweep). decorrelation=20 is similarly
        # safe across the whole T range.
        n_therm = args.n_thermalization or 2_000
        decorr = args.decorrelation or 20
        sim_fn = simulate_2d_wolff
    else:
        n_therm = args.n_thermalization or 5_000
        decorr = args.decorrelation or 10
        sim_fn = simulate_2d_metropolis

    def block_seed(L: int, T: float) -> int:
        # Order-independent: same (base, L, T) -> same seed, so regenerating a
        # subset of blocks does not collide with seeds in the original run.
        ss = np.random.SeedSequence([args.seed, int(L), int(round(T * 1_000_000))])
        return int(np.random.default_rng(ss).integers(1, 2**31 - 1))

    total = len(args.sizes) * len(temps)
    done = 0
    t_start = time.time()
    print(f"Generating {total} blocks ({args.algorithm}, T_c={T_C_2D:.4f}) -> {args.out}")

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
            write_samples(args.out, dim=2, L=L, result=result,
                          overwrite=args.overwrite)
            done += 1
            elapsed = time.time() - t0
            e_per_spin = result.energies.mean() / (L * L)
            m_per_spin = float(np.abs(result.magnetizations).mean()) / (L * L)
            print(f"  [{done:3d}/{total}] L={L:4d} T={T:6.4f}  "
                  f"<E>/N={e_per_spin:+.4f}  <|M|>/N={m_per_spin:.4f}  ({elapsed:.1f}s)")

    print(f"Done in {time.time() - t_start:.1f}s.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
