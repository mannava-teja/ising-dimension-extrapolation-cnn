"""Generate 5D Ising MC data -- a second held-out test set.

5D is above the upper critical dimension: mean-field exponents, no
logarithmic corrections. T_c(5D) ~ 8.778. Never used to train the CNN.

Usage:
    python scripts/generate_5d.py --out data/ising_5d.h5
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ising.metropolis_5d import simulate_5d_metropolis  # noqa: E402
from ising.wolff_5d import simulate_5d_wolff             # noqa: E402
from ising.storage import write_samples                  # noqa: E402

T_C_5D = 8.778  # consensus value; Lundow & Markstrom


def default_temps() -> np.ndarray:
    pieces = [
        np.linspace(5.5, 7.5, 4, endpoint=False),
        np.linspace(7.5, 8.2, 5, endpoint=False),
        np.linspace(8.2, 9.4, 14, endpoint=False),   # critical region, dense
        np.linspace(9.4, 10.5, 5, endpoint=False),
        np.linspace(10.5, 13.0, 5),
    ]
    return np.unique(np.round(np.concatenate(pieces), 4))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=REPO_ROOT / "data" / "ising_5d.h5")
    p.add_argument("--algorithm", choices=("wolff", "metropolis"), default="wolff")
    p.add_argument("--sizes", type=int, nargs="+", default=[4, 6, 8])
    p.add_argument("--temps", type=float, nargs="+", default=None)
    p.add_argument("--n-samples", type=int, default=1000)
    p.add_argument("--n-thermalization", type=int, default=None)
    p.add_argument("--decorrelation", type=int, default=None)
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
        sim_fn = simulate_5d_wolff
    else:
        n_therm = args.n_thermalization or 2_000
        decorr = args.decorrelation or 5
        sim_fn = simulate_5d_metropolis

    def block_seed(L: int, T: float) -> int:
        ss = np.random.SeedSequence([args.seed, int(L), int(round(T * 1_000_000))])
        return int(np.random.default_rng(ss).integers(1, 2**31 - 1))

    total = len(args.sizes) * len(temps)
    done = 0
    t_start = time.time()
    print(f"Generating {total} blocks ({args.algorithm}, T_c={T_C_5D:.4f}) -> {args.out}")

    for L in args.sizes:
        for T in temps:
            seed = block_seed(L, float(T))
            t0 = time.time()
            result = sim_fn(L=L, T=float(T), n_samples=args.n_samples,
                            n_thermalization=n_therm, decorrelation=decorr,
                            seed=seed)
            write_samples(args.out, dim=5, L=L, result=result,
                          overwrite=args.overwrite)
            done += 1
            N = L ** 5
            e = result.energies.mean() / N
            m = float(np.abs(result.magnetizations).mean()) / N
            print(f"  [{done:3d}/{total}] L={L:2d} T={T:7.4f}  "
                  f"<E>/N={e:+.4f}  <|M|>/N={m:.4f}  ({time.time()-t0:.1f}s)")

    print(f"Done in {time.time() - t_start:.1f}s.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
