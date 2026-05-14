"""Generate 1D Ising Metropolis data across lattice sizes and temperatures.

Usage:
    python scripts/generate_1d.py --out data/ising_1d.h5

Defaults follow Phase 1 of the project plan: L in {256, 512, 1024}, 30 temperatures
roughly spanning 0.5 .. 5.0 (1D has T_c = 0, so no special densification near T_c).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# Make `src/` importable without installing the package.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ising.metropolis_1d import simulate_1d  # noqa: E402
from ising.storage import write_samples       # noqa: E402


def adaptive_schedule(T: float, base_decorr: int, base_therm: int) -> tuple[int, int]:
    """Return (decorrelation, n_thermalization) scaled by tau_int ~ xi^2.

    1D Ising correlation length: xi = -1/ln(tanh(beta)). Single-spin-flip
    Metropolis has integrated autocorrelation time tau_int proportional to
    xi^2 (Family-Vicsek / standard random-walk-of-domain-walls argument).
    Use 2*xi^2 between samples and 20*xi^2 thermalization sweeps.
    """
    u = np.tanh(1.0 / T)
    xi = -1.0 / np.log(u) if u < 1.0 else 1.0
    tau = max(1.0, xi * xi)
    decorr = max(base_decorr, int(np.ceil(2.0 * tau)))
    therm = max(base_therm, int(np.ceil(20.0 * tau)))
    return decorr, therm


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=REPO_ROOT / "data" / "ising_1d.h5",
                   help="Output HDF5 path.")
    p.add_argument("--sizes", type=int, nargs="+", default=[256, 512, 1024],
                   help="Lattice sizes L to simulate.")
    p.add_argument("--temps", type=float, nargs="+", default=None,
                   help="Explicit list of temperatures. If omitted, uses --t-min/--t-max/--n-temps.")
    p.add_argument("--t-min", type=float, default=0.5)
    p.add_argument("--t-max", type=float, default=5.0)
    p.add_argument("--n-temps", type=int, default=30)
    p.add_argument("--n-samples", type=int, default=1000,
                   help="Independent configurations to keep per (L, T).")
    p.add_argument("--n-thermalization", type=int, default=10_000,
                   help="Floor on thermalization sweeps. Actual value is "
                        "max(this, 20 * xi^2) per (L, T) with --adaptive.")
    p.add_argument("--decorrelation", type=int, default=10,
                   help="Floor on sweeps between kept samples. Actual value is "
                        "max(this, 2 * xi^2) per (L, T) with --adaptive.")
    p.add_argument("--adaptive", action=argparse.BooleanOptionalAction, default=True,
                   help="Scale thermalization and decorrelation with the integrated "
                        "autocorrelation time tau_int ~ xi^2 of single-spin-flip "
                        "Metropolis. On by default.")
    p.add_argument("--seed", type=int, default=42,
                   help="Base seed; per-(L, T) seeds are derived deterministically.")
    p.add_argument("--overwrite", action="store_true",
                   help="Replace existing groups in the HDF5 file.")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if args.temps is not None:
        temps = sorted(set(args.temps))
    else:
        temps = list(np.linspace(args.t_min, args.t_max, args.n_temps))

    # Derive a unique seed per (L, T) so reruns are reproducible regardless of
    # iteration order.
    rng = np.random.default_rng(args.seed)
    seed_map = {(L, float(f"{T:.6f}")): int(rng.integers(1, 2**31 - 1))
                for L in args.sizes for T in temps}

    total = len(args.sizes) * len(temps)
    done = 0
    t_start = time.time()
    print(f"Generating {total} blocks -> {args.out}")

    for L in args.sizes:
        for T in temps:
            seed = seed_map[(L, float(f"{T:.6f}"))]
            if args.adaptive:
                decorr, therm = adaptive_schedule(
                    float(T), args.decorrelation, args.n_thermalization
                )
            else:
                decorr, therm = args.decorrelation, args.n_thermalization
            t0 = time.time()
            result = simulate_1d(
                L=L, T=float(T),
                n_samples=args.n_samples,
                n_thermalization=therm,
                decorrelation=decorr,
                seed=seed,
            )
            write_samples(args.out, dim=1, L=L, result=result,
                          algorithm="metropolis", overwrite=args.overwrite)
            done += 1
            elapsed = time.time() - t0
            mean_e_per_spin = result.energies.mean() / L
            print(f"  [{done:3d}/{total}] L={L:4d} T={T:6.4f}  "
                  f"therm={therm:6d} decorr={decorr:5d}  "
                  f"<E>/N={mean_e_per_spin:+.4f}  ({elapsed:.1f}s)")

    print(f"Done in {time.time() - t_start:.1f}s.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
