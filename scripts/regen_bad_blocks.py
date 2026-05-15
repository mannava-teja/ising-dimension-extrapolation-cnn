"""Find blocks with low effective sample size and regenerate them with Metropolis.

Some blocks just above T_c still have insufficiently decorrelated Wolff
samples because cluster size is O(xi^2) << L^d in that regime. Metropolis
at ~half acceptance is efficient there. Identifies blocks with n_eff < 200
(based on lag-1 autocorrelation of energy) and regenerates in place.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ising.metropolis_2d import simulate_2d_metropolis
from ising.metropolis_3d import simulate_3d_metropolis
from ising.storage import iter_blocks, write_samples

THRESHOLD = 200  # regen if n_eff < this


def integrated_autocorr(x: np.ndarray, max_lag: int = 200) -> float:
    x = np.asarray(x, np.float64) - np.asarray(x, np.float64).mean()
    var = (x * x).mean()
    if var == 0:
        return 0.5
    rho = np.array([1.0] + [(x[:-k] * x[k:]).mean() / var for k in range(1, max_lag + 1)])
    tau = 0.5 + rho[1:].cumsum()
    M = np.arange(1, max_lag + 1)
    cut = np.where(M >= 5 * tau)[0]
    return float(tau[cut[0]]) if cut.size else float(tau[-1])


def block_seed(base: int, L: int, T: float) -> int:
    ss = np.random.SeedSequence([base, int(L), int(round(T * 1_000_000))])
    return int(np.random.default_rng(ss).integers(1, 2**31 - 1))


def find_bad(path: Path, dim: int) -> list[tuple[int, float]]:
    bad = []
    for b in iter_blocks(path, dim=dim):
        tau = integrated_autocorr(b["energies"])
        n_eff = max(1.0, len(b["energies"]) / max(2.0 * tau, 1.0))
        if n_eff < THRESHOLD:
            bad.append((b["L"], float(b["T"])))
    return bad


def main():
    for dim, h5path in [(2, REPO_ROOT / "data" / "ising_2d.h5"),
                        (3, REPO_ROOT / "data" / "ising_3d.h5")]:
        bad = find_bad(h5path, dim)
        if not bad:
            print(f"dim={dim}: no blocks with n_eff < {THRESHOLD}. Skipping.")
            continue
        print(f"dim={dim}: {len(bad)} blocks below threshold; regenerating with Metropolis.")

        sim_fn = simulate_2d_metropolis if dim == 2 else simulate_3d_metropolis
        n_therm = 5_000 if dim == 2 else 2_000
        decorr = 10
        for i, (L, T) in enumerate(sorted(bad)):
            seed = block_seed(42, L, T)
            print(f"  [{i+1:2d}/{len(bad)}] L={L:4d} T={T:.4f}", end="  ", flush=True)
            result = sim_fn(L=L, T=T, n_samples=1000,
                            n_thermalization=n_therm,
                            decorrelation=decorr, seed=seed)
            write_samples(h5path, dim=dim, L=L, result=result, overwrite=True)
            print(f"<E>/N={result.energies.mean()/L**dim:+.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
