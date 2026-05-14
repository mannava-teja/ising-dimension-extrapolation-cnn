"""2D Ising Metropolis-Hastings on an LxL torus.

H = -J * sum_<i,j> s_i s_j    (J = 1, k_B = 1, periodic boundaries)

A spin flip at (i, j) changes the energy by 2 * s_{i,j} * (sum of 4 neighbors),
which lives in {-8, -4, 0, 4, 8}. We tabulate the Metropolis acceptance
probability for those five values up front.

This is intentionally simple and self-contained: production 2D runs should use
the Wolff cluster algorithm (see wolff_2d.py) because single-spin-flip
Metropolis suffers from critical slowing down near T_c (tau ~ L^2.17). This
file exists for algorithm-vs-algorithm cross-checks and as a high-T sanity
reference.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit


@dataclass(frozen=True)
class Sim2DResult:
    configurations: np.ndarray  # (n_samples, L, L) int8 in {-1, +1}
    energies: np.ndarray        # (n_samples,) float64, total energy
    magnetizations: np.ndarray  # (n_samples,) float64, total signed magnetization
    T: float
    L: int
    seed: int
    n_thermalization: int
    decorrelation: int
    algorithm: str = "metropolis"


@njit(cache=True, fastmath=False)
def _total_energy_2d(spins: np.ndarray) -> float:
    """Total energy with each bond counted once (right + down neighbors only)."""
    L = spins.shape[0]
    e = 0.0
    for i in range(L):
        ip = i + 1 if i + 1 < L else 0
        for j in range(L):
            jp = j + 1 if j + 1 < L else 0
            e -= spins[i, j] * spins[ip, j]
            e -= spins[i, j] * spins[i, jp]
    return e


@njit(cache=True, fastmath=False)
def _sweep_2d(spins: np.ndarray, accept: np.ndarray) -> None:
    """One MC sweep = L*L attempted single-spin flips at uniform-random sites."""
    L = spins.shape[0]
    for _ in range(L * L):
        i = np.random.randint(0, L)
        j = np.random.randint(0, L)
        # Four neighbors with PBC.
        up = spins[i - 1 if i > 0 else L - 1, j]
        down = spins[i + 1 if i + 1 < L else 0, j]
        left = spins[i, j - 1 if j > 0 else L - 1]
        right = spins[i, j + 1 if j + 1 < L else 0]
        nbr_sum = up + down + left + right       # in {-4,-2,0,2,4}
        dE = 2 * spins[i, j] * nbr_sum            # in {-8,-4,0,4,8}
        k = (dE + 8) // 4                         # 0..4
        if np.random.random() < accept[k]:
            spins[i, j] = -spins[i, j]


@njit(cache=True, fastmath=False)
def _run_2d(
    L: int,
    T: float,
    n_thermalization: int,
    n_samples: int,
    decorrelation: int,
    seed: int,
):
    np.random.seed(seed)
    spins = np.where(np.random.random((L, L)) < 0.5,
                     np.int8(-1), np.int8(1))

    # Acceptance table indexed by k = (dE + 8)/4 in {0..4}.
    accept = np.empty(5, dtype=np.float64)
    for k in range(5):
        dE = 4 * k - 8
        accept[k] = 1.0 if dE <= 0 else np.exp(-dE / T)

    for _ in range(n_thermalization):
        _sweep_2d(spins, accept)

    configs = np.empty((n_samples, L, L), dtype=np.int8)
    energies = np.empty(n_samples, dtype=np.float64)
    mags = np.empty(n_samples, dtype=np.float64)
    for s in range(n_samples):
        for _ in range(decorrelation):
            _sweep_2d(spins, accept)
        configs[s] = spins
        energies[s] = _total_energy_2d(spins)
        mags[s] = spins.astype(np.float64).sum()
    return configs, energies, mags


def simulate_2d_metropolis(
    L: int,
    T: float,
    n_samples: int,
    *,
    n_thermalization: int = 5_000,
    decorrelation: int = 5,
    seed: int = 0,
) -> Sim2DResult:
    """Run 2D Metropolis and return decorrelated samples plus their observables."""
    if L < 4:
        raise ValueError("L must be >= 4 for sensible PBC.")
    if T <= 0:
        raise ValueError("T must be positive.")
    if n_samples < 1 or n_thermalization < 0 or decorrelation < 1:
        raise ValueError("Bad sampling parameters.")

    configs, energies, mags = _run_2d(
        L, float(T), int(n_thermalization),
        int(n_samples), int(decorrelation), int(seed),
    )
    return Sim2DResult(
        configurations=configs,
        energies=energies,
        magnetizations=mags,
        T=float(T),
        L=int(L),
        seed=int(seed),
        n_thermalization=int(n_thermalization),
        decorrelation=int(decorrelation),
        algorithm="metropolis",
    )
