"""3D Ising Metropolis-Hastings on an LxLxL torus.

H = -J * sum_<i,j> s_i s_j    (J = 1, k_B = 1, periodic boundaries)

A spin flip at (i, j, k) changes the energy by 2 * s_{ijk} * (sum of 6 neighbors),
which lives in {-12, -8, -4, 0, 4, 8, 12}. We tabulate the 7-entry Metropolis
acceptance table up front.

As in 2D, production runs should use Wolff (see wolff_3d.py) -- single-spin-flip
Metropolis suffers critical slowing down near T_c. This file exists for
algorithm-vs-algorithm cross-checks and as a sanity reference.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit


@dataclass(frozen=True)
class Sim3DResult:
    configurations: np.ndarray  # (n_samples, L, L, L) int8 in {-1, +1}
    energies: np.ndarray        # (n_samples,) float64, total energy
    magnetizations: np.ndarray  # (n_samples,) float64, total signed magnetization
    T: float
    L: int
    seed: int
    n_thermalization: int
    decorrelation: int
    algorithm: str = "metropolis"


@njit(cache=True, fastmath=False)
def _total_energy_3d(spins: np.ndarray) -> float:
    """Total energy with each bond counted once (forward neighbors only)."""
    L = spins.shape[0]
    e = 0.0
    for i in range(L):
        ip = i + 1 if i + 1 < L else 0
        for j in range(L):
            jp = j + 1 if j + 1 < L else 0
            for k in range(L):
                kp = k + 1 if k + 1 < L else 0
                s = spins[i, j, k]
                e -= s * spins[ip, j, k]
                e -= s * spins[i, jp, k]
                e -= s * spins[i, j, kp]
    return e


@njit(cache=True, fastmath=False)
def _sweep_3d(spins: np.ndarray, accept: np.ndarray) -> None:
    """One MC sweep = L^3 attempted single-spin flips at uniform-random sites."""
    L = spins.shape[0]
    for _ in range(L * L * L):
        i = np.random.randint(0, L)
        j = np.random.randint(0, L)
        k = np.random.randint(0, L)
        nbr_sum = (
            spins[i - 1 if i > 0 else L - 1, j, k]
            + spins[i + 1 if i + 1 < L else 0, j, k]
            + spins[i, j - 1 if j > 0 else L - 1, k]
            + spins[i, j + 1 if j + 1 < L else 0, k]
            + spins[i, j, k - 1 if k > 0 else L - 1]
            + spins[i, j, k + 1 if k + 1 < L else 0]
        )
        # dE = 2 * s_{ijk} * nbr_sum, in {-12,-8,-4,0,4,8,12}
        dE = 2 * spins[i, j, k] * nbr_sum
        idx = (dE + 12) // 4         # 0..6
        if np.random.random() < accept[idx]:
            spins[i, j, k] = -spins[i, j, k]


@njit(cache=True, fastmath=False)
def _run_3d(
    L: int,
    T: float,
    n_thermalization: int,
    n_samples: int,
    decorrelation: int,
    seed: int,
):
    np.random.seed(seed)
    spins = np.where(np.random.random((L, L, L)) < 0.5,
                     np.int8(-1), np.int8(1))

    accept = np.empty(7, dtype=np.float64)
    for idx in range(7):
        dE = 4 * idx - 12
        accept[idx] = 1.0 if dE <= 0 else np.exp(-dE / T)

    for _ in range(n_thermalization):
        _sweep_3d(spins, accept)

    configs = np.empty((n_samples, L, L, L), dtype=np.int8)
    energies = np.empty(n_samples, dtype=np.float64)
    mags = np.empty(n_samples, dtype=np.float64)
    for s in range(n_samples):
        for _ in range(decorrelation):
            _sweep_3d(spins, accept)
        configs[s] = spins
        energies[s] = _total_energy_3d(spins)
        mags[s] = spins.astype(np.float64).sum()
    return configs, energies, mags


def simulate_3d_metropolis(
    L: int,
    T: float,
    n_samples: int,
    *,
    n_thermalization: int = 2_000,
    decorrelation: int = 5,
    seed: int = 0,
) -> Sim3DResult:
    """Run 3D Metropolis and return decorrelated samples plus observables."""
    if L < 4:
        raise ValueError("L must be >= 4 for sensible PBC.")
    if T <= 0:
        raise ValueError("T must be positive.")
    if n_samples < 1 or n_thermalization < 0 or decorrelation < 1:
        raise ValueError("Bad sampling parameters.")

    configs, energies, mags = _run_3d(
        L, float(T), int(n_thermalization),
        int(n_samples), int(decorrelation), int(seed),
    )
    return Sim3DResult(
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
