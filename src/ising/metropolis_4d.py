"""4D Ising Metropolis-Hastings on an L^4 hypercubic torus.

H = -J * sum_<i,j> s_i s_j    (J = 1, k_B = 1, periodic boundaries)

Each site has 8 neighbors (2 per axis x 4 axes). A spin flip changes the energy
by 2 * s * (sum of 8 neighbors), which lives in {-16,-12,-8,-4,0,4,8,12,16} --
a 9-entry acceptance table.

4D is the upper critical dimension of the Ising universality class:
T_c ~ 6.6803 (Lundow & Markstrom 2009), mean-field exponents with logarithmic
corrections. This dataset is the held-out test set for the dimension-
extrapolation experiment -- it is never used to train the network.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit


@dataclass(frozen=True)
class Sim4DResult:
    configurations: np.ndarray  # (n_samples, L, L, L, L) int8 in {-1, +1}
    energies: np.ndarray        # (n_samples,) float64, total energy
    magnetizations: np.ndarray  # (n_samples,) float64, total signed magnetization
    T: float
    L: int
    seed: int
    n_thermalization: int
    decorrelation: int
    algorithm: str = "metropolis"


@njit(cache=True, fastmath=False)
def _total_energy_4d(spins: np.ndarray) -> float:
    """Total energy, each bond counted once (forward neighbors on all 4 axes)."""
    L = spins.shape[0]
    e = 0.0
    for i in range(L):
        ip = i + 1 if i + 1 < L else 0
        for j in range(L):
            jp = j + 1 if j + 1 < L else 0
            for k in range(L):
                kp = k + 1 if k + 1 < L else 0
                for m in range(L):
                    mp = m + 1 if m + 1 < L else 0
                    s = spins[i, j, k, m]
                    e -= s * spins[ip, j, k, m]
                    e -= s * spins[i, jp, k, m]
                    e -= s * spins[i, j, kp, m]
                    e -= s * spins[i, j, k, mp]
    return e


@njit(cache=True, fastmath=False)
def _sweep_4d(spins: np.ndarray, accept: np.ndarray) -> None:
    """One MC sweep = L^4 attempted single-spin flips at uniform-random sites."""
    L = spins.shape[0]
    for _ in range(L * L * L * L):
        i = np.random.randint(0, L)
        j = np.random.randint(0, L)
        k = np.random.randint(0, L)
        m = np.random.randint(0, L)
        nbr_sum = (
            spins[i - 1 if i > 0 else L - 1, j, k, m]
            + spins[i + 1 if i + 1 < L else 0, j, k, m]
            + spins[i, j - 1 if j > 0 else L - 1, k, m]
            + spins[i, j + 1 if j + 1 < L else 0, k, m]
            + spins[i, j, k - 1 if k > 0 else L - 1, m]
            + spins[i, j, k + 1 if k + 1 < L else 0, m]
            + spins[i, j, k, m - 1 if m > 0 else L - 1]
            + spins[i, j, k, m + 1 if m + 1 < L else 0]
        )
        # dE = 2 * s * nbr_sum, in {-16,-12,-8,-4,0,4,8,12,16}
        dE = 2 * spins[i, j, k, m] * nbr_sum
        idx = (dE + 16) // 4         # 0..8
        if np.random.random() < accept[idx]:
            spins[i, j, k, m] = -spins[i, j, k, m]


@njit(cache=True, fastmath=False)
def _run_4d(
    L: int,
    T: float,
    n_thermalization: int,
    n_samples: int,
    decorrelation: int,
    seed: int,
):
    np.random.seed(seed)
    spins = np.where(np.random.random((L, L, L, L)) < 0.5,
                     np.int8(-1), np.int8(1))

    accept = np.empty(9, dtype=np.float64)
    for idx in range(9):
        dE = 4 * idx - 16
        accept[idx] = 1.0 if dE <= 0 else np.exp(-dE / T)

    for _ in range(n_thermalization):
        _sweep_4d(spins, accept)

    configs = np.empty((n_samples, L, L, L, L), dtype=np.int8)
    energies = np.empty(n_samples, dtype=np.float64)
    mags = np.empty(n_samples, dtype=np.float64)
    for s in range(n_samples):
        for _ in range(decorrelation):
            _sweep_4d(spins, accept)
        configs[s] = spins
        energies[s] = _total_energy_4d(spins)
        mags[s] = spins.astype(np.float64).sum()
    return configs, energies, mags


def simulate_4d_metropolis(
    L: int,
    T: float,
    n_samples: int,
    *,
    n_thermalization: int = 2_000,
    decorrelation: int = 5,
    seed: int = 0,
) -> Sim4DResult:
    """Run 4D Metropolis and return decorrelated samples plus observables."""
    if L < 3:
        raise ValueError("L must be >= 3 for sensible PBC.")
    if T <= 0:
        raise ValueError("T must be positive.")
    if n_samples < 1 or n_thermalization < 0 or decorrelation < 1:
        raise ValueError("Bad sampling parameters.")

    configs, energies, mags = _run_4d(
        L, float(T), int(n_thermalization),
        int(n_samples), int(decorrelation), int(seed),
    )
    return Sim4DResult(
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
