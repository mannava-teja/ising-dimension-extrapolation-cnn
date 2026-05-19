"""5D Ising Metropolis-Hastings on an L^5 hypercubic torus.

H = -J * sum_<i,j> s_i s_j    (J = 1, k_B = 1, periodic boundaries)

Each site has 10 neighbors (2 per axis x 5 axes). A spin flip changes the
energy by 2 * s * (sum of 10 neighbors), in {-20,-16,...,0,...,16,20} -- an
11-entry acceptance table.

5D is *above* the upper critical dimension (d_c = 4): the Ising model is
mean-field there, with exponents nu = 1/2, beta = 1/2, gamma = 1 and -- unlike
exactly d = 4 -- no logarithmic corrections. T_c(5D) ~ 8.778. This dataset is
a second held-out test set: if the network's nu(5D) matches nu(4D) ~ 1/2, the
critical exponents have frozen, directly demonstrating the upper critical
dimension.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit


@dataclass(frozen=True)
class Sim5DResult:
    configurations: np.ndarray  # (n_samples, L, L, L, L, L) int8 in {-1, +1}
    energies: np.ndarray        # (n_samples,) float64, total energy
    magnetizations: np.ndarray  # (n_samples,) float64, total signed magnetization
    T: float
    L: int
    seed: int
    n_thermalization: int
    decorrelation: int
    algorithm: str = "metropolis"


@njit(cache=True, fastmath=False)
def _total_energy_5d(spins: np.ndarray) -> float:
    """Total energy, each bond counted once (forward neighbors on all 5 axes)."""
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
                    for n in range(L):
                        np_ = n + 1 if n + 1 < L else 0
                        s = spins[i, j, k, m, n]
                        e -= s * spins[ip, j, k, m, n]
                        e -= s * spins[i, jp, k, m, n]
                        e -= s * spins[i, j, kp, m, n]
                        e -= s * spins[i, j, k, mp, n]
                        e -= s * spins[i, j, k, m, np_]
    return e


@njit(cache=True, fastmath=False)
def _sweep_5d(spins: np.ndarray, accept: np.ndarray) -> None:
    """One MC sweep = L^5 attempted single-spin flips at uniform-random sites."""
    L = spins.shape[0]
    for _ in range(L * L * L * L * L):
        i = np.random.randint(0, L)
        j = np.random.randint(0, L)
        k = np.random.randint(0, L)
        m = np.random.randint(0, L)
        n = np.random.randint(0, L)
        nbr_sum = (
            spins[i - 1 if i > 0 else L - 1, j, k, m, n]
            + spins[i + 1 if i + 1 < L else 0, j, k, m, n]
            + spins[i, j - 1 if j > 0 else L - 1, k, m, n]
            + spins[i, j + 1 if j + 1 < L else 0, k, m, n]
            + spins[i, j, k - 1 if k > 0 else L - 1, m, n]
            + spins[i, j, k + 1 if k + 1 < L else 0, m, n]
            + spins[i, j, k, m - 1 if m > 0 else L - 1, n]
            + spins[i, j, k, m + 1 if m + 1 < L else 0, n]
            + spins[i, j, k, m, n - 1 if n > 0 else L - 1]
            + spins[i, j, k, m, n + 1 if n + 1 < L else 0]
        )
        # dE = 2 * s * nbr_sum, in {-20,-16,...,16,20}
        dE = 2 * spins[i, j, k, m, n] * nbr_sum
        idx = (dE + 20) // 4         # 0..10
        if np.random.random() < accept[idx]:
            spins[i, j, k, m, n] = -spins[i, j, k, m, n]


@njit(cache=True, fastmath=False)
def _run_5d(L, T, n_thermalization, n_samples, decorrelation, seed):
    np.random.seed(seed)
    spins = np.where(np.random.random((L, L, L, L, L)) < 0.5,
                     np.int8(-1), np.int8(1))

    accept = np.empty(11, dtype=np.float64)
    for idx in range(11):
        dE = 4 * idx - 20
        accept[idx] = 1.0 if dE <= 0 else np.exp(-dE / T)

    for _ in range(n_thermalization):
        _sweep_5d(spins, accept)

    configs = np.empty((n_samples, L, L, L, L, L), dtype=np.int8)
    energies = np.empty(n_samples, dtype=np.float64)
    mags = np.empty(n_samples, dtype=np.float64)
    for s in range(n_samples):
        for _ in range(decorrelation):
            _sweep_5d(spins, accept)
        configs[s] = spins
        energies[s] = _total_energy_5d(spins)
        mags[s] = spins.astype(np.float64).sum()
    return configs, energies, mags


def simulate_5d_metropolis(L, T, n_samples, *, n_thermalization=2_000,
                           decorrelation=5, seed=0) -> Sim5DResult:
    """Run 5D Metropolis and return decorrelated samples plus observables."""
    if L < 3:
        raise ValueError("L must be >= 3 for sensible PBC.")
    if T <= 0:
        raise ValueError("T must be positive.")
    if n_samples < 1 or n_thermalization < 0 or decorrelation < 1:
        raise ValueError("Bad sampling parameters.")
    configs, energies, mags = _run_5d(
        L, float(T), int(n_thermalization), int(n_samples),
        int(decorrelation), int(seed))
    return Sim5DResult(
        configurations=configs, energies=energies, magnetizations=mags,
        T=float(T), L=int(L), seed=int(seed),
        n_thermalization=int(n_thermalization),
        decorrelation=int(decorrelation), algorithm="metropolis")
