"""1D Ising Metropolis-Hastings.

H = -J * sum_i s_i * s_{i+1}    (J = 1, k_B = 1, periodic boundaries)

A spin flip at site i changes the energy by 2 * s_i * (s_{i-1} + s_{i+1}),
which lives in {-4, -2, 0, 2, 4}. We tabulate the Metropolis acceptance
probability for those five values up front so the inner loop is branch-light.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit


@dataclass(frozen=True)
class Sim1DResult:
    configurations: np.ndarray  # (n_samples, L) int8 in {-1, +1}
    energies: np.ndarray        # (n_samples,) float64, total energy
    magnetizations: np.ndarray  # (n_samples,) float64, total magnetization
    T: float
    L: int
    seed: int
    n_thermalization: int
    decorrelation: int


@njit(cache=True, fastmath=False)
def _total_energy(spins: np.ndarray) -> float:
    L = spins.shape[0]
    e = 0.0
    for i in range(L):
        j = i + 1 if i + 1 < L else 0
        e -= spins[i] * spins[j]
    return e


@njit(cache=True, fastmath=False)
def _sweep(spins: np.ndarray, accept: np.ndarray) -> None:
    """One Monte Carlo sweep: L attempted single-spin flips at uniform-random sites.

    `accept[k]` holds the acceptance probability for delta_E = 2k - 4,
    indexed by k in {0, 1, 2, 3, 4} (so delta_E in {-4, -2, 0, 2, 4}).
    """
    L = spins.shape[0]
    for _ in range(L):
        i = np.random.randint(0, L)
        left = spins[i - 1] if i > 0 else spins[L - 1]
        right = spins[i + 1] if i + 1 < L else spins[0]
        # delta_E = 2 * s_i * (left + right), in {-4, -2, 0, 2, 4}
        dE = 2 * spins[i] * (left + right)
        k = (dE + 4) // 2  # 0..4
        if np.random.random() < accept[k]:
            spins[i] = -spins[i]


@njit(cache=True, fastmath=False)
def _run(
    L: int,
    T: float,
    n_thermalization: int,
    n_samples: int,
    decorrelation: int,
    seed: int,
):
    np.random.seed(seed)

    # Hot start: random +/- 1 spins. Cold start (all +1) equilibrates faster at
    # very low T but biases warm runs; for a uniform sweep across T, random is
    # the safer default.
    spins = np.where(np.random.random(L) < 0.5, np.int8(-1), np.int8(1))

    # Acceptance table indexed by k = (dE + 4)/2.
    accept = np.empty(5, dtype=np.float64)
    for k in range(5):
        dE = 2 * k - 4
        accept[k] = 1.0 if dE <= 0 else np.exp(-dE / T)

    for _ in range(n_thermalization):
        _sweep(spins, accept)

    configs = np.empty((n_samples, L), dtype=np.int8)
    energies = np.empty(n_samples, dtype=np.float64)
    mags = np.empty(n_samples, dtype=np.float64)

    for s in range(n_samples):
        for _ in range(decorrelation):
            _sweep(spins, accept)
        configs[s] = spins
        energies[s] = _total_energy(spins)
        # sum on int8 overflows for L > 127; cast first.
        mags[s] = spins.astype(np.float64).sum()

    return configs, energies, mags


def simulate_1d(
    L: int,
    T: float,
    n_samples: int,
    *,
    n_thermalization: int = 10_000,
    decorrelation: int = 10,
    seed: int = 0,
) -> Sim1DResult:
    """Run 1D Metropolis and return decorrelated samples plus their observables."""
    if L < 4:
        raise ValueError("L must be at least 4 for sensible PBC.")
    if T <= 0:
        raise ValueError("T must be positive (T_c = 0 in 1D, but T = 0 is degenerate).")
    if n_samples < 1 or n_thermalization < 0 or decorrelation < 1:
        raise ValueError("Bad sampling parameters.")

    configs, energies, mags = _run(
        L, float(T), int(n_thermalization), int(n_samples), int(decorrelation), int(seed)
    )
    return Sim1DResult(
        configurations=configs,
        energies=energies,
        magnetizations=mags,
        T=float(T),
        L=int(L),
        seed=int(seed),
        n_thermalization=int(n_thermalization),
        decorrelation=int(decorrelation),
    )
