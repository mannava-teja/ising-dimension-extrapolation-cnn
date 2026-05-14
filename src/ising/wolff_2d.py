"""2D Ising Wolff cluster algorithm (Wolff, Phys. Rev. Lett. 62, 361 (1989)).

At each step we pick a random seed site, grow a cluster of like-spin neighbors
with bond-add probability  p = 1 - exp(-2*beta)  (J = 1, k_B = 1), then flip
the entire cluster. This satisfies detailed balance for the Ising Boltzmann
distribution and largely defeats critical slowing down: tau_Wolff ~ L^0.25 at
T_c, vs ~ L^2.17 for single-spin-flip Metropolis.

Implementation notes:
  - Cluster grown via an explicit int32 stack of capacity L*L (max possible
    cluster size); BFS/DFS is the same up to traversal order under Wolff.
  - `in_cluster` mask and `stack` are allocated once outside the hot loop.
  - Energy and magnetization are recomputed only at sample time; doing it per
    cluster update would dominate runtime, especially near T_c where clusters
    are large.
"""

from __future__ import annotations

import numpy as np
from numba import njit

from ising.metropolis_2d import Sim2DResult


@njit(cache=True, fastmath=False)
def _total_energy_2d(spins: np.ndarray) -> float:
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
def _wolff_step(spins: np.ndarray, p_add: float,
                stack: np.ndarray, in_cluster: np.ndarray) -> int:
    """Build one Wolff cluster from a random seed and flip it. Returns cluster size."""
    L = spins.shape[0]
    # Reset mask. (np.zeros allocation would be slower; this is in-place.)
    for i in range(L):
        for j in range(L):
            in_cluster[i, j] = False

    si = np.random.randint(0, L)
    sj = np.random.randint(0, L)
    seed_spin = spins[si, sj]
    in_cluster[si, sj] = True
    stack[0] = si * L + sj
    sp = 1
    size = 1

    while sp > 0:
        sp -= 1
        idx = stack[sp]
        ci = idx // L
        cj = idx - ci * L

        # Four PBC neighbors, unrolled.
        ni = ci - 1 if ci > 0 else L - 1
        if spins[ni, cj] == seed_spin and not in_cluster[ni, cj]:
            if np.random.random() < p_add:
                in_cluster[ni, cj] = True
                stack[sp] = ni * L + cj
                sp += 1
                size += 1

        ni = ci + 1 if ci + 1 < L else 0
        if spins[ni, cj] == seed_spin and not in_cluster[ni, cj]:
            if np.random.random() < p_add:
                in_cluster[ni, cj] = True
                stack[sp] = ni * L + cj
                sp += 1
                size += 1

        nj = cj - 1 if cj > 0 else L - 1
        if spins[ci, nj] == seed_spin and not in_cluster[ci, nj]:
            if np.random.random() < p_add:
                in_cluster[ci, nj] = True
                stack[sp] = ci * L + nj
                sp += 1
                size += 1

        nj = cj + 1 if cj + 1 < L else 0
        if spins[ci, nj] == seed_spin and not in_cluster[ci, nj]:
            if np.random.random() < p_add:
                in_cluster[ci, nj] = True
                stack[sp] = ci * L + nj
                sp += 1
                size += 1

    # Flip the cluster in-place.
    for i in range(L):
        for j in range(L):
            if in_cluster[i, j]:
                spins[i, j] = -spins[i, j]

    return size


@njit(cache=True, fastmath=False)
def _run_wolff_2d(
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
    beta = 1.0 / T
    p_add = 1.0 - np.exp(-2.0 * beta)

    stack = np.empty(L * L, dtype=np.int64)
    in_cluster = np.empty((L, L), dtype=np.bool_)

    for _ in range(n_thermalization):
        _wolff_step(spins, p_add, stack, in_cluster)

    configs = np.empty((n_samples, L, L), dtype=np.int8)
    energies = np.empty(n_samples, dtype=np.float64)
    mags = np.empty(n_samples, dtype=np.float64)
    for s in range(n_samples):
        for _ in range(decorrelation):
            _wolff_step(spins, p_add, stack, in_cluster)
        configs[s] = spins
        energies[s] = _total_energy_2d(spins)
        mags[s] = spins.astype(np.float64).sum()

    return configs, energies, mags


def simulate_2d_wolff(
    L: int,
    T: float,
    n_samples: int,
    *,
    n_thermalization: int = 200,
    decorrelation: int = 5,
    seed: int = 0,
) -> Sim2DResult:
    """Run 2D Wolff and return decorrelated samples plus their observables.

    `n_thermalization` and `decorrelation` are counts of cluster updates, not
    sweeps. Cluster updates have much lower autocorrelation than Metropolis
    sweeps, so the defaults are correspondingly small.
    """
    if L < 4:
        raise ValueError("L must be >= 4 for sensible PBC.")
    if T <= 0:
        raise ValueError("T must be positive.")
    if n_samples < 1 or n_thermalization < 0 or decorrelation < 1:
        raise ValueError("Bad sampling parameters.")

    configs, energies, mags = _run_wolff_2d(
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
        algorithm="wolff",
    )
