"""5D Ising Wolff cluster algorithm on an L^5 hypercubic torus.

Same construction as the 2D/3D/4D Wolff, with ten PBC neighbors per site.
A site (i, j, k, m, n) is encoded as the flat index
    i*L^4 + j*L^3 + k*L^2 + m*L + n
for the explicit cluster stack. The ten neighbors are decoded and tested with
an axis loop (a strides array) rather than a fully unrolled block, to keep the
kernel readable at five dimensions.
"""

from __future__ import annotations

import numpy as np
from numba import njit

from ising.metropolis_5d import Sim5DResult


@njit(cache=True, fastmath=False)
def _total_energy_5d(spins: np.ndarray) -> float:
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
def _wolff_step_5d(spins, p_add, stack, in_cluster, strides):
    """Build one Wolff cluster from a random seed and flip it. Returns size."""
    L = spins.shape[0]
    in_cluster[:] = False

    coord = np.empty(5, dtype=np.int64)
    ncoord = np.empty(5, dtype=np.int64)
    for a in range(5):
        coord[a] = np.random.randint(0, L)
    seed_spin = spins[coord[0], coord[1], coord[2], coord[3], coord[4]]
    seed_flat = 0
    for a in range(5):
        seed_flat += coord[a] * strides[a]
    in_cluster[seed_flat] = True
    stack[0] = seed_flat
    sp = 1
    size = 1

    while sp > 0:
        sp -= 1
        idx = stack[sp]
        r = idx
        for a in range(5):
            coord[a] = r // strides[a]
            r -= coord[a] * strides[a]

        # Ten neighbors: each axis a, shift -1 and +1 (periodic).
        for a in range(5):
            for delta in (-1, 1):
                for b in range(5):
                    ncoord[b] = coord[b]
                ca = coord[a] + delta
                if ca < 0:
                    ca = L - 1
                elif ca >= L:
                    ca = 0
                ncoord[a] = ca
                if spins[ncoord[0], ncoord[1], ncoord[2],
                         ncoord[3], ncoord[4]] != seed_spin:
                    continue
                nflat = 0
                for b in range(5):
                    nflat += ncoord[b] * strides[b]
                if in_cluster[nflat]:
                    continue
                if np.random.random() < p_add:
                    in_cluster[nflat] = True
                    stack[sp] = nflat
                    sp += 1
                    size += 1

    # Flip the cluster.
    flat = 0
    for i in range(L):
        for j in range(L):
            for k in range(L):
                for m in range(L):
                    for n in range(L):
                        if in_cluster[flat]:
                            spins[i, j, k, m, n] = -spins[i, j, k, m, n]
                        flat += 1
    return size


@njit(cache=True, fastmath=False)
def _run_wolff_5d(L, T, n_thermalization, n_samples, decorrelation, seed):
    np.random.seed(seed)
    spins = np.where(np.random.random((L, L, L, L, L)) < 0.5,
                     np.int8(-1), np.int8(1))
    beta = 1.0 / T
    p_add = 1.0 - np.exp(-2.0 * beta)

    total = L ** 5
    stack = np.empty(total, dtype=np.int64)
    in_cluster = np.empty(total, dtype=np.bool_)
    strides = np.array([L ** 4, L ** 3, L ** 2, L, 1], dtype=np.int64)

    for _ in range(n_thermalization):
        _wolff_step_5d(spins, p_add, stack, in_cluster, strides)

    configs = np.empty((n_samples, L, L, L, L, L), dtype=np.int8)
    energies = np.empty(n_samples, dtype=np.float64)
    mags = np.empty(n_samples, dtype=np.float64)
    for s in range(n_samples):
        for _ in range(decorrelation):
            _wolff_step_5d(spins, p_add, stack, in_cluster, strides)
        configs[s] = spins
        energies[s] = _total_energy_5d(spins)
        mags[s] = spins.astype(np.float64).sum()
    return configs, energies, mags


def simulate_5d_wolff(L, T, n_samples, *, n_thermalization=500,
                      decorrelation=10, seed=0) -> Sim5DResult:
    """Run 5D Wolff and return decorrelated samples plus observables."""
    if L < 3:
        raise ValueError("L must be >= 3 for sensible PBC.")
    if T <= 0:
        raise ValueError("T must be positive.")
    if n_samples < 1 or n_thermalization < 0 or decorrelation < 1:
        raise ValueError("Bad sampling parameters.")
    configs, energies, mags = _run_wolff_5d(
        L, float(T), int(n_thermalization), int(n_samples),
        int(decorrelation), int(seed))
    return Sim5DResult(
        configurations=configs, energies=energies, magnetizations=mags,
        T=float(T), L=int(L), seed=int(seed),
        n_thermalization=int(n_thermalization),
        decorrelation=int(decorrelation), algorithm="wolff")
