"""4D Ising Wolff cluster algorithm on an L^4 hypercubic torus.

Same construction as the 2D / 3D Wolff: pick a random seed site, grow a cluster
of like-spin neighbors with bond-add probability p = 1 - exp(-2*beta), flip the
cluster. The only change in 4D is eight PBC neighbors per site instead of six.

A site (i, j, k, m) is encoded as the flat index
    i*L^3 + j*L^2 + k*L + m
for the explicit cluster stack (Numba has no dynamic containers).
"""

from __future__ import annotations

import numpy as np
from numba import njit

from ising.metropolis_4d import Sim4DResult


@njit(cache=True, fastmath=False)
def _total_energy_4d(spins: np.ndarray) -> float:
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
def _wolff_step_4d(spins: np.ndarray, p_add: float,
                   stack: np.ndarray, in_cluster: np.ndarray) -> int:
    """Build one Wolff cluster from a random seed and flip it. Returns cluster size."""
    L = spins.shape[0]
    L2 = L * L
    L3 = L2 * L

    # Reset mask in-place.
    for i in range(L):
        for j in range(L):
            for k in range(L):
                for m in range(L):
                    in_cluster[i, j, k, m] = False

    si = np.random.randint(0, L)
    sj = np.random.randint(0, L)
    sk = np.random.randint(0, L)
    sm = np.random.randint(0, L)
    seed_spin = spins[si, sj, sk, sm]
    in_cluster[si, sj, sk, sm] = True
    stack[0] = si * L3 + sj * L2 + sk * L + sm
    sp = 1
    size = 1

    while sp > 0:
        sp -= 1
        idx = stack[sp]
        ci = idx // L3
        rem = idx - ci * L3
        cj = rem // L2
        rem = rem - cj * L2
        ck = rem // L
        cm = rem - ck * L

        # Eight PBC neighbors, unrolled.
        # -i / +i
        ni = ci - 1 if ci > 0 else L - 1
        if spins[ni, cj, ck, cm] == seed_spin and not in_cluster[ni, cj, ck, cm]:
            if np.random.random() < p_add:
                in_cluster[ni, cj, ck, cm] = True
                stack[sp] = ni * L3 + cj * L2 + ck * L + cm
                sp += 1
                size += 1
        ni = ci + 1 if ci + 1 < L else 0
        if spins[ni, cj, ck, cm] == seed_spin and not in_cluster[ni, cj, ck, cm]:
            if np.random.random() < p_add:
                in_cluster[ni, cj, ck, cm] = True
                stack[sp] = ni * L3 + cj * L2 + ck * L + cm
                sp += 1
                size += 1
        # -j / +j
        nj = cj - 1 if cj > 0 else L - 1
        if spins[ci, nj, ck, cm] == seed_spin and not in_cluster[ci, nj, ck, cm]:
            if np.random.random() < p_add:
                in_cluster[ci, nj, ck, cm] = True
                stack[sp] = ci * L3 + nj * L2 + ck * L + cm
                sp += 1
                size += 1
        nj = cj + 1 if cj + 1 < L else 0
        if spins[ci, nj, ck, cm] == seed_spin and not in_cluster[ci, nj, ck, cm]:
            if np.random.random() < p_add:
                in_cluster[ci, nj, ck, cm] = True
                stack[sp] = ci * L3 + nj * L2 + ck * L + cm
                sp += 1
                size += 1
        # -k / +k
        nk = ck - 1 if ck > 0 else L - 1
        if spins[ci, cj, nk, cm] == seed_spin and not in_cluster[ci, cj, nk, cm]:
            if np.random.random() < p_add:
                in_cluster[ci, cj, nk, cm] = True
                stack[sp] = ci * L3 + cj * L2 + nk * L + cm
                sp += 1
                size += 1
        nk = ck + 1 if ck + 1 < L else 0
        if spins[ci, cj, nk, cm] == seed_spin and not in_cluster[ci, cj, nk, cm]:
            if np.random.random() < p_add:
                in_cluster[ci, cj, nk, cm] = True
                stack[sp] = ci * L3 + cj * L2 + nk * L + cm
                sp += 1
                size += 1
        # -m / +m
        nm = cm - 1 if cm > 0 else L - 1
        if spins[ci, cj, ck, nm] == seed_spin and not in_cluster[ci, cj, ck, nm]:
            if np.random.random() < p_add:
                in_cluster[ci, cj, ck, nm] = True
                stack[sp] = ci * L3 + cj * L2 + ck * L + nm
                sp += 1
                size += 1
        nm = cm + 1 if cm + 1 < L else 0
        if spins[ci, cj, ck, nm] == seed_spin and not in_cluster[ci, cj, ck, nm]:
            if np.random.random() < p_add:
                in_cluster[ci, cj, ck, nm] = True
                stack[sp] = ci * L3 + cj * L2 + ck * L + nm
                sp += 1
                size += 1

    # Flip the cluster.
    for i in range(L):
        for j in range(L):
            for k in range(L):
                for m in range(L):
                    if in_cluster[i, j, k, m]:
                        spins[i, j, k, m] = -spins[i, j, k, m]
    return size


@njit(cache=True, fastmath=False)
def _run_wolff_4d(
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
    beta = 1.0 / T
    p_add = 1.0 - np.exp(-2.0 * beta)

    stack = np.empty(L * L * L * L, dtype=np.int64)
    in_cluster = np.empty((L, L, L, L), dtype=np.bool_)

    for _ in range(n_thermalization):
        _wolff_step_4d(spins, p_add, stack, in_cluster)

    configs = np.empty((n_samples, L, L, L, L), dtype=np.int8)
    energies = np.empty(n_samples, dtype=np.float64)
    mags = np.empty(n_samples, dtype=np.float64)
    for s in range(n_samples):
        for _ in range(decorrelation):
            _wolff_step_4d(spins, p_add, stack, in_cluster)
        configs[s] = spins
        energies[s] = _total_energy_4d(spins)
        mags[s] = spins.astype(np.float64).sum()

    return configs, energies, mags


def simulate_4d_wolff(
    L: int,
    T: float,
    n_samples: int,
    *,
    n_thermalization: int = 500,
    decorrelation: int = 10,
    seed: int = 0,
) -> Sim4DResult:
    """Run 4D Wolff and return decorrelated samples plus observables.

    `n_thermalization` and `decorrelation` count cluster updates, not sweeps.
    """
    if L < 3:
        raise ValueError("L must be >= 3 for sensible PBC.")
    if T <= 0:
        raise ValueError("T must be positive.")
    if n_samples < 1 or n_thermalization < 0 or decorrelation < 1:
        raise ValueError("Bad sampling parameters.")

    configs, energies, mags = _run_wolff_4d(
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
        algorithm="wolff",
    )
