"""3D Ising Wolff cluster algorithm on an LxLxL torus.

Same idea as 2D Wolff: pick a random seed site, grow a cluster of like-spin
neighbors with bond-add probability p = 1 - exp(-2*beta), flip the cluster.
The difference here is six PBC neighbors instead of four, encoded inline.
T_c(3D) is ~ 4.5115 (Ferrenberg & Landau 1991); critical slowing down is
worse than 2D (z_Wolff ~ 0.4) but Wolff still vastly outperforms single-
spin-flip Metropolis near the transition.
"""

from __future__ import annotations

import numpy as np
from numba import njit

from ising.metropolis_3d import Sim3DResult


@njit(cache=True, fastmath=False)
def _total_energy_3d(spins: np.ndarray) -> float:
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
def _wolff_step_3d(spins: np.ndarray, p_add: float,
                   stack: np.ndarray, in_cluster: np.ndarray) -> int:
    """Build one Wolff cluster from a random seed and flip it. Returns cluster size."""
    L = spins.shape[0]
    # Reset mask in-place.
    for i in range(L):
        for j in range(L):
            for k in range(L):
                in_cluster[i, j, k] = False

    si = np.random.randint(0, L)
    sj = np.random.randint(0, L)
    sk = np.random.randint(0, L)
    seed_spin = spins[si, sj, sk]
    in_cluster[si, sj, sk] = True
    L2 = L * L
    stack[0] = si * L2 + sj * L + sk
    sp = 1
    size = 1

    while sp > 0:
        sp -= 1
        idx = stack[sp]
        ci = idx // L2
        rem = idx - ci * L2
        cj = rem // L
        ck = rem - cj * L

        # Six PBC neighbors, unrolled.
        # -i
        ni = ci - 1 if ci > 0 else L - 1
        if spins[ni, cj, ck] == seed_spin and not in_cluster[ni, cj, ck]:
            if np.random.random() < p_add:
                in_cluster[ni, cj, ck] = True
                stack[sp] = ni * L2 + cj * L + ck
                sp += 1
                size += 1
        # +i
        ni = ci + 1 if ci + 1 < L else 0
        if spins[ni, cj, ck] == seed_spin and not in_cluster[ni, cj, ck]:
            if np.random.random() < p_add:
                in_cluster[ni, cj, ck] = True
                stack[sp] = ni * L2 + cj * L + ck
                sp += 1
                size += 1
        # -j
        nj = cj - 1 if cj > 0 else L - 1
        if spins[ci, nj, ck] == seed_spin and not in_cluster[ci, nj, ck]:
            if np.random.random() < p_add:
                in_cluster[ci, nj, ck] = True
                stack[sp] = ci * L2 + nj * L + ck
                sp += 1
                size += 1
        # +j
        nj = cj + 1 if cj + 1 < L else 0
        if spins[ci, nj, ck] == seed_spin and not in_cluster[ci, nj, ck]:
            if np.random.random() < p_add:
                in_cluster[ci, nj, ck] = True
                stack[sp] = ci * L2 + nj * L + ck
                sp += 1
                size += 1
        # -k
        nk = ck - 1 if ck > 0 else L - 1
        if spins[ci, cj, nk] == seed_spin and not in_cluster[ci, cj, nk]:
            if np.random.random() < p_add:
                in_cluster[ci, cj, nk] = True
                stack[sp] = ci * L2 + cj * L + nk
                sp += 1
                size += 1
        # +k
        nk = ck + 1 if ck + 1 < L else 0
        if spins[ci, cj, nk] == seed_spin and not in_cluster[ci, cj, nk]:
            if np.random.random() < p_add:
                in_cluster[ci, cj, nk] = True
                stack[sp] = ci * L2 + cj * L + nk
                sp += 1
                size += 1

    # Flip the cluster.
    for i in range(L):
        for j in range(L):
            for k in range(L):
                if in_cluster[i, j, k]:
                    spins[i, j, k] = -spins[i, j, k]
    return size


@njit(cache=True, fastmath=False)
def _run_wolff_3d(
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
    beta = 1.0 / T
    p_add = 1.0 - np.exp(-2.0 * beta)

    stack = np.empty(L * L * L, dtype=np.int64)
    in_cluster = np.empty((L, L, L), dtype=np.bool_)

    for _ in range(n_thermalization):
        _wolff_step_3d(spins, p_add, stack, in_cluster)

    configs = np.empty((n_samples, L, L, L), dtype=np.int8)
    energies = np.empty(n_samples, dtype=np.float64)
    mags = np.empty(n_samples, dtype=np.float64)
    for s in range(n_samples):
        for _ in range(decorrelation):
            _wolff_step_3d(spins, p_add, stack, in_cluster)
        configs[s] = spins
        energies[s] = _total_energy_3d(spins)
        mags[s] = spins.astype(np.float64).sum()

    return configs, energies, mags


def simulate_3d_wolff(
    L: int,
    T: float,
    n_samples: int,
    *,
    n_thermalization: int = 500,
    decorrelation: int = 10,
    seed: int = 0,
) -> Sim3DResult:
    """Run 3D Wolff and return decorrelated samples plus observables.

    `n_thermalization` and `decorrelation` are counts of cluster updates.
    Defaults are larger than 2D because z_Wolff(3D) > z_Wolff(2D) -- but
    still much smaller than equivalent Metropolis sweeps.
    """
    if L < 4:
        raise ValueError("L must be >= 4 for sensible PBC.")
    if T <= 0:
        raise ValueError("T must be positive.")
    if n_samples < 1 or n_thermalization < 0 or decorrelation < 1:
        raise ValueError("Bad sampling parameters.")

    configs, energies, mags = _run_wolff_3d(
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
        algorithm="wolff",
    )
