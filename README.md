# ising-dimension-extrapolation-cnn

A CNN trained on Ising-model Monte Carlo configurations in 1D, 2D, and 3D,
tested on whether it can extrapolate to **4D** — a dimension it never saw
during training. 4D is the upper critical dimension of the Ising universality
class, where mean-field theory becomes exact (modulo logarithmic corrections),
making it a stringent test of whether a network can generalize physics
across dimensions.

## Background

Spins `s_i ∈ {-1, +1}` sit on the sites of a regular lattice. The Hamiltonian
`H = -J Σ_⟨ij⟩ s_i s_j` couples nearest neighbors. Configurations follow
`p(s) ∝ exp(-H/kT)`. Throughout: `J = 1`, `k_B = 1`, periodic boundaries.

| d | T_c | Transition |
|---|----:|------------|
| 1 | 0 | None |
| 2 | 2.2692 | Exact (Onsager 1944) |
| 3 | 4.5115 | MC-numerics (Ferrenberg & Landau 1991) |
| 4 | ≈ 6.68 | Upper critical dimension; mean-field + log corrections |

The interesting feature: 4D is exactly where the model's behavior crosses
over from non-trivial critical exponents to mean-field exponents. If a
network trained on d=1,2,3 can recognize the d=4 transition, that's
evidence the network has learned features that transcend the spatial
structure of each dimension.

Monte Carlo sampling uses both **Wolff cluster updates** (near and below T_c,
where they defeat critical slowing down) and **Metropolis-Hastings** (above
T_c, where cluster size is O(1) and Wolff is inefficient). Per-block metadata
records which algorithm produced each block.

## Datasets

| d | Configurations | Sizes × Temps | T_c estimate from Binder crossings | Reference |
|---|---:|---|---|---|
| 1 |  90,000 | 3 × 30 | n/a (no transition) | exact `-tanh(1/T)` |
| 2 | 160,000 | 4 × 40 | **2.2543** (vs 2.2692, 0.7% off) | Onsager |
| 3 | 120,000 | 3 × 40 | **4.5132** (vs 4.5115, 0.04% off) | Ferrenberg–Landau / Talapov–Blöte / Hasenbusch |

All three datasets pass:

- Bit-exact storage integrity (manual energy recomputation on 600 random samples)
- Format checks: `int8` configurations with values exactly `{-1, +1}`, PBC translation invariance on all spatial axes
- Energy convention identical across dimensions: `H = -Σ_⟨ij⟩ s_i s_j`, each bond counted once
- Per-block effective independent sample size > 100 across every block, median ≈ 950
- Z₂ symmetry sampling in the paramagnetic phase

Validation figures: [reports/figures/validate_1d.png](reports/figures/validate_1d.png), [reports/figures/validate_2d.png](reports/figures/validate_2d.png), [reports/figures/validate_3d.png](reports/figures/validate_3d.png).

## Code layout

```
src/ising/        Simulation, storage. Per-dim Metropolis + Wolff kernels.
scripts/          Runnable entry points: generation, validation, audit.
data/             HDF5 datasets (1d, 2d, 3d + cross-check + smoke files).
reports/figures/  Validation figures.
models/           Will hold CNN checkpoints (gitignored except .gitkeep).
notebooks/        Validation/analysis notebooks.
```

HDF5 schema:

```
/dim_<d>/L_<L>/T_<T:.4f>/
    configurations   (N, L, L, ..., L)  int8
    energies         (N,)               float64
    magnetizations   (N,)               float64
    attrs: T, L, dim, seed, algorithm, n_thermalization, decorrelation, n_samples
attrs at root: git_commit, created_utc, schema_version
```

Per-block seeds are derived from `SeedSequence([base, L, int(T*1e6)])` so any
single block can be regenerated bit-exactly without rerunning the others.

## Setup

```pwsh
python -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -r requirements.txt
```

Python 3.10+ recommended. On a Jetson Nano, install the NVIDIA-prebuilt PyTorch
wheel (ARM64 + CUDA 10.2) instead of the pinned `torch`.

## Regenerating the datasets

Each generation run is deterministic given the seed and produces the file
exactly as committed.

```pwsh
.venv\Scripts\python scripts\generate_1d.py --out data\ising_1d.h5
.venv\Scripts\python scripts\generate_2d.py --out data\ising_2d.h5
.venv\Scripts\python scripts\generate_3d.py --out data\ising_3d.h5
```

Wall time on a modern workstation: 1D ≈ 4 min, 2D ≈ 17 min, 3D ≈ 12 min.

## Validation

Per-dim physics checks against exact or literature values:

```pwsh
.venv\Scripts\python scripts\validate_physics_1d.py data\ising_1d.h5 --figure reports\figures\validate_1d.png
.venv\Scripts\python scripts\validate_physics_2d.py data\ising_2d.h5 --figure reports\figures\validate_2d.png
.venv\Scripts\python scripts\validate_physics_3d.py data\ising_3d.h5 --figure reports\figures\validate_3d.png
```

Cross-dimensional consistency audit (schema, dtype, energy convention,
effective sample size, Z₂ sampling, class balance, memory footprint):

```pwsh
.venv\Scripts\python scripts\audit_cross_dim.py
.venv\Scripts\python scripts\inspect_cnn_readiness.py
```

## References

- Onsager, L. (1944). *Crystal statistics I*. Phys. Rev. 65, 117.
- Wolff, U. (1989). *Collective Monte Carlo updating*. Phys. Rev. Lett. 62, 361.
- Ferrenberg, A. M. & Landau, D. P. (1991). *Critical behavior of the 3D Ising
  model*. Phys. Rev. B 44, 5081.
- Talapov, A. L. & Blöte, H. W. J. (1996). *Magnetization of the 3D Ising
  model*. J. Phys. A 29, 5727.
- Hasenbusch, M. (1999). *Improved estimators for the universal cumulant
  ratios*. J. Phys. A 32, 4851.
- Lundow, P. H. & Markström, K. (2009). *Critical behaviour of the Ising
  model on the 4D regular lattice*. Phys. Rev. E 80, 031104.
- Carrasquilla, J. & Melko, R. G. (2017). *Machine learning phases of matter*.
  Nature Physics 13, 431.
- Newman, M. E. J. & Barkema, G. T. (1999). *Monte Carlo Methods in
  Statistical Physics*.
