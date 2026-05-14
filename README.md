# ising-dimension-extrapolation-cnn

CNN model for predictions on dimensional upscaling of Ising models.

The goal: generate Monte Carlo data for the Ising model in 1D, 2D, and 3D; validate
against known physics; train CNNs to identify thermodynamic phases; analyze
cross-dimensional transfer with an eye toward predicting 4D behavior.

## Layout

```
src/ising/        Simulation, storage, and (later) model code
scripts/          Runnable entry points (data generation, validation)
notebooks/        Validation and analysis notebooks
data/             Generated HDF5 datasets (gitignored)
models/           Trained model checkpoints (gitignored)
reports/          Figures and writeups
```

## Setup

Python 3.10+ recommended.

```
python -m venv .venv
.venv\Scripts\activate          # PowerShell on Windows
pip install -r requirements.txt
```

On the Jetson Nano, install the NVIDIA-prebuilt PyTorch wheel for ARM64 + CUDA 10.2
instead of the version pinned in `requirements.txt`.

## Reproducibility

Every simulation run records its seed, algorithm, lattice parameters, thermalization
length, and decorrelation interval as HDF5 attributes on the dataset it produces.
See `src/ising/storage.py` for the schema.

For full reproducibility, also note the git commit hash and the package versions
in your environment (`pip freeze > reports/env-<date>.txt`).

## Phase 1 quickstart (1D)

Generate 1D data across lattice sizes and temperatures:

```
python scripts/generate_1d.py --out data/ising_1d.h5
```

Then sanity-check the energy curve against the analytic result `<E>/N = -tanh(1/T)`:

```
python scripts/quick_validate_1d.py data/ising_1d.h5
```

A full validation notebook (with 2D Onsager checks, Binder cumulants, etc.) lands
in `notebooks/validation.ipynb` during Phase 3.

## Conventions

- Spins are stored as `int8` in `{-1, +1}`.
- Temperature `T` is in units of `J / k_B` with `J = 1` and `k_B = 1`.
- Lattices use periodic boundary conditions everywhere. The CNN code in Phase 4+
  will use circular padding (`F.pad(..., mode='circular')`) to match.
