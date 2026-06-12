# ising-dimension-extrapolation-cnn

Can a neural network trained on low-dimensional Ising configurations
predict what happens in a higher dimension it has never seen?

A small (~22K parameter) dimension-agnostic CNN is trained to classify
ordered vs disordered Monte Carlo configurations from d = 1, 2, 3 and
evaluated on d = 4 and 5, which are held out. d = 4 is the upper
critical dimension of the Ising class, where the critical exponents
stop running and freeze at mean-field values, so there's an exact,
known answer to be right or wrong against.

Headline numbers (3 seeds, mean +/- std): trained on 1D+2D+3D, the
network puts T_c(4D) at 6.682 +/- 0.025 against the literature 6.6803,
and T_c(5D) at 8.89 +/- 0.16 against 8.778 - dimensions it never saw.
A linear extrapolation of T_c(d) through d=2,3 is off by 1.1%, so the
network is doing considerably better than the obvious baseline.

The more interesting part is *why* it works: configurations from
different dimensions stay completely separated in the network's feature
space, but the direction separating ordered from disordered within each
dimension's cluster rotates smoothly with d, at roughly 33 degrees per
dimension regardless of which dimensions were trained on. That rotation
predicts which extrapolations work and how precisely. Details and
caveats (there are real ones, including a clearly wrong nu(2D)) in
[reports/RESULTS.md](reports/RESULTS.md).

## Background

| d | T_c | nu | notes |
|---|----:|----:|-----------|
| 1 | 0 | - | no transition |
| 2 | 2.2692 | 1 | Onsager exact |
| 3 | 4.5115 | 0.630 | MC numerics |
| 4 | 6.6803 | 0.5 | upper critical dimension, mean-field + log corrections |
| 5 | 8.778 | 0.5 | clean mean-field |

Locating T_c with a CNN inside one dimension is a solved problem
(Carrasquilla & Melko 2017). The question here is whether one set of
weights learns something about criticality that transfers *across*
dimension, and whether the extrapolation reproduces the
upper-critical-dimension behaviour. The architecture makes this
possible: the "convolution" is a per-site linear map of each spin and
the sum of its nearest neighbours (gathered with torch.roll, which on a
periodic lattice is exactly circular padding), so the same weights run
on input of any dimensionality. Global average pooling makes the
feature vector independent of lattice size and dimension.

## Datasets

Monte Carlo configurations validated against exact or literature
results before any training. Wolff cluster updates near/below T_c,
Metropolis above (Wolff is inefficient at high T), recorded per block.

| d | configs | T_c from Binder crossings | validated against |
|---|---:|---|---|
| 1 | 90,000 | n/a | exact E/N = -tanh(1/T) |
| 2 | 160,000 | 2.2543 (0.7% off) | Onsager |
| 3 | 120,000 | 4.5132 (0.04% off) | Ferrenberg-Landau, Talapov-Blote, Hasenbusch |
| 4 | 184,000 | held out | Lundow-Markstrom |
| 5 | 184,000 | 8.768/8.762 (0.1-0.2% off) | Lundow-Markstrom |

The 5D file (194 MB) and the extended 4D file (L=10,12, for the
finite-size-scaling checks) are over GitHub's file size limit, so they
are gitignored and regenerated from scripts/generate_5d.py /
generate_4d.py. Generation is deterministic per block
(SeedSequence([base, L, T]) style), so any block can be rebuilt
bit-exactly.

All datasets pass a cross-dimensional audit: int8 spins in {-1,+1},
identical energy convention, periodic-boundary translation invariance,
effective sample size > 100 per block, Z2 symmetry in the paramagnetic
phase.

## Layout

```
src/ising/        simulators (metropolis + wolff per dim), storage, dataset, CNN, training
scripts/          generation, validation, training, multi-seed aggregation
data/             HDF5 datasets (1-4D committed, 5D + extended 4D local)
reports/          RESULTS.md, notes.md, figures
models/           checkpoints (gitignored)
```

## Setup

```
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

Python 3.10+, CPU is fine (a 3-seed training run is ~6h on a laptop;
the model is tiny).

## Reproducing

Datasets:

```
python scripts/generate_1d.py --out data/ising_1d.h5
python scripts/generate_2d.py --out data/ising_2d.h5
python scripts/generate_3d.py --out data/ising_3d.h5
python scripts/generate_4d.py --out data/ising_4d.h5
python scripts/generate_4d.py --sizes 10 12 --out data/ising_4d_extended.h5
python scripts/generate_5d.py --out data/ising_5d.h5
```

Validation:

```
python scripts/validate_physics_2d.py data/ising_2d.h5 --figure reports/figures/validate_2d.png
python scripts/audit_cross_dim.py
```

(same pattern for the other dimensions). Training and analysis commands
are in reports/RESULTS.md.

## References

- Onsager (1944), Phys. Rev. 65, 117
- Wolff (1989), PRL 62, 361
- Ferrenberg & Landau (1991), PRB 44, 5081
- Lundow & Markstrom (2009), PRE 80, 031104
- Carrasquilla & Melko (2017), Nature Physics 13, 431
- Lundow et al. (2024), arXiv:2408.15230 (log corrections to 4D FSS)
- Newman & Barkema, Monte Carlo Methods in Statistical Physics
