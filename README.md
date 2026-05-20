# ising-dimension-extrapolation-cnn

**Does a neural network learn the universal structure of phase transitions
well enough to extrapolate across spatial dimension?**

This project tests that question on the Ising model. A convolutional network
is trained on Monte Carlo configurations from dimensions *d* = 1, 2, 3 and
evaluated on *d* = 4 — a dimension it never sees during training. Because
*d* = 4 is the **upper critical dimension** of the Ising universality class,
the theory makes an exact, non-trivial prediction there: critical exponents
stop running with dimension and collapse to mean-field values. That makes 4D
a *falsifiable* test — the network can be measurably right or wrong against a
known answer.

## Why this is a physics question, not just a machine-learning demo

The renormalization group says critical phenomena are governed by universal
features independent of microscopic detail. But the universality *class*
itself depends on spatial dimension — the critical exponents run with *d*
until *d* = 4, where they freeze at mean-field values:

| d | T_c | β | γ | ν | Character |
|---|----:|----:|----:|----:|-----------|
| 1 | 0 | — | — | — | No transition |
| 2 | 2.2692 | 0.125 | 1.75 | 1 | Onsager-exact, non-trivial |
| 3 | 4.5115 | 0.326 | 1.237 | 0.630 | Non-trivial, MC-numerics |
| 4 | ≈ 6.68 | 0.5 | 1 | 0.5 | **Upper critical dimension — mean-field + log corrections** |

A network that reproduces a known T_c in a *single* dimension is a solved
problem (Carrasquilla & Melko, *Nature Physics* 2017). The open question — and
the contribution this project targets — is whether a network learns a
representation of criticality *universal enough to transfer across dimension*,
and whether, extrapolating to *d* = 4, it reproduces the signature of the
upper critical dimension. A correct extrapolation is data-driven evidence that
the network has internalized the renormalization-group structure of the
problem; an incorrect one is an informative limit on machine-learned physics.

## What the project measures

Each measurement has a *known correct answer*, so the result is a verifiable
yes/no rather than an unfalsifiable number. The first three of these are
quantitative; the fourth is a structural property of the trained network.

1. **Extrapolated transition temperature** `T_c(4D)` — ground truth 6.68
   (Lundow & Markström 2009). Compared *against trivial physics-statistic
   baselines* (linear and quadratic fits of `T_c(d)`), not just against
   literature: a 2-point linear fit already gets 1%, and the value of the
   CNN extrapolation is the *gap* over that baseline.
2. **Trend of the correlation-length exponent `ν(d)`** as `d` increases.
   Literature: `ν(2D)=1, ν(3D)≈0.63, ν(4D)=1/2`. We extract an effective
   `ν` from the network's classification-crossover width at each lattice
   size and ask whether it descends across dimension. *Not* a precision
   `ν(4D)` claim — the 1-loop Wilson-Fisher ε-expansion already gives
   `ν(4D)=1/2` exactly, for free, so the paper does not try to beat that.
   The defensible statement is qualitative: "the network's effective `ν`
   decreases with `d` toward the mean-field value".
3. **Upper-critical-dimension signature**, made sharp by `d=5`. Above the
   upper critical dimension exponents *freeze* at mean-field. If the network
   gives `ν(5D) ≈ ν(4D) ≈ 1/2` — a flat plateau, not a continued descent —
   it has reproduced the freezing.
4. **The transfer mechanism: a rotating decision axis.** *Not* universality
   collapse (the data actually rules that out: configurations from different
   dimensions form *segregated* clusters in feature space, with
   cross-dimension nearest-neighbour mixing ≈ 0). What the data does show
   is that the ordered→disordered direction within each dimension's cluster
   rotates *smoothly* with `d`: cos(2D, 3D) = 0.82, cos(3D, 4D) = 0.70,
   cos(2D, 4D) = 0.37. The shared decision axis between adjacent
   dimensions is what transfers; that rotation mechanistically explains
   the staged result in #1.

The 4D dataset is generated, validated against literature, and then **sealed
as a held-out test set** — it is never used to train or tune the network.
5D is held-out too.

## Datasets

Monte Carlo configurations for *d* = 1, 2, 3, validated against exact or
literature physics before any network sees them. Sampling uses **Wolff cluster
updates** near and below `T_c` (where they defeat critical slowing down) and
**Metropolis-Hastings** above `T_c` (where clusters are O(1) and Wolff is
inefficient); per-block metadata records which algorithm produced each block.

| d | Configurations | Sizes × Temps | T_c from Binder crossings | Validated against |
|---|---:|---|---|---|
| 1 |  90,000 | 3 × 30 | n/a (no transition) | exact `⟨E⟩/N = -tanh(1/T)` |
| 2 | 160,000 | 4 × 40 | **2.2543** (vs 2.2692, 0.7% off) | Onsager exact solution |
| 3 | 120,000 | 3 × 40 | **4.5132** (vs 4.5115, 0.04% off) | Ferrenberg–Landau, Talapov–Blöte, Hasenbusch |

All three datasets pass a cross-dimensional audit: bit-exact storage integrity,
`int8` configurations with values exactly `{-1, +1}`, periodic-boundary
translation invariance on every axis, an energy convention identical across
dimensions (`H = -Σ_⟨ij⟩ s_i s_j`, each bond counted once), effective
independent sample size > 100 in every block, and Z₂ symmetry in the
paramagnetic phase. Multiple lattice sizes per dimension is deliberate — it is
exactly what the finite-size-scaling exponent extraction (measurement 2) needs.

Validation figures: [1D](reports/figures/validate_1d.png),
[2D](reports/figures/validate_2d.png), [3D](reports/figures/validate_3d.png).

## Scope, honestly

This is a machine-learning-for-physics methods contribution, validated on
known ground truth — not new Ising physics, and not a claim to discover an
unknown number. Its novelty is the *dimension* axis and the
*upper-critical-dimension* framing: per-dimension phase classification is
settled, cross-dimensional extrapolation with a falsifiable 4D test is not.
The implicit promise of the method is for systems where the target *cannot*
be cheaply simulated; 4D Ising is the proof-of-concept precisely because its
answer is independently known.

In its current state the project is a **rigorous prototype, not a finished
paper**. A workshop-paper-tier methods contribution (ML for physical
sciences / NeurIPS-workshop level, not *Nature Physics*) is realistic — but
it needs the hardening listed below first.

## Path to publication — known gaps and how to close them

External review identified six items that separate "impressive prototype on
one unreplicated number" from "defensible methods paper":

1. **Multi-seed runs with error bars.** Every reported number (T_c = 6.676,
   the decision-axis cosines 0.82 / 0.70 / 0.37, ν ≈ 0.57) is from a *single*
   training seed. Without error bars, none of them are individually
   trustworthy. Plan: 3–5 seeds × Stage B and aggregate.
2. **Physics-statistic baselines.** A two-point linear fit through
   T_c(2D) = 2.27 and T_c(3D) = 4.51 already predicts T_c(4D) ≈ 6.75
   (within 1%). The CNN gets 0.06%, but the *gap* over a trivial baseline
   is what the paper has to defend. Same applies to ν via the ε-expansion.
   Plan: `scripts/baselines.py` runs the trivial extrapolations and a single
   figure shows CNN vs baselines vs literature.
3. **Measurement #2 hardening.** The "resolution floor" `c = 0.055` is a
   single-parameter post-hoc fit chosen to minimise residual. Either it is
   derived from first principles (network smoothness × sample fluctuation
   scale) or the precise `ν(4D) ≈ 0.57` claim is downgraded to "exponents
   trend toward mean-field" (which is true even with the *naive* uncorrected
   fits).
4. **Measurement #4 reframed.** The data does not show universality
   collapse in the feature space (it shows the opposite — dimension-
   segregated blobs). What it *does* show is that the ordered→disordered
   decision axis rotates smoothly with dimension (`cos(2D,3D) = 0.82`,
   `cos(3D,4D) = 0.70`, `cos(2D,4D) = 0.37`). That rotation mechanistically
   explains the staged improvement — and it should be the headline of #4,
   not a consolation finding.
5. **Bigger 4D lattices + a `train123` ablation.** Finite-size scaling at
   4D presently rests on three lattices `{4, 6, 8}`; with the upper-critical
   logarithmic corrections present, three points is not enough for a
   credible fit. A GPU run targeting `L ∈ {6, 8, 10, 12}` is the proper
   resolution. Likewise, an ablation training on 1D + 2D + 3D (with 1D as
   a transition-free control) tests whether *more* training dimensions help
   or whether 1D's absence of a transition adds noise.
6. **5D as a second held-out dimension.** Already generated and validated
   (Binder crossings at 8.768 / 8.762 vs literature 8.778, 0.1–0.2% off).
   If `ν(5D) ≈ ν(4D) ≈ 1/2` from the network — i.e. exponents *freeze* for
   `d ≥ 4` — that directly demonstrates the upper critical dimension,
   replacing the qualitative #3 claim with a sharp one.

These are weeks of focused work, not new infrastructure. The architecture
is sound; what is missing is uncertainty quantification, baselines, and
honest reframing.

## Code layout

```
src/ising/        Monte Carlo simulation and HDF5 storage.
scripts/          Generation, physics validation, cross-dimensional audit.
data/             HDF5 datasets (1d, 2d, 3d, + cross-check and smoke files).
reports/figures/  Validation figures.
models/           CNN checkpoints (gitignored except .gitkeep).
notebooks/        Validation and analysis notebooks.
```

HDF5 schema:

```
/dim_<d>/L_<L>/T_<T:.4f>/
    configurations   (N, L, ..., L)   int8, values in {-1, +1}
    energies         (N,)             float64
    magnetizations   (N,)             float64
    attrs: T, L, dim, seed, algorithm, n_thermalization, decorrelation, n_samples
root attrs: git_commit, created_utc, schema_version
```

Per-block seeds derive from `SeedSequence([base, L, int(T*1e6)])`, so any single
block can be regenerated bit-exactly without rerunning the others.

## Setup

```pwsh
python -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -r requirements.txt
```

Python 3.10+. On a Jetson Nano, install the NVIDIA-prebuilt PyTorch wheel
(ARM64 + CUDA 10.2) instead of the pinned `torch`.

## Reproducing the datasets

Generation is deterministic given the seed; each run reproduces the committed
file exactly.

```pwsh
.venv\Scripts\python scripts\generate_1d.py --out data\ising_1d.h5
.venv\Scripts\python scripts\generate_2d.py --out data\ising_2d.h5
.venv\Scripts\python scripts\generate_3d.py --out data\ising_3d.h5
```

Wall time on a modern workstation: 1D ≈ 4 min, 2D ≈ 17 min, 3D ≈ 12 min.

## Validating the physics

Per-dimension checks against exact or literature values:

```pwsh
.venv\Scripts\python scripts\validate_physics_1d.py data\ising_1d.h5 --figure reports\figures\validate_1d.png
.venv\Scripts\python scripts\validate_physics_2d.py data\ising_2d.h5 --figure reports\figures\validate_2d.png
.venv\Scripts\python scripts\validate_physics_3d.py data\ising_3d.h5 --figure reports\figures\validate_3d.png
```

Cross-dimensional consistency audit (schema, dtype, energy convention,
effective sample size, Z₂ sampling, class balance):

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
- Lundow, P. H. & Markström, K. (2009). *Critical behaviour of the Ising model
  on the 4D regular lattice*. Phys. Rev. E 80, 031104.
- Mehta, P. & Schwab, D. J. (2014). *An exact mapping between the variational
  renormalization group and deep learning*. arXiv:1410.3831.
- Carrasquilla, J. & Melko, R. G. (2017). *Machine learning phases of matter*.
  Nature Physics 13, 431.
- Newman, M. E. J. & Barkema, G. T. (1999). *Monte Carlo Methods in
  Statistical Physics*.
