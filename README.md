# ising-dimension-extrapolation-cnn

**Does a neural network learn the universal structure of phase transitions
well enough to extrapolate across spatial dimension — and can we *measure*
how it does so?**

This project tests that question on the Ising model. A 22 K-parameter
dimension-agnostic convolutional network is trained on Monte Carlo
configurations from dimensions *d* = 1, 2, 3 and evaluated on *d* = 4 and
*d* = 5 — dimensions it never sees during training. Because *d* = 4 is the
**upper critical dimension** of the Ising universality class, the theory
makes an exact, non-trivial prediction there: critical exponents stop
running with dimension and collapse to mean-field values. That makes
*d* = 4 a *falsifiable* test — the network can be measurably right or
wrong against a known answer.

The reoriented headline of this work is that we **measure a real-valued
geometric observable** of the cross-dimensional Ising representation —
the *rotation rate of the network's decision axis*, ~33 ± 6 degrees per
dimension — and show it **predicts** the variance-shrinkage scaling of
the held-out T_c extrapolation. The traditional T_c prediction results
(held-out T_c(4D) = 6.682 ± 0.025, **0.03% off literature** in Stage C;
held-out T_c(5D) = 8.708 ± 0.007, **0.80% off** in Stage D) become a
*validation* of the underlying representation, not the main contribution.

See [`reports/RESULTS.md`](reports/RESULTS.md) for the full
reasoning document, and
[`reports/council-review.md`](reports/council-review.md) for adversarial
critique with concrete edit suggestions.

## Why this is a physics question, not just a machine-learning demo

The renormalization group says critical phenomena are governed by
universal features independent of microscopic detail. But the universality
*class* itself depends on spatial dimension — the critical exponents run
with *d* until *d* = 4, where they freeze at mean-field values:

| d | T_c | β | γ | ν | Character |
|---|----:|----:|----:|----:|-----------|
| 1 | 0 | — | — | — | No transition |
| 2 | 2.2692 | 0.125 | 1.75 | 1 | Onsager-exact, non-trivial |
| 3 | 4.5115 | 0.326 | 1.237 | 0.630 | Non-trivial, MC numerics |
| 4 | 6.6803 | 0.5 | 1 | 0.5 | **Upper critical dimension — mean-field + log corrections** |
| 5 | 8.778 | 0.5 | 1 | 0.5 | Above d_c — clean mean-field |

A network that reproduces a known T_c in a *single* dimension is a
solved problem (Carrasquilla & Melko, *Nature Physics* 2017). The open
question is whether a network learns a representation of criticality
*universal enough to transfer across dimension*, and whether,
extrapolating to *d* = 4 (and *d* = 5), it reproduces the
upper-critical-dimension signature. A correct extrapolation is data-driven
evidence that the network has internalised the renormalisation-group
structure of the problem; an incorrect one is an informative limit on
machine-learned physics. *Either outcome is publishable.* Ours is the
former, with characterised limitations.

## What the project measures

Each measurement has a *known correct answer*, so the result is a
verifiable yes/no rather than an unfalsifiable number. Headline numbers
below are 3-seed mean ± standard deviation.

1. **Extrapolated transition temperature** `T_c(4D)` — ground truth
   6.6803 (Lundow & Markström 2009). Stage B (train d = 2, 3) gives
   6.716 ± 0.051 (0.54% off). Stage C (train d = 1, 2, 3, *with 1D as
   transition-free control*) gives **6.682 ± 0.025**, **0.03% off** —
   ~37× better than the linear fit baseline.
2. **Trend of the correlation-length exponent `ν(d)`**. Literature: ν(2D)=1,
   ν(3D)≈0.63, ν(4D)=1/2. Floor-corrected network estimates are 0.756 ± 0.054
   (2D), 0.594 ± 0.069 (3D), **0.522 ± 0.102 (4D, Stage B) / 0.473 ± 0.034
   (4D, Stage C)** — d = 3 and d = 4 within 1σ of literature; d = 2 a real
   ~5σ systematic that gets reported honestly. The exponents do *not*
   continue running past d = 4 in the network's readout.
3. **Upper-critical-dimension signature, made sharp by `d=5`**. Held-out
   T_c(5D) shrinks with training-set breadth: Stage B 9.20 ± 0.53, Stage C
   8.89 ± 0.16, **Stage D 8.708 ± 0.007** (0.80% off literature 8.778).
   The "transfer horizon" is a *scaling law*, not a wall. The sharp ν(5D)
   plateau test on the Stage D checkpoints came back negative
   (ν(5D) = 0.400 ± 0.021, continued descent rather than freezing) — read
   as a small-lattice limitation (L ≤ 8 in 5D), with larger 5D lattices
   as the decisive follow-up. See RESULTS.md Measurement #3.
4. **The transfer mechanism: a rotating decision axis.** *Not*
   universality collapse (the data actually rules that out: configurations
   from different dimensions form *segregated* clusters in feature space).
   What it does show is that the ordered → disordered direction within
   each dimension's cluster rotates *smoothly* with d, at a **measured
   rate of ~33 ± 6 degrees per dimension** that is stable across all
   three multi-seed training configurations. That stability is what
   suggests the rotation rate is a *geometric observable* of the
   universality class as encoded by this architecture.

The 4D and 5D datasets are generated, validated against literature, and
then **sealed as held-out test sets** — they are never used to train or
tune the network.

## Datasets

Monte Carlo configurations for *d* = 1, 2, 3, validated against exact or
literature physics before any network sees them. Sampling uses **Wolff
cluster updates** near and below `T_c` (where they defeat critical slowing
down) and **Metropolis-Hastings** above `T_c` (where clusters are O(1)
and Wolff is inefficient); per-block metadata records which algorithm
produced each block.

| d | Configurations | Sizes × Temps | T_c from Binder | Validated against |
|---|---:|---|---|---|
| 1 | 90,000 | 3 × 30 | n/a | exact ⟨E⟩/N = −tanh(1/T) |
| 2 | 160,000 | 4 × 40 | **2.2543** (0.7% off 2.2692) | Onsager exact |
| 3 | 120,000 | 3 × 40 | **4.5132** (0.04% off 4.5115) | Ferrenberg–Landau, Talapov–Blöte, Hasenbusch |
| 4 (test) | 184,000 | 3 × 46 | (held out) | Lundow–Markström 2009 (6.6803) |
| 5 (test, local-only) | 184,000 | 3 × 35 | **8.768 / 8.762** (0.1–0.2% off 8.778) | Lundow–Markström |

Plus extended-L 4D data (`data/ising_4d_extended.h5`, L = 10, 12) generated
locally for FSS hardening; gitignored due to size (~127 MB).

All datasets pass a cross-dimensional audit: bit-exact storage integrity,
`int8` configurations with values exactly `{−1, +1}`, periodic-boundary
translation invariance on every axis, identical energy convention
(H = −Σ_⟨ij⟩ s_i s_j, each bond counted once), effective independent
sample size > 100 in every block, and Z₂ symmetry in the paramagnetic
phase.

Validation figures: [1D](reports/figures/validate_1d.png),
[2D](reports/figures/validate_2d.png), [3D](reports/figures/validate_3d.png),
[4D](reports/figures/validate_4d.png),
[5D](reports/figures/validate_5d.png).

## Scope, honestly

This is a machine-learning-for-physics methods contribution validated on
known ground truth — not new Ising physics, and not a claim to discover an
unknown number. Its novelty is the *dimension* axis, the
*upper-critical-dimension* framing, and the *rotation-rate-as-observable*
methodology. Per-dimension phase classification is settled;
cross-dimensional extrapolation with a falsifiable 4D test, and a measured
geometric observable that predicts the extrapolation's behaviour, is what
is new.

The implicit promise of the method is for systems where the target *cannot*
be cheaply simulated; 4D and 5D Ising are the proof of concept precisely
because their answers are independently known. The architecture and
measurement principle should generalise to other universality classes
(XY, Potts, Heisenberg) and other continuous physical control parameters
(coupling, field, symmetry-breaking).

In its current state the project is a **rigorous prototype targeting a
workshop submission** (e.g. NeurIPS ML4PS). Honest limitations are
documented in [`reports/RESULTS.md`](reports/RESULTS.md) § *Honest
limitations*; the most material ones are the 2D ν systematic, the
not-yet-ablated 1D-as-control claim, and the not-yet-fitted log-corrected
FSS form.

## What this revision delivered (path-to-publication audit)

External review identified six items that separate "impressive prototype
on one unreplicated number" from "defensible methods paper." All six have
now been addressed:

1. **Multi-seed error bars.** ✅ Stage B, C, and D each run with 3 seeds;
   all reported numbers are mean ± std.
2. **Physics-statistic baselines.** ✅ [`scripts/baselines.py`](scripts/baselines.py)
   compares CNN to linear / quadratic / asymptotic fits; figure committed.
3. **Measurement #2 hardening.** ✅ Floor c = 0.0552 ± 0.0035 (Stage B),
   0.0513 ± 0.0012 (Stage C) is rock-stable across seeds; ν(4D) is now
   reported with error bars that *include* mean-field 0.5 in 1σ. Honest
   limit: derivation from first principles still future work.
4. **Measurement #4 reframed.** ✅ Universality-collapse hypothesis ruled
   out by data; *rotating decision axis* established with a measured rate
   (33 ± 6°/dim) stable across stages — promoted to the headline observable.
5. **Bigger 4D lattices.** ✅ L = 10, 12 generated locally
   (`data/ising_4d_extended.h5`); FSS rerun shows the broadening signature
   consistent with the known log corrections at d = 4. Honest limit: the
   *direct* log-corrected fit is future work.
6. **train123 (and train1234) ablations.** ✅ Stage C (train d = 1, 2, 3)
   and Stage D (train d = 1, 2, 3, 4) both run multi-seed. Stage C is the
   single biggest design improvement found (18× on T_c(4D)); Stage D
   produces the sharp d = 5 freezing-test checkpoint (ν(5D) aggregation
   in progress).

Plus three new findings that emerged from the multi-seed runs and were
not anticipated:

- **The transfer-horizon scaling law** — three-point monotonic variance
  shrinkage on held-out T_c(5D) (factor ~80 across Stages B → C → D),
  predicted quantitatively by the rotation-rate mechanism.
- **The in-training-vs-held-out paradox** — Stage D's *in-training*
  T_c(4D) (0.79% off) is *worse* than Stage C's *held-out* T_c(4D)
  (0.03% off). Methodological consequence: for the most accurate
  single-d readout, hold that d out.
- **The transition-free control trick** as a named, transferable
  methodology recipe.

## Code layout

```
src/ising/        Monte Carlo simulation and HDF5 storage.
scripts/          Generation, validation, training, multi-seed aggregation.
data/             HDF5 datasets (1d, 2d, 3d, 4d committed; 5d + extended-4d gitignored).
reports/          RESULTS.md, council-review.md, reasoning-log.md, figures.
models/           CNN checkpoints (gitignored except .gitkeep).
notebooks/        Colab notebook for cloud-GPU runs.
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

Per-block seeds derive from `SeedSequence([base, L, int(T*1e6)])`, so any
single block can be regenerated bit-exactly without rerunning the others.

## Setup

```pwsh
python -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -r requirements.txt
```

Python 3.10+. For Colab cloud-GPU runs, see
[`notebooks/colab_multiseed.ipynb`](notebooks/colab_multiseed.ipynb).

## Reproducing the datasets

Generation is deterministic given the seed; each run reproduces the
committed file exactly.

```pwsh
.venv\Scripts\python scripts\generate_1d.py --out data\ising_1d.h5
.venv\Scripts\python scripts\generate_2d.py --out data\ising_2d.h5
.venv\Scripts\python scripts\generate_3d.py --out data\ising_3d.h5
.venv\Scripts\python scripts\generate_4d.py --out data\ising_4d.h5
.venv\Scripts\python scripts\generate_4d.py --sizes 10 12 --out data\ising_4d_extended.h5
.venv\Scripts\python scripts\generate_5d.py --out data\ising_5d.h5
```

Wall time on a modern laptop: 1D ≈ 4 min, 2D ≈ 17 min, 3D ≈ 12 min,
4D core ≈ 15 min, 4D extended ≈ 17 min, 5D ≈ 25 min.

## Reproducing the multi-seed training and analysis

See [`reports/RESULTS.md`](reports/RESULTS.md) § *Reproduction* for the
end-to-end commands. Three lines of training (Stages B, C, D × 3 seeds
each) plus a handful of aggregations. On a CPU laptop the staged training
runs are ~6 hours each; on a Colab GPU they are minutes.

## Validating the physics

Per-dimension checks against exact or literature values:

```pwsh
.venv\Scripts\python scripts\validate_physics_1d.py data\ising_1d.h5 --figure reports\figures\validate_1d.png
.venv\Scripts\python scripts\validate_physics_2d.py data\ising_2d.h5 --figure reports\figures\validate_2d.png
.venv\Scripts\python scripts\validate_physics_3d.py data\ising_3d.h5 --figure reports\figures\validate_3d.png
.venv\Scripts\python scripts\validate_physics_4d.py data\ising_4d.h5 --figure reports\figures\validate_4d.png
.venv\Scripts\python scripts\validate_physics_5d.py data\ising_5d.h5 --figure reports\figures\validate_5d.png
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
- Lundow et al. (2024). *Logarithmic finite-size scaling of the 4D Ising model*.
  arXiv:2408.15230.
- Newman, M. E. J. & Barkema, G. T. (1999). *Monte Carlo Methods in
  Statistical Physics*.
