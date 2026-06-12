# Results

Summary of where the project ended up. Numbers are mean +/- std over 3
training seeds unless noted. Stages: A = trained on 2D only, B = 2D+3D,
C = 1D+2D+3D, D = 1D+2D+3D+4D. 4D and 5D were never used for training
except in stage D (4D).

## Main numbers

T_c at held-out dimensions, vs literature (4D: 6.6803, 5D: 8.778,
Lundow & Markstrom):

| stage | trained on | T_c(4D) | T_c(5D) |
|---|---|---|---|
| A | 2 | 5.872 (12.1% off, single seed) | - |
| B | 2,3 | 6.716 +/- 0.051 (0.54%) | 9.20 +/- 0.53 (4.8%) |
| C | 1,2,3 | 6.682 +/- 0.025 (0.03%) | 8.89 +/- 0.16 (1.2%) |
| D | 1,2,3,4 | 6.628 +/- 0.002 (0.79%, in-training) | 8.708 +/- 0.007 (0.80%) |

For reference, a linear fit of T_c(d) through d=2,3 predicts 6.754 (1.1%
off) and a quadratic through 1,2,3 gets 6.727 (0.7%). So stage B only
beats the trivial baseline by about 2x, but stage C beats it by ~35x.

A note on the 0.03%: the eval temperature grid near T_c(4D) has spacing
~0.06, and 0.03% is ~0.002 in temperature units, well below that. The
estimate comes from linearly interpolating the classifier output between
grid points, so read it as "within grid resolution", not as real
sub-grid precision.

Things I didn't expect:

1. Adding 1D helped a lot (B -> C is an 18x improvement on the 4D gap).
   1D doesn't even have a phase transition. My best guess is that having
   a transition-free dimension in training calibrates the classifier
   baseline across dimensions. I haven't ablated this against e.g.
   shuffled 1D data, so I can't yet rule out that it's just "more data".

2. Stage D's in-training 4D estimate (0.79% off) is worse than stage C's
   held-out one (0.03%). Training on the target dimension made the point
   estimate worse, though much more stable across seeds. I think the
   shared head has to compromise across all training dims, which biases
   each individual readout.

3. The 5D error bar collapses as training dims get closer to 5:
   0.53 -> 0.16 -> 0.007 across B/C/D. That's roughly an 80x reduction
   and matches what you'd expect from the decision-axis picture below.

## The exponent nu

I extract an effective correlation-length exponent from finite-size
scaling of the classifier crossover width (the temperature window where
P(disordered) goes from 0.25 to 0.75; width ~ L^(-1/nu) for a clean
transition).

The raw fits give nu way too big at low d (the network's decision
boundary has a finite sharpness, so the crossover can't shrink below
some floor). Fitting a shared floor c per checkpoint and redoing the
fits, with L in {4,6,8}:

| dim | stage B | stage C | literature |
|---|---|---|---|
| 2D | 0.756 +/- 0.054 | 0.754 +/- 0.051 | 1.0 |
| 3D | 0.594 +/- 0.069 | 0.648 +/- 0.096 | 0.6301 |
| 4D | 0.522 +/- 0.102 | 0.473 +/- 0.034 | 0.5 |

3D and 4D land within 1 sigma of literature. 4D is the held-out one, so
that's the result that matters: the exponent stops running at the upper
critical dimension and the network's readout reflects that. Stage C's
*uncorrected* nu(4D) is 0.521 +/- 0.037, also within 1 sigma, which I
find more convincing than the floor-corrected number.

2D is bad: ~0.75 vs the exact answer 1.0, consistently across seeds and
stages. That's a real systematic in the method, in the dimension with
the most training data, and I don't have a verified explanation
(presumably the decision-boundary smoothness matters most where the
true transition is sharpest, but I haven't tested that).

Adding the bigger 4D lattices (L=10,12) made the power-law fits *worse*
(stage B 4D: 0.522 -> 0.621 +/- 0.180; stage C: 0.618 +/- 0.059). At
first I thought this was a bug. It's probably the known logarithmic
corrections to FSS at exactly d=4 (arXiv:2408.15230) showing up once
there are enough lattice sizes to resolve them, but I haven't fit the
log-corrected form to check, so treat that as a plausible reading and
not a demonstrated one.

## The 5D freezing test

The sharp version of the upper-critical-dimension test is: does
nu(5D) = nu(4D) = 1/2 (the exponents freeze above d=4)? Stage B and C
checkpoints can't answer this because their 5D classifier doesn't
produce a usable crossover at L=4,6. The stage D checkpoint (trained
through 4D) does, on all three lattice sizes. Result, naive fits:

- nu(4D) = 0.494 +/- 0.017 (in-training, basically exactly 1/2)
- nu(5D) = 0.400 +/- 0.021 (held-out, ~5 sigma below 1/2)

So no plateau: the effective exponent keeps descending past d=4. I
don't think this kills the freezing story, because the 5D lattices are
tiny (L=8 max, and 4^5 = 1024 spins at the small end) and finite-size
corrections at that scale are big. But as measured, the sharp test
failed, and larger 5D lattices are the obvious next step.

## The decision axis

The original hypothesis for *how* the network transfers across
dimension was wrong. I expected near-critical configurations from
different dimensions to overlap in feature space ("universality
collapse"). They don't, at all: each dimension forms its own cluster,
and nearest-neighbour mixing between dimensions is ~0 at every
temperature.

What actually transfers is the direction within each cluster that
separates ordered from disordered (the decision axis). Cosine
similarities between dimensions' axes (stage B):

cos(2,3) = 0.84 +/- 0.06, cos(3,4) = 0.77 +/- 0.09,
cos(2,4) = 0.45 +/- 0.15.

Converting to angles and fitting angle = rate * |d_i - d_j| gives a
rotation rate of roughly 33 degrees per dimension, and that number
barely moves across the three training stages (33.0 +/- 5.8, 34.3 +/-
1.7, 32.3 +/- 5.1 for B, C, D). So it looks like a property of the
problem, not of the training set, though I should check per-seed
correlations across stages before leaning on that too hard - the
stability could partly be coincidence of means.

This picture explains most of the other observations: the staged
improvement in T_c (each added training dim leaves less rotation to
extrapolate), the 5D error-bar collapse, and why stage B/C classifiers
fail at 5D small-L while stage D's works.

## Known problems / todo

- nu(2D) is ~5 sigma off. Unexplained systematic.
- The 1D-as-control improvement isn't ablated against a synthetic
  transition-free dataset.
- The log-correction reading of the extended-L 4D fits is plausible but
  unverified (no log-corrected fit done).
- The resolution floor c (~0.05, stable across seeds) is fitted, not
  derived.
- Stage A is single-seed.
- Everything here is one universality class. XY or Potts would be the
  natural test of whether any of this generalizes.

## Reproducing

```
python scripts/baselines.py

python scripts/run_multiseed.py --train 2 3     --eval 3 4 5 --seeds 1 2 3 --out-prefix cnn_train23
python scripts/run_multiseed.py --train 1 2 3   --eval 3 4 5 --seeds 1 2 3 --out-prefix cnn_train123
python scripts/run_multiseed.py --train 1 2 3 4 --eval 3 4 5 --seeds 1 2 3 --out-prefix cnn_train1234

python scripts/multiseed_aggregate.py    --pattern "cnn_train23_seed*.pt"
python scripts/multiseed_nu_aggregate.py --pattern "cnn_train23_seed*.pt" --dims 2 3 4
python scripts/rotation_rate.py          --pattern "cnn_train23_seed*.pt" --dims 2 3 4
```

(same aggregate commands with the other two prefixes; add --dims 2 3 4 5
for the stage D nu run). Each 3-seed training run is ~6h on a laptop
CPU. Checkpoints and the 5D / extended-4D h5 files are gitignored but
regenerable from scripts/generate_*.py.
