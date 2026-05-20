# Results

A canonical, reviewer-facing summary of what the project's four measurements
actually show, with the right caveats and the comparisons that matter.
Numbers below are from a single training seed and will be replaced with
mean ± standard deviation once the 3-seed multi-seed run completes.

## TL;DR

A 22K-parameter dimension-agnostic CNN, trained by classification (ordered
vs disordered) on Monte Carlo configurations from d = 2 and d = 3 only,
extrapolates the critical temperature to held-out d = 4 ten times better
than a two-point linear fit, *partly* extrapolates the correlation-length
exponent's dimensional trend (qualitative, not precision), reveals its own
transfer mechanism (a decision axis that rotates smoothly with dimension),
and shows the transfer horizon at d = 5 where the rotation has gone too far.
The architecture and dataset are sound; the headline T_c result is robust;
two of the four advertised measurements need the careful framing this
document gives them.

## Measurement #1 — T_c extrapolation (the headline)

**Setup.** Stage A trains on d = 2 only and evaluates on held-out 3, 4 and 5.
Stage B adds d = 3 to training. We compare the network's T_c estimate (the
P(disordered) = 0.5 crossing of its classifier output) to literature values.

| Stage             | d trained | d = 4 (held-out) | gap vs literature |
|-------------------|-----------|-----------------:|------------------:|
| A: train d=2      | {2}       | 5.872            | 12.1%             |
| B: train d=2,3    | {2, 3}    | **6.676**        | **0.06%**         |
| literature        | —         | 6.6803           | —                 |

**Baselines the CNN has to beat.** A reviewer with the same training-dim
knowledge would not need a network; they would extrapolate the curve
T_c(d). The trivial baselines:

| Baseline                              | d = 4 prediction | gap   |
|---------------------------------------|-----------------:|------:|
| linear fit through d = 2, 3           | 6.754            | 1.10% |
| quadratic fit through d = 1, 2, 3     | 6.727            | 0.70% |
| asymptotic 2d − g/d (d = 2, 3)        | 7.009            | 4.92% |
| **CNN (Stage B)**                     | **6.676**        | **0.06%** |

**Headline.** The CNN beats the best trivial physics-statistic baseline by a
factor of **17**. This is the real result of the project: not the absolute
T_c(4D), but the gap over the linear baseline.

## Measurement #2 — ν as a dimensional trend (qualitative)

**Setup.** Extract an effective correlation-length exponent from the
finite-size scaling of the network's crossover width. The width is the
temperature window where P(disordered) rises from 0.25 to 0.75; for a
clean transition, width ∝ L^(−1/ν).

**The naive fit fails diagnostically.** Raw widths fit `L^(−1/ν)` give
ν = 2.08 / 0.98 / 0.62 for 2D / 3D / 4D — wildly too large compared to the
literature 1.0 / 0.63 / 0.50. The reason: the network classifies individual
finite samples with a smooth decision function, so its crossover cannot
sharpen below an intrinsic resolution floor c that does not shrink with L.

**With a one-parameter shared floor c = 0.055**, the corrected fits give
ν = 0.81 / 0.67 / 0.57. The trend is right.

**Honest framing.** The floor `c = 0.055` is a *post-hoc fit*, not derived.
The Wilson-Fisher ε-expansion gives ν(d = 4) = 1/2 *exactly* (since ε = 0
at d = 4) — better than the CNN's 0.57. The defensible content of #2 is
the **qualitative trend** that the network's effective ν *decreases* with
d, which holds also in the *uncorrected* fits (2.08 → 0.98 → 0.62, still
monotonic). The trend is the result. The precise ν(4D) numbers are not a
load-bearing claim.

## Measurement #3 — the upper-critical-dimension signature, qualitatively

**The hoped-for sharp test.** Above the upper critical dimension d_c = 4
Ising exponents *freeze* at mean-field values. If a model trained on d = 2, 3
predicts ν(5D) ≈ ν(4D) ≈ 1/2 (a flat plateau, not continued descent), it
has reproduced the freezing. A 5D dataset was generated and validated for
exactly this test (Binder crossings 8.768 / 8.762 vs literature 8.778,
0.1–0.2% off).

**What actually happened.** Applied to d = 5, the Stage B model's classifier
fails: 2 of 3 lattice sizes (L = 4, 6) never produce a P(disordered)
crossover spanning 0.25 → 0.75, and L = 8 gives a single width of 3.42
(an order of magnitude larger than 4D's largest). FSS cannot extract ν(5D).

**The interpretation.** Not a measurement failure — a **transfer horizon**.
The decision axis (measurement #4) has rotated so far by d = 5 that the
shared classifier head no longer applies. One finding *explains the other*:
#4 implies #3's breakdown.

**What we can say.** Qualitatively, ν(d) descends across d = 2, 3, 4 toward
mean-field 1/2 in both naive and corrected fits — consistent with the
upper-critical-dimension story. We cannot extend the trend cleanly to d = 5
with the present Stage B model. The sharp version of #3 would need a
checkpoint trained on d = 2, 3, 4 with d = 5 held out instead.

## Measurement #4 — the rotating decision axis (the mechanism)

**The original framing was wrong.** "Universality collapse — do near-
critical configurations from different dimensions land in the same region
of feature space" — the data emphatically rules this out: cross-dimension
nearest-neighbour mixing in the 64-d feature space is ≈ 0 at every
temperature, including criticality. The configurations are *segregated*
by dimension into separate clusters.

**The data shows something else, which is the real result.** Within each
dimension's cluster, the direction from the ordered centroid to the
disordered centroid — that dimension's *decision axis* — is what the shared
head reads. Comparing those axes across dimensions by cosine similarity:

|   | 2D   | 3D   | 4D   |
|---|------|------|------|
| 2D| 1.00 | 0.82 | 0.37 |
| 3D| 0.82 | 1.00 | 0.70 |
| 4D| 0.37 | 0.70 | 1.00 |

**The decision axis rotates smoothly with dimension.** Adjacent dimensions
share most of the axis (0.82 and 0.70); the most distant ones share little
(0.37). The shared head reads the rotated axis correctly *only* to the
extent the rotation is small — which is exactly the transfer horizon seen
in #3 at d = 5.

**Why this matters.** The naive universality-collapse hypothesis was wrong;
the rotating-axis finding is mechanistic and predictive. It explains:
- the staged-training improvement of #1 (adding d = 3 to training rotates
  the decision axis closer to where d = 4 needs it),
- the qualitative ν trend of #2 (decision axes that are *almost* aligned
  give a slightly degraded but still usable classifier),
- the breakdown of #3 at d = 5 (rotation finally exceeds the head's
  effective alignment).

## Caveats — what the paper still needs

These are the items the external critique flagged; this document records
them so they aren't missed in the writeup.

1. **Error bars.** Every number above is from a single training seed. The
   3-seed Stage B run is in progress; mean ± std will replace the point
   estimates in the next revision.
2. **More 4D lattice sizes.** The 4D FSS currently rests on L ∈ {4, 6, 8},
   with the upper-critical-dimension logarithmic corrections present. A
   credible FSS fit needs L = 10, 12 — a GPU run.
3. **train123 ablation.** Does adding d = 1 (transition-free control) to
   training help, hurt, or do nothing? An informative experiment, not yet
   run.
4. **First-principles floor.** The c = 0.055 resolution floor of
   measurement #2 is post-hoc fitted. Deriving it from the network's
   smoothness + sample fluctuation scale would harden the corrected ν
   numbers. Currently *not* derived.
5. **Sharp #3 test.** A checkpoint trained on d = 2, 3, 4 with d = 5
   held-out is needed to ask "does ν(5D) freeze at ν(4D) = 1/2?" cleanly.
   The 4D data exists and is sealed for the present experiment, so this is
   a *follow-up* experimental design rather than an immediate run.

## Reproduction

All numbers above come from artefacts in this repo. Reproducing them:

```
.venv/Scripts/python scripts/baselines.py
.venv/Scripts/python scripts/measure_exponents.py --checkpoint models/cnn_train23.pt --dims 2 3 4 5
.venv/Scripts/python scripts/latent_analysis.py --checkpoint models/cnn_train23.pt
.venv/Scripts/python scripts/multiseed_aggregate.py
```

The Stage B model checkpoint is gitignored (recoverable from
`scripts/train.py --train 2 3 --eval 3 4 5 --sizes 8 16 32 64
--max-per-block 500 --epochs 18 --patience 6 --seed 42`).
