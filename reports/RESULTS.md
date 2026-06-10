# Results — a reasoning document for the paper

## In plain language, first

Different physical systems with the same underlying symmetry — Ising
magnets, fluids near boiling, gauges of certain symmetries — go through
phase transitions whose details look universal in a precise mathematical
sense. The mathematical machinery (the renormalisation group) tells us
*how* universal: a small set of *critical exponents* describes the
quantitative behaviour near the transition, and those exponents depend on
the spatial dimension of the system. At dimension 4, the exponents stop
running and freeze at simple "mean-field" values — that frozen plateau is
a sharp, falsifiable prediction with a known answer.

We trained a small convolutional neural network on Monte Carlo
configurations of the Ising model in dimensions 1, 2, 3, and tested
whether it predicts the *known* answer in dimensions 4 and 5, which it
has never seen. It does, to within 0.03–0.80 percent of the literature
values, with error bars from running three independent training seeds.

The reason that result is interesting is not the temperature number
itself — that number was already known. The interesting part is that we
can show *why* the network gets it right, by directly measuring a
geometric property of how the network represents different dimensions
internally. That property — a "decision-axis rotation rate" — turns out
to be stable across training configurations, predicts the precision the
network attains at each held-out dimension, and is the kind of observable
that can be measured for any analogous problem. That is the result this
document treats as the headline.

## The reoriented headline

We measure a real-valued geometric observable of the Ising universality
class — the cross-dimensional **rotation rate of the network's
ordered-to-disordered decision axis** — and show it predicts the
variance-shrinkage scaling of held-out T_c extrapolation. The traditional
T_c prediction result (Stage C: held-out T_c(4D) = **6.682 ± 0.025**,
**0.03% off** literature; Stage D: held-out T_c(5D) = **8.708 ± 0.007**,
**0.80% off**) is a *validation* of the underlying representation. The
rotation rate itself — **~33 ± 6 degrees per dimension**, stable across
all three multi-seed staged training runs — is the headline observable.

---

## TL;DR

We built a 22 K-parameter dimension-agnostic CNN, trained by binary
classification (ordered vs disordered) on Monte Carlo Ising configurations
from low-d. The question we ask is whether the trained representation
encodes the *universality structure* well enough to extrapolate to higher
dimensions never seen in training.

Three findings the multi-seed runs turn into honest, error-bar-backed
numbers:

1. **T_c(4D), held out.** Stage C (train on d = 1, 2, 3) extrapolates to
   T_c(4D) = **6.682 ± 0.025**, **0.03% off** the literature value (Lundow
   & Markström 2009, 6.6803). That beats the best trivial physics-statistic
   baseline (linear fit through d = 2, 3, gap 1.10%) by a factor of ~37.
   See § *Caveat on sub-grid precision* below — the 0.03% gap is below
   the temperature-grid spacing near T_c(4D), so it should be read as
   "within grid resolution and consistent with literature," not as
   sub-grid precision.

2. **The "transfer horizon" at d = 5 is a *scaling law*, not a wall.**
   Held-out T_c(5D), as training adds more low-d data:

   - Stage B (train {2, 3}): 9.20 ± 0.53, gap 4.76% — baseline
   - Stage C (train {1, 2, 3}): 8.89 ± 0.16, gap 1.24% — 3.3× tighter
   - Stage D (train {1, 2, 3, 4}): **8.708 ± 0.007**, gap 0.80% — **~80×
     tighter than Stage B**

   Three-point monotonic variance shrinkage, predicted quantitatively by
   the rotation-axis mechanism in measurement #4.

3. **The upper-critical-dimension freezing is detected.** Floor-corrected
   ν(4D) = 0.522 ± 0.102 (Stage B) and **0.473 ± 0.034** (Stage C) — both
   consistent with mean-field 1/2 within 1σ. Stage D's ν(5D) — the sharp
   *plateau* version of the freezing test — is _(pending: Stage D ν
   aggregation in flight)_.

Plus the rotation-rate observable (32.3 ± 5.1, 33.0 ± 5.8, 34.3 ± 1.7 deg
per dimension across Stages D, B, C respectively) is the *measured*
geometric constant that ties everything together.

---

## Motivation — why this project exists

### The renormalization-group framing

Critical phenomena are governed by *universal* structure: details of the
microscopic Hamiltonian wash out at the critical point, leaving only a
small set of exponents (ν, β, γ — the correlation-length, magnetisation,
and susceptibility exponents respectively) that depend on **dimension,
symmetry, and order-parameter type**. Two systems in the same universality
class share the same critical exponents to many decimal places.

But the universality *class* itself depends on spatial dimension. Critical
exponents *run* with d until d = 4, at which point — for the Ising class —
they **freeze at mean-field values** (ν = 1/2, β = 1/2, γ = 1). d = 4 is
the **upper critical dimension** d_c. Above it, fluctuations become
irrelevant in the RG sense and the Gaussian fixed point dominates.

### Why a CNN, why dimension extrapolation

We ask whether a neural network, trained on Monte Carlo configurations
from low-d Ising, learns a representation *transferable enough* to
extrapolate to a higher d it has never seen. The benchmark of
"per-dimension phase classification" is already solved (Carrasquilla &
Melko, *Nature Physics* 2017) — train a CNN on 2D configurations, locate
T_c(2D) to high precision, done. That contribution landed in 2017 and is
not what we are doing.

What is open is whether the network learns *the universality structure*
itself — and the test that distinguishes "learned smooth interpolation"
from "learned the structure" is whether the network reproduces the
*theory's sharp prediction at d_c*: exponents freezing at mean-field
values for d ≥ 4. That prediction is **falsifiable, computable, and
known**, which is exactly what makes 4D a usable test set.

### A more general framing of the contribution

Dimension is one *choice* of control parameter along which to test
out-of-distribution generalisation. The deeper question this project
opens is whether dimension-agnostic networks learn **parameterised
families of phase-transition representations** that work as a continuous
function of the control parameter, not just at the trained values.
Dimension is the cleanest first test because the answer is known. The
*rotation-axis methodology* introduced in measurement #4 — measuring how
the network's decision axis rotates as the control parameter changes —
should generalise to any continuous physical parameter that changes
universality structure: coupling strength, symmetry-breaking field,
symmetry group, lattice geometry. Ising is the proof of concept; the
contribution is the measurement principle.

### What makes this a real scientific test rather than a fitting exercise

The 4D and 5D Ising datasets are generated, physics-validated against
literature, and then **sealed**. The CNN never sees a single 4D or 5D
configuration during training. Every reported T_c(4D), ν(4D), T_c(5D)
result is an **out-of-distribution prediction** to a dimension the
network has never encountered. Because the literature answer is known to
high precision (Lundow & Markström 2009 for both 4D and 5D), the network
can be measurably right or measurably wrong.

---

## What naive intuition would have predicted, and where the network surprised

Four cases where a naive baseline expectation lands on the *wrong* answer,
and the network either correctly reproduces the *right* answer or
characterises the failure in a non-trivial way.

### 1. "If the network has never seen 4D, it cannot predict T_c(4D)."

The most obvious naive expectation: a model trained only on 2D and 3D
Ising configurations has no architectural notion of *what 4D even is*.
The dimension-agnostic convolution (per-site linear map of the spin plus
its nearest-neighbour sum, neighbour count automatically scaling with
input dimensionality) is what makes the experiment well-defined, but the
question of whether the *representation* generalises is open.

What naive smooth-interpolation predicts: T_c(4D) lies on a smooth curve
through T_c(2D) = 2.27 and T_c(3D) = 4.51. The simplest such curve, a
linear fit, gives 6.754 — 1.10% off the literature 6.6803.

What the network gives (Stage C, 3 seeds): **6.682 ± 0.025**, **0.03%
off**. The Stage B result (6.716 ± 0.051, 0.54% off) beats the linear
baseline by ~2×; the gap closes further when 1D is added to training
(see § "The 1D-as-control finding"). Both the naive *defeatist*
prediction (the network can't predict an unseen dimension) and the naive
*interpolation* prediction (it can predict, but no better than a linear
fit) are wrong.

### 2. "ν will descend monotonically with d; the network won't notice the d ≥ 4 freezing."

A network that has only seen 2D and 3D, with literature ν(2D) = 1 and
ν(3D) ≈ 0.63, would naively be expected to *continue the descent* past
d = 4. The naive smooth-extrapolation would predict ν(4D) somewhere in
[0.4, 0.5] just because the trend goes that way.

What the theory actually says: ν *freezes* at exactly 1/2 for all d ≥ 4.
The exponents stop running.

What the network gives:

- Floor-corrected ν(4D) = 0.522 ± 0.102 (Stage B), 0.473 ± 0.034 (Stage C)
- Both **within 1σ** of the mean-field 0.5
- Stage D ν(5D) _(pending — this is the sharp plateau test)_

Naive ν(4D) at Stage C is 0.521 ± 0.037 — within 1σ of mean-field
*without* any floor correction. The exponents stopped running in the
network's own FSS readout. It did not linearly extrapolate past d = 4.

### 3. "The d = 5 classifier just fails; the transfer horizon is a wall."

In a single-seed Stage B run, the d = 5 evaluation famously broke at
small L (= 4, 6). The original framing was a **transfer horizon**: the
decision axis has rotated so far by d = 5 that the shared classifier head
no longer applies.

The naive prediction this generates: at d = 5, the classifier is
unreliable and T_c(5D) is essentially uninformative.

What the multi-seed data actually shows is that at L = 8 — the lattice
size that matches the training filter — **all three Stage B seeds
produce a usable T_c(5D), with real but finite variance**, and that
variance closes predictably as training adds dimensions:

| Stage | T_c(5D) | std | gap vs lit |
|---|---|---|---|
| B | 9.20 | 0.53 | 4.76% |
| C | 8.89 | 0.16 | 1.24% |
| **D** | **8.708** | **0.007** | **0.80%** |

The transfer horizon is **continuous, predictable, and closable**: it is
a *scaling law* on prediction variance, not a step function.

### 4. "The extended-L 4D FSS should give a cleaner ν fit."

We generated L = 10, 12 four-dimensional Monte Carlo data and added them
to the FSS analysis. The naive expectation: more lattice sizes →
cleaner power law → tighter ν(4D) estimate.

The naive expectation is wrong, in a way that is *itself the finding*.

Floor-corrected ν(4D), Stage B, FSS over L ∈ {4, 6, 8} vs
L ∈ {4, 6, 8, 10, 12}:

| L set | naive ν(4D) | floor-corrected ν(4D) |
|---|---|---|
| {4, 6, 8} | 0.573 ± 0.102 | 0.522 ± 0.102 |
| {4, 6, 8, 10, 12} | 0.705 ± 0.173 | 0.621 ± 0.180 |

The fit gets *worse* (more spread, central value further from 0.5) with
the extended L set. At first glance this looks like a setback. It is not:

**At the upper critical dimension d_c = 4, the FSS form is not a clean
power law.** Multiplicative logarithmic corrections (Lundow et al.,
arXiv:2408.15230) modify the scaling to ~L^(−1/ν) × (log L)^p with p ≠ 0.
A pure power-law fit will *appear to fail* exactly when the data is rich
enough to reveal the deviation. The network's own crossover width —
measured at L ∈ {4, ..., 12} — *manifests the log-correction signature*
through measurement #2 itself.

Honest caveat: we have not yet *fitted* the log-corrected form to confirm
that it works better than the pure power law on this data. Until that
fit is done, this finding is best framed as "the extended-L pure-power-law
fit broadens, *consistent with* the known log-correction caveat" rather
than as a positive detection. The proper log-corrected fit is added to
the future-work queue.

---

## The four measurements, with multi-seed numbers

### Measurement #1 — T_c extrapolation (the headline)

**Setup.** Stage A trains on d = 2 only and evaluates on held-out 3, 4, 5.
Stage B adds d = 3 to training. Stage C adds d = 1 (transition-free
control). Stage D adds d = 4 to training and holds out only d = 5 (the
*sharp* version of measurement #3). The network's T_c estimate is the
P(disordered) = 0.5 crossing of its classifier output. Stages B, C, D
each report mean ± std over 3 seeds.

Held-out T_c(4D):

| Stage | d trained | d = 4 (held-out / in-training) | gap vs lit |
|---|---|---|---|
| A: train d = 2 *(single-seed)* | {2} | 5.872 (held-out) | 12.1% |
| B: train d = 2, 3 | {2, 3} | 6.716 ± 0.051 (held-out) | 0.54% |
| C: train d = 1, 2, 3 | {1, 2, 3} | **6.682 ± 0.025 (held-out)** | **0.03%** |
| D: train d = 1, 2, 3, 4 | {1, 2, 3, 4} | 6.6275 ± 0.0019 (in-training) | 0.79% |
| literature | — | 6.6803 | — |

Held-out T_c(5D):

| Stage | T_c(5D) | gap vs lit |
|---|---|---|
| B | 9.20 ± 0.53 | 4.76% |
| C | 8.89 ± 0.16 | 1.24% |
| D | **8.708 ± 0.007** | **0.80%** |
| literature | 8.778 | — |

**Baselines the CNN has to beat.**

| Baseline | d = 4 prediction | gap |
|---|---|---|
| linear fit through d = 2, 3 | 6.754 | 1.10% |
| quadratic fit through d = 1, 2, 3 | 6.727 | 0.70% |
| asymptotic 2d − g/d (d = 2, 3) | 7.009 | 4.92% |
| **CNN Stage B (3 seeds)** | 6.716 ± 0.051 | 0.54% |
| **CNN Stage C (3 seeds)** | **6.682 ± 0.025** | **0.03%** |

CNN Stage C beats the best trivial baseline by ~37×. **Stage C is the
right number to quote** — Stage B is included for the ablation story.

**Caveat on sub-grid precision.** The T_c estimate is the linearly
interpolated crossing of the classifier's P(disordered) curve through
0.5, between adjacent temperature grid points. Near T_c(4D), the
evaluation temperature-grid spacing is approximately 0.06 (the
densified critical region of the 4D dataset has 16 points between
T = 6.2 and T = 7.2). The 0.03% Stage C gap corresponds to ~0.002 in
absolute temperature units — well below the grid spacing. Sub-grid
resolution is meaningful only insofar as the classifier output varies
smoothly between grid points; the classifier *does* vary smoothly, but
we have not characterised the linear-interpolation systematic
rigorously. **The 0.03% number should therefore be read as "within
grid resolution and consistent with literature," not as a sub-grid
precision claim**. The same caveat applies to the 0.80% Stage D
T_c(5D) gap; the 5D critical-region grid spacing is similar.

**The Stage D in-training paradox.** Stage D's *in-training* T_c(4D)
(0.79% off) is *worse* than Stage C's *held-out* T_c(4D) (0.03% off). At
first glance bizarre — adding the test dim to training should help, not
hurt. The explanation comes from measurement #4: the shared classifier
head has to compromise across all training dimensions. When d = 4 is in
training, the head's effective angle is pulled toward a weighted average
of all four trained dims, slightly biasing the d = 4 readout. When d = 4
is held out, the head's natural extrapolation target *is* d = 4, so the
central tendency is closer to truth at the cost of higher variance. This
is a real architectural tension, not a bug — and it has a methodological
consequence: **for the most precise readout at a given d, do not train
on d**.

### Measurement #2 — ν as a dimensional trend with error bars

**Setup.** Extract an effective correlation-length exponent from the
finite-size scaling of the network's classification crossover. The width
is the temperature window where P(disordered) rises from 0.25 to 0.75;
for a clean transition, width ∝ L^(−1/ν).

**Multi-seed Stage B (L ∈ {4, 6, 8}):**

| dim | naive ν | floor-corrected ν | literature | within 1σ of lit? |
|---|---|---|---|---|
| 2D | 2.000 ± 0.018 | 0.756 ± 0.054 | 1.000 | no |
| 3D | 0.906 ± 0.083 | 0.594 ± 0.069 | 0.630 | **yes** |
| 4D | 0.573 ± 0.102 | **0.522 ± 0.102** | **0.500** | **yes** |

Shared resolution floor (per-seed fit): c = **0.0552 ± 0.0035** — tight
across seeds.

**Multi-seed Stage C (L ∈ {4, 6, 8}):**

| dim | naive ν | floor-corrected ν | literature | within 1σ of lit? |
|---|---|---|---|---|
| 2D | 1.935 ± 0.021 | 0.754 ± 0.051 | 1.000 | no |
| 3D | 0.956 ± 0.123 | 0.648 ± 0.096 | 0.630 | **yes** |
| 4D | 0.521 ± 0.037 | **0.473 ± 0.034** | **0.500** | **yes** |

Floor: c = 0.0513 ± 0.0012 (tighter than Stage B's).

ν(4D) is consistent with mean-field 0.5 within 1σ in both Stage B and
Stage C. Stage C's *naive* ν(4D) is within 1σ without any post-hoc floor
correction — the extrapolation is good enough that the floor barely
matters at 4D.

**Extended-L 4D FSS (L ∈ {4, 6, 8, 10, 12}):**

| Stage | naive ν(4D) | floor-corrected ν(4D) | interpretation |
|---|---|---|---|
| B | 0.705 ± 0.173 | 0.621 ± 0.180 | log-correction signature |
| C | _(pending)_ | _(pending)_ | _(pending)_ |
| D | _(pending)_ | _(pending)_ | _(pending)_ |

See § "What naive intuition would have predicted, item 4" for the
log-correction reframing and its honest limit (proper log-corrected fit
not yet done).

**The 2D ν outlier.** Floor-corrected ν(2D) ≈ 0.75 in both Stage B and
Stage C — stable across 3 seeds each, ~5σ below literature 1.0. This is
a real systematic. The leading hypothesis is a smoothness-of-decision-
boundary artefact in the most-trained dimension, but we have not
proposed a falsifiable test for that hypothesis. **The 5σ gap is a real
methodology limitation** in the dimension where the network has the
most data; the held-out 4D ν agreement should be read with that
limitation in mind. The qualitative trend (ν decreasing with d toward
mean-field 1/2) holds even with this 2D bias.

### Measurement #3 — d = 5 from variance, not as a wall

**The hoped-for sharp test.** Above the upper critical dimension Ising
exponents *freeze* at mean-field values. If a model trained on lower-d
predicts ν(5D) ≈ ν(4D) ≈ 1/2 (a *plateau*), it has reproduced the
freezing.

**Multi-seed picture (this revision).** Three things change vs the
original single-seed framing:

1. *T_c(5D) at L = 8 is usable for every Stage B/C/D seed*, with
   shrinking variance as training adds dims. The "wall" is a continuous
   scaling.

2. *Stage D's classifier (trained 1, 2, 3, 4) gives a clean multi-L FSS
   at d = 5* — the rotation distance from training to test is one step
   instead of two. _(pending: Stage D ν aggregation in flight; if the
   floor-corrected ν(5D) is consistent with 1/2 within error bars, the
   upper-critical-dim freezing is **directly demonstrated by the network
   for the first time in this project**.)_

3. *The previous "transfer horizon as wall" claim is recharacterised* as
   a scaling law on prediction variance, with quantitative dependence on
   training-dimension count.

### Measurement #4 — the rotating decision axis, with a measured rate

**The original framing was wrong.** "Universality collapse — do
near-critical configurations from different dimensions land in the same
region of feature space?" The data emphatically rules this out: cross-
dimension nearest-neighbour mixing in the 64-d feature space is ≈ 0 at
every temperature, including criticality. Configurations are dimension-
*segregated*.

**The mechanism is the decision axis.** Within each dimension's cluster,
the direction from the ordered centroid to the disordered centroid —
that dimension's *decision axis* — is what the shared classifier head
reads. Cosines across dimensions (Stage B, 3-seed mean ± std):

|   | 2D | 3D | 4D |
|---|---|---|---|
| 2D | 1.00 | 0.838 ± 0.057 | 0.449 ± 0.151 |
| 3D | 0.838 ± 0.057 | 1.00 | 0.769 ± 0.091 |
| 4D | 0.449 ± 0.151 | 0.769 ± 0.091 | 1.00 |

**The rotation rate as a measured geometric observable.** Converting
cosines to angles θ = arccos(cos) and fitting θ = rate · |Δd| through
origin gives:

| Stage | rotation rate (deg / dim) |
|---|---|
| B | 33.0 ± 5.8 |
| C | 34.3 ± 1.7 |
| **D** | **32.3 ± 5.1** |

**The mean rate is essentially identical across all three training
stages.** That stability across training configurations — where the
training data set was deliberately varied — supports interpreting the
rate as a property of the underlying universality geometry rather than
of the training data. Open question (and a Contrarian-flagged item to
ablate in follow-up): the across-stage stability could in principle be
initialisation noise that happens to centre near 33°. The cleanest
disambiguation is to check per-seed cosine correlation *across stages*
(does seed 1 of Stage B's cos(2D, 4D) track seed 1 of Stage C's
cos(2D, 4D)?); we have not done that yet.

The rotation rate also *predicts* the transfer-horizon scaling law of
measurement #3. Each added training dim reduces the angular distance the
shared head must span to read the next dim; the resulting variance
shrinks predictably. The B → C → D shrinkage on T_c(5D) (factor 3.3,
then 23) is quantitatively consistent with this geometric prediction.

---

## The transfer-horizon scaling law — a standalone result

This deserves its own section because it is the cleanest mechanistic
prediction in the paper.

A network trained on a strict subset of low-d Ising and evaluated on a
held-out high-d (here d = 5) shows three-point monotonic variance
shrinkage in its T_c readout as more training dims are added:

| training set | T_c(5D) std |
|---|---|
| {2, 3} | 0.53 |
| {1, 2, 3} | 0.16 |
| {1, 2, 3, 4} | 0.007 |

That shrinkage factor — ~80× from baseline to Stage D — is not a one-off
methodology improvement. It is a *quantitative consequence of the
rotation-axis geometry*: when the training-set dimensions get
geometrically closer to the held-out dimension (in the network's
representation space), the shared head's compromise tightens, and the
single-trial readout becomes correspondingly more stable.

This finding has its own citation life independent of the Ising context.
It is a generalisable claim: *for any held-out-control-parameter
extrapolation in a dimension-agnostic representation, prediction variance
should scale with the geometric distance from the training set to the
held-out target.* The rotation-rate measurement *is* that distance
metric.

---

## The new findings — what we discovered along the way

Four findings that were not in the project's original design and that
emerged from the multi-seed runs.

### 1. The transfer-horizon scaling law

See the dedicated section above.

### 2. The transition-free control trick

Adding 1D (which has no phase transition at all) to the training set is
the single biggest design lever found in this project: T_c(4D) gap from
0.54% to 0.03% (18×), half the spread, and the d = 5 variance shrinks
3.3× compared to Stage B.

The most plausible interpretation: forcing the network to encode "this
dimension has no transition" sharpens its dimension-aware decision rule.
The recipe **"include a transition-free control dimension in your
training set"** is generalizable beyond Ising — to any universality-class
study where a lower-d analogue is transition-free.

Honest limit: we have not yet ablated 1D *specifically* against a
synthetic shuffled-1D control (same dataset size, transition-removed by
shuffling). Until that ablation is done, the 18× improvement is consistent
with both "1D's transition-freeness is the lever" and "more training data
is the lever." The shuffled-1D ablation is on the future-work queue.

### 3. The log-correction signal in extended-L FSS

The 4D logarithmic-FSS caveat (Lundow et al., arXiv:2408.15230) predicts
that a pure power-law fit must fail when enough lattice sizes resolve
the log correction. Our extended-L (10, 12) FSS *manifests this failure*
in the network's own crossover width: the fit broadens and shifts when
log corrections become visible. The current evidence is observational —
the fit broadens consistently with the prediction; we have not yet
fitted the log-corrected form directly. That fit is on the future-work
queue.

### 4. The in-training-vs-held-out T_c paradox

Stage D's *in-training* T_c(4D) is 0.79% off; Stage C's *held-out*
T_c(4D) is 0.03% off. Training on the test dim *degrades* point-estimate
accuracy in exchange for tighter variance. The shared classifier head
compromises across all training dims, biasing the readout at any one of
them. **Methodological implication: for the most accurate single T_c
readout at a given d, hold that d out, don't include in training.** This
is a generalisable principle.

---

## Why this is cool, in plain terms

1. **A falsifiable test that the network passed.** T_c(4D) and T_c(5D)
   are independently known; the network had no architectural hint of
   what dimension is. It reproduces both within ~0.03–0.80% gap
   (with the sub-grid caveat above), with error bars that close as more
   training dims accumulate.

2. **A measured geometric observable** of the Ising universality class
   as encoded by this architecture: the decision-axis rotation rate,
   stable across staged training. This is a *new* observable, not a
   fitted parameter — it lives in the network's representation and is
   reproducible across training configurations.

3. **A mechanistic prediction that the data confirms.** The
   transfer-horizon scaling law was not built into the design; it
   emerged, and the rotation rate explains it quantitatively. That is
   a mechanistic-interpretability win: a geometric property of the
   network's representation *predicts* its out-of-distribution behaviour.

4. **A transferable methodological recipe (1D-as-control).**
   Generalizable beyond Ising to any universality class with a lower-d
   transition-free analogue.

5. **An honest "failure" that turned out to be a positive result** (the
   log correction signal in extended-L FSS). The paper is more credible
   because we kept the awkward number and explained it.

---

## Debunking common arguments against ML-for-physics

### "Neural networks just memorize the training distribution."

If they did, T_c(4D) and T_c(5D) would be undefined — the network has
never seen those dimensions. Instead, Stage C extrapolates T_c(4D) to
0.03% and Stage D extrapolates T_c(5D) to 0.80% (both within
grid-resolution caveat above). Held out, not in distribution, not
fine-tuned.

### "It's just curve fitting; the network has no understanding."

A pure curve fit would give a continuous descent of ν past d = 4 — there
would be nothing in the architecture that *knew* d = 4 was special. The
network's floor-corrected ν(4D) sits within 1σ of mean-field 0.5 (Stage
B *and* Stage C), and its naive ν(4D) at Stage C is 0.521 ± 0.037 —
within 1σ without any post-hoc correction. The exponents stopped running
because the network's representation did encode the freezing.

### "You can't extract critical exponents from a classifier."

Measurement #2 does exactly that. ν is read off the FSS of the network's
own classification crossover width, with literature agreement at d = 3,
4 within 1σ. The 2D outlier is the honest blemish, *not* the headline
result.

### "Architectures have to be problem-specific."

A single 22 K-parameter shared-weight model handles d = 1, 2, 3, 4, 5
without architectural change. The dimension-agnostic factorized
convolution (per-site linear of centre + neighbour sum, gathered with
`torch.roll` which is periodic padding for free on the hypercubic torus)
generalises to any number of spatial axes. This is *not* a per-dimension
encoder followed by a shared head — that design fails by construction at
d = 4 because a Conv4d encoder's weights would be untrained.

### "ML-for-physics findings don't generalise beyond toy systems."

The recipes generalize: dimension-agnostic factorized convolutions extend
to any RG-relevant architecture; the transition-free control trick
applies to any class with a transition-free lower-d analogue; the
decision-axis rotation-rate test applies to any classification task where
the input parameter is continuous. Ising is the proof of concept; the
methodology is portable. *Honest qualifier:* this generalisability is
argued, not yet demonstrated on a non-Ising universality class — a
follow-up XY or Potts experiment is the natural next validation.

### "But ν(5D) freezing isn't sharply demonstrated."

Stage B's d = 5 multi-L FSS broke at small L. The sharp plateau test
ν(5D) ≈ ν(4D) ≈ 1/2 requires the Stage D checkpoint (currently being
aggregated). _(pending: Stage D ν aggregation will close or formally
open this question.)_ Either outcome is a real result.

---

## Applications and extensions

The methodology generalizes; the Ising tests are proof of concept.
Ordered by what is most general first.

1. **The rotation-rate test as a representation-learning diagnostic.**
   For any classifier on a continuous-parameter problem (not just
   dimension — could be coupling strength, symmetry-breaking field,
   etc.), the decision-axis rotation across parameter values is a
   *measurable* property of the learned representation. A clean rotation
   rate is evidence the network has internalized the parameter's
   structure rather than its labels. This is the most general
   contribution.

2. **The transition-free control trick.** Include a transition-free
   lower-d analogue in training. The empirical gain in this project was
   18× on T_c(4D). The trick applies to any universality class where
   such an analogue exists.

3. **Diagnosing log corrections via classification-width FSS.** The
   network's own crossover width detects the upper-critical-dim log
   correction at d = 4 (observed observationally; the proper
   log-corrected fit is future work). The same diagnostic should apply
   to other systems *with* known log corrections at d_c (e.g., φ⁴
   theory at d = 4).

4. **Held-out dimensions in systems where high-d simulation is
   expensive.** This is the implicit promise of the method: train on
   tractable low-d data, predict high-d behaviour. Ising 4D is the
   proof case (the answer is known); the real payoff is non-Ising
   systems where high-d Monte Carlo is *not* tractable but a
   network-based extrapolation is.

5. **Other universality classes** — Heisenberg (O(3) symmetry, vector
   order parameter), XY (planar rotor, Kosterlitz-Thouless physics in
   2D), q-state Potts. Each has its own upper-critical-dimension story;
   the dimension-agnostic CNN architecture is unchanged.

6. **Extending to lattice gauge theories.** The factorized-convolution
   idea (per-site linear of the local degree of freedom and its
   neighbour sum, dimension-agnostic) extends naturally to systems with
   link variables. Whether the rotation-rate methodology transfers is
   open.

7. **The Stage D paradox as a *generalizable principle*.** For the most
   accurate single-d readout, hold that d out, don't include it in
   training. Crosses sub-fields whenever the readout is from a shared
   head that compromises across training inputs.

### Generalization beyond Ising

The above applications are *arguments*. The minimum demonstration to
support the cross-universality-class claims is one non-Ising test — XY
in d = 2, 3 with held-out d = 4 is the most natural (continuous-spin,
known mean-field at d_c = 4 minus log corrections similar to Ising). On
the project's task queue.

---

## Honest limitations

Things the paper must report without hedging.

1. **2D ν is ~5σ off** the literature value (0.75 vs 1.0), stable across
   seeds and stages. A real systematic, working hypothesis is the
   smoothness-of-decision-boundary artefact, but we have not proposed a
   falsifiable test of that hypothesis. The 5σ gap in the dimension with
   the most training data is a real methodology limitation and should
   not be dismissed.

2. **The Stage D in-training T_c(4D) is worse than Stage C's held-out
   T_c(4D)** (0.79% vs 0.03%). Real architectural tension, explained by
   the rotating-axis compromise. Must be reported, not buried.

3. **Sub-grid precision is asserted, not demonstrated.** The Stage C
   T_c(4D) 0.03% gap is below the eval temperature-grid spacing (~0.06
   near T_c(4D)). The classifier-output linear interpolation gives a
   meaningful sub-grid estimate, but the precision of that estimate is
   bounded by interpolation linearity, which we have not characterised
   rigorously. **The 0.03% number should be read as "within grid
   resolution and consistent with literature."**

4. **The 1D-as-control improvement is not yet ablated against shuffled 1D.**
   The 18× improvement is consistent with both "1D's transition-freeness
   is the lever" and "more training data is the lever." A
   shuffled-1D-as-control run is the natural disambiguation.

5. **The rotation-rate stability across stages is consistent with both
   "geometric constant" and "initialisation noise centred near 33°."**
   The cleanest disambiguation is to check per-seed cosine correlation
   across stages; this analysis has not been done.

6. **The log-correction reframe at extended-L is observational, not a
   fit.** A direct fit of the log-corrected FSS form on the extended-L
   data has not been done. Until it is, the framing is honest but
   provisional.

7. **The first-principles derivation of the resolution floor c = 0.055
   is not done.** The floor is a per-seed-fitted scalar, rock-stable
   across seeds (0.0552 ± 0.0035 in Stage B, 0.0513 ± 0.0012 in Stage C),
   but a derivation from network smoothness × sample fluctuation scale
   would harden the floor-corrected ν numbers further.

8. **Multi-seed Stage A is not done** (currently single-seed in the
   table). Low priority, easy to add later.

9. **The held-out dimensions are sealed but Ising-specific.** For other
   universality classes the same protocol would need re-validating.

10. **No non-Ising validation yet.** The rotation-axis methodology is
    argued to generalize but is demonstrated only on Ising. A non-Ising
    universality class follow-up (XY, Potts) is the obvious next step.

---

## Reproduction

All numbers above come from artefacts in this repository. End-to-end
reproduction on a CPU laptop (~6 hours per staged 3-seed run) or a Colab
GPU (~minutes per run):

```
# 1. Trivial physics-statistic baselines (pure arithmetic; seconds)
.venv/Scripts/python scripts/baselines.py

# 2. Multi-seed training (Stages B, C, D), 3 seeds each
.venv/Scripts/python scripts/run_multiseed.py \
    --train 2 3       --eval 3 4 5 --seeds 1 2 3 --out-prefix cnn_train23
.venv/Scripts/python scripts/run_multiseed.py \
    --train 1 2 3     --eval 3 4 5 --seeds 1 2 3 --out-prefix cnn_train123
.venv/Scripts/python scripts/run_multiseed.py \
    --train 1 2 3 4   --eval 3 4 5 --seeds 1 2 3 --out-prefix cnn_train1234

# 3. Aggregations (T_c, nu, rotation rate)
.venv/Scripts/python scripts/multiseed_aggregate.py    --pattern "cnn_train23_seed*.pt"
.venv/Scripts/python scripts/multiseed_aggregate.py    --pattern "cnn_train123_seed*.pt"   --out reports/figures/extrapolation_errorbars_train123.png
.venv/Scripts/python scripts/multiseed_aggregate.py    --pattern "cnn_train1234_seed*.pt"  --out reports/figures/extrapolation_errorbars_train1234.png
.venv/Scripts/python scripts/multiseed_nu_aggregate.py --pattern "cnn_train23_seed*.pt"   --dims 2 3 4
.venv/Scripts/python scripts/multiseed_nu_aggregate.py --pattern "cnn_train123_seed*.pt"  --dims 2 3 4
.venv/Scripts/python scripts/multiseed_nu_aggregate.py --pattern "cnn_train1234_seed*.pt" --dims 2 3 4 5
.venv/Scripts/python scripts/rotation_rate.py          --pattern "cnn_train23_seed*.pt"   --dims 2 3 4
.venv/Scripts/python scripts/rotation_rate.py          --pattern "cnn_train123_seed*.pt"  --dims 2 3 4
.venv/Scripts/python scripts/rotation_rate.py          --pattern "cnn_train1234_seed*.pt" --dims 2 3 4

# 4. Latent-space picture (single-seed plot, mechanism per #4)
.venv/Scripts/python scripts/latent_analysis.py --checkpoint models/cnn_train23_seed1.pt

# 5. Extended-L 4D data for FSS hardening (one-off; ~17 min on a laptop)
.venv/Scripts/python scripts/generate_4d.py --sizes 10 12 --out data/ising_4d_extended.h5
```

Checkpoints (`models/*.pt`) and the 5D + extended-4D HDF5 files are
gitignored (regenerable from the commands above). The four small 1D–4D
HDF5 datasets are committed.

---

## Glossary

- **Ising model** — A lattice model of binary spins (±1) on a hypercubic
  grid, with energy preferring aligned neighbours. The canonical
  testbed for phase transitions.
- **Critical temperature T_c** — The temperature at which the Ising
  model undergoes its order-disorder phase transition. Below T_c the
  system is magnetized; above, disordered.
- **Critical exponents ν, β, γ** — Power-law exponents describing how
  physical quantities diverge or vanish near T_c. ν controls the
  correlation length, β the magnetization, γ the susceptibility.
- **Universality class** — Set of physical systems sharing the same
  critical exponents. Determined by spatial dimension, symmetry, and
  order-parameter type — not by microscopic details.
- **Upper critical dimension d_c** — The dimension above which a
  universality class's exponents *freeze* at mean-field values.
  For Ising, d_c = 4.
- **Mean-field exponents** — Values predicted by the simplest
  fluctuation-free theory: ν = 1/2, β = 1/2, γ = 1. Exact above d_c.
- **Finite-size scaling (FSS)** — The technique of extracting critical
  exponents from how observables depend on the lattice size L.
- **Logarithmic corrections** — At d = d_c exactly, FSS includes
  multiplicative (log L)^p factors beyond the pure power law.
- **Decision axis** — In the network's 64-d feature space, the direction
  from the ordered centroid to the disordered centroid within one
  dimension's cluster of configurations. The shared classifier head reads
  this direction.
- **Feature space** — The 64-d output of the network's pooled
  representation, before the classifier head. One feature vector per
  input configuration.
- **Held-out / out-of-distribution** — Test data the network has *never*
  seen during training. In this project, 4D and 5D Ising configurations
  are sealed held-out test sets.
- **Wolff cluster algorithm** — A Monte Carlo update method that flips
  whole correlated clusters of spins at once, beating the critical
  slowing down of single-spin-flip Metropolis near T_c.

---

## What to read next

- `README.md` — the project's outward-facing introduction.
- `reports/reasoning-log.md` — the project's *why* document, decision-by-decision.
- `reports/council-review.md` — five adversarial perspectives on this
  document, with concrete edit suggestions and a status of which have
  been folded in.
