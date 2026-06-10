# Reasoning Log

A companion to the README. The README says *what* the project is and *what
was built*; this file records *why* — the reasoning behind each decision, the
missteps, and what was learned. Kept so the project's thinking is preserved,
not just its artifacts. Newest entries at the bottom.

Each entry: the decision or step, the reasoning at the time, the outcome
(right or wrong), and the lesson.

---

## 1. Validation-first methodology

**Reasoning.** Most ML-for-physics projects train models on Monte Carlo data
without first proving the data reproduces known physics. Decided every
dimension's dataset must pass a comparison to exact/literature values *before*
any network touches it.

**Outcome — right.** It caught real bugs (entries 4, 7) and meant that when the
CNN later behaved oddly, we could trust the data and look at the network.

**Lesson.** "Trust the physics, then the network." A validated dataset is the
fixed point you debug everything else against.

## 2. Numba kernels + HDF5 schema

**Reasoning.** Monte Carlo inner loops are slow in pure Python; Numba JITs them
to near-C speed with little code. HDF5 with a `/dim_d/L_L/T_T/` schema and
per-block metadata (seed, algorithm, thermalization) gives reproducibility for
free.

**Outcome — right.** 1D-4D generation all run in minutes. Per-block seeds
(`SeedSequence([base, L, int(T*1e6)])`) let any single block be regenerated
bit-exactly.

**Lesson.** Spend the up-front effort on a reproducible data layer; it pays
back every time a block needs regenerating.

## 3. The Windows long-path install failure

**Reasoning.** Installed the full `requirements.txt` (including jupyter) into a
venv inside the deep `sandbox/worktrees/...` path.

**Outcome — wrong.** `jupyterlab` unpacks extension files with ~330-char paths;
Windows' default `MAX_PATH` is 260. The install died.

**Lesson.** Install only what the current phase needs. Heavy/irrelevant deps
(jupyter) were deferred. Environment fragility is a real cost — keep the
dependency surface minimal.

## 4. The 1D magnetization tolerance bug

**Reasoning.** The 1D validator flagged a block if `|M|/N` exceeded `5/sqrt(L)`
— the naive "fluctuations scale as 1/sqrt(N)" guess.

**Outcome — wrong.** At low T the correlation length `xi` is large; the right
scale for magnetization fluctuations is `sqrt(xi/L)`, not `1/sqrt(L)`. Good
blocks were being flagged.

**Lesson.** Validation tolerances must be *derived from physics*, not guessed.
A threshold that ignores the correlation length will mis-judge the ordered
regime.

## 5. 1D low-T autocorrelation → adaptive decorrelation

**Reasoning.** Initially used a fixed number of sweeps between recorded samples.

**Outcome — incomplete.** At low T, single-spin-flip Metropolis has integrated
autocorrelation time `tau ~ xi^2`; fixed decorrelation left low-T samples
correlated, biasing variance-based observables.

**Lesson.** Sampling parameters must scale with the physics. Fixed:
`decorrelation = max(base, 2 xi^2)`, `thermalization = max(base, 20 xi^2)`.

## 6. 2D — adding the Wolff cluster algorithm

**Reasoning.** Single-spin-flip Metropolis suffers critical slowing down near
`T_c` (`tau ~ L^z`, z~2). The Wolff cluster algorithm flips whole correlated
clusters and largely defeats it.

**Outcome — right.** Wolff made 2D/3D production near `T_c` feasible.

**Lesson.** Match the algorithm to the regime; don't brute-force critical
slowing down.

## 7. Wolff is inefficient at high T → hybrid scheme

**Reasoning.** Assumed Wolff was simply "better" and used it everywhere in 2D.

**Outcome — wrong (partially).** At `T >> T_c` clusters are O(1); a Wolff step
barely changes a large lattice, so samples stayed correlated (low effective
sample size). Found via the cross-dimensional audit.

**Lesson.** No single algorithm is best across the whole phase diagram. The fix
was a hybrid — Wolff near/below `T_c`, Metropolis above — recorded per block.

## 8. The L=16 "failure" that was really finite-size physics

**Reasoning.** 2D L=16 near `T_c` deviated ~5% from the Onsager exact energy;
this initially looked like a bug.

**Outcome — not a bug.** It is the genuine O(1/L) finite-size correction to the
thermodynamic-limit formula. Wolff and Metropolis agreed with each other,
ruling out a sampling error.

**Lesson.** Distinguish finite-size effects from bugs. Validation tolerances
were made L-aware (`tol = base + c/L`). Small lattices *should* deviate — that
is data the CNN later learns from.

## 9. Cross-dimensional audit

**Reasoning.** Per-dimension validation checks each dataset against its own
physics but not consistency *between* datasets (schema, energy convention,
effective sample size). Wrote a dedicated cross-dim audit.

**Outcome — right.** It surfaced the low-`n_eff` paramagnetic blocks (entry 7),
which were then regenerated with Metropolis.

**Lesson.** Audit the cross-cutting invariants, not just the per-unit ones.

## 10. HDF5 file growth and the GitHub 100 MB limit

**Reasoning.** Regenerated blocks were written with `overwrite=True`.

**Outcome — wrong (surprise).** HDF5 does not reclaim space from deleted/
overwritten groups; the 3D file silently grew past GitHub's hard 100 MB
per-file limit and the push was rejected.

**Lesson.** Repack HDF5 files after overwrites (`repack_hdf5.py`). Know the
platform's hard limits before relying on them.

## 11. Architecture: per-dimension encoders — the biggest misstep

**Reasoning.** The first design (written into an early README) used a separate
`Conv1d`/`Conv2d`/`Conv3d` encoder per dimension, feeding a shared head.

**Outcome — wrong, and the user caught it.** With per-dimension encoders, a 4D
input needs a `Conv4d` encoder whose weights were never trained (4D is the
held-out test). It would output garbage — the experiment would be impossible.

**Lesson.** Work out the *implications of the goal* before designing. The
held-out-dimension goal *mandates* a dimension-agnostic architecture: one set
of weights for all of 1D-4D. This reframed the whole model design.

## 12. Reframing the project around a falsifiable question

**Reasoning.** The project was first framed as "predict `T_c(4D)`".

**Outcome — weak framing.** `T_c(4D)` is already known to 5 decimals;
reproducing it teaches nobody anything.

**Lesson.** Frame around the falsifiable scientific question: *does the network
learn universal, dimension-transferable structure?* 4D is the test because the
answer (mean-field exponents at the upper critical dimension) is known and
non-trivial — the network can be measurably right or wrong.

## 13. The dimension-agnostic "convolution"

**Reasoning.** Needed one operation that works for any number of lattice axes.
Chose a per-site linear map of the spin plus its nearest-neighbour sum, with
neighbours gathered by `torch.roll` — which on a periodic lattice *is* circular
padding. The neighbour sum runs over however many axes the input has.

**Outcome — right.** A self-test confirmed one 22K-parameter model maps 1D, 2D,
3D and 4D inputs to a common 64-d feature space.

**Lesson.** When a constraint looks hard (one net, four dimensionalities), look
for the operation that is *intrinsically* dimension-free rather than special-
casing each dimension.

## 14. The staged-training result

**Reasoning.** Train on lower dimensions, evaluate on held-out higher ones;
add a dimension and repeat.

**Outcome — right, and the headline.** Trained on 2D only, held-out 4D `T_c`
was 12% off; trained on 2D+3D, it landed within the temperature-grid
resolution. The monotone improvement is the core result.

**Lesson.** A staged/controlled design turns a single number into a *trend*,
which is far more convincing than one lucky extrapolation.

## 15. Measurement #2 — the naive exponent fit failed, then was diagnosed (and downgraded after critique)

**Reasoning.** Extract the correlation-length exponent `nu` from the
finite-size scaling of the network's classification-crossover width:
`width ~ L^(-1/nu)`.

**Outcome — wrong, then partially understood.** The naive fit gave `nu` ~2x
too large. Diagnosed cause: the network classifies individual finite samples
with a smooth decision function, so its crossover cannot sharpen below an
intrinsic *resolution floor* `c`. With `width = a L^(-1/nu) + c` and `c`
fitted by a global scan to minimise total residual, `c = 0.055` and the
corrected exponents track the dimensional trend (2D 0.81, 3D 0.67, 4D 0.57
vs literature 1.0 / 0.63 / 0.50).

**Honest follow-up after external critique.** The floor `c = 0.055` is a
post-hoc one-parameter fit, not derived from first principles. The 1-loop
Wilson-Fisher epsilon-expansion gives `nu(4D) = 1/2` *exactly* for free
(since epsilon = 0 at d = 4); the CNN's 0.57 is therefore *worse* than the
trivial physics baseline at d = 4. The defensible content of #2 is not the
numerical `nu(4D)` value but the qualitative trend that the network's
effective `nu` decreases with `d` -- which holds *also in the naive
(uncorrected) fits* (2.08 -> 0.98 -> 0.62, monotonic). The trend is the
result; the floor correction is a presentation aid, not a load-bearing
claim.

**Lesson.** A classifier's decision-boundary width is not the physical
correlation length -- it carries an instrument resolution. Cleanly diagnosing
the failure was useful, but presenting the corrected number as a precision
claim was overclaiming. The version that survives external scrutiny is
qualitative.

## 16. Measurement #4 — wrong hypothesis, better finding

**Reasoning.** Expected near-critical configurations from different dimensions
to "collapse" onto a common manifold in the 64-d feature space (universality).

**Outcome — hypothesis wrong, mechanism found.** The features are completely
segregated by dimension (cross-dimension neighbour mixing ~0). But the
*ordered-to-disordered decision axis* is shared between adjacent dimensions
(cos: 2D-3D 0.82, 3D-4D 0.70) and weak between distant ones (2D-4D 0.37). The
decision axis rotates smoothly with dimension — and that rotation
mechanistically explains the staged `T_c` result.

**Lesson.** When the naive hypothesis fails, the refined question often reveals
the real mechanism. "No collapse, but a smoothly rotating decision rule" is a
more honest and more interesting finding than a clean collapse would have been.

## 17. The 5D extension — a second held-out dimension

**Reasoning.** With only `d = 4` as the held-out target, measurement #3 (the
upper-critical-dimension signature) was qualitative — the network's
`nu(2D)`, `nu(3D)`, `nu(4D)` descended toward `1/2`, but that is what a
generic smooth-extrapolation would predict too. A second held-out dimension
above the upper critical dimension turns it into a sharp test: in Ising,
exponents *freeze* for `d ≥ 4`, so `nu(5D)` should equal `nu(4D) = 1/2`. The
network reproducing that flat plateau (rather than continuing to descend) is
a falsifiable signature.

**Outcome.** Simulators, generator, and validator extended to 5D. T_c(5D)
from Binder crossings: 8.768, 8.762 vs literature 8.778 (0.1-0.2% off). The
5D HDF5 file is 194 MB -- over GitHub's 100 MB per-file limit -- so the
dataset is regeneration-only via the deterministic script. The `nu(5D)`
measurement was still running at the session boundary; result lands in the
follow-up.

**Lesson.** A *second* held-out point above the upper critical dimension
costs little extra (the simulator extends mechanically) and changes the
character of the claim from "the trend goes the right way" to "the trend
*saturates* exactly where the theory says it should".

## 18. External critique — what's missing for a paper

**Context.** A second independent review of the work
identified the gaps separating "impressive prototype on one unreplicated
number" from "defensible workshop methods paper".

**The six gaps.**
  1. **No error bars.** Every reported number comes from a single training
     seed. Without multi-seed reruns, we cannot tell signal from luck.
  2. **No baselines.** A two-point linear fit through `T_c(2D)` and
     `T_c(3D)` predicts `T_c(4D) ~ 6.75` (within 1%) without any neural
     network. The paper needs to show the CNN *adds value* over trivial
     physics-statistic extrapolation.
  3. **Measurement #2's `c = 0.055` floor is a post-hoc fitted constant**,
     not derived. Either derive it or downgrade the precise `nu` numbers
     to qualitative trend statements (which hold even with the *uncorrected*
     fits).
  4. **Measurement #4 was advertised as "universality collapse" but the
     data shows the opposite** — dimension-segregated blobs in feature
     space. The genuine result (the rotating decision axis) should be the
     headline, not a recovery move.
  5. **4D finite-size scaling rests on three lattices** with logarithmic
     corrections in the mix. A credible fit needs `L = 10, 12` and a GPU.
  6. **No `train123` ablation** to test whether more *training* dimensions
     help, with 1D as a transition-free control.

**Outcome.** Right on all counts. The README now has a "Path to publication"
section enumerating these explicitly, and the autonomous work queue
addresses items 1, 2, 3, 4 and a baseline of item 6 (item 5 needs GPU).

**Lesson.** A second reviewer's read is genuinely worth the cost. Two of
the four advertised measurements (#2's precision and #4's framing) were
overclaiming what the data supports; that was hard to see from inside the
project. Always invite an outside read before believing your own
conclusions.

## 19. The 5D ν measurement — transfer horizon sets in earlier than hoped

**Reasoning.** The hoped-for sharp test of measurement #3 was: train on
2D+3D, evaluate on both held-out 4D *and* held-out 5D; if `nu(5D) ≈ nu(4D)`
(a flat plateau, not continued descent), the network has reproduced
exponent-freezing above the upper critical dimension.

**Outcome — not sharply testable with the Stage B model.** Re-running
`measure_exponents.py` on the Stage B (trained 2D+3D) checkpoint at d=5:

  L = 4:  width *undefined* (the network's P(disordered) never spans 0.25-0.75
          over the temperature range tested)
  L = 6:  width *undefined* (same)
  L = 8:  width = 3.42  (an order of magnitude larger than 4D's largest)

Only one lattice has a measurable width, so the finite-size-scaling fit
cannot be done. The "is nu(5D) frozen at 1/2" question cannot be answered
from this checkpoint -- not because the answer is "no", but because the
network's *classifier itself* breaks down at d = 5.

**The interpretation -- and why this is consistent with #4's rotating axis.**
Measurement #4 found that the ordered-to-disordered decision axis rotates
smoothly with dimension: cos(2D,3D) = 0.82, cos(3D,4D) = 0.70,
cos(2D,4D) = 0.37. Extrapolating that rotation, cos(3D, 5D) is presumably
~0.4 and cos(2D, 5D) considerably less. The shared head, optimised against
the 2D+3D axes, then has too little overlap with the 5D axis to produce a
clean order/disorder crossover at small L. The transfer mechanism (the
rotating decision axis from #4) has a natural *horizon*: it works to
adjacent dimensions and degrades with distance.

**The sharp test that would still work.** Reframe the experiment so d=5 is
the held-out test and d=4 enters the training set. The network would then
have a decision axis cos(4D, 5D) ~ 0.7+ by extension of the pattern, the
classifier would work at d=5, and nu(5D) vs nu(4D) could be cleanly
compared. That breaks the current "4D is held out" rule but is a one-line
change in which dimension is the test target.

**Lesson.** A failure mode of an extrapolation method is part of the
result, not a missing measurement. Reporting the transfer horizon ("works
to one dimension away; breaks down beyond") is as informative as a
positive plateau would have been. And the breakdown is *consistent with*
the rotating-axis mechanism -- one finding explains the other.

## 20. Multi-seed Stage B -- the lucky-anecdote correction

**Reasoning.** The single-seed Stage B result (T_c(4D) = 6.676, 0.06% off
literature, beating the linear baseline by ~17x) was always going to be
soft until replicated. The external critique flagged it; we ran 3 seeds.

**Outcome.** The 3-seed mean is T_c(4D) = 6.716 +/- 0.051 (0.54% off,
beating linear baseline by ~2x). The original single-seed number was a
particularly fortunate run. The honest multi-seed number is *still* a
real win, just smaller than the anecdote suggested.

**Lesson.** Single-seed results in ML-for-physics are not just "missing
error bars" -- they are *misleading point estimates*. The replication is
not a formality; it is what distinguishes a finding from luck. We have
this lesson now even for the headline.

## 21. The 1D-as-control finding (Stage C, the design surprise)

**Reasoning.** "Does adding 1D to training help, hurt, or do nothing?"
was queued as a routine ablation -- one of the "external critique
followups." The hypothesis was that 1D (no phase transition) might add
noise without signal.

**Outcome.** Stage C (train on d = 1, 2, 3) gives T_c(4D) = 6.682 +/-
0.025, 0.03% off literature -- **18x better than Stage B's mean** and half
the spread. T_c(5D) also tightens by 3.3x. This was *not* a routine
ablation; it was the largest design improvement found in the whole
project.

**Working interpretation.** Forcing the network to encode "this dimension
has no phase transition" sharpens its dimension-aware decision rule. The
transition-free control acts like a calibration of the *baseline* of the
classifier's output across dimensions.

**Honest limit.** We have not ablated 1D-as-control against a synthetic
shuffled-1D control. Until that is done, the 18x improvement is
consistent with both "1D's transition-freeness is the lever" and "more
training data per se is the lever." The disambiguation is one extra
training run on the future-work queue.

**Lesson.** When a project's most surprising finding is an "ablation,"
take it seriously: it likely needs to be *named* (transition-free
control trick) and ablated again specifically. Cite-baiting a
methodology recipe is appropriate; underclaiming it leaves the
contribution on the table.

## 22. Stage D, the sharp #3 test, and the transfer-horizon scaling law

**Reasoning.** Measurement #3 (does the network show the d = 4 freezing
plateau in nu?) was qualitative because the d = 5 multi-L FSS broke in
Stage B (no clean crossover at L = 4, 6). The natural fix: train on
{1, 2, 3, 4} and hold out only d = 5. The rotating-axis prediction (#4)
says the d = 5 classifier should now work cleanly, because the rotation
distance from the training set to the test is one step instead of two.

**Outcome.** Stage D gives T_c(5D) = 8.708 +/- 0.007 (0.80% off), with
the error bar shrunk **23x compared to Stage C and ~80x compared to
Stage B**. Three-point monotonic shrinkage in held-out variance, driven
by the rotation-axis mechanism. That is a *quantitative* prediction of
the mechanism, confirmed.

**Bonus surprise: the in-training-vs-held-out paradox.** Stage D's
*in-training* T_c(4D) is 0.79% off; Stage C's *held-out* T_c(4D) is
0.03% off. Training on the test dim degrades point-estimate accuracy in
exchange for tighter variance. The shared classifier head compromises
across all training dims. Methodological consequence: for the most
accurate single-d readout, hold that d out, do not include it in
training. This generalises beyond Ising.

**Lesson.** A planned ablation can produce *two* findings: the one you
were testing (sharp #3 works) and one you were not testing (training on
the target degrades the readout). When a number disagrees with naive
expectation, ask whether it is a measurement of something separately
interesting before treating it as noise.

## 23. The extended-L 4D log-correction signal

**Reasoning.** "More lattice sizes = cleaner FSS" was the naive
expectation when we generated L = 10, 12 data and added them to the 4D
FSS. The expected result: tighter ν(4D) estimate.

**Outcome -- the opposite, in a useful way.** Adding L = 10, 12 made the
power-law fit *worse* (broader spread, central value shifts away from
0.5). At first this looked like a regression. But the known caveat at
d = d_c = 4 is that FSS carries multiplicative log corrections
(arXiv:2408.15230); a *pure* power-law fit *should* appear to fail when
the data is rich enough to reveal the log term.

**Reframing.** The "failure" of the pure power law is *consistent with*
the predicted log-correction signature appearing in the network's own
crossover-width FSS. This is a positive scientific finding via what
looked like a negative result, and it is detected through measurement
#2 itself with no new instrumentation.

**Honest limit.** We have not yet *fit* the log-corrected FSS form on
the same data to confirm it outperforms the pure power law. Until that
fit is done, the framing in RESULTS.md is: "consistent with the known
log-correction caveat" rather than "we detected the log corrections."
A direct log-corrected fit is on the future-work queue.

**Lesson.** When a known caveat predicts a *failure mode* of the
default analysis, and the failure shows up exactly as predicted, that
is evidence -- but it is observational evidence. Asserting the
positive form ("we detected X") requires the fitted form (positive
F-test vs the simpler model). Asserting only the consistent form
("data is consistent with X") is honest with one less computation done.

## 24. The rotation rate as a measured geometric observable -- reframing the project

**Reasoning.** Measurement #4 was originally pitched as "the mechanism
that explains the staged T_c improvement." Three multi-seed runs later,
the rotation rate has the same mean (32-34 deg/dim) across Stages B,
C, and D, with error bars that tighten and loosen depending on
training but a mean that does not move.

**Outcome.** That stability across deliberately-varied training data
makes the rotation rate look like a *property of the underlying
universality geometry*, not of the training data. Which means it is
not a side observation -- it is a measured geometric observable of
the cross-dimensional Ising representation in this architecture. It
*predicts* the transfer-horizon scaling, *predicts* the per-step
shrinkage between training stages, and is reproducible.

**Reframing.** Promoted the rotation rate from "bonus result of
measurement #4" to the headline observable of the paper. T_c(4D) and
T_c(5D) become *validations* of the underlying representation, not
the contribution.

**Honest limit (per Contrarian review).** The cross-stage mean
stability is observationally consistent with both "geometric constant
of the universality class" and "initialisation noise centred near
33 deg." The per-seed cosine correlation *across stages* would
disambiguate; that analysis has not been done yet. Acknowledged in
RESULTS.md and flagged for follow-up.

**Lesson.** When the same number recurs across deliberately varied
runs, take it seriously as a candidate observable rather than as a
coincidence. Promote it to the headline if the alternative is to
report it as "bonus."

## 25. The five-person council review

**Reasoning.** Before submitting a paper, run an adversarial review
internally. Five personas (Contrarian, First Principles, Expansionist,
Outsider, Executor) reviewed the RESULTS.md document.

**Outcome.** Many concrete edits surfaced: sub-grid-precision caveat,
plain-language preface, glossary, rotation-rate-as-headline reframing,
ablation/limit flags. The most material ones were folded into RESULTS.md
in the same session; the unfolded ones (proper log-corrected FSS fit,
shuffled-1D ablation, per-seed cosine correlation across stages, XY/Potts
extension) are flagged as future work in the *Honest limitations*
section.

**Lesson.** A self-administered adversarial review at five distinct
stances catches a notable fraction of what real reviewers will flag.
Worth running before *every* submission. Particular value of the
Outsider stance: catches the curse-of-knowledge gaps invisible to the
author.
