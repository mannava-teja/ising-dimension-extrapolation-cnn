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

## 15. Measurement #2 — the naive exponent fit failed, then was diagnosed

**Reasoning.** Extract the correlation-length exponent `nu` from the
finite-size scaling of the network's classification-crossover width:
`width ~ L^(-1/nu)`.

**Outcome — wrong, then understood.** The naive fit gave `nu` ~2x too large.
Diagnosed cause: the network classifies individual finite samples with a
smooth decision function, so its crossover cannot sharpen below an intrinsic
*resolution floor* `c ~ 0.055`. With `width = a L^(-1/nu) + c`, the corrected
exponents track the dimensional trend.

**Lesson.** A classifier's decision-boundary width is *not* the physical
correlation length — it carries an instrument resolution. A failed measurement
that is cleanly diagnosed is itself a result.

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
