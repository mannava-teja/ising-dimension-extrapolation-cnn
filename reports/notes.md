# Project notes

Running notes on decisions and mistakes, roughly chronological. Mostly
for my own memory.

**Validate the data before training anything.** Decided early that every
dataset has to reproduce known physics before a network touches it.
Caught several real bugs this way and meant that when the CNN did
something weird later, I could rule out the data.

**Numba + HDF5.** MC inner loops in numba (near-C speed), one HDF5 file
per dimension with /dim/L/T blocks and per-block seeds so any block can
be regenerated alone. Worth the setup time.

**1D validation tolerance bug.** Flagged blocks where |M|/N exceeded
5/sqrt(L) - wrong scale at low T where the correlation length is large,
should be sqrt(xi/L). Lesson: derive tolerances from the physics, don't
guess them.

**Low-T autocorrelation.** Fixed decorrelation sweeps left low-T 1D
samples correlated (tau ~ xi^2). Made thermalization/decorrelation scale
with xi^2.

**Wolff everywhere was wrong.** Wolff beats critical slowing down near
T_c but at high T the clusters are O(1) and samples stay correlated.
Found via the effective-sample-size audit. Now: Wolff near/below T_c,
Metropolis above.

**The L=16 "bug" that wasn't.** 2D L=16 deviated ~5% from the Onsager
thermodynamic-limit energy near T_c. It's just the finite-size
correction. Made tolerances L-aware.

**HDF5 doesn't reclaim space.** Overwriting blocks grew the 3D file past
GitHub's 100MB limit and the push failed. repack_hdf5.py fixes it.

**Biggest design mistake: per-dimension encoders.** First architecture
had a separate ConvNd encoder per dimension feeding a shared head.
Useless for the actual goal - the 4D encoder would be untrained at test
time. The held-out-dimension experiment forces a dimension-agnostic
model: per-site linear of (spin, neighbour-sum), neighbours via
torch.roll, global average pool. One set of weights for every d.

**Reframing.** "Predict T_c(4D)" is a weak goal (the number is known to
5 decimals). The defensible question is whether the network learns
dimension-transferable structure, with 4D as the falsifiable test since
mean-field freezing is an exact prediction there.

**Staged training works.** Train on 2D only: held-out 4D T_c is 12% off.
Add 3D: lands basically on the literature value. The trend across stages
is more convincing than any single number.

**The naive nu fit fails, instructively.** width ~ L^(-1/nu) gives nu
about 2x too big - the classifier's decision boundary has finite
sharpness, so the crossover width has a floor. Fitting a shared floor c
fixes the trend. But the floor is fitted, not derived, and the 1-loop
epsilon expansion gives nu(4D)=1/2 for free, so the corrected numbers
are a presentation aid, not a precision claim. The qualitative trend
(nu descends toward 1/2) survives even in the uncorrected fits.

**Universality collapse is wrong, rotation is right.** Expected
near-critical configs from different d to overlap in feature space.
They're completely segregated (kNN mixing ~ 0). But the
ordered->disordered axis within each cluster is shared between adjacent
dimensions (cos ~ 0.8) and decays with distance (cos(2,4) ~ 0.4). The
rotation is the transfer mechanism.

**5D.** Generated as a second held-out dimension to sharpen the freezing
test. Binder crossings validate fine. But the stage B classifier breaks
down at 5D for small L - the axis has rotated too far. Reported as a
transfer horizon rather than hiding it.

**Multi-seed reality check.** The single-seed stage B headline
(T_c(4D) 0.06% off) was a lucky seed - 3-seed mean is 0.54%. Always
multi-seed before believing a number.

**1D helps, a lot.** Adding 1D to training (stage C) improved the
held-out 4D estimate 18x. Did not see that coming - 1D has no
transition. Possibly it calibrates the classifier baseline. Needs an
ablation against shuffled 1D before claiming the mechanism.

**Stage D surprises.** Trained through 4D with 5D held out: the 5D
classifier now works at all L, and the 5D error bar collapses ~80x vs
stage B. But the in-training 4D estimate got *worse* than stage C's
held-out one (0.79% vs 0.03%) - the shared head compromises across
training dims. So: for the most accurate readout at a dimension, hold
it out.

**The sharp freezing test failed.** Stage D nu(5D) = 0.40 +/- 0.02, not
0.5 - continued descent rather than a plateau. The 5D lattices are small
(L <= 8), so I read it as finite-size corrections rather than absent
freezing, but as measured the test is negative. Bigger 5D lattices would
settle it.

**Extended-L 4D made the fits worse.** Adding L=10,12 broadened the nu
fits instead of tightening them. Probably the known log corrections at
d=4 becoming resolvable, but I haven't fit the log-corrected form, so
that stays a hypothesis.

**Rotation rate is stable.** ~33 deg/dim across all three training
stages. If that holds up (per-seed cross-stage check pending) it's a
property of the problem, not the training set, and it's the single most
interesting thing in the project.
