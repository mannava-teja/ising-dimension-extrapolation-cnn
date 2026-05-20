"""Physics-statistic baselines for T_c(d) and nu(d).

The external critique pointed out that the paper's headline -- a CNN trained
on d=2+3 extrapolates T_c(4D) to 0.06% -- only stands if it *beats* trivial
physics-statistic extrapolations of T_c and nu. A two-point linear fit of
T_c(d) through the known T_c(2D), T_c(3D) values already predicts T_c(4D)
within 1%, with no neural network. Likewise, leading-order epsilon expansion
predicts nu(4D) = 1/2 *exactly* (it is the Wilson-Fisher fixed point at the
upper critical dimension d=4).

This script computes all the relevant baselines and reports them alongside
the CNN's values. Pure arithmetic -- no training, no forward passes through
the network -- so it runs in seconds.

Numbers used:
  - Literature T_c: 1D=0, 2D=2.2692 (Onsager), 3D=4.5115 (Ferrenberg-Landau),
    4D=6.6803 (Lundow-Markstrom), 5D=8.778 (Lundow-Markstrom).
  - Literature nu: 2D=1.0, 3D=0.6301, 4D=0.5 (mean-field, with log corrections),
    5D=0.5 (clean mean-field).
  - CNN values, Stage B (trained 2D+3D), single seed:
      T_c: 2D=2.253, 3D=4.488, 4D=6.676 (held-out)
      nu (floor-corrected): 2D=0.813, 3D=0.667, 4D=0.569 (held-out)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]

# ------ literature values ------
T_C_LIT = {1: 0.0, 2: 2.2692, 3: 4.5115, 4: 6.6803, 5: 8.778}
NU_LIT = {2: 1.0, 3: 0.6301, 4: 0.5, 5: 0.5}

# ------ CNN values (single seed, Stage B trained 2D+3D) ------
T_C_CNN = {2: 2.253, 3: 4.488, 4: 6.676}    # 5D pending; will be filled if available
NU_CNN_NAIVE = {2: 2.076, 3: 0.975, 4: 0.619}     # raw FSS fit
NU_CNN_FLOOR = {2: 0.813, 3: 0.667, 4: 0.569}     # floor-corrected (c=0.055)


# ---------- T_c baselines ----------

def linear_tc_baseline(d_train=(2, 3)):
    """Two-point linear fit T_c(d) = a + b*d through literature values
    at d_train. Returns the prediction function."""
    xs = np.array(d_train, float)
    ys = np.array([T_C_LIT[d] for d in d_train])
    b, a = np.polyfit(xs, ys, 1)
    return lambda d: a + b * d, (a, b)


def quadratic_tc_baseline(d_train=(1, 2, 3)):
    """Three-point quadratic fit through literature values."""
    xs = np.array(d_train, float)
    ys = np.array([T_C_LIT[d] for d in d_train])
    c, b, a = np.polyfit(xs, ys, 2)
    return lambda d: a + b * d + c * d * d, (a, b, c)


def mean_field_deficit_baseline(d_train=(2, 3)):
    """Asymptotic baseline: T_c(d) approaches 2d (mean-field) from below.
    Fit T_c(d) = 2d - g/d using d_train to find g, then extrapolate.

    The 1/d correction goes the right way (deficit shrinks with d) but is a
    naive parameterisation. Gives a different number from the linear fit and
    is a useful second baseline."""
    xs = np.array(d_train, float)
    ys = np.array([T_C_LIT[d] for d in d_train])
    # Solve 2d - g/d = T_c(d) => g = d * (2d - T_c(d))
    gs = xs * (2 * xs - ys)
    g = float(gs.mean())
    return lambda d: 2 * d - g / d, g


# ---------- nu baselines ----------

def epsilon_expansion_nu(order=1):
    """Wilson-Fisher epsilon-expansion for nu in d = 4 - epsilon dimensions.
    1-loop:  nu = 1/2 + epsilon/12
    2-loop:  nu = 1/2 + epsilon/12 + 7 epsilon^2 / 162
    At d=4 (epsilon=0) returns 1/2 *exactly*.
    For d>4 (negative epsilon) the expansion is formally an extrapolation."""
    def predict(d):
        eps = 4 - d
        nu = 0.5 + eps / 12.0
        if order >= 2:
            nu += 7.0 * eps * eps / 162.0
        return nu
    return predict


def linear_nu_baseline(d_train=(2, 3)):
    """Two-point linear fit nu(d) through literature values."""
    xs = np.array(d_train, float)
    ys = np.array([NU_LIT[d] for d in d_train])
    b, a = np.polyfit(xs, ys, 1)
    return lambda d: a + b * d, (a, b)


# ---------- reporting ----------

def pct_off(pred, truth):
    return abs(pred - truth) / abs(truth) * 100.0 if truth != 0 else float("nan")


def main():
    print("=" * 78)
    print("BASELINES vs CNN  --  T_c(d) extrapolation")
    print("=" * 78)
    lin, (a, b) = linear_tc_baseline()
    quad, (qa, qb, qc) = quadratic_tc_baseline()
    deficit, g = mean_field_deficit_baseline()

    print(f"  linear T_c through d=2,3:    T_c(d) = {a:.3f} + {b:.3f}*d")
    print(f"  quadratic through d=1,2,3:   T_c(d) = {qa:.3f} + {qb:.3f}*d + {qc:.4f}*d^2")
    print(f"  mean-field deficit 2d - g/d: g = {g:.3f}")
    print()
    print(f"  {'d':>2}  {'literature':>11}  {'linear(2,3)':>12}  "
          f"{'quad(1,2,3)':>12}  {'2d - g/d':>10}  {'CNN':>10}")
    for d in (4, 5):
        lit = T_C_LIT[d]
        l = lin(d); q = quad(d); m = deficit(d)
        cnn = T_C_CNN.get(d)
        cnn_str = f"{cnn:7.3f}" if cnn else "    --"
        print(f"  {d:>2}  {lit:>11.4f}  "
              f"{l:>7.3f}  {pct_off(l, lit):5.2f}%   "
              f"{q:>7.3f}  {pct_off(q, lit):5.2f}%   "
              f"{m:>6.3f}  {pct_off(m, lit):5.2f}%   "
              f"{cnn_str}  {pct_off(cnn, lit):5.2f}%" if cnn else
              f"  {d:>2}  {lit:>11.4f}  "
              f"{l:>7.3f}  {pct_off(l, lit):5.2f}%   "
              f"{q:>7.3f}  {pct_off(q, lit):5.2f}%   "
              f"{m:>6.3f}  {pct_off(m, lit):5.2f}%       --      --")

    print()
    print("=" * 78)
    print("BASELINES vs CNN  --  nu(d) extrapolation")
    print("=" * 78)
    eps1 = epsilon_expansion_nu(order=1)
    eps2 = epsilon_expansion_nu(order=2)
    linnu, (nua, nub) = linear_nu_baseline()
    print(f"  linear nu through d=2,3:                nu(d) = {nua:.3f} + {nub:.3f}*d")
    print(f"  1-loop epsilon-expansion:               nu = 1/2 + (4-d)/12")
    print(f"  2-loop epsilon-expansion:               nu = 1/2 + (4-d)/12 + 7(4-d)^2/162")
    print()
    print(f"  {'d':>2}  {'literature':>11}  {'linear(2,3)':>12}  "
          f"{'1-loop eps':>12}  {'2-loop eps':>12}  "
          f"{'CNN-naive':>11}  {'CNN-floor':>11}")
    for d in (2, 3, 4, 5):
        lit = NU_LIT[d]
        ln = linnu(d); e1 = eps1(d); e2 = eps2(d)
        cn = NU_CNN_NAIVE.get(d); cf = NU_CNN_FLOOR.get(d)
        cn_str = f"{cn:6.3f}  {pct_off(cn, lit):4.1f}%" if cn else "        --"
        cf_str = f"{cf:6.3f}  {pct_off(cf, lit):4.1f}%" if cf else "        --"
        print(f"  {d:>2}  {lit:>11.4f}  "
              f"{ln:>6.3f}  {pct_off(ln, lit):5.1f}%   "
              f"{e1:>6.3f}  {pct_off(e1, lit):5.1f}%   "
              f"{e2:>6.3f}  {pct_off(e2, lit):5.1f}%    "
              f"{cn_str}    {cf_str}")

    # ------ figure: CNN vs baselines vs literature ------
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(14, 6))
    ds = np.linspace(1.0, 5.5, 100)

    # T_c panel
    axL.plot(ds, [lin(d) for d in ds], "C0--", lw=1, label="linear (d=2,3)")
    axL.plot(ds, [quad(d) for d in ds], "C2--", lw=1, label="quadratic (d=1,2,3)")
    axL.plot(ds, [deficit(d) for d in ds], "C4--", lw=1, label="2d - g/d (d=2,3)")
    axL.plot(list(T_C_LIT.keys()), list(T_C_LIT.values()),
             "k*", ms=14, label="literature")
    for d, t in T_C_CNN.items():
        held = d not in (2, 3)
        axL.plot(d, t, "o", color="C3" if held else "C1", ms=12,
                 markeredgecolor="k",
                 label=("CNN, held-out" if held and d == 4 else
                        "CNN, trained" if not held and d == 2 else None))
    axL.set_xlabel("dimension d")
    axL.set_ylabel(r"$T_c$")
    axL.set_title("T_c(d): CNN vs trivial physics-statistic baselines\n"
                  "(the linear fit predicts T_c(4D) to 1% with no NN at all)")
    axL.legend(fontsize=9)
    axL.grid(True, alpha=0.3)

    # nu panel
    ds_nu = np.linspace(1.5, 5.5, 100)
    axR.plot(ds_nu, [linnu(d) for d in ds_nu], "C0--", lw=1, label="linear (d=2,3)")
    axR.plot(ds_nu, [eps1(d) for d in ds_nu], "C2--", lw=1, label="1-loop $\\epsilon$-exp")
    axR.plot(ds_nu, [eps2(d) for d in ds_nu], "C5--", lw=1, label="2-loop $\\epsilon$-exp")
    axR.plot(list(NU_LIT.keys()), list(NU_LIT.values()),
             "k*", ms=14, label="literature")
    axR.axhline(0.5, color="r", lw=0.6, ls=":", label="mean-field floor (d>=4)")
    for d, n in NU_CNN_FLOOR.items():
        held = d not in (2, 3)
        axR.plot(d, n, "o", color="C3" if held else "C1", ms=12,
                 markeredgecolor="k",
                 label=("CNN-floor-corrected, held-out" if held and d == 4 else
                        "CNN-floor-corrected, trained" if not held and d == 2 else None))
    for d, n in NU_CNN_NAIVE.items():
        held = d not in (2, 3)
        axR.plot(d, n, "s", color="C3" if held else "C1", ms=8, alpha=0.4,
                 label=("CNN-naive, held-out" if held and d == 4 else
                        "CNN-naive, trained" if not held and d == 2 else None))
    axR.set_xlabel("dimension d")
    axR.set_ylabel(r"$\nu$")
    axR.set_title("nu(d): CNN vs trivial physics-statistic baselines\n"
                  "(the 1-loop $\\epsilon$-expansion gets nu(4D)=1/2 *exactly* for free)")
    axR.legend(fontsize=8, loc="lower left")
    axR.grid(True, alpha=0.3)

    fig.suptitle("CNN extrapolation vs physics-statistic baselines  "
                 "(the gap is what the paper must defend)", fontsize=13)
    out = REPO_ROOT / "reports" / "figures" / "baselines.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out, dpi=130)
    print(f"\nfigure -> {out}")

    # ------ headline summary ------
    print()
    print("=" * 78)
    print("HEADLINE")
    print("=" * 78)
    lit4 = T_C_LIT[4]
    print(f"  T_c(4D) literature: {lit4:.4f}")
    print(f"    linear baseline:  {lin(4):.4f}  ({pct_off(lin(4), lit4):.2f}% off)")
    print(f"    quadratic baseln: {quad(4):.4f}  ({pct_off(quad(4), lit4):.2f}% off)")
    print(f"    CNN (Stage B):    {T_C_CNN[4]:.4f}  ({pct_off(T_C_CNN[4], lit4):.2f}% off)")
    print(f"    -> CNN beats the trivial linear baseline by a factor of "
          f"{pct_off(lin(4), lit4) / pct_off(T_C_CNN[4], lit4):.1f}")
    print()
    print(f"  nu(4D) literature: {NU_LIT[4]:.4f}")
    print(f"    1-loop epsilon:   {eps1(4):.4f}  ({pct_off(eps1(4), NU_LIT[4]):.2f}% off, "
          f"trivially correct at d=4)")
    print(f"    CNN floor-corr:   {NU_CNN_FLOOR[4]:.4f}  "
          f"({pct_off(NU_CNN_FLOOR[4], NU_LIT[4]):.2f}% off)")
    print(f"    -> epsilon-expansion is *better* than the CNN at d=4 for nu;")
    print(f"       the CNN nu story does not outperform the trivial physics baseline.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
