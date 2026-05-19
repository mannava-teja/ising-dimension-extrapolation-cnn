"""Physics + format validation for the 5D Ising HDF5 dataset.

5D is above the upper critical dimension: the Ising model is mean-field, with
exponents nu = 1/2, beta = 1/2, gamma = 1, and -- unlike exactly d = 4 -- no
logarithmic corrections. The quantitative anchor is the Binder-cumulant
crossing, which should sit near

    T_c(5D) ~ 8.778

Checks mirror validate_physics_4d.py: format, storage integrity, Binder
crossings, energy-curve sanity (physical range [-5, 0]; 5 bonds/site).

Usage:
    python scripts/validate_physics_5d.py data/ising_5d.h5 \\
        --figure reports/figures/validate_5d.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ising.storage import iter_blocks  # noqa: E402

T_C_5D = 8.778


def mc_specific_heat_per_spin(energies, N, T):
    return energies.var(ddof=1) / (N * T * T)


def mc_susceptibility_per_spin(mags, N, T):
    return ((mags ** 2).mean() - (mags.mean()) ** 2) / (N * T)


def binder_cumulant(mags):
    m2 = (mags ** 2).mean()
    m4 = (mags ** 4).mean()
    return 0.0 if m2 == 0 else 1.0 - m4 / (3.0 * m2 * m2)


def manual_energy_5d(spins):
    s = spins.astype(np.int64)
    return float(sum(-(s * np.roll(s, -1, axis=ax)).sum() for ax in range(5)))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("h5_path", type=Path)
    p.add_argument("--n-integrity-checks", type=int, default=150)
    p.add_argument("--figure", type=Path, default=None)
    p.add_argument("--tc-tol", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=12345)
    return p.parse_args()


def main():
    args = parse_args()
    if not args.h5_path.exists():
        print(f"File not found: {args.h5_path}", file=sys.stderr)
        return 2
    blocks = list(iter_blocks(args.h5_path, dim=5))
    if not blocks:
        print("No dim_5 blocks found.", file=sys.stderr)
        return 2
    rng = np.random.default_rng(args.seed)

    # ---- format ----
    print("=" * 78)
    print(f"FORMAT / CNN-readiness (T_c = {T_C_5D:.4f})")
    print("=" * 78)
    fmt = pbc = 0
    seen = set()
    for b in blocks:
        cfg = b["configurations"]; L = b["L"]
        if cfg.ndim != 6 or cfg.shape[1:] != (L,) * 5:
            print(f"  L={L} T={b['T']:.4f} shape {cfg.shape}"); fmt += 1
        if cfg.dtype != np.int8:
            print(f"  L={L} dtype {cfg.dtype}"); fmt += 1
        if not (set(np.unique(cfg).tolist()) <= {-1, 1}):
            print(f"  L={L} bad spin values"); fmt += 1
        if b["seed"] in seen:
            print(f"  duplicate seed {b['seed']}"); fmt += 1
        seen.add(b["seed"])
        cf = cfg.astype(np.float64)
        for ax in range(1, 6):
            g1 = (cf * np.roll(cf, -1, axis=ax)).mean()
            gL = (cf * np.roll(cf, -(L - 1), axis=ax)).mean()
            if abs(g1 - gL) / max(abs(g1), 1e-8) > 0.05:
                print(f"  L={L} T={b['T']:.4f} axis={ax} PBC broken"); pbc += 1
    print(f"  format issues: {fmt}   PBC issues: {pbc}")

    # ---- storage integrity ----
    print()
    print("=" * 78)
    print(f"STORAGE INTEGRITY ({args.n_integrity_checks} random samples)")
    print("=" * 78)
    integ = 0
    me = mm = 0.0
    for _ in range(args.n_integrity_checks):
        b = blocks[rng.integers(0, len(blocks))]
        i = int(rng.integers(0, b["configurations"].shape[0]))
        de = abs(float(b["energies"][i]) - manual_energy_5d(b["configurations"][i]))
        dm = abs(float(b["magnetizations"][i])
                 - float(b["configurations"][i].astype(np.int64).sum()))
        me = max(me, de); mm = max(mm, dm)
        if de > 0 or dm > 0:
            integ += 1
    print(f"  max |E diff| = {me}   max |M diff| = {mm}   failures: {integ}")

    # ---- physics ----
    print()
    print("=" * 78)
    print("PHYSICS vs 5D Ising literature (mean-field; no log corrections)")
    print("=" * 78)
    rows = []
    for b in blocks:
        L = b["L"]; T = b["T"]; N = L ** 5
        E = b["energies"]; M = b["magnetizations"]
        rows.append({"L": L, "T": T,
                     "e": E.mean() / N,
                     "abs_m": float(np.abs(M).mean()) / N,
                     "c": mc_specific_heat_per_spin(E, N, T),
                     "chi": mc_susceptibility_per_spin(M, N, T),
                     "U4": binder_cumulant(M)})
    Ls = sorted({r["L"] for r in rows})
    e_range = (min(r["e"] for r in rows), max(r["e"] for r in rows))
    e_issue = 0 if (-5.001 <= e_range[0] and e_range[1] <= 0.001) else 1
    print(f"  <E>/N range: [{e_range[0]:+.3f}, {e_range[1]:+.3f}]  "
          f"(physical [-5, 0]; issue={e_issue})")

    n_tc_fail = 0
    crossings = []
    if len(Ls) >= 2:
        print(f"  Binder crossings (literature T_c = {T_C_5D:.4f}):")
        per_L = {L: sorted([(r["T"], r["U4"]) for r in rows if r["L"] == L])
                 for L in Ls}
        for a in range(len(Ls) - 1):
            L1, L2 = Ls[a], Ls[a + 1]
            Ts = sorted(set(t for t, _ in per_L[L1]) & set(t for t, _ in per_L[L2]))
            if not Ts:
                continue
            U1 = np.array([next(u for t, u in per_L[L1] if t == tt) for tt in Ts])
            U2 = np.array([next(u for t, u in per_L[L2] if t == tt) for tt in Ts])
            diff = U1 - U2
            cr = []
            for k in range(len(Ts) - 1):
                if diff[k] * diff[k + 1] < 0:
                    cr.append(Ts[k] - diff[k] * (Ts[k + 1] - Ts[k])
                              / (diff[k + 1] - diff[k]))
            if cr:
                best = min(cr, key=lambda t: abs(t - T_C_5D))
                crossings.append(best)
                rel = abs(best - T_C_5D) / T_C_5D
                ok = "OK" if rel <= args.tc_tol else "FAIL"
                if ok == "FAIL":
                    n_tc_fail += 1
                print(f"    L={L1} vs L={L2}: nearest-T_c crossing "
                      f"{best:.4f}  (rel err {rel:.4f})  {ok}")
            else:
                print(f"    L={L1} vs L={L2}: no crossing"); n_tc_fail += 1

    # ---- figure ----
    if args.figure is not None:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib missing; figure skipped")
        else:
            fig, axes = plt.subplots(2, 2, figsize=(12, 9))
            for ax, (key, ylab, title) in zip(axes.flat, [
                ("e", r"$\langle E\rangle/N$", "Energy per spin"),
                ("abs_m", r"$\langle|m|\rangle$", "Magnetization"),
                ("chi", r"$\chi/N$", "Susceptibility"),
                ("U4", r"$U_4$", "Binder cumulant"),
            ]):
                for L in Ls:
                    sub = sorted([r for r in rows if r["L"] == L],
                                 key=lambda r: r["T"])
                    ax.plot([r["T"] for r in sub], [r[key] for r in sub],
                            "o-", ms=3, label=f"L={L}")
                ax.axvline(T_C_5D, color="r", lw=0.6, ls="--")
                ax.set_xlabel("T"); ax.set_ylabel(ylab); ax.set_title(title)
                ax.legend(fontsize=8)
            fig.suptitle("5D Ising validation (held-out test set, mean-field)",
                         fontsize=12)
            fig.tight_layout()
            args.figure.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(args.figure, dpi=130)
            print(f"\nfigure -> {args.figure}")

    print()
    print("=" * 78)
    total = fmt + pbc + integ + e_issue + n_tc_fail
    if crossings:
        print(f"Binder-crossing T_c estimates: "
              f"{', '.join(f'{t:.4f}' for t in crossings)}  (lit {T_C_5D})")
    print(f"TOTAL issues: {total}")
    print("=" * 78)
    return 0 if total == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
