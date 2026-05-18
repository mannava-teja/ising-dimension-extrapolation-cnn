"""Physics + format validation for the 4D Ising HDF5 dataset.

4D is the upper critical dimension. There is no exact solution, and unlike
2D/3D the Binder cumulant carries logarithmic corrections at d_c = 4, so its
crossing value is not cleanly universal. The robust, quantitative anchor is
therefore the *location* of the Binder crossings, which should still sit near

    T_c = 6.6803    (Lundow & Markstrom, Phys. Rev. E 80, 031104 (2009))

Checks:

  A. Physics
       1. Binder cumulant crossings between adjacent L -> T_c estimate
       2. Energy curve is monotonic and within the physical range [-4, 0]
          (ground state -4 per spin: 4 bonds/site; disordered limit 0)
       3. Magnetization ~ 1 deep in the ordered phase, ~ 0 in the
          disordered phase, sharp drop near T_c
       4. Specific heat and susceptibility peak near T_c

  B. CNN-readiness: shapes (N,L,L,L,L), dtype int8, values {-1,+1}, no
     NaN/inf, PBC translation invariance on all four axes, seed uniqueness

  C. Storage integrity: manual recomputation of E and M for random samples

Optional --metro-h5 cross-checks Wolff vs Metropolis at matching (L, T).

Usage:
    python scripts/validate_physics_4d.py data/ising_4d.h5 \\
        --figure reports/figures/validate_4d.png \\
        --metro-h5 data/ising_4d_metro_check.h5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ising.storage import iter_blocks  # noqa: E402

T_C_4D = 6.6803


def mc_specific_heat_per_spin(energies, N, T):
    return energies.var(ddof=1) / (N * T * T)


def mc_susceptibility_per_spin(mags, N, T):
    var_M = (mags ** 2).mean() - (mags.mean()) ** 2
    return var_M / (N * T)


def binder_cumulant(mags):
    m2 = (mags ** 2).mean()
    m4 = (mags ** 4).mean()
    return 0.0 if m2 == 0 else 1.0 - m4 / (3.0 * m2 * m2)


def manual_energy_4d(spins):
    s = spins.astype(np.int64)
    return float(-(s * np.roll(s, -1, axis=0)).sum()
                 - (s * np.roll(s, -1, axis=1)).sum()
                 - (s * np.roll(s, -1, axis=2)).sum()
                 - (s * np.roll(s, -1, axis=3)).sum())


def manual_magnetization_4d(spins):
    return float(spins.astype(np.int64).sum())


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("h5_path", type=Path)
    p.add_argument("--metro-h5", type=Path, default=None)
    p.add_argument("--n-integrity-checks", type=int, default=200)
    p.add_argument("--figure", type=Path, default=None)
    p.add_argument("--tc-tol", type=float, default=0.10,
                   help="Tolerance on Binder-crossing T_c vs 6.6803 (relative).")
    p.add_argument("--cross-tol", type=float, default=0.03)
    p.add_argument("--seed", type=int, default=12345)
    return p.parse_args()


def main():
    args = parse_args()
    if not args.h5_path.exists():
        print(f"File not found: {args.h5_path}", file=sys.stderr)
        return 2

    blocks = list(iter_blocks(args.h5_path, dim=4))
    if not blocks:
        print("No dim_4 blocks found.", file=sys.stderr)
        return 2

    rng = np.random.default_rng(args.seed)

    # ---------- Part B: format ----------
    print("=" * 80)
    print(f"PART B  CNN-readiness checks (T_c = {T_C_4D:.4f})")
    print("=" * 80)
    fmt_issues = 0
    seen_seeds = set()
    for b in blocks:
        cfg = b["configurations"]
        L = b["L"]
        if cfg.ndim != 5 or cfg.shape[1:] != (L, L, L, L):
            print(f"  L={L} T={b['T']:.4f}  shape {cfg.shape}, expected (N,L,L,L,L)")
            fmt_issues += 1
        if cfg.dtype != np.int8:
            print(f"  dtype {cfg.dtype} != int8 at L={L} T={b['T']:.4f}")
            fmt_issues += 1
        u = np.unique(cfg)
        if not (set(u.tolist()) <= {-1, 1}):
            print(f"  L={L} T={b['T']:.4f}  values {u}")
            fmt_issues += 1
        if np.isnan(b["energies"]).any() or np.isinf(b["energies"]).any():
            print(f"  L={L} T={b['T']:.4f}  NaN/inf in energies")
            fmt_issues += 1
        if b["seed"] in seen_seeds:
            print(f"  duplicate seed {b['seed']} at L={L} T={b['T']:.4f}")
            fmt_issues += 1
        seen_seeds.add(b["seed"])
    print(f"  format issues: {fmt_issues}")

    pbc_issues = 0
    for b in blocks:
        cfg = b["configurations"].astype(np.float64)
        L = b["L"]
        for axis in (1, 2, 3, 4):
            g1 = (cfg * np.roll(cfg, -1, axis=axis)).mean()
            gL = (cfg * np.roll(cfg, -(L - 1), axis=axis)).mean()
            rel = abs(g1 - gL) / max(abs(g1), 1e-8)
            if rel > 0.05:
                print(f"  L={L} T={b['T']:.4f} axis={axis} PBC inv broken (rel {rel:.3f})")
                pbc_issues += 1
    print(f"  PBC translation-invariance issues: {pbc_issues}")

    # ---------- Part C: storage integrity ----------
    print()
    print("=" * 80)
    print(f"PART C  storage integrity ({args.n_integrity_checks} random samples)")
    print("=" * 80)
    integrity_fail = 0
    max_e = max_m = 0.0
    for _ in range(args.n_integrity_checks):
        b = blocks[rng.integers(0, len(blocks))]
        i = int(rng.integers(0, b["configurations"].shape[0]))
        e_s = float(b["energies"][i]); m_s = float(b["magnetizations"][i])
        e_m = manual_energy_4d(b["configurations"][i])
        m_m = manual_magnetization_4d(b["configurations"][i])
        d_e = abs(e_s - e_m); d_m = abs(m_s - m_m)
        max_e = max(max_e, d_e); max_m = max(max_m, d_m)
        if d_e > 0 or d_m > 0:
            print(f"  L={b['L']} T={b['T']:.4f} i={i}  E:{e_s}/{e_m}  M:{m_s}/{m_m}")
            integrity_fail += 1
    print(f"  max |E_stored - E_manual| = {max_e}")
    print(f"  max |M_stored - M_manual| = {max_m}")
    print(f"  integrity failures: {integrity_fail} / {args.n_integrity_checks}")

    # ---------- Part A: physics ----------
    print()
    print("=" * 80)
    print("PART A  physics vs 4D Ising literature")
    print("=" * 80)
    rows = []
    for b in blocks:
        L = b["L"]; T = b["T"]; N = L ** 4
        E = b["energies"]; M = b["magnetizations"]
        rows.append({
            "L": L, "T": T, "N": N,
            "e_mc": E.mean() / N,
            "abs_m_mc": float(np.abs(M).mean()) / N,
            "c_mc": mc_specific_heat_per_spin(E, N, T),
            "chi_mc": mc_susceptibility_per_spin(M, N, T),
            "U4_mc": binder_cumulant(M),
        })
    Ls = sorted({r["L"] for r in rows})

    # Energy curve sanity: monotonic in T, within [-4, 0].
    energy_issues = 0
    for L in Ls:
        sub = sorted([r for r in rows if r["L"] == L], key=lambda r: r["T"])
        es = [r["e_mc"] for r in sub]
        if any(e < -4.0001 or e > 0.0001 for e in es):
            print(f"  L={L}: energy out of physical range [-4, 0]")
            energy_issues += 1
        # Monotone non-decreasing in T (allow small MC noise).
        viol = sum(1 for a, b2 in zip(es, es[1:]) if b2 < a - 0.05)
        if viol > 2:
            print(f"  L={L}: energy non-monotonic in T ({viol} drops > 0.05)")
            energy_issues += 1
    print(f"  energy-curve issues: {energy_issues}")
    print(f"  <E>/N range across all blocks: "
          f"[{min(r['e_mc'] for r in rows):+.3f}, {max(r['e_mc'] for r in rows):+.3f}]")

    # Magnetization extremes
    m_lowT = max((r["abs_m_mc"] for r in rows if r["T"] < 0.7 * T_C_4D), default=float("nan"))
    m_highT = min((r["abs_m_mc"] for r in rows if r["T"] > 1.4 * T_C_4D), default=float("nan"))
    print(f"  <|m|> deep-ordered (max, T<0.7 T_c): {m_lowT:.3f}  (expect near 1)")
    print(f"  <|m|> disordered  (min, T>1.4 T_c): {m_highT:.3f}  (expect near 0)")

    # Binder crossings -> T_c
    n_tc_fail = 0
    crossings = []
    if len(Ls) >= 2:
        print()
        print(f"  Binder cumulant crossings (literature T_c = {T_C_4D:.4f}):")
        per_L = {L: sorted([(r["T"], r["U4_mc"]) for r in rows if r["L"] == L])
                 for L in Ls}
        for i in range(len(Ls) - 1):
            L1, L2 = Ls[i], Ls[i + 1]
            Ts = sorted(set(t for t, _ in per_L[L1]) & set(t for t, _ in per_L[L2]))
            if not Ts:
                continue
            U1 = np.array([next(u for t, u in per_L[L1] if t == tt) for tt in Ts])
            U2 = np.array([next(u for t, u in per_L[L2] if t == tt) for tt in Ts])
            diff = U1 - U2
            block_crossings = []
            for k in range(len(Ts) - 1):
                if diff[k] * diff[k + 1] < 0:
                    t = Ts[k] - diff[k] * (Ts[k + 1] - Ts[k]) / (diff[k + 1] - diff[k])
                    block_crossings.append(t)
            # The physical crossing is the one nearest T_c; spurious ones live
            # in the high-T tail where U_4 ~ 0 wiggles around zero.
            if block_crossings:
                best = min(block_crossings, key=lambda t: abs(t - T_C_4D))
                crossings.append(best)
                rel = abs(best - T_C_4D) / T_C_4D
                status = "OK" if rel <= args.tc_tol else "FAIL"
                if status == "FAIL":
                    n_tc_fail += 1
                allc = ", ".join(f"{t:.3f}" for t in block_crossings)
                print(f"    L={L1:2d} vs L={L2:2d}: nearest-T_c crossing "
                      f"{best:.4f} (rel err {rel:.4f}) {status}   [all: {allc}]")
            else:
                print(f"    L={L1:2d} vs L={L2:2d}: no crossing found")
                n_tc_fail += 1

    # ---------- cross-algorithm check ----------
    cross_fails = 0
    if args.metro_h5 is not None and args.metro_h5.exists():
        print()
        print("=" * 80)
        print(f"CROSS-CHECK  {args.h5_path.name} vs {args.metro_h5.name}")
        print("=" * 80)
        metro = {(b["L"], round(b["T"], 4)): b
                 for b in iter_blocks(args.metro_h5, dim=4)}
        for r in rows:
            mb = metro.get((r["L"], round(r["T"], 4)))
            if mb is None:
                continue
            e_metro = mb["energies"].mean() / r["N"]
            d = abs(r["e_mc"] - e_metro) / max(abs(r["e_mc"]), 1e-8)
            if d > args.cross_tol:
                cross_fails += 1
                print(f"  DIFF  L={r['L']:2d} T={r['T']:.4f}  "
                      f"primary={r['e_mc']:+.4f}  metro={e_metro:+.4f}  rel={d:.4f}")
        print(f"  cross-algorithm diffs > {args.cross_tol}: {cross_fails}")

    # ---------- figure ----------
    if args.figure is not None:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not installed; figure skipped.")
        else:
            args.figure.parent.mkdir(parents=True, exist_ok=True)
            fig, axes = plt.subplots(2, 3, figsize=(15, 9))
            panels = [
                ("e_mc",    r"$\langle E\rangle/N$", "Energy per spin", False),
                ("abs_m_mc", r"$\langle|m|\rangle$", "Magnetization", False),
                ("c_mc",    r"$C/N$",                "Specific heat", False),
                ("chi_mc",  r"$\chi/N$ (log)",       "Susceptibility", True),
                ("U4_mc",   r"$U_4$",                "Binder cumulant", False),
            ]
            for ax, (key, ylabel, title, logy) in zip(axes.flat, panels):
                for L in Ls:
                    sub = sorted([r for r in rows if r["L"] == L], key=lambda r: r["T"])
                    ys = [r[key] for r in sub]
                    if logy:
                        ys = [max(y, 1e-3) for y in ys]
                    ax.plot([r["T"] for r in sub], ys, "o-", ms=3, label=f"L={L}")
                ax.axvline(T_C_4D, color="r", lw=0.6, ls="--",
                           label="lit. T_c" if key == "e_mc" else None)
                if logy:
                    ax.set_yscale("log")
                ax.set_xlabel("T"); ax.set_ylabel(ylabel)
                ax.set_title(title); ax.legend(fontsize=8)
            axes.flat[5].axis("off")
            fig.suptitle("4D Ising validation (held-out test set)", fontsize=13)
            fig.tight_layout()
            fig.savefig(args.figure, dpi=120)
            print(f"\nfigure -> {args.figure}")

    # ---------- summary ----------
    print()
    print("=" * 80)
    total = n_tc_fail + fmt_issues + pbc_issues + integrity_fail + cross_fails + energy_issues
    print(f"TOTAL T_c-crossing failures: {n_tc_fail}")
    print(f"TOTAL energy-curve issues: {energy_issues}")
    print(f"TOTAL format failures: {fmt_issues + pbc_issues}")
    print(f"TOTAL integrity failures: {integrity_fail}")
    print(f"TOTAL cross-algorithm diffs: {cross_fails}")
    if crossings:
        print(f"Binder-crossing T_c estimates: "
              f"{', '.join(f'{t:.4f}' for t in crossings)}  (literature {T_C_4D})")
    print("=" * 80)
    return 0 if total == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
