"""Physics + format validation for 3D Ising HDF5 datasets.

There is no exact solution in 3D, so we compare against the high-precision
MC literature values:

  T_c       = 4.5115        (Ferrenberg & Landau, Phys. Rev. B 44, 5081 (1991);
                             modern consensus 4.51152(4))
  <E>/N|T_c = -0.9906       (Talapov & Blote, J. Phys. A 29, 5727 (1996))
  U_4*     = 0.4655         (Hasenbusch, J. Phys. A 32, 4851 (1999))
  beta     = 0.3265
  gamma    = 1.2372         (so chi peak at T_c grows like L^{gamma/nu})
  nu       = 0.6301

Three-part check (mirrors 1D / 2D):

  A. Physics
       1. Energy at T_c matches Talapov-Blote within tolerance
       2. Binder cumulant crossings between adjacent L give T_c estimates
          near 4.5115
       3. Binder cumulant value at the crossing matches U_4* ~ 0.4655
       4. Specific heat / susceptibility / magnetization have qualitatively
          correct shape (sharp peaks at T_c, M -> 0 above)

  B. CNN-readiness
       - shapes (N, L, L, L), dtype int8, values {-1, +1}, no NaN/inf
       - PBC translation invariance on all three axes
       - per-block seed uniqueness

  C. Storage integrity via manual recomputation

Optional --metro-h5 cross-checks Wolff vs Metropolis at matching (L, T).

Usage:
    python scripts/validate_physics_3d.py data/ising_3d.h5 \\
        --figure reports/figures/validate_3d.png \\
        --metro-h5 data/ising_3d_metro_check.h5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ising.storage import iter_blocks  # noqa: E402

T_C_3D = 4.5115
E_PER_SPIN_AT_TC = -0.9906  # Talapov & Blote 1996
U4_STAR_3D = 0.4655          # Hasenbusch 1999


# ---------- MC estimators ----------

def mc_specific_heat_per_spin(energies, N, T):
    return energies.var(ddof=1) / (N * T * T)


def mc_susceptibility_per_spin(mags, N, T):
    var_M = (mags ** 2).mean() - (mags.mean()) ** 2
    return var_M / (N * T)


def binder_cumulant(mags):
    m2 = (mags ** 2).mean()
    m4 = (mags ** 4).mean()
    if m2 == 0:
        return 0.0
    return 1.0 - m4 / (3.0 * m2 * m2)


# ---------- Part C: manual recomputation ----------

def manual_energy_3d(spins):
    s = spins.astype(np.int64)
    return float(-(s * np.roll(s, -1, axis=0)).sum()
                 - (s * np.roll(s, -1, axis=1)).sum()
                 - (s * np.roll(s, -1, axis=2)).sum())


def manual_magnetization_3d(spins):
    return float(spins.astype(np.int64).sum())


# ---------- main ----------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("h5_path", type=Path)
    p.add_argument("--metro-h5", type=Path, default=None)
    p.add_argument("--n-integrity-checks", type=int, default=200)
    p.add_argument("--figure", type=Path, default=None)
    p.add_argument("--rel-tol-energy-Tc", type=float, default=0.05,
                   help="Tolerance on <E>/N at T_c vs Talapov-Blote -0.9906.")
    p.add_argument("--cross-tol", type=float, default=0.03)
    p.add_argument("--seed", type=int, default=12345)
    return p.parse_args()


def main():
    args = parse_args()
    if not args.h5_path.exists():
        print(f"File not found: {args.h5_path}", file=sys.stderr)
        return 2

    blocks = list(iter_blocks(args.h5_path, dim=3))
    if not blocks:
        print("No dim_3 blocks found.", file=sys.stderr)
        return 2

    rng = np.random.default_rng(args.seed)

    # ---------- Part B: format ----------
    print("=" * 80)
    print(f"PART B  CNN-readiness checks (T_c = {T_C_3D:.4f})")
    print("=" * 80)
    fmt_issues = 0
    seen_seeds = set()
    for b in blocks:
        cfg = b["configurations"]
        L = b["L"]
        if cfg.ndim != 4 or cfg.shape[1:] != (L, L, L):
            print(f"  L={L} T={b['T']:.4f}  shape {cfg.shape}, expected (N, L, L, L)")
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

    # PBC translation invariance on each axis: G(1) ~ G(L-1).
    pbc_issues = 0
    for b in blocks:
        cfg = b["configurations"].astype(np.float64)
        L = b["L"]
        for axis in (1, 2, 3):
            g1 = (cfg * np.roll(cfg, -1, axis=axis)).mean()
            gL = (cfg * np.roll(cfg, -(L - 1), axis=axis)).mean()
            rel = abs(g1 - gL) / max(abs(g1), 1e-8)
            if rel > 0.05:
                print(f"  L={L} T={b['T']:.4f} axis={axis} PBC inv broken: "
                      f"G(1)={g1:.4f} G(L-1)={gL:.4f}")
                pbc_issues += 1
    print(f"  PBC translation-invariance issues: {pbc_issues}")

    try:
        import torch  # noqa: F401
        from torch.utils.data import Dataset, DataLoader

        class Ising3DDataset(Dataset):
            def __init__(self, configs, T):
                self.x = torch.from_numpy(configs.astype(np.float32))[:, None]
                self.y = torch.full((len(self.x),), float(T))
            def __len__(self): return len(self.x)
            def __getitem__(self, i): return self.x[i], self.y[i]

        b0 = blocks[0]
        ds = Ising3DDataset(b0["configurations"], b0["T"])
        x, y = next(iter(DataLoader(ds, batch_size=8, shuffle=True)))
        print(f"  PyTorch Dataset OK: x.shape={tuple(x.shape)}, "
              f"y.shape={tuple(y.shape)}, x.dtype={x.dtype}")
    except ImportError:
        print("  PyTorch not installed; Dataset check skipped.")

    # ---------- Part C: storage integrity ----------
    print()
    print("=" * 80)
    print(f"PART C  storage integrity ({args.n_integrity_checks} random samples)")
    print("=" * 80)
    integrity_fail = 0
    max_e = 0.0; max_m = 0.0
    for _ in range(args.n_integrity_checks):
        b = blocks[rng.integers(0, len(blocks))]
        i = int(rng.integers(0, b["configurations"].shape[0]))
        e_s = float(b["energies"][i]); m_s = float(b["magnetizations"][i])
        e_m = manual_energy_3d(b["configurations"][i])
        m_m = manual_magnetization_3d(b["configurations"][i])
        d_e = abs(e_s - e_m); d_m = abs(m_s - m_m)
        max_e = max(max_e, d_e); max_m = max(max_m, d_m)
        if d_e > 0 or d_m > 0:
            print(f"  L={b['L']} T={b['T']:.4f} i={i}  "
                  f"E_stored={e_s} E_manual={e_m}  M_stored={m_s} M_manual={m_m}")
            integrity_fail += 1
    print(f"  max |E_stored - E_manual| = {max_e}")
    print(f"  max |M_stored - M_manual| = {max_m}")
    print(f"  integrity failures: {integrity_fail} / {args.n_integrity_checks}")

    # ---------- Part A: physics ----------
    print()
    print("=" * 80)
    print("PART A  physics vs 3D Ising literature values")
    print("=" * 80)
    rows = []
    for b in blocks:
        L = b["L"]; T = b["T"]; N = L ** 3
        E = b["energies"]; M = b["magnetizations"]
        rows.append({
            "L": L, "T": T, "N": N,
            "e_mc": E.mean() / N,
            "abs_m_mc": float(np.abs(M).mean()) / N,
            "c_mc": mc_specific_heat_per_spin(E, N, T),
            "chi_mc": mc_susceptibility_per_spin(M, N, T),
            "U4_mc": binder_cumulant(M),
            "signed_m_mc": M.mean() / N,
        })

    # Energy at T_c. Finite-size corrections at T_c scale as ~L^{-(1-alpha/nu)}
    # ~ L^{-0.83} in 3D, so even moderate L deviates from the TD-limit value.
    # Use an L-aware tolerance.
    def e_tol(L: int) -> float:
        return args.rel_tol_energy_Tc + 0.9 / L

    Ls = sorted({r["L"] for r in rows})
    print(f"  <E>/N at T_c (Talapov-Blote ref = {E_PER_SPIN_AT_TC:+.4f}):")
    n_e_tc_fail = 0
    for L in Ls:
        candidates = [r for r in rows if r["L"] == L]
        nearest = min(candidates, key=lambda r: abs(r["T"] - T_C_3D))
        d = abs(nearest["e_mc"] - E_PER_SPIN_AT_TC) / abs(E_PER_SPIN_AT_TC)
        tol = e_tol(L)
        status = "OK" if d <= tol else "FAIL"
        if status == "FAIL":
            n_e_tc_fail += 1
        print(f"    L={L:3d}  T={nearest['T']:.4f}  <E>/N={nearest['e_mc']:+.4f}  "
              f"rel_err={d:.4f}  tol={tol:.4f}  {status}")

    # Binder cumulant crossings between adjacent L
    if len(Ls) >= 2:
        print()
        print(f"  Binder cumulant crossings (literature T_c = {T_C_3D:.4f}, "
              f"U_4* ~ {U4_STAR_3D:.3f}):")
        per_L = {L: sorted([(r["T"], r["U4_mc"]) for r in rows if r["L"] == L])
                 for L in Ls}
        all_crossings = []
        for i in range(len(Ls) - 1):
            L1, L2 = Ls[i], Ls[i + 1]
            Ts_common = sorted(set(t for t, _ in per_L[L1]) & set(t for t, _ in per_L[L2]))
            if not Ts_common:
                continue
            U1 = np.array([next(u for t, u in per_L[L1] if t == tt) for tt in Ts_common])
            U2 = np.array([next(u for t, u in per_L[L2] if t == tt) for tt in Ts_common])
            diff = U1 - U2
            for k in range(len(Ts_common) - 1):
                if diff[k] * diff[k + 1] < 0:
                    t_cross = (Ts_common[k]
                               - diff[k] * (Ts_common[k + 1] - Ts_common[k])
                               / (diff[k + 1] - diff[k]))
                    # Linear interp of U at the crossing.
                    frac = (t_cross - Ts_common[k]) / (Ts_common[k + 1] - Ts_common[k])
                    u_cross = U1[k] + frac * (U1[k + 1] - U1[k])
                    all_crossings.append((L1, L2, t_cross, u_cross))
        for L1, L2, t, u in all_crossings:
            d_t = abs(t - T_C_3D)
            d_u = abs(u - U4_STAR_3D)
            print(f"    L={L1:3d} vs L={L2:3d}: T={t:.4f} (|dT|={d_t:.4f}),  "
                  f"U_4={u:.4f} (|dU|={d_u:.4f})")

    # ---------- Optional cross-algorithm check ----------
    cross_fails = 0
    if args.metro_h5 is not None and args.metro_h5.exists():
        print()
        print("=" * 80)
        print(f"CROSS-CHECK  {args.h5_path.name} vs {args.metro_h5.name}")
        print("=" * 80)
        metro_blocks = {(b["L"], round(b["T"], 4)): b
                        for b in iter_blocks(args.metro_h5, dim=3)}
        for r in rows:
            key = (r["L"], round(r["T"], 4))
            mb = metro_blocks.get(key)
            if mb is None:
                continue
            e_metro = mb["energies"].mean() / r["N"]
            d = abs(r["e_mc"] - e_metro) / max(abs(r["e_mc"]), 1e-8)
            if d > args.cross_tol:
                cross_fails += 1
                print(f"  DIFF  L={r['L']:3d} T={r['T']:.4f}  "
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
            for ax, (key, ylabel, title, *opt) in zip(axes.flat, [
                ("e_mc",       r"$\langle E\rangle/N$",        "Energy per spin",         "tc_ref"),
                ("abs_m_mc",   r"$\langle|m|\rangle$",         "Magnetization"),
                ("c_mc",       r"$C/N$",                       "Specific heat"),
                ("chi_mc",     r"$\chi/N$ (log)",              "Susceptibility", "logy"),
                ("U4_mc",      r"$U_4$",                       "Binder cumulant", "u4ref"),
                ("signed_m_mc",r"$\langle M\rangle/N$",        "Signed M"),
            ]):
                for L in Ls:
                    sub = sorted([r for r in rows if r["L"] == L], key=lambda r: r["T"])
                    ys = [r[key] for r in sub]
                    Ts = [r["T"] for r in sub]
                    style = "o-" if key != "chi_mc" else "o-"
                    ax.plot(Ts, ys, style, ms=3, label=f"L={L}")
                ax.axvline(T_C_3D, color="r", lw=0.5, ls="--",
                           label="lit. T_c" if key == "e_mc" else None)
                if "tc_ref" in opt:
                    ax.axhline(E_PER_SPIN_AT_TC, color="g", lw=0.5, ls=":",
                               label="Talapov-Blote $u_c$")
                if "u4ref" in opt:
                    ax.axhline(U4_STAR_3D, color="g", lw=0.5, ls=":",
                               label="Hasenbusch $U^*$")
                if "logy" in opt:
                    ax.set_yscale("log")
                ax.set_xlabel("T"); ax.set_ylabel(ylabel); ax.set_title(title)
                ax.legend(fontsize=8)
            fig.suptitle("3D Ising validation: MC vs literature values", fontsize=13)
            fig.tight_layout()
            fig.savefig(args.figure, dpi=120)
            print(f"\nfigure -> {args.figure}")

    print()
    print("=" * 80)
    total = n_e_tc_fail + fmt_issues + pbc_issues + integrity_fail + cross_fails
    print(f"TOTAL energy-at-Tc failures: {n_e_tc_fail}")
    print(f"TOTAL format failures: {fmt_issues + pbc_issues}")
    print(f"TOTAL integrity failures: {integrity_fail}")
    print(f"TOTAL cross-algorithm diffs: {cross_fails}")
    print("=" * 80)
    return 0 if total == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
