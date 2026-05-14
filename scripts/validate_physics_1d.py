"""Full physics + format validation for the 1D Ising HDF5 dataset.

Three parts:

  A. Physics accuracy vs the exact 1D Ising analytical solution
       1. Energy per spin       <E>/N           = -tanh(beta)
       2. Specific heat         C/N             = beta^2 / cosh^2(beta)
       3. Susceptibility        chi/N           ~ beta * exp(2*beta)   (large L)
       4. Two-point correlator  G(r) = <s_0 s_r> = tanh(beta)^r
       5. Correlation length    xi              = -1 / ln(tanh(beta))
       6. Signed magnetization  <M>             = 0 (Z2 symmetry)

  B. CNN readiness for the dimension-upscaling problem
       - shapes, dtypes, value range, NaN/inf
       - PBC translation invariance: <s_0 s_{L-1}> ~ <s_0 s_1>
       - per-block seed uniqueness
       - PyTorch-style Dataset sanity (one batch)

  C. Storage integrity (random samples + manual recomputation)
       - For K random (L, T, sample_idx) triples, recompute E and M from the
         stored configurations and compare BIT-EXACTLY to the stored
         energies/magnetizations columns. Any nonzero diff = bug.

Usage:
    python scripts/validate_physics_1d.py data/ising_1d.h5
    python scripts/validate_physics_1d.py data/ising_1d.h5 --figure reports/figures/validate_1d.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ising.storage import iter_blocks  # noqa: E402


# ---------- exact 1D Ising analytical results ----------

def exact_energy_per_spin(T: float) -> float:
    return -np.tanh(1.0 / T)


def exact_specific_heat_per_spin(T: float) -> float:
    beta = 1.0 / T
    return (beta / np.cosh(beta)) ** 2


def exact_susceptibility_per_spin(T: float) -> float:
    """Large-L limit: chi/N = beta * (1+u)/(1-u) = beta * exp(2*beta), u = tanh(beta)."""
    beta = 1.0 / T
    u = np.tanh(beta)
    return beta * (1.0 + u) / (1.0 - u)


def exact_correlation_length(T: float) -> float:
    u = np.tanh(1.0 / T)
    return -1.0 / np.log(u) if u < 1.0 else float("inf")


# ---------- MC observable estimators ----------

def mc_specific_heat_per_spin(energies: np.ndarray, L: int, T: float) -> float:
    """C/N = (<E^2> - <E>^2) / (N * T^2). `energies` is total energy per sample."""
    var_E = energies.var(ddof=1)
    return var_E / (L * T * T)


def mc_susceptibility_per_spin(mags: np.ndarray, L: int, T: float) -> float:
    """chi/N = beta * (<M^2> - <M>^2) / N, using signed M.

    1D Ising is paramagnetic at every T > 0, so <M> -> 0 by Z2 symmetry; the
    subtraction is essentially <M^2> for large enough samples. (Subtracting
    <|M|>^2 would be the ferromagnetic-phase formula and is wrong here --
    <|M|>^2 is order N, which would zero out the signal.)
    """
    var_M = (mags ** 2).mean() - (mags.mean()) ** 2
    return var_M / (L * T)


def mc_correlator(configs: np.ndarray, r_max: int) -> np.ndarray:
    """G(r) = <s_0 s_r> averaged over translations and samples, for r in [0, r_max]."""
    N_samples, L = configs.shape
    r_max = min(r_max, L // 2)
    cfg = configs.astype(np.float64)
    G = np.empty(r_max + 1)
    for r in range(r_max + 1):
        # Average over all starting sites then over samples.
        G[r] = (cfg * np.roll(cfg, -r, axis=1)).mean()
    return G


def xi_from_G1(G: np.ndarray) -> float | None:
    """Estimate xi from the nearest-neighbor correlator: G(1) ~ tanh(beta) ~ exp(-1/xi).

    Exact in the open-chain limit. For PBC, accurate when xi << L (wrap-around
    contribution to G(1) is order u^{L-1} = exp(-(L-1)/xi), negligible unless
    xi approaches L/2). Robust where a linear fit of ln G(r) over a broad
    range of r fails because of PBC wrap-around (G has a non-zero floor near
    r = L/2).
    """
    if len(G) < 2:
        return None
    g1 = G[1]
    if not (0.0 < g1 < 1.0):
        return None
    return -1.0 / np.log(g1)


# ---------- Part C: storage integrity via manual recomputation ----------

def manual_energy(spin_row: np.ndarray) -> float:
    """E = -sum_i s_i * s_{(i+1) mod L}. spin_row is 1D int array of +/-1."""
    s = spin_row.astype(np.int64)
    return float(-(s * np.roll(s, -1)).sum())


def manual_magnetization(spin_row: np.ndarray) -> float:
    return float(spin_row.astype(np.int64).sum())


# ---------- main ----------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("h5_path", type=Path)
    p.add_argument("--n-integrity-checks", type=int, default=100,
                   help="Number of random (L, T, sample_idx) triples for Part C.")
    p.add_argument("--figure", type=Path, default=None,
                   help="If given, write a 6-subplot validation figure to this path.")
    p.add_argument("--rel-tol-energy", type=float, default=0.02)
    p.add_argument("--rel-tol-cv", type=float, default=0.15,
                   help="C/N is a variance estimator -- noisier than the mean, "
                        "especially at low T where Metropolis autocorrelation hurts. "
                        "Default 15%.")
    p.add_argument("--rel-tol-chi", type=float, default=0.20,
                   help="chi/N uses the large-L limit formula. Default 20%.")
    p.add_argument("--rel-tol-xi", type=float, default=0.15,
                   help="xi estimated from G(1); PBC wrap-around adds error when "
                        "xi approaches L. Default 15%.")
    p.add_argument("--seed", type=int, default=12345)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not args.h5_path.exists():
        print(f"File not found: {args.h5_path}", file=sys.stderr)
        return 2

    # ---------- collect blocks ----------
    blocks = list(iter_blocks(args.h5_path, dim=1))
    if not blocks:
        print("No dim_1 blocks found.", file=sys.stderr)
        return 2

    rng = np.random.default_rng(args.seed)

    # ---------- Part B (cheap): format / shape / dtype / values ----------
    print("=" * 78)
    print("PART B  CNN-readiness checks")
    print("=" * 78)

    seen_seeds = set()
    format_issues = 0
    for b in blocks:
        cfg = b["configurations"]
        if cfg.dtype != np.int8:
            print(f"  L={b['L']} T={b['T']:.4f}  configs dtype is {cfg.dtype}, expected int8")
            format_issues += 1
        unique = np.unique(cfg)
        if not (set(unique.tolist()) <= {-1, 1}):
            print(f"  L={b['L']} T={b['T']:.4f}  configs contain values {unique}")
            format_issues += 1
        if np.isnan(b["energies"]).any() or np.isinf(b["energies"]).any():
            print(f"  L={b['L']} T={b['T']:.4f}  NaN/inf in energies")
            format_issues += 1
        if b["seed"] in seen_seeds:
            print(f"  L={b['L']} T={b['T']:.4f}  duplicate seed {b['seed']}")
            format_issues += 1
        seen_seeds.add(b["seed"])
    print(f"  format issues: {format_issues}")

    # PBC translation invariance: G(L-1) should equal G(1) (both nearest-neighbor by wrap).
    pbc_issues = 0
    for b in blocks:
        cfg = b["configurations"].astype(np.float64)
        L = b["L"]
        g1 = (cfg * np.roll(cfg, -1, axis=1)).mean()
        gLm1 = (cfg * np.roll(cfg, -(L - 1), axis=1)).mean()
        rel = abs(g1 - gLm1) / max(abs(g1), 1e-8)
        if rel > 0.05:
            print(f"  L={L} T={b['T']:.4f}  PBC inv broken: G(1)={g1:.4f}, G(L-1)={gLm1:.4f}")
            pbc_issues += 1
    print(f"  PBC translation-invariance issues: {pbc_issues}")

    # PyTorch Dataset sanity (skip if torch is not installed -- it's deferred).
    try:
        import torch  # noqa: F401
        from torch.utils.data import Dataset, DataLoader

        class IsingHDF5Dataset(Dataset):
            def __init__(self, configs: np.ndarray, T: float):
                self.x = torch.from_numpy(configs.astype(np.float32))[:, None, :]
                self.y = torch.full((len(self.x),), float(T))

            def __len__(self):
                return len(self.x)

            def __getitem__(self, idx):
                return self.x[idx], self.y[idx]

        sample_block = blocks[0]
        ds = IsingHDF5Dataset(sample_block["configurations"], sample_block["T"])
        dl = DataLoader(ds, batch_size=16, shuffle=True)
        x_batch, y_batch = next(iter(dl))
        print(f"  PyTorch Dataset OK: x.shape={tuple(x_batch.shape)}, "
              f"y.shape={tuple(y_batch.shape)}, x.dtype={x_batch.dtype}")
    except ImportError:
        print("  PyTorch not installed; Dataset check skipped (expected for Phase 1).")

    # ---------- Part C: storage integrity ----------
    print()
    print("=" * 78)
    print(f"PART C  storage integrity ({args.n_integrity_checks} random samples)")
    print("=" * 78)
    integrity_failures = 0
    max_e_diff = 0.0
    max_m_diff = 0.0
    for _ in range(args.n_integrity_checks):
        b = blocks[rng.integers(0, len(blocks))]
        i = int(rng.integers(0, b["configurations"].shape[0]))
        e_stored = float(b["energies"][i])
        m_stored = float(b["magnetizations"][i])
        e_manual = manual_energy(b["configurations"][i])
        m_manual = manual_magnetization(b["configurations"][i])
        d_e = abs(e_stored - e_manual)
        d_m = abs(m_stored - m_manual)
        max_e_diff = max(max_e_diff, d_e)
        max_m_diff = max(max_m_diff, d_m)
        if d_e > 0 or d_m > 0:
            print(f"  L={b['L']} T={b['T']:.4f} i={i}  "
                  f"E_stored={e_stored} E_manual={e_manual}  "
                  f"M_stored={m_stored} M_manual={m_manual}")
            integrity_failures += 1
    print(f"  max |E_stored - E_manual| = {max_e_diff}")
    print(f"  max |M_stored - M_manual| = {max_m_diff}")
    print(f"  integrity failures: {integrity_failures} / {args.n_integrity_checks}")

    # ---------- Part A: physics ----------
    print()
    print("=" * 78)
    print("PART A  physics accuracy vs exact 1D Ising solution")
    print("=" * 78)

    rows = []
    for b in blocks:
        L = b["L"]; T = b["T"]
        E = b["energies"]; M = b["magnetizations"]

        e_mc = E.mean() / L
        c_mc = mc_specific_heat_per_spin(E, L, T)
        chi_mc = mc_susceptibility_per_spin(M, L, T)
        m_signed_mc = M.mean() / L
        # G(r) for the plot; xi estimated from G(1) for robustness under PBC.
        G = mc_correlator(b["configurations"], r_max=min(L // 2, 64))
        xi_fit = xi_from_G1(G)

        e_ex = exact_energy_per_spin(T)
        c_ex = exact_specific_heat_per_spin(T)
        chi_ex = exact_susceptibility_per_spin(T)
        xi_ex = exact_correlation_length(T)

        rows.append({
            "L": L, "T": T,
            "e_mc": e_mc, "e_ex": e_ex,
            "c_mc": c_mc, "c_ex": c_ex,
            "chi_mc": chi_mc, "chi_ex": chi_ex,
            "xi_mc": xi_fit, "xi_ex": xi_ex,
            "m_signed_mc": m_signed_mc,
            "G": G,
        })

    def rel_diff(mc, ex):
        return abs(mc - ex) / abs(ex) if ex != 0 else abs(mc)

    # Aggregate stats per observable
    def summarize(name, key_mc, key_ex, tol, *, allow_skip=False, lin_or_rel="rel"):
        diffs = []
        fails = []
        for r in rows:
            mc = r[key_mc]; ex = r[key_ex]
            if mc is None or (isinstance(mc, float) and not np.isfinite(mc)):
                if allow_skip:
                    continue
                else:
                    fails.append((r["L"], r["T"], float("inf"), mc, ex))
                    diffs.append(float("inf"))
                    continue
            d = rel_diff(mc, ex) if lin_or_rel == "rel" else abs(mc - ex)
            diffs.append(d)
            if d > tol:
                fails.append((r["L"], r["T"], d, mc, ex))
        diffs = np.array(diffs, dtype=float) if diffs else np.array([0.0])
        finite = diffs[np.isfinite(diffs)]
        print(f"  {name:24s}  blocks={len(rows)}  "
              f"max={finite.max() if len(finite) else float('nan'):.4f}  "
              f"mean={finite.mean() if len(finite) else float('nan'):.4f}  "
              f"fail={len(fails)}  tol={tol}")
        for L, T, d, mc, ex in fails:
            print(f"      fail: L={L:4d} T={T:6.4f}  rel_err={d:.4f}  mc={mc:.4f}  ex={ex:.4f}")
        return len(fails)

    fails = 0
    fails += summarize("energy per spin",       "e_mc",   "e_ex",   args.rel_tol_energy)
    fails += summarize("specific heat per spin","c_mc",   "c_ex",   args.rel_tol_cv)
    fails += summarize("susceptibility per spin","chi_mc","chi_ex", args.rel_tol_chi)
    fails += summarize("correlation length",    "xi_mc",  "xi_ex",  args.rel_tol_xi, allow_skip=True)

    # Signed magnetization: should average to 0 by Z2 symmetry. Standard error
    # of <M>/N is sqrt(<M^2>/N^2) / sqrt(n_eff) ~ sqrt(xi/L) / sqrt(n_eff).
    # Single-spin-flip Metropolis at low T has integrated autocorrelation time
    # of order xi^2 in 1D, so n_eff ~ n_samples * decorrelation / max(xi^2, 1).
    # Wolff would fix this; for now we widen the threshold accordingly.
    n_m_fail = 0
    n_samples = 1000
    decorr = 10
    for r in rows:
        L = r["L"]
        xi_ex = r["xi_ex"] if np.isfinite(r["xi_ex"]) else float(L)
        n_eff = max(1.0, n_samples * decorr / max(xi_ex ** 2, 1.0))
        sigma_M_per_N = np.sqrt(min(xi_ex, L) / L)
        scale = 5.0 * sigma_M_per_N / np.sqrt(n_eff)
        if abs(r["m_signed_mc"]) > max(scale, 0.01):
            n_m_fail += 1
    print(f"  signed magnetization     blocks={len(rows)}  "
          f"max|<M>/N|={max(abs(r['m_signed_mc']) for r in rows):.4f}  "
          f"fail={n_m_fail}")
    fails += n_m_fail

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
            Ls = sorted({r["L"] for r in rows})
            exact_fns = {
                "e_ex":   exact_energy_per_spin,
                "c_ex":   exact_specific_heat_per_spin,
                "chi_ex": exact_susceptibility_per_spin,
                "xi_ex":  exact_correlation_length,
            }
            panels = [
                ("e_mc",         "e_ex",   r"$\langle E\rangle/N$", "Energy per spin"),
                ("c_mc",         "c_ex",   r"$C/N$",                "Specific heat per spin"),
                ("chi_mc",       "chi_ex", r"$\chi/N$",             "Susceptibility (large-L limit)"),
                ("xi_mc",        "xi_ex",  r"$\xi$",                "Correlation length (log y)"),
                ("m_signed_mc",  None,     r"$\langle M\rangle/N$", "Signed M (should be 0)"),
                ("G",            None,    r"$|G(r)|$ at $T\approx 1.5$", "Correlation function"),
            ]
            fig, axes = plt.subplots(2, 3, figsize=(15, 9))
            T_min = min(r["T"] for r in rows); T_max = max(r["T"] for r in rows)
            Ts_fine = np.linspace(T_min, T_max, 200)

            for ax, (key_mc, key_ex, ylabel, title) in zip(axes.flat, panels):
                if key_mc == "G":
                    target = min(rows, key=lambda r: abs(r["T"] - 1.5))
                    Lt, Tt, Gt = target["L"], target["T"], target["G"]
                    r_arr = np.arange(len(Gt))
                    G_ex = np.tanh(1.0 / Tt) ** r_arr
                    rmax = min(Lt // 2, len(Gt))
                    ax.semilogy(r_arr[:rmax], np.abs(Gt[:rmax]), "o", ms=3, label="MC |G|")
                    ax.semilogy(r_arr[:rmax], G_ex[:rmax], "-",
                                label=f"exact $u^r$, T={Tt:.2f}, L={Lt}")
                    ax.set_xlabel("r")
                elif key_mc == "m_signed_mc":
                    for L in Ls:
                        sub = sorted((r for r in rows if r["L"] == L), key=lambda r: r["T"])
                        ax.plot([r["T"] for r in sub], [r["m_signed_mc"] for r in sub],
                                "o-", ms=3, label=f"L={L}")
                    ax.axhline(0.0, color="k", lw=0.5)
                    ax.set_xlabel("T")
                else:
                    for L in Ls:
                        sub = sorted((r for r in rows if r["L"] == L), key=lambda r: r["T"])
                        ys = [r[key_mc] for r in sub]
                        ys = [np.nan if v is None else v for v in ys]
                        ax.plot([r["T"] for r in sub], ys, "o", ms=3, label=f"L={L}")
                    if key_ex is not None:
                        ax.plot(Ts_fine, [exact_fns[key_ex](T) for T in Ts_fine],
                                "k-", lw=1, label="exact")
                    ax.set_xlabel("T")
                    if key_mc == "xi_mc":
                        ax.set_yscale("log")
                ax.set_ylabel(ylabel); ax.set_title(title); ax.legend(fontsize=8)

            fig.suptitle("1D Ising validation: MC vs exact analytical solution", fontsize=13)
            fig.tight_layout()
            fig.savefig(args.figure, dpi=120)
            print(f"figure -> {args.figure}")

    print()
    print("=" * 78)
    print(f"TOTAL physics failures (across observables): {fails}")
    print(f"TOTAL format failures: {format_issues + pbc_issues}")
    print(f"TOTAL integrity failures: {integrity_failures}")
    print("=" * 78)

    return 0 if (fails + format_issues + pbc_issues + integrity_failures) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
