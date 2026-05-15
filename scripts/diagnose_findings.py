"""Deep dive into the audit findings.

Two questions:
  Q1. Where is the lag-1 autocorrelation high? Critical region vs deep
      order vs high T? Quantify effective sample size per block.
  Q2. Where is <M> biased away from 0? Is it just the deep-ordered phase
      where Z2-flipping is rare under Wolff at finite chain length, or
      does it also affect the disordered phase where it would be a bug?
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ising.storage import iter_blocks  # noqa: E402

T_C = {1: 0.0, 2: 2.0 / np.log(1.0 + np.sqrt(2.0)), 3: 4.5115}


def lag1(x):
    x = np.asarray(x, np.float64)
    x = x - x.mean()
    d = (x * x).sum()
    return 0.0 if d == 0 else float((x[:-1] * x[1:]).sum() / d)


def integrated_autocorr(x: np.ndarray, max_lag: int = 100) -> float:
    """Naive integrated autocorrelation time: tau = 1 + 2 sum_{k=1}^M rho(k).
    Standard Sokal automatic-window truncation: stop at first lag where the
    cumulative sum's window M satisfies M >= c * tau (c=5)."""
    x = np.asarray(x, np.float64)
    x = x - x.mean()
    var = (x * x).mean()
    if var == 0:
        return 0.5  # constant -> nominally tau = 0 in the integrated sense
    rho = np.empty(max_lag + 1)
    rho[0] = 1.0
    for k in range(1, max_lag + 1):
        rho[k] = (x[:-k] * x[k:]).mean() / var
    tau = 0.5 + rho[1:].cumsum()
    M = np.arange(1, max_lag + 1)
    cutoff = np.where(M >= 5 * tau)[0]
    return float(tau[cutoff[0]]) if cutoff.size else float(tau[-1])


def main():
    for dim in (2, 3):
        path = REPO_ROOT / "data" / f"ising_{dim}d.h5"
        tc = T_C[dim]
        print()
        print("=" * 80)
        print(f"dim={dim}  T_c={tc:.4f}  reading {path.name}")
        print("=" * 80)

        rows = []
        for b in iter_blocks(path, dim=dim):
            L, T = b["L"], b["T"]
            N = L ** dim
            E = b["energies"]
            M = b["magnetizations"]
            rho_e = lag1(E)
            rho_m = lag1(M)
            # tau_int on energy with automatic-window truncation:
            tau_e = integrated_autocorr(E, max_lag=200)
            n_eff = max(1.0, len(E) / max(2.0 * tau_e, 1.0))
            m_mean = M.mean() / N
            m_abs_mean = np.abs(M).mean() / N
            # z-score on signed M
            m_std = np.abs(M).std(ddof=1)
            z = abs(M.mean()) / max(m_std / np.sqrt(len(M)), 1e-12)
            rows.append({
                "L": L, "T": T,
                "rho_E": rho_e, "rho_M": rho_m,
                "tau_E": tau_e, "n_eff": n_eff,
                "m_signed": m_mean, "m_abs": m_abs_mean,
                "z": z,
            })

        # Group by phase regime
        deep_order = [r for r in rows if r["T"] < 0.85 * tc]
        near_tc   = [r for r in rows if 0.85 * tc <= r["T"] <= 1.15 * tc]
        para      = [r for r in rows if r["T"] > 1.15 * tc]

        for name, group in [("deep order (T<0.85 T_c)", deep_order),
                            ("near T_c (0.85..1.15 T_c)", near_tc),
                            ("paramagnetic (T>1.15 T_c)", para)]:
            if not group:
                continue
            rho_e_vals = [r["rho_E"] for r in group]
            tau_e_vals = [r["tau_E"] for r in group]
            n_eff_vals = [r["n_eff"] for r in group]
            z_vals = [r["z"] for r in group]
            print(f"  {name}  blocks={len(group)}")
            print(f"    lag-1 rho(E):  mean={np.mean(rho_e_vals):.3f}  "
                  f"median={np.median(rho_e_vals):.3f}  "
                  f"max={np.max(rho_e_vals):.3f}")
            print(f"    tau_int(E):    mean={np.mean(tau_e_vals):.1f}  "
                  f"median={np.median(tau_e_vals):.1f}  "
                  f"max={np.max(tau_e_vals):.1f}")
            print(f"    n_eff:         mean={np.mean(n_eff_vals):.0f}  "
                  f"median={np.median(n_eff_vals):.0f}  "
                  f"min={np.min(n_eff_vals):.0f}")
            z_above5 = sum(1 for z in z_vals if z > 5)
            print(f"    z>5 count:     {z_above5} / {len(group)}")

        # Spot inspect worst-rho blocks
        print()
        print("  Worst-5 lag-1 rho(E) blocks in this dim:")
        worst = sorted(rows, key=lambda r: -r["rho_E"])[:5]
        for r in worst:
            print(f"    L={r['L']:4d} T={r['T']:6.4f}  rho_E={r['rho_E']:.3f}  "
                  f"tau_E={r['tau_E']:6.1f}  n_eff={r['n_eff']:5.0f}  "
                  f"<M>/N={r['m_signed']:+.4f}  <|M|>/N={r['m_abs']:.4f}")

        # Spot inspect worst-z blocks restricted to paramagnetic phase
        print()
        print("  Worst-5 z blocks restricted to T > 1.15*T_c (paramagnetic):")
        para_worst = sorted([r for r in rows if r["T"] > 1.15 * tc],
                            key=lambda r: -r["z"])[:5]
        for r in para_worst:
            print(f"    L={r['L']:4d} T={r['T']:6.4f}  z={r['z']:8.2f}  "
                  f"<M>/N={r['m_signed']:+.4f}  <|M|>/N={r['m_abs']:.4f}")


if __name__ == "__main__":
    main()
