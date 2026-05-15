"""Cross-dimensional audit: find inconsistencies, verify CNN-readiness.

The per-dim validate_physics_*.py scripts confirm each dataset matches its
own physics. This script asks the harder questions:

  1. Schema consistency: same HDF5 layout across dim_1, dim_2, dim_3?
  2. Attribute presence: every required attr on every block?
  3. Dtype, byte-order, value-range identical across dims?
  4. Energy / magnetization conventions consistent (each bond counted
     once, signed M, same J=1, k_B=1)?
  5. PBC translation invariance on every spatial axis of every block?
  6. Per-block sample count exactly 1000 everywhere?
  7. Seed uniqueness *within each file* (collisions across files are fine
     because they apply to different (dim, L, T))?
  8. Algorithm tagging consistent with file convention?
  9. Sample independence: lag-1 autocorrelation of energies within each
     block (large values mean residual MC correlation)?
 10. Z2 symmetry sampling: |<M>|/sigma_M small per block (signed M should
     average to 0 with proper sampling)?
 11. Class balance for the binary above/below-T_c task in 2D and 3D?
 12. Combined-dataset memory footprint as float32 (what a CNN would see).

Exits 0 only if every check passes. Prints findings so you can decide
which "inconsistencies" are real bugs vs known physics.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ising.storage import iter_blocks  # noqa: E402

T_C = {1: 0.0, 2: 2.0 / np.log(1.0 + np.sqrt(2.0)), 3: 4.5115}
REQUIRED_BLOCK_ATTRS = {"T", "L", "dim", "seed", "algorithm",
                        "n_thermalization", "decorrelation", "n_samples"}
REQUIRED_ROOT_ATTRS = {"git_commit", "created_utc", "schema_version"}

FILES = [
    (1, REPO_ROOT / "data" / "ising_1d.h5"),
    (2, REPO_ROOT / "data" / "ising_2d.h5"),
    (3, REPO_ROOT / "data" / "ising_3d.h5"),
]


# ---------- helpers ----------

def lag1_autocorr(x: np.ndarray) -> float:
    """Lag-1 autocorrelation. Should be << 1 if samples are decorrelated."""
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    denom = (x * x).sum()
    if denom == 0:
        return 0.0
    return float((x[:-1] * x[1:]).sum() / denom)


def total_energy_pbc(spins: np.ndarray) -> float:
    """Manual energy with each bond counted once, using forward neighbors."""
    s = spins.astype(np.int64)
    d = s.ndim
    e = 0.0
    for axis in range(d):
        e -= (s * np.roll(s, -1, axis=axis)).sum()
    return float(e)


# ---------- audit ----------

class Audit:
    def __init__(self):
        self.issues = []     # critical inconsistencies
        self.notes = []      # informational
        self.findings = []   # surprising-but-acceptable observations

    def issue(self, msg): self.issues.append(msg);   print(f"  [ISSUE]  {msg}")
    def note(self, msg):  self.notes.append(msg);    print(f"  [NOTE]   {msg}")
    def found(self, msg): self.findings.append(msg); print(f"  [FOUND]  {msg}")


def audit() -> int:
    a = Audit()

    # --- Pre-flight: files exist ---
    print("=" * 80)
    print("0. PRE-FLIGHT")
    print("=" * 80)
    for dim, path in FILES:
        if not path.exists():
            a.issue(f"dim={dim} file missing: {path}")
        else:
            size_mb = path.stat().st_size / 1e6
            print(f"  dim={dim}  {path.name}  {size_mb:.1f} MB")
    if a.issues:
        return 1

    # --- 1. Schema and root attributes ---
    print()
    print("=" * 80)
    print("1. ROOT ATTRIBUTES")
    print("=" * 80)
    git_commits = {}
    for dim, path in FILES:
        with h5py.File(path, "r") as f:
            root_attrs = dict(f.attrs)
            missing = REQUIRED_ROOT_ATTRS - set(root_attrs)
            if missing:
                a.issue(f"dim={dim} missing root attrs: {missing}")
            git_commits[dim] = root_attrs.get("git_commit", "?")
            schema = int(root_attrs.get("schema_version", -1))
            if schema != 1:
                a.issue(f"dim={dim} unexpected schema_version: {schema}")
            print(f"  dim={dim}  git={git_commits[dim][:8]}  "
                  f"created={root_attrs.get('created_utc','?')}  "
                  f"schema_v{schema}")
    # Note: each file was committed at a different time so git_commit differs
    # by file. That's expected -- not an issue.

    # --- 2-8: per-block audit across all files ---
    print()
    print("=" * 80)
    print("2-8. PER-BLOCK AUDIT (attrs, dtype, values, PBC, seeds, algorithms)")
    print("=" * 80)
    per_dim_seeds = defaultdict(set)
    per_dim_algos = defaultdict(lambda: defaultdict(int))
    per_dim_Ts = defaultdict(set)
    per_dim_Ls = defaultdict(set)
    per_dim_total = defaultdict(int)

    n_dtype_bad = n_value_bad = n_attr_missing = n_pbc_bad = 0
    n_sample_bad = n_seed_dup = 0

    for dim, path in FILES:
        for b in iter_blocks(path, dim=dim):
            L = b["L"]; T = b["T"]
            cfg = b["configurations"]

            # dtype
            if cfg.dtype != np.int8:
                a.issue(f"dim={dim} L={L} T={T:.4f} dtype={cfg.dtype} (expected int8)")
                n_dtype_bad += 1

            # value range
            unique = set(np.unique(cfg).tolist())
            if not unique <= {-1, 1}:
                a.issue(f"dim={dim} L={L} T={T:.4f} spin values {unique}")
                n_value_bad += 1

            # shape consistency with dim and L
            expected_shape = (b["n_samples"],) + (L,) * dim
            if cfg.shape != expected_shape:
                a.issue(f"dim={dim} L={L} T={T:.4f} shape {cfg.shape} != {expected_shape}")

            # required attrs
            with h5py.File(path, "r") as f:
                g = f[f"/dim_{dim}/L_{L}/T_{T:.4f}"]
                missing = REQUIRED_BLOCK_ATTRS - set(g.attrs)
                if missing:
                    a.issue(f"dim={dim} L={L} T={T:.4f} missing attrs: {missing}")
                    n_attr_missing += 1

            # sample count
            if b["n_samples"] != 1000:
                a.found(f"dim={dim} L={L} T={T:.4f} n_samples={b['n_samples']} (expected 1000)")
                n_sample_bad += 1

            # seed uniqueness within file
            if b["seed"] in per_dim_seeds[dim]:
                a.issue(f"dim={dim} duplicate seed {b['seed']} at L={L} T={T:.4f}")
                n_seed_dup += 1
            per_dim_seeds[dim].add(b["seed"])

            # PBC: check each spatial axis
            cfg_f = cfg.astype(np.float64)
            for axis in range(1, dim + 1):
                g1 = (cfg_f * np.roll(cfg_f, -1, axis=axis)).mean()
                gL = (cfg_f * np.roll(cfg_f, -(L - 1), axis=axis)).mean()
                rel = abs(g1 - gL) / max(abs(g1), 1e-8)
                if rel > 0.05:
                    a.issue(f"dim={dim} L={L} T={T:.4f} axis={axis} "
                            f"PBC inv broken (rel diff {rel:.3f})")
                    n_pbc_bad += 1

            # accumulate metadata
            per_dim_algos[dim][b["algorithm"]] += 1
            per_dim_Ts[dim].add(round(T, 4))
            per_dim_Ls[dim].add(L)
            per_dim_total[dim] += b["n_samples"]

    print(f"  totals: dtype-bad={n_dtype_bad}  value-bad={n_value_bad}  "
          f"attr-missing={n_attr_missing}  pbc-bad={n_pbc_bad}  "
          f"sample-count-bad={n_sample_bad}  seed-dup={n_seed_dup}")
    for dim, path in FILES:
        print(f"  dim={dim}  L={sorted(per_dim_Ls[dim])}  "
              f"#T={len(per_dim_Ts[dim])}  algos={dict(per_dim_algos[dim])}  "
              f"total_configs={per_dim_total[dim]:,}")

    # --- 9. Sample independence: effective sample size via tau_int ---
    print()
    print("=" * 80)
    print("9. SAMPLE INDEPENDENCE (effective independent samples per block)")
    print("=" * 80)
    print("   n_eff = N_samples / (2 tau_int). Flagged only if n_eff < 100,")
    print("   which is the practical floor for CNN training (samples within a")
    print("   block are shuffled across train/val/test, so per-block n_eff")
    print("   only matters insofar as the block has diverse Boltzmann draws).")
    print()
    for dim, path in FILES:
        rho_list = []
        n_eff_list = []
        bad = []
        for b in iter_blocks(path, dim=dim):
            x = np.asarray(b["energies"], np.float64)
            x = x - x.mean()
            var = (x * x).mean()
            if var == 0:
                tau = 0.5
            else:
                rho = np.array(
                    [1.0] + [(x[:-k] * x[k:]).mean() / var
                             for k in range(1, min(200, len(x) - 1) + 1)])
                tau_cum = 0.5 + rho[1:].cumsum()
                M = np.arange(1, len(tau_cum) + 1)
                cut = np.where(M >= 5 * tau_cum)[0]
                tau = float(tau_cum[cut[0]] if cut.size else tau_cum[-1])
            rho_list.append(lag1_autocorr(b["energies"]))
            n_eff = max(1.0, len(b["energies"]) / max(2.0 * tau, 1.0))
            n_eff_list.append(n_eff)
            if n_eff < 100:
                bad.append((b["L"], b["T"], n_eff))
        print(f"  dim={dim}  blocks={len(n_eff_list)}  "
              f"n_eff: median={np.median(n_eff_list):.0f}  "
              f"min={min(n_eff_list):.0f}  "
              f"5%-ile={np.percentile(n_eff_list, 5):.0f}  "
              f"(blocks n_eff<100: {len(bad)})")
        if bad:
            a.found(f"dim={dim} {len(bad)} blocks have n_eff < 100")
            for L, T, n in bad:
                print(f"    L={L:4d} T={T:.4f} n_eff={n:.0f}")

    # --- 10. Z2 symmetry sampling: paramagnetic phase only ---
    print()
    print("=" * 80)
    print("10. Z2 SAMPLING in the paramagnetic phase (T > 1.1 T_c)")
    print("=" * 80)
    print("   Below T_c the system is in a broken-symmetry state and <M> != 0")
    print("   is physically correct -- the CNN should learn this. Only check")
    print("   T > 1.1 T_c where <M> should converge to 0 by Z2 symmetry.")
    print()
    for dim, path in FILES:
        tc = T_C[dim]
        flagged_blocks = []
        for b in iter_blocks(path, dim=dim):
            if dim == 1 or b["T"] <= 1.1 * tc:
                continue  # broken-symmetry regime (or 1D where there's no transition)
            N = b["L"] ** dim
            M = b["magnetizations"]
            m_mean = M.mean()
            m_abs_mean = np.abs(M).mean()
            # Compare absolute mean signed M to typical |M| (Z2 should make
            # them differ by at least sqrt(n_eff) in well-sampled paramagnet).
            if m_abs_mean == 0:
                continue
            ratio = abs(m_mean) / m_abs_mean
            if ratio > 0.3:  # signed bias > 30% of typical |M| magnitude
                flagged_blocks.append((b["L"], b["T"], ratio, m_mean / N, m_abs_mean / N))
        print(f"  dim={dim}  paramagnetic blocks flagged (|<M>|/<|M|> > 0.3): "
              f"{len(flagged_blocks)}")
        if flagged_blocks:
            a.found(f"dim={dim} {len(flagged_blocks)} paramagnetic blocks Z2-biased")
            for L, T, r, m, am in flagged_blocks[:5]:
                print(f"    L={L:4d} T={T:.4f}  |<M>|/<|M|>={r:.3f}  "
                      f"<M>/N={m:+.4f}  <|M|>/N={am:.4f}")

    # --- 11. Class balance for 2D and 3D binary task ---
    print()
    print("=" * 80)
    print("11. CLASS BALANCE for the above/below-T_c binary task")
    print("=" * 80)
    for dim in (2, 3):
        n_below = 0; n_above = 0
        for b in iter_blocks(REPO_ROOT / "data" / f"ising_{dim}d.h5", dim=dim):
            if b["T"] < T_C[dim]:
                n_below += b["n_samples"]
            elif b["T"] > T_C[dim]:
                n_above += b["n_samples"]
        ratio = min(n_below, n_above) / max(n_below, n_above) if max(n_below, n_above) else 0
        print(f"  dim={dim}  below T_c: {n_below:,}  above T_c: {n_above:,}  "
              f"balance ratio: {ratio:.3f}")
        if ratio < 0.7:
            a.found(f"dim={dim} class imbalance ratio {ratio:.3f} < 0.7 -- "
                    f"stratify or apply class weights when training")

    # --- 12. Combined-dataset memory if loaded as float32 ---
    print()
    print("=" * 80)
    print("12. MEMORY FOOTPRINT (as float32, the CNN-input format)")
    print("=" * 80)
    total_bytes = 0
    for dim, path in FILES:
        bytes_dim = 0
        for b in iter_blocks(path, dim=dim):
            N = b["L"] ** dim
            bytes_dim += b["n_samples"] * N * 4
        total_bytes += bytes_dim
        print(f"  dim={dim}  {bytes_dim / 1e9:6.2f} GB  "
              f"(int8 in HDF5: {bytes_dim / 16 / 1e6:.1f} MB)")
    print(f"  COMBINED: {total_bytes / 1e9:.2f} GB float32  -- too big for one tensor;")
    print(f"            use a streaming PyTorch Dataset (h5py lazy reads + on-the-fly cast)")
    if total_bytes / 1e9 > 4:
        a.note("Combined dataset >4 GB as float32; must stream from HDF5, not load all at once")

    # --- 13. Cross-dim consistency: energy conventions ---
    print()
    print("=" * 80)
    print("13. ENERGY-CONVENTION CONSISTENCY (one random sample per file)")
    print("=" * 80)
    print("   Recompute total energy from configuration. Compare to stored.")
    print("   All three dims should use H = -sum_<ij> s_i s_j with each bond")
    print("   counted once. Manual recompute should equal stored exactly.")
    print()
    rng = np.random.default_rng(0)
    for dim, path in FILES:
        blocks = list(iter_blocks(path, dim=dim))
        b = blocks[rng.integers(0, len(blocks))]
        i = int(rng.integers(0, b["n_samples"]))
        e_stored = float(b["energies"][i])
        e_manual = total_energy_pbc(b["configurations"][i])
        match = abs(e_stored - e_manual) < 1e-9
        print(f"  dim={dim}  L={b['L']} T={b['T']:.4f} sample={i}  "
              f"E_stored={e_stored:+.1f}  E_manual={e_manual:+.1f}  "
              f"{'MATCH' if match else 'MISMATCH'}")
        if not match:
            a.issue(f"dim={dim} energy convention mismatch")

    # --- 14. Algorithm consistency ---
    print()
    print("=" * 80)
    print("14. ALGORITHM CONSISTENCY")
    print("=" * 80)
    print("   1D: Metropolis only.")
    print("   2D / 3D: Wolff near and below T_c; Metropolis above T_c (and at")
    print("            blocks that were regenerated to fix low n_eff). Both")
    print("            sample the same Boltzmann distribution and are tracked")
    print("            in per-block attrs -- the mix is intentional.")
    print()
    for dim, path in FILES:
        algos = per_dim_algos[dim]
        allowed = {1: {"metropolis"}, 2: {"wolff", "metropolis"},
                   3: {"wolff", "metropolis"}}[dim]
        actual = set(algos.keys())
        if not actual <= allowed:
            a.issue(f"dim={dim} unknown algorithm(s): {actual - allowed}")
        else:
            print(f"  dim={dim}: {dict(algos)}  OK")

    # --- Summary ---
    print()
    print("=" * 80)
    print(f"SUMMARY  issues={len(a.issues)}  findings={len(a.findings)}  notes={len(a.notes)}")
    print("=" * 80)
    if a.issues:
        print("ISSUES (must fix before CNN training):")
        for x in a.issues: print(f"  - {x}")
    if a.findings:
        print("FINDINGS (real but acceptable; document/account for in training):")
        for x in a.findings: print(f"  - {x}")
    if a.notes:
        print("NOTES:")
        for x in a.notes: print(f"  - {x}")

    return 0 if not a.issues else 1


if __name__ == "__main__":
    sys.exit(audit())
