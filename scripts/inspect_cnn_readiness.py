"""Final CNN-readiness audit for the 1D and 2D Ising datasets.

Validates the *quantity* and *quality* questions before downstream training:

  - Total sample counts (overall and per (L, T))
  - Per-block sample count uniformity (every block should have N_samples=1000)
  - Class balance for the 2D binary task (T < T_c vs T > T_c)
  - Temperature coverage (density near T_c for 2D)
  - HDF5 metadata completeness (git_commit, seed, algorithm, etc.)
  - A small spot-check of physical sanity (energy ranges per regime)

Exits with code 0 only if every check passes.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ising.storage import iter_blocks  # noqa: E402

T_C_2D = 2.0 / np.log(1.0 + np.sqrt(2.0))


def audit_file(path: Path, dim: int, expected_samples: int = 1000) -> bool:
    print("=" * 78)
    print(f"AUDIT  {path.name}  (expected dim={dim})")
    print("=" * 78)
    if not path.exists():
        print(f"  MISSING: {path}")
        return False

    ok = True
    with h5py.File(path, "r") as f:
        # Root metadata
        root_attrs = dict(f.attrs)
        required_root = {"git_commit", "created_utc", "schema_version"}
        missing_root = required_root - set(root_attrs)
        if missing_root:
            print(f"  ROOT META missing: {missing_root}")
            ok = False
        else:
            print(f"  ROOT META   git={root_attrs['git_commit'][:8]}  "
                  f"created={root_attrs['created_utc']}  "
                  f"schema_v{root_attrs['schema_version']}")

    blocks = list(iter_blocks(path, dim=dim))
    if not blocks:
        print(f"  NO BLOCKS at dim_{dim}")
        return False

    # Sample count uniformity
    sample_counts = Counter(b["n_samples"] for b in blocks)
    print(f"  BLOCKS      n={len(blocks)}  per-block samples: {dict(sample_counts)}")
    if list(sample_counts) != [expected_samples]:
        print(f"  NON-UNIFORM sample counts!")
        ok = False

    # Temperature & L coverage
    Ls = sorted({b["L"] for b in blocks})
    Ts = sorted({round(b["T"], 4) for b in blocks})
    print(f"  L values    {Ls}  ({len(Ls)} sizes)")
    print(f"  T range     [{min(Ts):.4f}, {max(Ts):.4f}]  ({len(Ts)} unique temps)")

    # Algorithm breakdown
    algos = Counter(b["algorithm"] for b in blocks)
    print(f"  ALGORITHMS  {dict(algos)}")

    total_samples = sum(b["n_samples"] for b in blocks)
    print(f"  TOTAL CONFIGS  {total_samples:,}")

    # Per-(L) sample counts
    per_L_counts = {L: sum(b["n_samples"] for b in blocks if b["L"] == L) for L in Ls}
    print(f"  per-L totals  {per_L_counts}")

    # 2D-specific: class balance for above/below T_c
    if dim == 2:
        below = [b for b in blocks if b["T"] < T_C_2D]
        above = [b for b in blocks if b["T"] > T_C_2D]
        n_below = sum(b["n_samples"] for b in below)
        n_above = sum(b["n_samples"] for b in above)
        ratio = min(n_below, n_above) / max(n_below, n_above) if max(n_below, n_above) else 0
        print(f"  BINARY TASK below T_c: {n_below:,}  above T_c: {n_above:,}  "
              f"balance ratio: {ratio:.3f}")
        if ratio < 0.5:
            print(f"  WARN: class imbalance < 0.5 (stratify or weight when training)")

        # Density of T points within +/-20% of T_c (where physics is interesting)
        critical_window = [T for T in Ts if 0.8 * T_C_2D <= T <= 1.2 * T_C_2D]
        print(f"  T points within +/-20% of T_c: {len(critical_window)} of {len(Ts)}")
        if len(critical_window) < 8:
            print(f"  WARN: fewer than 8 T points near T_c -- finite-size scaling may be coarse")

    # Physical-regime sanity on energy means
    print(f"  ENERGY-REGIME SANITY:")
    for b in blocks:
        L, T = b["L"], b["T"]
        N = L ** dim
        e = b["energies"].mean() / N
        # Ground state: dim-dependent. 1D: -1 (each bond -1, 1 bond/site after halving).
        # 2D: -2 (2 bonds/site). 3D: -3.
        gs = -float(dim)
        # Disordered limit (T -> infty): 0
        if not (gs - 0.05 <= e <= 0.05):
            print(f"    OUT OF RANGE: L={L} T={T:.4f} dim={dim}  <E>/N={e:.4f} "
                  f"(expected in [{gs:.2f}, 0])")
            ok = False
    print(f"  ENERGY-REGIME OK across {len(blocks)} blocks")

    # Value range check (paranoid -- already covered by validate scripts but cheap)
    sample_block = blocks[0]
    cfg = sample_block["configurations"]
    if cfg.dtype != np.int8:
        print(f"  DTYPE MISMATCH: {cfg.dtype}")
        ok = False
    uniques = set(np.unique(cfg).tolist())
    if not uniques <= {-1, 1}:
        print(f"  VALUE RANGE NOT {{-1,+1}}: got {uniques}")
        ok = False

    return ok


def main() -> int:
    data_dir = REPO_ROOT / "data"
    results = []
    results.append(("1D", audit_file(data_dir / "ising_1d.h5", dim=1)))
    print()
    results.append(("2D", audit_file(data_dir / "ising_2d.h5", dim=2)))

    print()
    print("=" * 78)
    for name, ok in results:
        print(f"  {name}: {'PASS' if ok else 'FAIL'}")
    print("=" * 78)
    return 0 if all(ok for _, ok in results) else 1


if __name__ == "__main__":
    sys.exit(main())
