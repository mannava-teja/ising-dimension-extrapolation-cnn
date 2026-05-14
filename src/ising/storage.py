"""HDF5 storage for Ising Monte Carlo data.

Schema:

    /dim_<d>/L_<L>/T_<T:.4f>/
        configurations  (N, *spatial)  int8
        energies        (N,)           float64
        magnetizations  (N,)           float64
        attrs: T, L, dim, seed, algorithm, n_thermalization, decorrelation,
               timestamp_utc, git_commit (root only)

Temperatures are encoded with four decimal places (e.g. T_2.2690) so that the
dense sampling near T_c in 2D/3D doesn't collide on naming.
"""

from __future__ import annotations

import datetime as _dt
import os
import subprocess
from pathlib import Path
from typing import Iterator

import h5py

from ising.metropolis_1d import Sim1DResult

T_FMT = "{:.4f}"


def _t_key(T: float) -> str:
    return "T_" + T_FMT.format(T)


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=2, check=False,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return "unknown"


def _ensure_root_attrs(f: h5py.File) -> None:
    if "git_commit" not in f.attrs:
        f.attrs["git_commit"] = _git_commit()
    if "created_utc" not in f.attrs:
        f.attrs["created_utc"] = _dt.datetime.utcnow().isoformat(timespec="seconds")
    if "schema_version" not in f.attrs:
        f.attrs["schema_version"] = 1


def write_samples(
    h5_path: os.PathLike | str,
    dim: int,
    L: int,
    result: Sim1DResult,
    *,
    algorithm: str = "metropolis",
    overwrite: bool = False,
) -> str:
    """Write a single (dim, L, T) block to the HDF5 file. Returns the group path."""
    path = Path(h5_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    group_path = f"/dim_{dim}/L_{L}/{_t_key(result.T)}"

    with h5py.File(path, "a") as f:
        _ensure_root_attrs(f)
        if group_path in f:
            if not overwrite:
                raise FileExistsError(
                    f"{group_path} already exists in {path}. Pass overwrite=True to replace."
                )
            del f[group_path]
        g = f.create_group(group_path)

        g.create_dataset(
            "configurations",
            data=result.configurations,
            dtype="int8",
            chunks=True,
            compression="gzip",
            compression_opts=4,
            shuffle=True,
        )
        g.create_dataset("energies", data=result.energies, dtype="float64")
        g.create_dataset("magnetizations", data=result.magnetizations, dtype="float64")

        g.attrs["T"] = float(result.T)
        g.attrs["L"] = int(L)
        g.attrs["dim"] = int(dim)
        g.attrs["seed"] = int(result.seed)
        g.attrs["algorithm"] = algorithm
        g.attrs["n_thermalization"] = int(result.n_thermalization)
        g.attrs["decorrelation"] = int(result.decorrelation)
        g.attrs["n_samples"] = int(result.configurations.shape[0])
        g.attrs["written_utc"] = _dt.datetime.utcnow().isoformat(timespec="seconds")

    return group_path


def iter_blocks(h5_path: os.PathLike | str, dim: int | None = None) -> Iterator[dict]:
    """Yield one dict per (dim, L, T) block, eagerly loading arrays into memory."""
    path = Path(h5_path)
    with h5py.File(path, "r") as f:
        dims = [f"dim_{dim}"] if dim is not None else sorted(f.keys())
        for dkey in dims:
            if dkey not in f:
                continue
            for lkey in sorted(f[dkey].keys()):
                for tkey in sorted(f[dkey][lkey].keys()):
                    g = f[dkey][lkey][tkey]
                    yield {
                        "dim": int(g.attrs["dim"]),
                        "L": int(g.attrs["L"]),
                        "T": float(g.attrs["T"]),
                        "seed": int(g.attrs["seed"]),
                        "algorithm": str(g.attrs["algorithm"]),
                        "n_samples": int(g.attrs["n_samples"]),
                        "configurations": g["configurations"][...],
                        "energies": g["energies"][...],
                        "magnetizations": g["magnetizations"][...],
                    }
