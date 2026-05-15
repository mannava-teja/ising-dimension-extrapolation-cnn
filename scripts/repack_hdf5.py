"""Repack an HDF5 file to reclaim space from overwritten groups.

HDF5 does not free space when groups are deleted or overwritten -- the
file just grows. Copying every group into a fresh file recovers the
unused space and applies stronger gzip (level 9) on the way.

Usage:
    python scripts/repack_hdf5.py data/ising_2d.h5 data/ising_3d.h5
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import h5py


def repack(path: Path) -> None:
    tmp = path.with_suffix(".repack.h5")
    size_before = path.stat().st_size
    print(f"Repacking {path.name} ({size_before / 1e6:.1f} MB) ...", flush=True)

    with h5py.File(path, "r") as src, h5py.File(tmp, "w") as dst:
        # Root attrs
        for k, v in src.attrs.items():
            dst.attrs[k] = v

        def copy_groups(src_grp, dst_grp):
            for name, obj in src_grp.items():
                if isinstance(obj, h5py.Group):
                    new = dst_grp.create_group(name)
                    for k, v in obj.attrs.items():
                        new.attrs[k] = v
                    copy_groups(obj, new)
                else:  # Dataset
                    kwargs = {}
                    if obj.chunks is not None:
                        kwargs["chunks"] = obj.chunks
                    if obj.compression is not None:
                        # Bump gzip level to 9 for ~5-15% smaller binary spin data.
                        kwargs["compression"] = obj.compression
                        kwargs["compression_opts"] = 9 if obj.compression == "gzip" else obj.compression_opts
                        kwargs["shuffle"] = obj.shuffle
                    new_ds = dst_grp.create_dataset(
                        name, data=obj[...], dtype=obj.dtype, **kwargs
                    )
                    for k, v in obj.attrs.items():
                        new_ds.attrs[k] = v

        copy_groups(src, dst)

    shutil.move(tmp, path)
    size_after = path.stat().st_size
    print(f"  -> {size_after / 1e6:.1f} MB  "
          f"({100 * (size_before - size_after) / size_before:.1f}% smaller)")


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__, file=sys.stderr)
        return 2
    for arg in sys.argv[1:]:
        path = Path(arg)
        if not path.exists():
            print(f"Missing: {path}", file=sys.stderr)
            return 1
        repack(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
