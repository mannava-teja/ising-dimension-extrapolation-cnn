"""PyTorch datasets for Ising configurations.

Design notes:

  - The configurations are int8 on disk. The whole multi-dimensional corpus
    is only ~430 MB as int8, so each dataset loads its blocks fully into RAM
    and casts a single sample to float32 in __getitem__. No streaming needed.

  - Configurations of different lattice size L -- and of different
    dimensionality d -- cannot share a batch tensor. `ShapeBatchSampler`
    buckets sample indices by shape so every batch is homogeneous. Over an
    epoch the model still sees all shapes, interleaved.

  - Augmentation uses the exact symmetries of the Ising Hamiltonian on a
    hypercubic torus: reflections, axis permutations, periodic translations,
    and the global Z2 spin flip. All leave energy (hence the label)
    invariant, so they are free, label-preserving data.

  - Train / validation split is per block: each block's 1000 samples are
    shuffled with a fixed seed and split, so the split is reproducible and
    identical across runs. The 4D set is never split here -- it is the
    held-out test set, loaded separately and only at evaluation time.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from ising.storage import iter_blocks

# Critical temperatures (J = k_B = 1). 1D has no transition.
T_C = {1: 0.0,
       2: 2.0 / np.log(1.0 + np.sqrt(2.0)),  # 2.2691853...
       3: 4.5115,
       4: 6.6803,
       5: 8.778}

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_H5 = {d: REPO_ROOT / "data" / f"ising_{d}d.h5" for d in (1, 2, 3, 4, 5)}


class IsingDataset(Dataset):
    """Ising configurations from one or more dimensions.

    Parameters
    ----------
    dims : list[int]
        Which dimensionalities to include (each loaded from data/ising_<d>d.h5).
    task : {"classify", "regress"}
        "classify" -> label 1 if T > T_c(dim) else 0 (ordered/disordered).
        "regress"  -> label is the temperature T.
    split : {"train", "val", "all"}
        Per-block split. "all" is used for held-out evaluation sets.
    val_frac : float
        Fraction of each block reserved for validation.
    augment : bool
        Apply random lattice-symmetry + Z2 augmentation in __getitem__.
    sizes : list[int] | None
        Optional filter on lattice size L (useful for fast smoke runs).
    max_per_block : int | None
        If set, use at most this many samples from each (L, T) block (the
        train/val split is taken within that cap). Caps CPU training cost.
    """

    def __init__(self, dims, *, task="classify", split="all", val_frac=0.15,
                 augment=False, sizes=None, max_per_block=None,
                 h5_paths=None, seed=0):
        if task not in ("classify", "regress"):
            raise ValueError(f"unknown task: {task}")
        if split not in ("train", "val", "all"):
            raise ValueError(f"unknown split: {split}")
        self.task = task
        self.augment = augment
        self._rng = np.random.default_rng(seed)
        h5_paths = h5_paths or DEFAULT_H5

        split_rng = np.random.default_rng(20260517)  # fixed -> reproducible split
        self.buckets = []      # each: dict(dim, L, T, configs int8, labels)
        self._index = []       # (bucket_idx, sample_idx) for every usable sample

        for d in dims:
            path = Path(h5_paths[d])
            if not path.exists():
                raise FileNotFoundError(f"dim {d} dataset missing: {path}")
            for b in iter_blocks(path, dim=d):
                if sizes is not None and b["L"] not in sizes:
                    continue
                cfg = b["configurations"]          # (N, *spatial) int8
                N = cfg.shape[0]
                T = float(b["T"])

                perm = split_rng.permutation(N)
                if max_per_block is not None:      # cap samples used per block
                    perm = perm[:max_per_block]
                n_use = len(perm)
                n_val = int(round(val_frac * n_use))
                if split == "train":
                    sel = perm[n_val:]
                elif split == "val":
                    sel = perm[:n_val]
                else:
                    sel = perm

                if task == "classify":
                    cls = 1 if T > T_C[d] else 0
                    labels = np.full(len(sel), cls, dtype=np.int64)
                else:
                    labels = np.full(len(sel), T, dtype=np.float32)

                self.buckets.append({
                    "dim": d, "L": b["L"], "T": T,
                    "configs": cfg[sel], "labels": labels,
                })
                bi = len(self.buckets) - 1
                self._index.extend((bi, si) for si in range(len(sel)))

        if not self._index:
            raise ValueError("dataset is empty -- check dims / sizes filters")

    # -- augmentation -------------------------------------------------------

    def _augment(self, cfg: np.ndarray) -> np.ndarray:
        d = cfg.ndim
        for ax in range(d):                       # reflections
            if self._rng.random() < 0.5:
                cfg = np.flip(cfg, ax)
        cfg = np.transpose(cfg, self._rng.permutation(d))  # axis permutation
        shifts = [int(self._rng.integers(0, s)) for s in cfg.shape]  # translation
        cfg = np.roll(cfg, shifts, axis=tuple(range(d)))
        if self._rng.random() < 0.5:              # global Z2 spin flip
            cfg = -cfg
        return np.ascontiguousarray(cfg)

    # -- Dataset protocol ---------------------------------------------------

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, i):
        bi, si = self._index[i]
        bucket = self.buckets[bi]
        cfg = bucket["configs"][si].astype(np.float32)
        if self.augment:
            cfg = self._augment(cfg)
        x = torch.from_numpy(cfg).unsqueeze(0)    # (1, *spatial)
        label = bucket["labels"][si]
        if self.task == "classify":
            y = torch.tensor(int(label), dtype=torch.long)
        else:
            y = torch.tensor(float(label), dtype=torch.float32)
        return x, y

    # -- helpers for per-block evaluation ----------------------------------

    def block_summary(self):
        """One row per bucket: (dim, L, T, n_samples). For evaluation/reporting."""
        return [(b["dim"], b["L"], b["T"], len(b["labels"])) for b in self.buckets]


class ShapeBatchSampler(Sampler):
    """Yield batches whose samples all share one (dim, L) shape.

    A batch tensor must be homogeneous in shape; configurations from different
    dimensions or lattice sizes cannot stack. This sampler groups the dataset's
    global indices by shape and forms batches within each group, then shuffles
    the batch order so the model sees shapes interleaved.
    """

    def __init__(self, dataset: IsingDataset, batch_size: int,
                 shuffle: bool = True, seed: int = 0):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self._epoch = 0
        groups: dict[tuple[int, int], list[int]] = {}
        for gi, (bi, _si) in enumerate(dataset._index):
            b = dataset.buckets[bi]
            groups.setdefault((b["dim"], b["L"]), []).append(gi)
        self.groups = groups

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self._epoch)
        self._epoch += 1
        batches = []
        for idxs in self.groups.values():
            idxs = np.array(idxs)
            if self.shuffle:
                rng.shuffle(idxs)
            for s in range(0, len(idxs), self.batch_size):
                batches.append(idxs[s:s + self.batch_size].tolist())
        if self.shuffle:
            rng.shuffle(batches)
        return iter(batches)

    def __len__(self) -> int:
        return sum((len(idxs) + self.batch_size - 1) // self.batch_size
                   for idxs in self.groups.values())
