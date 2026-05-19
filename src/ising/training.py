"""Training loop and per-block evaluation for the dimension-agnostic CNN."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ising.datasets import IsingDataset, ShapeBatchSampler, T_C


@dataclass
class TrainHistory:
    epochs: list[dict] = field(default_factory=list)
    best_epoch: int = -1
    best_val: float = float("inf")


def _run_epoch(model, loader, task, loss_fn, device, optimizer=None):
    """One pass over `loader`. Trains if `optimizer` is given, else evaluates."""
    training = optimizer is not None
    model.train(training)
    total_loss, total_n, total_correct = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.set_grad_enabled(training):
            out = model(x)
            if task == "classify":
                loss = loss_fn(out, y)
                total_correct += (out.argmax(-1) == y).sum().item()
            else:
                loss = loss_fn(out.squeeze(-1), y)
        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * len(y)
        total_n += len(y)
    metrics = {"loss": total_loss / total_n}
    if task == "classify":
        metrics["acc"] = total_correct / total_n
    return metrics


def train_model(model, train_ds: IsingDataset, val_ds: IsingDataset, *,
                task: str = "classify", epochs: int = 30, batch_size: int = 64,
                lr: float = 1e-3, patience: int = 8, device: str = "cpu",
                seed: int = 0, verbose: bool = True) -> TrainHistory:
    """Train `model`, early-stopping on validation loss. Restores best weights."""
    torch.manual_seed(seed)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss() if task == "classify" else nn.MSELoss()

    train_loader = DataLoader(
        train_ds, batch_sampler=ShapeBatchSampler(train_ds, batch_size,
                                                  shuffle=True, seed=seed))
    val_loader = DataLoader(
        val_ds, batch_sampler=ShapeBatchSampler(val_ds, batch_size,
                                                shuffle=False, seed=seed))

    history = TrainHistory()
    best_state = None
    bad_epochs = 0

    for epoch in range(epochs):
        t0 = time.time()
        tr = _run_epoch(model, train_loader, task, loss_fn, device, optimizer)
        va = _run_epoch(model, val_loader, task, loss_fn, device, None)
        row = {"epoch": epoch, "train": tr, "val": va, "secs": time.time() - t0}
        history.epochs.append(row)

        if verbose:
            msg = (f"  epoch {epoch:3d}  "
                   f"train loss {tr['loss']:.4f}  val loss {va['loss']:.4f}")
            if task == "classify":
                msg += f"  train acc {tr['acc']:.3f}  val acc {va['acc']:.3f}"
            msg += f"  ({row['secs']:.1f}s)"
            print(msg, flush=True)

        if va["loss"] < history.best_val - 1e-5:
            history.best_val = va["loss"]
            history.best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                if verbose:
                    print(f"  early stopping at epoch {epoch} "
                          f"(best epoch {history.best_epoch})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return history


@torch.no_grad()
def evaluate_per_block(model, dataset: IsingDataset, *, task: str = "classify",
                       batch_size: int = 128, device: str = "cpu") -> list[dict]:
    """Run the model on every block of `dataset`, one block at a time.

    Returns one row per (dim, L, T) block with the mean prediction. For
    classification: `p_disordered`, the mean softmax probability of the
    disordered class -- sweeping this vs T and finding the 0.5 crossing is
    how T_c is read off. For regression: `pred_T`, the mean predicted value.
    """
    model = model.to(device)
    model.eval()
    rows = []
    for b in dataset.buckets:
        cfg = b["configs"]               # (N, *spatial) int8
        N = cfg.shape[0]
        preds = []
        for s in range(0, N, batch_size):
            chunk = cfg[s:s + batch_size].astype(np.float32)
            x = torch.from_numpy(chunk).unsqueeze(1).to(device)  # (n,1,*spatial)
            out = model(x)
            if task == "classify":
                p = torch.softmax(out, dim=-1)[:, 1]   # P(disordered)
                preds.append(p.cpu().numpy())
            else:
                preds.append(out.squeeze(-1).cpu().numpy())
        preds = np.concatenate(preds)
        row = {"dim": b["dim"], "L": b["L"], "T": b["T"], "n": N}
        if task == "classify":
            row["p_disordered"] = float(preds.mean())
            true_cls = 1 if b["T"] > T_C[b["dim"]] else 0
            row["accuracy"] = float(((preds > 0.5).astype(int) == true_cls).mean())
        else:
            row["pred_T"] = float(preds.mean())
            row["pred_T_std"] = float(preds.std())
        rows.append(row)
    return rows


def estimate_tc(block_rows: list[dict], dim: int) -> float | None:
    """Estimate T_c for one dimension from the P(disordered)=0.5 crossing.

    `block_rows` are evaluate_per_block outputs (classification). The crossing
    of the accuracy/probability curve through 0.5, interpolated linearly, is
    the network's estimate of T_c -- the standard Carrasquilla-Melko readout.
    """
    pts = sorted(((r["T"], r["p_disordered"])
                  for r in block_rows if r["dim"] == dim))
    if len(pts) < 2:
        return None
    # Average over L at each T (block_rows may include multiple L).
    by_T: dict[float, list[float]] = {}
    for T, p in pts:
        by_T.setdefault(T, []).append(p)
    curve = sorted((T, float(np.mean(ps))) for T, ps in by_T.items())
    for (t0, p0), (t1, p1) in zip(curve, curve[1:]):
        if (p0 - 0.5) * (p1 - 0.5) <= 0 and p1 != p0:
            return t0 + (0.5 - p0) * (t1 - t0) / (p1 - p0)
    return None
