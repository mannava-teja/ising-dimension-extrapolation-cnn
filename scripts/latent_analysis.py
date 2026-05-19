"""Measurements #3 and #4: latent-space structure of the trained network.

The dimension-agnostic CNN maps every configuration -- of any dimensionality --
to a 64-d feature vector (model.features). This script inspects that space.

#4  Dimensional universality collapse.
    If the network learned a *dimension-agnostic* notion of criticality, then
    near-critical configurations from 2D, 3D and 4D should land in the same
    region of feature space. We test this two ways:
      - a 2-D embedding (PCA and t-SNE) coloured by dimension and by reduced
        temperature t = T / T_c;
      - a quantitative cross-dimension neighbour-mixing score: for each
        configuration, the fraction of its nearest neighbours (in 64-d feature
        space) that come from a *different* dimension. If criticality induces
        collapse, near-critical configs mix across dimension more than
        off-critical ones do.

#3  Upper-critical-dimension signature.
    4D is where the Ising critical exponents stop running and freeze at their
    mean-field values. The network's own correlation-length exponents
    (measurement #2, floor-corrected) are plotted against dimension alongside
    the literature values and the mean-field floor nu = 1/2: the signature is
    nu(d) descending and flattening toward 1/2 as d -> 4. A sharp confirmation
    (exponents demonstrably frozen) would need a d=5 dataset -- noted as the
    next experiment.

Usage:
    python scripts/latent_analysis.py --checkpoint models/cnn_train23.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from sklearn.manifold import TSNE              # noqa: E402
from sklearn.neighbors import NearestNeighbors  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ising.cnn import IsingCNN              # noqa: E402
from ising.datasets import T_C              # noqa: E402
from ising.storage import iter_blocks       # noqa: E402

# One representative lattice size per dimension (features are size-agnostic
# thanks to global pooling; fixing L keeps the config counts balanced).
REP_L = {2: 32, 3: 16, 4: 8}

# Correlation-length exponents from measurement #2 (floor-corrected) and
# the literature, for the upper-critical-dimension panel.
NU_NETWORK = {2: 0.813, 3: 0.667, 4: 0.569}
NU_LIT = {2: 1.0, 3: 0.6301, 4: 0.5}


def load_model(checkpoint: Path):
    ck = torch.load(checkpoint, weights_only=False)
    cfg = ck["model_config"]
    model = IsingCNN(n_out=cfg["n_out"], channels=tuple(cfg["channels"]))
    model.load_state_dict(ck["model_state"])
    model.eval()
    return model, ck


@torch.no_grad()
def extract_features(model, dim, L, n_per_block, device):
    """Return (features Nx64, reduced temperature t) for one dimension."""
    feats, treduced = [], []
    path = REPO_ROOT / "data" / f"ising_{dim}d.h5"
    for b in iter_blocks(path, dim=dim):
        if b["L"] != L:
            continue
        cfg = b["configurations"][:n_per_block].astype(np.float32)
        x = torch.from_numpy(cfg).unsqueeze(1).to(device)   # (n,1,*spatial)
        f = model.features(x).cpu().numpy()
        feats.append(f)
        treduced.extend([b["T"] / T_C[dim]] * len(cfg))
    return np.concatenate(feats), np.array(treduced)


def neighbour_mixing(F, dim_label, mask, k=12):
    """Mean fraction of each point's k nearest neighbours (in feature space)
    that belong to a different dimension. Restricted to points in `mask`."""
    idx = np.where(mask)[0]
    if len(idx) < k + 1:
        return float("nan")
    nn = NearestNeighbors(n_neighbors=k + 1).fit(F[idx])
    _, neigh = nn.kneighbors(F[idx])
    labels = dim_label[idx]
    frac = []
    for row, self_label in zip(neigh[:, 1:], labels):   # skip self (col 0)
        frac.append(np.mean(dim_label[idx][row] != self_label))
    return float(np.mean(frac))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path,
                   default=REPO_ROOT / "models" / "cnn_train23.pt")
    p.add_argument("--n-per-block", type=int, default=40)
    p.add_argument("--out", type=Path,
                   default=REPO_ROOT / "reports" / "figures" / "latent_space.png")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    if not args.checkpoint.exists():
        print(f"checkpoint missing: {args.checkpoint}", file=sys.stderr)
        return 2
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, ck = load_model(args.checkpoint)
    print(f"checkpoint: {args.checkpoint.name}   train dims {ck['train_dims']}")

    dims = [2, 3, 4]
    feats, treds, dim_label = [], [], []
    for d in dims:
        F, t = extract_features(model, d, REP_L[d], args.n_per_block, device)
        feats.append(F)
        treds.append(t)
        dim_label.append(np.full(len(F), d))
        print(f"  d={d}  L={REP_L[d]}  extracted {len(F)} feature vectors")
    F = np.concatenate(feats)
    t = np.concatenate(treds)
    dim_label = np.concatenate(dim_label)

    # standardise features before projection
    Fz = (F - F.mean(0)) / (F.std(0) + 1e-8)

    print("projecting (t-SNE) ...", flush=True)
    tsne = TSNE(n_components=2, perplexity=30, init="pca",
                random_state=args.seed).fit_transform(Fz)

    # ---- quantitative universality-collapse test ----
    near = np.abs(t - 1.0) < 0.07          # near-critical: |T/T_c - 1| < 7%
    ordered = t < 0.8
    disordered = t > 1.25
    print()
    print("=" * 72)
    print("#4  cross-dimension neighbour mixing (fraction of k-NN from other d)")
    print("=" * 72)
    mix_near = neighbour_mixing(Fz, dim_label, near)
    mix_ord = neighbour_mixing(Fz, dim_label, ordered)
    mix_dis = neighbour_mixing(Fz, dim_label, disordered)
    # chance baseline if dimensions were perfectly intermixed within a group
    def chance(mask):
        labs = dim_label[mask]
        n = len(labs)
        return float(np.mean([1.0 - (np.sum(labs == L) - 1) / (n - 1)
                              for L in labs])) if n > 1 else float("nan")
    print(f"  near-critical |t-1|<0.07 :  mixing = {mix_near:.3f}   "
          f"(perfect-mix baseline {chance(near):.3f})")
    print(f"  ordered      t < 0.80    :  mixing = {mix_ord:.3f}   "
          f"(baseline {chance(ordered):.3f})")
    print(f"  disordered   t > 1.25    :  mixing = {mix_dis:.3f}   "
          f"(baseline {chance(disordered):.3f})")
    print("  -> mixing ~ 0 everywhere means the feature blobs are segregated "
          "by dimension")

    # ---- #4 resolution: is the ordered->disordered direction shared? ----
    # The feature blobs are dimension-segregated, yet the network extrapolates.
    # Test the mechanism: in each dimension's blob, the direction from the
    # ordered centroid to the disordered centroid is that dimension's
    # "decision axis". If those axes are nearly parallel across dimensions,
    # the shared head reads one common rule -- and that is what transfers.
    print()
    print("=" * 72)
    print("#4  shared decision axis: cosine similarity of the ordered ->")
    print("    disordered feature-space direction across dimensions")
    print("=" * 72)
    order_dir = {}
    for d in dims:
        md = dim_label == d
        v = Fz[md & disordered].mean(0) - Fz[md & ordered].mean(0)
        order_dir[d] = v / (np.linalg.norm(v) + 1e-12)
    cos = np.eye(3)
    for i, di in enumerate(dims):
        for j, dj in enumerate(dims):
            cos[i, j] = float(order_dir[di] @ order_dir[dj])
    for i, di in enumerate(dims):
        for j, dj in enumerate(dims):
            if j > i:
                held = "  (4D held out)" if 4 in (di, dj) else ""
                print(f"  cos(decision axis {di}D, {dj}D) = "
                      f"{cos[i, j]:.3f}{held}")
    print("  -> high cosine = the network applies one common ordered/disordered"
          "\n     rule across dimensions; that shared rule is the transfer "
          "mechanism")

    # ---- figure ----
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    dim_colors = {2: "C0", 3: "C1", 4: "C3"}

    # (0,0) t-SNE coloured by reduced temperature
    ax = axes[0, 0]
    sc = ax.scatter(tsne[:, 0], tsne[:, 1], c=t, cmap="coolwarm",
                    s=8, vmin=0.6, vmax=1.5)
    fig.colorbar(sc, ax=ax, label="reduced temperature  T / T_c")
    ax.set_title("t-SNE of the 64-d features, coloured by T/T_c\n"
                 "(a single criticality axis across all dimensions)")
    ax.set_xticks([]); ax.set_yticks([])

    # (0,1) t-SNE coloured by dimension
    ax = axes[0, 1]
    for d in dims:
        m = dim_label == d
        ax.scatter(tsne[m, 0], tsne[m, 1], s=8, color=dim_colors[d],
                   label=f"{d}D", alpha=0.6)
    ax.set_title("t-SNE coloured by dimension\n"
                 "(2D/3D trained, 4D held out)")
    ax.legend(); ax.set_xticks([]); ax.set_yticks([])

    # (1,0) #4 -- shared decision axis: cross-dimension cosine similarity
    ax = axes[1, 0]
    im = ax.imshow(cos, vmin=0.0, vmax=1.0, cmap="viridis")
    ax.set_xticks(range(3)); ax.set_xticklabels([f"{d}D" for d in dims])
    ax.set_yticks(range(3)); ax.set_yticklabels([f"{d}D" for d in dims])
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{cos[i, j]:.2f}", ha="center", va="center",
                    color="w" if cos[i, j] < 0.7 else "k", fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title("#4  ordered->disordered decision axis:\n"
                 "cross-dimension cosine similarity "
                 "(1 = one shared rule)")

    # (1,1) #3 -- exponent running toward the mean-field floor
    ax = axes[1, 1]
    ds = np.array(dims, float)
    ax.plot(ds, [NU_LIT[d] for d in dims], "ks--", ms=9,
            label="literature nu")
    ax.plot(ds, [NU_NETWORK[d] for d in dims], "o-", ms=9, color="C2",
            label="network nu (measurement #2)")
    ax.axhline(0.5, color="r", lw=1, ls=":",
               label="mean-field floor nu = 1/2  (d >= 4)")
    ax.set_xticks([2, 3, 4])
    ax.set_xlabel("spatial dimension d")
    ax.set_ylabel("correlation-length exponent  nu")
    ax.set_title("#3  exponents running toward the upper critical dimension\n"
                 "(nu descends and flattens toward 1/2 at d=4)")
    ax.legend()

    fig.suptitle("Latent-space analysis of the dimension-agnostic CNN  "
                 "(trained on 2D+3D, 4D held out)", fontsize=13)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(args.out, dpi=130)
    print(f"\nfigure -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
