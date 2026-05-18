"""Dimension-agnostic CNN for Ising configurations.

The whole project hinges on one architectural property: **the same trained
weights must process 1D, 2D, 3D, and 4D lattices**. Only then is "train on
d=1,2,3, test on d=4" a meaningful experiment -- a per-dimension encoder would
need a 4D encoder whose weights were never trained, producing garbage.

How this network achieves dimension-agnosticism:

  - A "convolution" here is a per-site linear map of (a) the spin itself and
    (b) the sum of its nearest neighbors. The neighbor sum is taken with
    `torch.roll`, which on a periodic lattice IS circular padding -- it
    matches the Ising periodic boundary conditions exactly, for free.
  - The neighbor sum runs over *however many spatial axes the input has*.
    1D: 2 neighbors. 2D: 4. 3D: 6. 4D: 8. The weights never change.
  - Reflection symmetry of the Ising Hamiltonian is baked in: roll(+1) and
    roll(-1) are summed before the linear map, so the network cannot tell
    +axis from -axis (as it shouldn't).
  - Global average pooling over the spatial axes makes the network agnostic
    to both lattice size L and dimensionality d -- the pooled feature vector
    is always (N, C) regardless of the input shape.

Parameter count is therefore a single number, independent of dimension
(~22K with the defaults). That is the point.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FactorizedConv(nn.Module):
    """Dimension-agnostic nearest-neighbor convolution.

    out = W_center @ x  +  W_neighbor @ (sum over the 2d nearest neighbors)  + b

    Implemented with two 1x1 'convolutions' (per-site linear maps), so the
    weight tensors have no spatial extent and do not depend on dimension.
    Neighbors are gathered with torch.roll == periodic padding.
    """

    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.center = nn.Linear(c_in, c_out)
        self.neighbor = nn.Linear(c_in, c_out, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, C, *spatial) with d spatial axes at positions 2 .. x.dim()-1
        nbr = torch.zeros_like(x)
        for ax in range(2, x.dim()):
            nbr = nbr + torch.roll(x, 1, ax) + torch.roll(x, -1, ax)
        # 1x1 conv == Linear on the channel axis; move C last, then back.
        xc = x.movedim(1, -1)        # (N, *spatial, C_in)
        nc = nbr.movedim(1, -1)
        out = self.center(xc) + self.neighbor(nc)   # (N, *spatial, C_out)
        return out.movedim(-1, 1)    # (N, C_out, *spatial)


class IsingCNN(nn.Module):
    """Dimension-agnostic CNN.

    Input:  (N, 1, *spatial), spatial = (L,) | (L,L) | (L,L,L) | (L,L,L,L)
    Output: (N, n_out)

    `features()` exposes the pooled pre-head representation -- used for the
    latent-space universality-collapse analysis (do critical configurations
    from different dimensions land in the same region of feature space?).
    """

    def __init__(self, n_out: int = 2,
                 channels: tuple[int, ...] = (16, 32, 64),
                 head_hidden: int = 128):
        super().__init__()
        self.blocks = nn.ModuleList()
        c_prev = 1
        for c in channels:
            self.blocks.append(FactorizedConv(c_prev, c))
            c_prev = c
        self.feature_dim = c_prev
        self.head = nn.Sequential(
            nn.Linear(c_prev, head_hidden), nn.ReLU(),
            nn.Linear(head_hidden, head_hidden // 2), nn.ReLU(),
            nn.Linear(head_hidden // 2, n_out),
        )

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """Pooled feature vector (N, feature_dim) for any input dimensionality."""
        h = x
        for blk in self.blocks:
            h = F.relu(blk(h))
        spatial_axes = tuple(range(2, h.dim()))
        return h.mean(dim=spatial_axes)   # global average pool -> (N, C)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Self-test: one model instance must process all four dimensionalities.
    torch.manual_seed(0)
    model = IsingCNN(n_out=2)
    print(f"IsingCNN parameters: {count_parameters(model):,}")
    print("same weights applied to every dimension:")
    for d, L in [(1, 64), (2, 16), (3, 8), (4, 4)]:
        x = torch.randint(0, 2, (4, 1) + (L,) * d, dtype=torch.float32) * 2 - 1
        with torch.no_grad():
            feat = model.features(x)
            out = model(x)
        print(f"  d={d}  input {str(tuple(x.shape)):<22}  "
              f"features {tuple(feat.shape)}  output {tuple(out.shape)}")
