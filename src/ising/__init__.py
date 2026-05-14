"""Ising model Monte Carlo simulation, storage, and (later) CNN code."""

from ising.metropolis_1d import simulate_1d
from ising.metropolis_2d import simulate_2d_metropolis
from ising.wolff_2d import simulate_2d_wolff
from ising.metropolis_3d import simulate_3d_metropolis
from ising.wolff_3d import simulate_3d_wolff

__all__ = [
    "simulate_1d",
    "simulate_2d_metropolis",
    "simulate_2d_wolff",
    "simulate_3d_metropolis",
    "simulate_3d_wolff",
]
