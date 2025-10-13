"""
Algorithm registry exposing solver functions for the travelling salesman problem.
"""

from .base import AlgorithmResult
from .branch_and_bound import branch_and_bound
from .cutting_plane import cutting_plane
from .genetic_algorithm import genetic_algorithm
from .greedy import greedy
from .held_karp import held_karp
from .simulated_annealing import simulated_annealing
from .concorde_approx import concorde_approx

ALGORITHMS = {
    "branch_and_bound": branch_and_bound,
    "held_karp": held_karp,
    "cutting_plane": cutting_plane,
    "simulated_annealing": simulated_annealing,
    "genetic_algorithm": genetic_algorithm,
    "concorde_approx": concorde_approx,
    "greedy": greedy,
}

__all__ = [
    "AlgorithmResult",
    "ALGORITHMS",
    "branch_and_bound",
    "held_karp",
    "cutting_plane",
    "simulated_annealing",
    "genetic_algorithm",
    "concorde_approx",
    "greedy",
]
