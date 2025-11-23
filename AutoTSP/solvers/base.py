from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence, Type

import numpy as np

from AutoTSP.utils.taxonomy import AlgorithmFamily


@dataclass
class AlgorithmResult:
    """Container capturing the outcome of running a TSP solver."""

    name: str
    path: List[int] | None
    cost: float | None
    elapsed: float
    status: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class TimeLimitExpired(Exception):
    """Raised when an algorithm exceeds the allotted wall clock budget."""


def current_time() -> float:
    return time.perf_counter()


def remaining_budget(start_time: float, time_limit: float) -> float:
    return time_limit - (current_time() - start_time)


def enforce_time_budget(start_time: float, time_limit: float) -> None:
    if remaining_budget(start_time, time_limit) <= 0:
        raise TimeLimitExpired("Time budget exhausted")


def compute_cycle_cost(dist_matrix: np.ndarray, cycle: Sequence[int]) -> float:
    """Compute tour cost (including return leg)."""
    if not cycle:
        return float("inf")
    cost = 0.0
    for i in range(len(cycle)):
        a = cycle[i]
        b = cycle[(i + 1) % len(cycle)]
        cost += float(dist_matrix[a, b])
    return cost


def best_cycle(points: Sequence[int]) -> List[int]:
    cycle = list(points)
    if cycle and cycle[0] != cycle[-1]:
        cycle.append(cycle[0])
    return cycle


@dataclass(frozen=True)
class SolverSpec:
    """Metadata describing a solver implementation."""

    name: str
    cls: Type["BaseSolver"]
    family: AlgorithmFamily
    supports_directed: bool = True


class BaseSolver:
    """Common interface for AutoTSP solvers."""

    name: str
    family: AlgorithmFamily
    supports_directed: bool = True

    def solve(self, graph: np.ndarray, time_limit: float = 5.0) -> AlgorithmResult:  # noqa: D401
        """Solve a TSP instance represented as a distance matrix."""
        raise NotImplementedError

    def __call__(self, graph: np.ndarray, time_limit: float = 5.0) -> AlgorithmResult:
        return self.solve(graph, time_limit=time_limit)


__all__ = [
    "AlgorithmResult",
    "AlgorithmFamily",
    "BaseSolver",
    "SolverSpec",
    "TimeLimitExpired",
    "best_cycle",
    "compute_cycle_cost",
    "current_time",
    "enforce_time_budget",
    "remaining_budget",
]
