from __future__ import annotations

import itertools

import numpy as np

from .base import (
    AlgorithmResult,
    TimeLimitExpired,
    best_cycle,
    compute_cycle_cost,
    current_time,
    enforce_time_budget,
)


def cutting_plane(dist_matrix: np.ndarray, time_limit: float = 5.0) -> AlgorithmResult:
    start_time = current_time()
    n = dist_matrix.shape[0]
    best_cost = float("inf")
    best_path: list[int] | None = None
    permutations_checked = 0

    try:
        for perm in itertools.permutations(range(1, n)):
            enforce_time_budget(start_time, time_limit)
            path = [0, *perm]
            cycle = best_cycle(path)
            cost = compute_cycle_cost(dist_matrix, cycle)
            permutations_checked += 1
            if cost < best_cost:
                best_cost = cost
                best_path = cycle
    except TimeLimitExpired:
        status = "timeout"
        elapsed = current_time() - start_time
        return AlgorithmResult(
            name="cutting_plane",
            path=best_path,
            cost=best_cost if best_path else None,
            elapsed=elapsed,
            status=status,
            metadata={"permutations_checked": permutations_checked},
        )

    elapsed = current_time() - start_time
    status = "complete" if best_path is not None else "failed"
    return AlgorithmResult(
        name="cutting_plane",
        path=best_path,
        cost=best_cost if best_path else None,
        elapsed=elapsed,
        status=status,
        metadata={"permutations_checked": permutations_checked},
    )
