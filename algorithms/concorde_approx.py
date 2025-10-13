from __future__ import annotations

import numpy as np

from .base import (
    AlgorithmResult,
    TimeLimitExpired,
    best_cycle,
    compute_cycle_cost,
    current_time,
    enforce_time_budget,
)


def concorde_approx(dist_matrix: np.ndarray, time_limit: float = 5.0) -> AlgorithmResult:
    start_time = current_time()
    n = dist_matrix.shape[0]
    visited = [0]

    try:
        for _ in range(1, n):
            enforce_time_budget(start_time, time_limit)
            last = visited[-1]
            candidates = [
                (float(dist_matrix[last, city]), city)
                for city in range(n)
                if city not in visited
            ]
            if not candidates:
                break
            _, next_city = min(candidates)
            visited.append(next_city)
    except TimeLimitExpired:
        return AlgorithmResult(
            name="concorde_approx",
            path=best_cycle(visited),
            cost=compute_cycle_cost(dist_matrix, best_cycle(visited)),
            elapsed=current_time() - start_time,
            status="timeout",
            metadata={"nodes_visited": len(visited)},
        )

    cycle = best_cycle(visited)
    return AlgorithmResult(
        name="concorde_approx",
        path=cycle,
        cost=compute_cycle_cost(dist_matrix, cycle),
        elapsed=current_time() - start_time,
        status="complete",
        metadata={"nodes_visited": len(visited)},
    )
