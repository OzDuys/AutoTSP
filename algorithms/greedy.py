from __future__ import annotations

from .base import (
    AlgorithmResult,
    TimeLimitExpired,
    best_cycle,
    compute_cycle_cost,
    current_time,
    enforce_time_budget,
)


def greedy(dist_matrix: np.ndarray, time_limit: float = 5.0) -> AlgorithmResult:
    start_time = current_time()
    n = dist_matrix.shape[0]
    best_cost = float("inf")
    best_path = None

    try:
        for start in range(n):
            enforce_time_budget(start_time, time_limit)
            path = [start]
            remaining = set(range(n)) - {start}
            while remaining:
                enforce_time_budget(start_time, time_limit)
                last = path[-1]
                next_city = min(remaining, key=lambda city: float(dist_matrix[last, city]))
                path.append(next_city)
                remaining.remove(next_city)
            cycle = best_cycle(path)
            cost = compute_cycle_cost(dist_matrix, cycle)
            if cost < best_cost:
                best_cost = cost
                best_path = cycle
    except TimeLimitExpired:
        return AlgorithmResult(
            name="greedy",
            path=best_path,
            cost=best_cost if best_path else None,
            elapsed=current_time() - start_time,
            status="timeout",
            metadata={"best_start": best_path[0] if best_path else None},
        )

    return AlgorithmResult(
        name="greedy",
        path=best_path,
        cost=best_cost if best_path else None,
        elapsed=current_time() - start_time,
        status="complete" if best_path else "failed",
        metadata={"best_start": best_path[0] if best_path else None},
    )
