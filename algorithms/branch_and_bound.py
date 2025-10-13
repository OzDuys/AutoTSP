from __future__ import annotations

from typing import List

import numpy as np

from .base import (
    AlgorithmResult,
    TimeLimitExpired,
    best_cycle,
    compute_cycle_cost,
    current_time,
    enforce_time_budget,
)


def branch_and_bound(dist_matrix: np.ndarray, time_limit: float = 5.0) -> AlgorithmResult:
    start_time = current_time()
    n = dist_matrix.shape[0]
    best_cost = float("inf")
    best_path: List[int] | None = None
    nodes_explored = 0

    def bound(path: List[int]) -> float:
        cost = 0.0
        for i in range(len(path) - 1):
            cost += float(dist_matrix[path[i], path[i + 1]])
        remaining = [city for city in range(n) if city not in path]
        if remaining:
            cost += min(float(dist_matrix[path[-1], j]) for j in remaining)
            if len(remaining) > 1:
                for city in remaining:
                    others = [o for o in remaining if o != city]
                    if others:
                        cost += min(float(dist_matrix[city, o]) for o in others) / 2.0
        return cost

    def dfs(path: List[int], cost_so_far: float) -> None:
        nonlocal best_cost, best_path, nodes_explored
        enforce_time_budget(start_time, time_limit)
        nodes_explored += 1

        if len(path) == n:
            total_cost = cost_so_far + float(dist_matrix[path[-1], path[0]])
            if total_cost < best_cost:
                best_cost = total_cost
                best_path = best_cycle(path)
            return

        remaining = [city for city in range(n) if city not in path]
        for next_city in remaining:
            new_cost = cost_so_far + float(dist_matrix[path[-1], next_city])
            if new_cost >= best_cost:
                continue
            prospective = path + [next_city]
            if bound(prospective) >= best_cost:
                continue
            dfs(prospective, new_cost)

    status = "complete"
    try:
        dfs([0], 0.0)
    except TimeLimitExpired:
        status = "timeout"

    elapsed = current_time() - start_time
    cost = best_cost if best_path else None
    return AlgorithmResult(
        name="branch_and_bound",
        path=best_path,
        cost=cost,
        elapsed=elapsed,
        status=status,
        metadata={"nodes_explored": nodes_explored},
    )
