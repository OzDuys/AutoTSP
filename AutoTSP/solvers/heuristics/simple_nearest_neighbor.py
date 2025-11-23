from __future__ import annotations

import numpy as np

from AutoTSP.solvers.base import (
    AlgorithmResult,
    BaseSolver,
    TimeLimitExpired,
    best_cycle,
    compute_cycle_cost,
    current_time,
    enforce_time_budget,
)
from AutoTSP.utils.taxonomy import AlgorithmFamily


class SimpleNearestNeighborSolver(BaseSolver):
    name = "simple_nearest_neighbor"
    family = AlgorithmFamily.HEURISTIC
    supports_directed = True

    def solve(self, graph: np.ndarray, time_limit: float = 5.0) -> AlgorithmResult:
        dist_matrix = np.asarray(graph, dtype=float)
        start_time = current_time()
        n = dist_matrix.shape[0]
        visited = [0]

        try:
            for _ in range(1, n):
                enforce_time_budget(start_time, time_limit)
                last = visited[-1]
                candidates = [(float(dist_matrix[last, city]), city) for city in range(n) if city not in visited]
                if not candidates:
                    break
                _, next_city = min(candidates)
                visited.append(next_city)
        except TimeLimitExpired:
            cycle = best_cycle(visited)
            return AlgorithmResult(
                name=self.name,
                path=cycle,
                cost=compute_cycle_cost(dist_matrix, cycle),
                elapsed=current_time() - start_time,
                status="timeout",
                metadata={"nodes_visited": len(visited)},
            )

        cycle = best_cycle(visited)
        return AlgorithmResult(
            name=self.name,
            path=cycle,
            cost=compute_cycle_cost(dist_matrix, cycle),
            elapsed=current_time() - start_time,
            status="complete",
            metadata={"nodes_visited": len(visited)},
        )


__all__ = ["SimpleNearestNeighborSolver"]
