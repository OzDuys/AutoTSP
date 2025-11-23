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


class TwoOptSolver(BaseSolver):
    name = "two_opt"
    family = AlgorithmFamily.HEURISTIC
    supports_directed = False

    def solve(self, graph: np.ndarray, time_limit: float = 5.0, max_iterations: int = 1000) -> AlgorithmResult:
        dist_matrix = np.asarray(graph, dtype=float)
        start_time = current_time()
        n = dist_matrix.shape[0]
        if n <= 2:
            cycle = best_cycle(list(range(n)))
            cost = compute_cycle_cost(dist_matrix, cycle)
            return AlgorithmResult(
                name=self.name,
                path=cycle,
                cost=cost,
                elapsed=current_time() - start_time,
                status="complete",
                metadata={"iterations": 0},
            )

        path = list(range(n))
        best_cycle_path = best_cycle(path)
        best_cost = compute_cycle_cost(dist_matrix, best_cycle_path)
        iterations = 0

        try:
            while iterations < max_iterations:
                improved = False
                iterations += 1
                for i in range(1, n - 2):
                    for j in range(i + 1, n):
                        enforce_time_budget(start_time, time_limit)
                        if j - i == 1:
                            continue
                        new_path = path[:]
                        new_path[i:j] = reversed(path[i:j])
                        new_cycle = best_cycle(new_path)
                        new_cost = compute_cycle_cost(dist_matrix, new_cycle)
                        if new_cost + 1e-9 < best_cost:
                            path = new_path
                            best_cost = new_cost
                            best_cycle_path = new_cycle
                            improved = True
                if not improved:
                    break
        except TimeLimitExpired:
            return AlgorithmResult(
                name=self.name,
                path=best_cycle_path,
                cost=best_cost,
                elapsed=current_time() - start_time,
                status="timeout",
                metadata={"iterations": iterations},
            )

        return AlgorithmResult(
            name=self.name,
            path=best_cycle_path,
            cost=best_cost,
            elapsed=current_time() - start_time,
            status="complete",
            metadata={"iterations": iterations},
        )


__all__ = ["TwoOptSolver"]
