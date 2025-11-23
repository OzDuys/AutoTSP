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


class ThreeOptSolver(BaseSolver):
    name = "three_opt"
    family = AlgorithmFamily.HEURISTIC
    supports_directed = False

    def solve(self, graph: np.ndarray, time_limit: float = 5.0, max_iterations: int = 200) -> AlgorithmResult:
        dist_matrix = np.asarray(graph, dtype=float)
        start_time = current_time()
        n = dist_matrix.shape[0]

        if n <= 3:
            cycle = best_cycle(list(range(n)))
            return AlgorithmResult(
                name=self.name,
                path=cycle,
                cost=compute_cycle_cost(dist_matrix, cycle),
                elapsed=current_time() - start_time,
                status="complete",
                metadata={"iterations": 0},
            )

        path = list(range(n))
        best_path = best_cycle(path)
        best_cost = compute_cycle_cost(dist_matrix, best_path)
        iterations = 0

        try:
            improved = True
            while improved and iterations < max_iterations:
                improved = False
                iterations += 1
                for i in range(n - 2):
                    for j in range(i + 1, n - 1):
                        for k in range(j + 1, n):
                            enforce_time_budget(start_time, time_limit)
                            new_paths = self._three_opt_variations(path, i, j, k)
                            for candidate in new_paths:
                                cycle = best_cycle(candidate)
                                cost = compute_cycle_cost(dist_matrix, cycle)
                                if cost + 1e-9 < best_cost:
                                    best_cost = cost
                                    best_path = cycle
                                    path = candidate
                                    improved = True
                                    break
                            if improved:
                                break
                        if improved:
                            break
                    if improved:
                        break
        except TimeLimitExpired:
            return AlgorithmResult(
                name=self.name,
                path=best_path,
                cost=best_cost,
                elapsed=current_time() - start_time,
                status="timeout",
                metadata={"iterations": iterations},
            )

        return AlgorithmResult(
            name=self.name,
            path=best_path,
            cost=best_cost,
            elapsed=current_time() - start_time,
            status="complete",
            metadata={"iterations": iterations},
        )

    def _three_opt_variations(self, path: list[int], i: int, j: int, k: int) -> list[list[int]]:
        a, b = path[i:j], path[j:k]
        c = path[k:]
        return [
            path[:i] + a[::-1] + b + c,  # reverse first segment
            path[:i] + a + b[::-1] + c,  # reverse second
            path[:i] + b + a + c,  # swap first two
            path[:i] + b[::-1] + a + c,  # swap+reverse
            path[:i] + c + b + a,  # rotate
        ]


__all__ = ["ThreeOptSolver"]
