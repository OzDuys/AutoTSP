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


class IteratedLocalSearchSolver(BaseSolver):
    name = "iterated_local_search"
    family = AlgorithmFamily.METAHEURISTIC
    supports_directed = False

    def solve(
        self,
        graph: np.ndarray,
        time_limit: float = 5.0,
        max_iterations: int = 50,
        perturbation_size: int = 3,
    ) -> AlgorithmResult:
        dist_matrix = np.asarray(graph, dtype=float)
        start_time = current_time()
        n = dist_matrix.shape[0]
        rng = np.random.default_rng(0)

        def two_opt_local(path: list[int]) -> tuple[list[int], float]:
            improved = True
            best_path_local = path
            best_cost_local = compute_cycle_cost(dist_matrix, best_cycle(path))
            while improved:
                improved = False
                for i in range(1, n - 2):
                    for j in range(i + 1, n):
                        if j - i == 1:
                            continue
                        enforce_time_budget(start_time, time_limit)
                        candidate = best_path_local[:]
                        candidate[i:j] = reversed(candidate[i:j])
                        cycle = best_cycle(candidate)
                        cost = compute_cycle_cost(dist_matrix, cycle)
                        if cost + 1e-9 < best_cost_local:
                            best_cost_local = cost
                            best_path_local = candidate
                            improved = True
                            break
                    if improved:
                        break
            return best_path_local, best_cost_local

        # Initial solution: simple nearest neighbor path
        path = list(range(n))
        best_path, best_cost = two_opt_local(path)
        iterations = 0

        try:
            while iterations < max_iterations:
                iterations += 1
                enforce_time_budget(start_time, time_limit)
                candidate = self._perturb(best_path, rng, perturbation_size)
                candidate, candidate_cost = two_opt_local(candidate)
                if candidate_cost + 1e-9 < best_cost:
                    best_cost = candidate_cost
                    best_path = candidate
        except TimeLimitExpired:
            return AlgorithmResult(
                name=self.name,
                path=best_cycle(best_path),
                cost=best_cost,
                elapsed=current_time() - start_time,
                status="timeout",
                metadata={"iterations": iterations},
            )

        return AlgorithmResult(
            name=self.name,
            path=best_cycle(best_path),
            cost=best_cost,
            elapsed=current_time() - start_time,
            status="complete",
            metadata={"iterations": iterations},
        )

    def _perturb(self, path: list[int], rng: np.random.Generator, size: int) -> list[int]:
        if len(path) <= size:
            return path[:]
        idx = sorted(rng.choice(len(path), size=size, replace=False))
        new_path = path[:]
        segment = new_path[idx[0] : idx[-1] + 1]
        rng.shuffle(segment)
        new_path[idx[0] : idx[-1] + 1] = segment
        return new_path


__all__ = ["IteratedLocalSearchSolver"]
