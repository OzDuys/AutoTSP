from __future__ import annotations

import itertools

import numpy as np

from AutoTSP.solvers.base import (
    AlgorithmResult,
    BaseSolver,
    TimeLimitExpired,
    best_cycle,
    current_time,
    enforce_time_budget,
)
from AutoTSP.utils.taxonomy import AlgorithmFamily


class HeldKarpSolver(BaseSolver):
    name = "held_karp"
    family = AlgorithmFamily.EXACT
    supports_directed = True

    def solve(self, graph: np.ndarray, time_limit: float = 5.0) -> AlgorithmResult:
        dist_matrix = np.asarray(graph, dtype=float)
        start_time = current_time()
        n = dist_matrix.shape[0]
        all_indices = range(n)
        dp: dict[tuple[int, int], tuple[float, int]] = {}

        for k in all_indices:
            if k == 0:
                continue
            dp[(1 << k) | 1, k] = (float(dist_matrix[0, k]), 0)

        try:
            for subset_size in range(3, n + 1):
                for subset in itertools.combinations(all_indices[1:], subset_size - 1):
                    mask = 1
                    for city in subset:
                        mask |= 1 << city
                    for k in subset:
                        enforce_time_budget(start_time, time_limit)
                        prev_mask = mask & ~(1 << k)
                        best = (float("inf"), -1)
                        for m in subset:
                            if m == k:
                                continue
                            prev = dp.get((prev_mask, m))
                            if prev is None:
                                continue
                            cost = prev[0] + float(dist_matrix[m, k])
                            if cost < best[0]:
                                best = (cost, m)
                        dp[(mask, k)] = best

            best_cost = float("inf")
            best_last = -1
            full_mask = (1 << n) - 1
            for k in all_indices:
                if k == 0:
                    continue
                enforce_time_budget(start_time, time_limit)
                prev = dp.get((full_mask, k))
                if prev is None:
                    continue
                cost = prev[0] + float(dist_matrix[k, 0])
                if cost < best_cost:
                    best_cost = cost
                    best_last = k

            if not np.isfinite(best_cost):
                return AlgorithmResult(
                    name=self.name,
                    path=None,
                    cost=None,
                    elapsed=current_time() - start_time,
                    status="failed",
                    metadata={},
                )

            path = [0]
            mask = full_mask
            last = best_last
            while last != 0:
                path.append(last)
                _, prev = dp[(mask, last)]
                mask &= ~(1 << last)
                last = prev
            path = list(reversed(path))
            cycle = best_cycle(path)
            elapsed = current_time() - start_time
            return AlgorithmResult(
                name=self.name,
                path=cycle,
                cost=best_cost,
                elapsed=elapsed,
                status="complete",
                metadata={},
            )
        except TimeLimitExpired:
            return AlgorithmResult(
                name=self.name,
                path=None,
                cost=None,
                elapsed=current_time() - start_time,
                status="timeout",
                metadata={},
            )


__all__ = ["HeldKarpSolver"]
