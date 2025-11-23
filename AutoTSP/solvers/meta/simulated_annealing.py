from __future__ import annotations

import math

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


class SimulatedAnnealingSolver(BaseSolver):
    name = "simulated_annealing"
    family = AlgorithmFamily.METAHEURISTIC
    supports_directed = True

    def solve(
        self,
        graph: np.ndarray,
        time_limit: float = 5.0,
        initial_temp_factor: float = 0.1,
    ) -> AlgorithmResult:
        dist_matrix = np.asarray(graph, dtype=float)
        start_time = current_time()
        n = dist_matrix.shape[0]
        rng = np.random.default_rng()

        # Construct a greedy nearest-neighbor tour for a decent starting point.
        def greedy_start() -> list[int]:
            if n == 0:
                return []
            unvisited = set(range(n))
            path_local = [0]
            unvisited.remove(0)
            while unvisited:
                last = path_local[-1]
                next_city = min(unvisited, key=lambda c: dist_matrix[last, c])
                path_local.append(next_city)
                unvisited.remove(next_city)
            return path_local

        path = greedy_start()
        if len(path) < n:
            # Fallback to random permutation if greedy failed for any reason.
            path = list(np.arange(n))
            rng.shuffle(path[1:])

        best_path = best_cycle(path)
        current_cost = best_cost = compute_cycle_cost(dist_matrix, best_path)

        # Scale temperature to the current tour cost; lower alpha for large n.
        init_temp = max(initial_temp_factor * current_cost / max(1, n), 1e-6)
        temperature = init_temp

        # Adaptive iteration budget relative to size to stay within time limit.
        max_moves = int(max(5000, min(100000, 5_000_000 // max(1, n))))
        target_cool = 0.01
        decay = math.exp(math.log(target_cool) / max_moves)

        iterations = 0
        accepted = 0
        improved = 0
        symmetric = np.allclose(dist_matrix, dist_matrix.T, atol=1e-9)

        try:
            while iterations < max_moves and temperature > 1e-6:
                enforce_time_budget(start_time, time_limit)
                i, j = sorted(rng.integers(1, n, size=2))
                if i == j:
                    continue
                if symmetric:
                    a = path[i - 1]
                    b = path[i]
                    c = path[j]
                    d = path[(j + 1) % n]
                    delta = (dist_matrix[a, c] + dist_matrix[b, d]) - (dist_matrix[a, b] + dist_matrix[c, d])
                    candidate_cost = current_cost + delta
                else:
                    candidate = path[:]
                    candidate[i : j + 1] = reversed(candidate[i : j + 1])
                    candidate_cost = compute_cycle_cost(dist_matrix, best_cycle(candidate))
                    delta = candidate_cost - current_cost

                accept = False
                if delta < 0:
                    accept = True
                    improved += 1
                else:
                    prob = math.exp(-delta / max(temperature, 1e-9))
                    accept = rng.random() < prob

                if accept:
                    accepted += 1
                    if symmetric:
                        path[i:j + 1] = reversed(path[i:j + 1])
                        current_cost = candidate_cost
                    else:
                        path = candidate
                        current_cost = candidate_cost
                    if current_cost + 1e-9 < best_cost:
                        best_cost = current_cost
                        best_path = best_cycle(path)

                temperature *= decay
                iterations += 1
        except TimeLimitExpired:
            return AlgorithmResult(
                name=self.name,
                path=best_path,
                cost=best_cost,
                elapsed=current_time() - start_time,
                status="timeout",
                metadata={"iterations": iterations, "accepted": accepted, "improved": improved},
            )

        return AlgorithmResult(
            name=self.name,
            path=best_path,
            cost=best_cost,
            elapsed=current_time() - start_time,
            status="complete",
            metadata={
                "iterations": iterations,
                "accepted": accepted,
                "improved": improved,
                "initial_temp": init_temp,
            },
        )


__all__ = ["SimulatedAnnealingSolver"]
