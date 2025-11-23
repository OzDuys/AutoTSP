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


class AntColonySolver(BaseSolver):
    name = "ant_colony"
    family = AlgorithmFamily.METAHEURISTIC
    supports_directed = True

    def solve(
        self,
        graph: np.ndarray,
        time_limit: float = 5.0,
        num_ants: int = 20,
        evaporation: float = 0.5,
        alpha: float = 1.0,
        beta: float = 2.0,
        max_iterations: int = 300,
    ) -> AlgorithmResult:
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

        pheromone = np.ones((n, n))
        heuristic = 1.0 / (dist_matrix + np.eye(n))
        np.fill_diagonal(heuristic, 0.0)

        rng = np.random.default_rng()
        best_path = None
        best_cost = float("inf")
        iterations = 0

        try:
            for _ in range(max_iterations):
                enforce_time_budget(start_time, time_limit)
                iterations += 1
                paths = []
                costs = []

                for _ in range(num_ants):
                    enforce_time_budget(start_time, time_limit)
                    start_city = rng.integers(n)
                    unvisited = set(range(n))
                    path = [start_city]
                    unvisited.remove(start_city)
                    current = start_city
                    while unvisited:
                        weights = []
                        candidates = []
                        for city in unvisited:
                            tau = pheromone[current, city] ** alpha
                            eta = heuristic[current, city] ** beta
                            weight = tau * eta
                            weights.append(weight)
                            candidates.append(city)
                        weights = np.array(weights, dtype=float)
                        total = weights.sum()
                        if total <= 0:
                            weights = np.ones_like(weights) / len(weights)
                        else:
                            weights = weights / total
                        next_city = rng.choice(candidates, p=weights)
                        path.append(next_city)
                        unvisited.remove(next_city)
                        current = next_city
                    cycle = best_cycle(path)
                    cost = compute_cycle_cost(dist_matrix, cycle)
                    paths.append(path)
                    costs.append(cost)
                    if cost < best_cost:
                        best_cost = cost
                        best_path = cycle

                pheromone *= (1.0 - evaporation)
                for path, cost in zip(paths, costs):
                    deposit = 1.0 / max(cost, 1e-9)
                    for i in range(len(path)):
                        a = path[i]
                        b = path[(i + 1) % len(path)]
                        # For directed graphs, reinforce the forward direction only.
                        pheromone[a, b] += deposit
                if current_time() - start_time >= time_limit:
                    break
        except TimeLimitExpired:
            return AlgorithmResult(
                name=self.name,
                path=best_path,
                cost=best_cost if best_path is not None else None,
                elapsed=current_time() - start_time,
                status="timeout",
                metadata={"iterations": iterations},
            )

        status = "complete" if best_path is not None else "failed"
        return AlgorithmResult(
            name=self.name,
            path=best_path,
            cost=best_cost if best_path is not None else None,
            elapsed=current_time() - start_time,
            status=status,
            metadata={"iterations": iterations},
        )


__all__ = ["AntColonySolver"]
