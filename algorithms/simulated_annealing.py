from __future__ import annotations

import math

import numpy as np

from .base import (
    AlgorithmResult,
    TimeLimitExpired,
    best_cycle,
    compute_cycle_cost,
    current_time,
    enforce_time_budget,
)


def simulated_annealing(
    dist_matrix: np.ndarray,
    time_limit: float = 5.0,
    initial_temp: float = 1000.0,
    cooling_rate: float = 0.995,
    max_iter: int = 10000,
) -> AlgorithmResult:
    start_time = current_time()
    n = dist_matrix.shape[0]
    rng = np.random.default_rng()
    path = np.arange(n)
    rng.shuffle(path[1:])
    best_path = best_cycle(path.tolist())
    best_cost = compute_cycle_cost(dist_matrix, best_path)
    current_cost = best_cost
    temperature = initial_temp
    iterations = 0

    try:
        while iterations < max_iter and temperature > 1e-3:
            enforce_time_budget(start_time, time_limit)
            i, j = rng.integers(1, n, size=2)
            if i == j:
                continue
            path[i], path[j] = path[j], path[i]
            candidate_cycle = best_cycle(path.tolist())
            candidate_cost = compute_cycle_cost(dist_matrix, candidate_cycle)
            delta = candidate_cost - current_cost
            if delta < 0 or rng.random() < math.exp(-delta / max(temperature, 1e-6)):
                current_cost = candidate_cost
                if candidate_cost < best_cost:
                    best_cost = candidate_cost
                    best_path = candidate_cycle
            else:
                path[i], path[j] = path[j], path[i]
            temperature *= cooling_rate
            iterations += 1
    except TimeLimitExpired:
        return AlgorithmResult(
            name="simulated_annealing",
            path=best_path,
            cost=best_cost,
            elapsed=current_time() - start_time,
            status="timeout",
            metadata={"iterations": iterations},
        )

    return AlgorithmResult(
        name="simulated_annealing",
        path=best_path,
        cost=best_cost,
        elapsed=current_time() - start_time,
        status="complete",
        metadata={"iterations": iterations},
    )
