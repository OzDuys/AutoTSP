from __future__ import annotations

import numpy as np

from .base import (
    AlgorithmResult,
    TimeLimitExpired,
    best_cycle,
    compute_cycle_cost,
    current_time,
    enforce_time_budget,
)


def genetic_algorithm(
    dist_matrix: np.ndarray,
    time_limit: float = 5.0,
    pop_size: int = 50,
    generations: int = 200,
    mutation_rate: float = 0.1,
    elite: int = 2,
) -> AlgorithmResult:
    start_time = current_time()
    n = dist_matrix.shape[0]
    rng = np.random.default_rng()

    def random_chromosome() -> np.ndarray:
        chromo = np.arange(n)
        rng.shuffle(chromo[1:])
        return chromo

    def crossover(parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        cut = rng.integers(1, n)
        child = -np.ones(n, dtype=int)
        child[:cut] = parent1[:cut]
        fill = [city for city in parent2 if city not in child[:cut]]
        child[cut:] = fill
        return child

    def mutate(chromo: np.ndarray) -> None:
        if rng.random() < mutation_rate:
            i, j = rng.integers(1, n, size=2)
            chromo[i], chromo[j] = chromo[j], chromo[i]

    population = [random_chromosome() for _ in range(pop_size)]
    fitness = np.empty(pop_size)
    best_path = None
    best_cost = float("inf")

    try:
        for generation in range(generations):
            enforce_time_budget(start_time, time_limit)
            for idx, chromo in enumerate(population):
                cycle = best_cycle(chromo.tolist())
                fitness[idx] = compute_cycle_cost(dist_matrix, cycle)
                if fitness[idx] < best_cost:
                    best_cost = fitness[idx]
                    best_path = cycle

            order = np.argsort(fitness)
            population = [population[i].copy() for i in order]
            fitness = fitness[order]

            next_population = population[:elite]
            while len(next_population) < pop_size:
                parents = rng.choice(len(population[: pop_size // 2]), size=2, replace=False)
                child = crossover(population[parents[0]], population[parents[1]])
                mutate(child)
                next_population.append(child)
            population = next_population
    except TimeLimitExpired:
        return AlgorithmResult(
            name="genetic_algorithm",
            path=best_path,
            cost=best_cost if best_path is not None else None,
            elapsed=current_time() - start_time,
            status="timeout",
            metadata={"population_size": pop_size, "generations_completed": generation if 'generation' in locals() else 0},
        )

    return AlgorithmResult(
        name="genetic_algorithm",
        path=best_path,
        cost=best_cost if best_path is not None else None,
        elapsed=current_time() - start_time,
        status="complete",
        metadata={"population_size": pop_size, "generations_completed": generations},
    )
