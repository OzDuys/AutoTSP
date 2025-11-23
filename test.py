import time
import math
import random
from dataclasses import dataclass
from typing import List, Callable, Tuple, Dict

import numpy as np


# ---------------------------
# TSP instance + utilities
# ---------------------------

@dataclass
class TSPInstance:
    coords: np.ndarray  # shape (n, 2)
    dist_matrix: np.ndarray  # shape (n, n)


def generate_euclidean_tsp(n_cities: int, seed: int = None) -> TSPInstance:
    """Generate a random Euclidean TSP instance with cities in [0,1] x [0,1]."""
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    coords = rng.random((n_cities, 2))
    diff = coords[:, None, :] - coords[None, :, :]
    dist_matrix = np.linalg.norm(diff, axis=2)
    return TSPInstance(coords=coords, dist_matrix=dist_matrix)


def tour_length(tour: List[int], dist_matrix: np.ndarray) -> float:
    """Compute the total length of a tour (closed: last to first)."""
    n = len(tour)
    total = 0.0
    for i in range(n):
        total += dist_matrix[tour[i], tour[(i + 1) % n]]
    return total


# ---------------------------
# TSP solvers
# ---------------------------

def nearest_neighbor_solver(instance: TSPInstance, start: int = 0) -> List[int]:
    """Simple nearest neighbor heuristic for TSP."""
    n = instance.dist_matrix.shape[0]
    if start < 0 or start >= n:
        start = 0

    unvisited = set(range(n))
    tour = [start]
    unvisited.remove(start)
    current = start

    dist_matrix = instance.dist_matrix

    while unvisited:
        # pick the closest unvisited city
        next_city = min(unvisited, key=lambda j: dist_matrix[current, j])
        tour.append(next_city)
        unvisited.remove(next_city)
        current = next_city

    return tour


def two_opt_solver(instance: TSPInstance,
                   initial_tour: List[int],
                   max_iterations: int = 10_000,
                   improvement_threshold: float = 1e-9) -> List[int]:
    """
    2-opt local search starting from an initial tour.
    For large n, it can be expensive, so we cap iterations.
    """
    tour = initial_tour[:]
    dist = instance.dist_matrix
    n = len(tour)

    def delta_2opt(i: int, k: int) -> float:
        """Change in tour length if we reverse tour[i:k+1]."""
        a, b = tour[i - 1], tour[i]
        c, d = tour[k], tour[(k + 1) % n]
        before = dist[a, b] + dist[c, d]
        after = dist[a, c] + dist[b, d]
        return after - before

    iteration = 0
    improved = True
    while improved and iteration < max_iterations:
        improved = False
        iteration += 1

        # simple nested loop; you can tweak ranges for speed
        for i in range(1, n - 2):
            for k in range(i + 1, n - 1):
                change = delta_2opt(i, k)
                if change < -improvement_threshold:
                    # apply 2-opt move: reverse segment [i, k]
                    tour[i:k + 1] = reversed(tour[i:k + 1])
                    improved = True
                    break  # restart search from beginning
            if improved:
                break

    return tour


# ---------------------------
# Benchmarking framework
# ---------------------------

SolverFunc = Callable[[TSPInstance], List[int]]


def make_nn_solver() -> SolverFunc:
    def solver(instance: TSPInstance) -> List[int]:
        return nearest_neighbor_solver(instance, start=0)
    return solver


def make_two_opt_solver(max_iterations: int = 10_000) -> SolverFunc:
    def solver(instance: TSPInstance) -> List[int]:
        initial = nearest_neighbor_solver(instance, start=0)
        return two_opt_solver(instance, initial, max_iterations=max_iterations)
    return solver


@dataclass
class BenchmarkResult:
    n_cities: int
    solver_name: str
    times: List[float]
    lengths: List[float]

    @property
    def avg_time(self) -> float:
        return sum(self.times) / len(self.times)

    @property
    def best_length(self) -> float:
        return min(self.lengths)

    @property
    def avg_length(self) -> float:
        return sum(self.lengths) / len(self.lengths)


def benchmark_solvers(
    sizes: List[int],
    solvers: Dict[str, SolverFunc],
    n_repeats: int = 3,
    seed: int = 42
) -> List[BenchmarkResult]:
    results: List[BenchmarkResult] = []
    rng = random.Random(seed)

    for n in sizes:
        print(f"\n=== Benchmark: n_cities = {n} ===")
        for solver_name, solver in solvers.items():
            times = []
            lengths = []

            for r in range(n_repeats):
                inst_seed = rng.randint(0, 10**9)
                instance = generate_euclidean_tsp(n, seed=inst_seed)

                start_t = time.perf_counter()
                tour = solver(instance)
                end_t = time.perf_counter()

                t = end_t - start_t
                L = tour_length(tour, instance.dist_matrix)

                times.append(t)
                lengths.append(L)
                print(f"  [{solver_name}] repeat {r+1}/{n_repeats}: "
                      f"time = {t:.4f}s, length = {L:.4f}")

            result = BenchmarkResult(
                n_cities=n,
                solver_name=solver_name,
                times=times,
                lengths=lengths,
            )
            results.append(result)

    return results


def print_summary_table(results: List[BenchmarkResult]) -> None:
    print("\n================ SUMMARY ================\n")
    header = f"{'n_cities':>8} | {'solver':>10} | {'avg_time (s)':>12} | {'best_len':>10} | {'avg_len':>10}"
    print(header)
    print("-" * len(header))
    for res in results:
        print(
            f"{res.n_cities:8d} | "
            f"{res.solver_name:>10} | "
            f"{res.avg_time:12.4f} | "
            f"{res.best_length:10.4f} | "
            f"{res.avg_length:10.4f}"
        )


if __name__ == "__main__":
    # You can tweak these values depending on how heavy you want the benchmark to be.
    sizes = [100, 300, 600]   # "large" synthetic TSP instances
    n_repeats = 3             # run each solver 3 times per size

    solvers = {
        "NN": make_nn_solver(),
        # For very large n, you might want to lower max_iterations to keep runtime reasonable
        "2-opt": make_two_opt_solver(max_iterations=5000),
    }

    results = benchmark_solvers(sizes=sizes, solvers=solvers, n_repeats=n_repeats)
    print_summary_table(results)
