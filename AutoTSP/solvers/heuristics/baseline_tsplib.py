from __future__ import annotations

"""
Baseline spatial heuristic for Euclidean-style TSP instances.
Adapts the provided TSPLIB baseline to the project's algorithm interface.
"""

from typing import Callable, Dict, Tuple

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

morton_bits = 16


def _part1by1_uint64(a: np.ndarray) -> np.ndarray:
    """Interleave lower 32 bits of a with zeros (vectorized)."""
    a = a.astype(np.uint64)
    a = (a | (a << np.uint64(16))) & np.uint64(0x0000FFFF0000FFFF)
    a = (a | (a << np.uint64(8))) & np.uint64(0x00FF00FF00FF00FF)
    a = (a | (a << np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    a = (a | (a << np.uint64(2))) & np.uint64(0x3333333333333333)
    a = (a | (a << np.uint64(1))) & np.uint64(0x5555555555555555)
    return a


def _morton_indices(coords: np.ndarray, bits: int) -> np.ndarray:
    """Map 2D points to Morton (Z-order) indices using `bits` bits per coordinate."""
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    span = np.where(maxs > mins, maxs - mins, 1.0)
    grid_max = (1 << bits) - 1
    scaled = ((coords - mins) / span * grid_max).astype(np.uint64)
    x = scaled[:, 0]
    y = scaled[:, 1]
    px = _part1by1_uint64(x)
    py = _part1by1_uint64(y)
    morton = px | (py << np.uint64(1))
    return morton


def _serpentine_order(coords: np.ndarray, bins: int | None = None) -> np.ndarray:
    """Quantize Y into bins and alternate X direction per bin (boustrophedon)."""
    n = coords.shape[0]
    xs = coords[:, 0]
    ys = coords[:, 1]
    if bins is None:
        bins = int(np.clip(np.sqrt(float(n)) / 2.0, 64, 2048))
    y_min = ys.min()
    y_range = ys.max() - y_min
    if y_range <= 0.0:
        y_range = 1.0
    y_rel = (ys - y_min) / y_range
    y_bin = np.floor(y_rel * (bins - 1)).astype(np.int64, copy=False)
    x_key = np.where((y_bin & 1) == 0, xs, -xs)
    order = np.lexsort((x_key, y_bin))
    return order.astype(np.int64, copy=False)


def _approx_cycle_length(coords: np.ndarray, order: np.ndarray) -> float:
    """Fast approximate Euclidean cycle length for a given order (vectorized)."""
    p = coords[order]
    q = np.roll(p, -1, axis=0)
    d = p - q
    return float(np.hypot(d[:, 0], d[:, 1]).sum())


def _local_2opt(
    coords: np.ndarray,
    tour: np.ndarray,
    start_time: float,
    time_limit: float,
    window: int = 20,
    max_passes: int = 2,
) -> None:
    """
    Improved cheap local 2-opt with a small distance cache and early stopping.
    Operates in-place on `tour`.
    """
    n = tour.shape[0]
    if n < 4:
        return
    xs = coords[:, 0]
    ys = coords[:, 1]

    def make_dist() -> Tuple[Callable[[int, int], float], Dict[tuple[int, int], float]]:
        cache: Dict[tuple[int, int], float] = {}

        def dist(a_idx: int, b_idx: int) -> float:
            if a_idx == b_idx:
                return 0.0
            key = (a_idx, b_idx) if a_idx <= b_idx else (b_idx, a_idx)
            v = cache.get(key)
            if v is None:
                dx = xs[a_idx] - xs[b_idx]
                dy = ys[a_idx] - ys[b_idx]
                v = (dx * dx + dy * dy) ** 0.5
                cache[key] = v
            return v

        return dist, cache

    cur_window = window
    for _ in range(max_passes):
        enforce_time_budget(start_time, time_limit)
        dist, cache = make_dist()
        cache.clear()
        swapped_any = False
        for i in range(n - 1):
            enforce_time_budget(start_time, time_limit)
            a = int(tour[i])
            b = int(tour[i + 1])
            ab = dist(a, b)
            j_max = min(n - 1, i + cur_window)
            j = i + 2
            while j <= j_max:
                enforce_time_budget(start_time, time_limit)
                c = int(tour[j])
                d = int(tour[j + 1]) if j + 1 < n else int(tour[0])
                cd = dist(c, d)
                ac = dist(a, c)
                bd = dist(b, d)
                if ac + bd + 1e-12 < ab + cd:
                    tour[i + 1 : j + 1] = tour[i + 1 : j + 1][::-1]
                    swapped_any = True
                    b = int(tour[i + 1])
                    ab = dist(a, b)
                j += 1
        if not swapped_any:
            break
        cur_window = min(200, int(cur_window * 1.5) + 1)


def _tour(coords: np.ndarray, start_time: float, time_limit: float) -> Tuple[np.ndarray, int]:
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("coords must be an (n, 2) array")
    n = coords.shape[0]
    enforce_time_budget(start_time, time_limit)

    candidates: list[np.ndarray] = []
    try:
        candidates.append(np.argsort(_morton_indices(coords, morton_bits), kind="mergesort").astype(np.int64, copy=False))
        candidates.append(
            np.argsort(_morton_indices(coords[:, [1, 0]], morton_bits), kind="mergesort").astype(np.int64, copy=False)
        )
        theta = np.deg2rad(30.0)
        c, s = np.cos(theta), np.sin(theta)
        rx = coords[:, 0] * c - coords[:, 1] * s
        ry = coords[:, 0] * s + coords[:, 1] * c
        rc = np.column_stack((rx, ry))
        candidates.append(np.argsort(_morton_indices(rc, morton_bits), kind="mergesort").astype(np.int64, copy=False))
    except Exception:
        pass
    candidates.append(_serpentine_order(coords))
    candidates.append(np.lexsort((coords[:, 1], coords[:, 0])).astype(np.int64, copy=False))
    candidates.append(np.lexsort((coords[:, 0], coords[:, 1])).astype(np.int64, copy=False))

    best_order = candidates[0]
    best_len = _approx_cycle_length(coords, best_order)
    for cand in candidates[1:]:
        enforce_time_budget(start_time, time_limit)
        cand_len = _approx_cycle_length(coords, cand)
        if cand_len < best_len:
            best_len = cand_len
            best_order = cand

    base_window = int(max(5, min(200, 500000 // max(1, n))))
    _local_2opt(coords, best_order, start_time=start_time, time_limit=time_limit, window=base_window, max_passes=2)
    return best_order, len(candidates)


def _recover_coordinates(dist_matrix: np.ndarray) -> np.ndarray:
    """Approximate 2D coordinates from a distance matrix via classical MDS."""
    n = dist_matrix.shape[0]
    if n == 0:
        return np.empty((0, 2), dtype=float)
    if n == 1:
        return np.zeros((1, 2), dtype=float)

    dist = np.asarray(dist_matrix, dtype=float)
    dist = np.maximum(dist, 0.0)
    dist = (dist + dist.T) * 0.5

    dist_sq = dist**2
    np.fill_diagonal(dist_sq, 0.0)
    J = np.eye(n) - np.full((n, n), 1.0 / n)
    B = -0.5 * J @ dist_sq @ J
    try:
        eigvals, eigvecs = np.linalg.eigh(B)
    except np.linalg.LinAlgError:
        indices = np.argsort(dist.mean(axis=1))
        coords_1d = np.linspace(0.0, 1.0, n)
        coords = np.column_stack((coords_1d[indices], np.zeros(n)))
        return coords[indices.argsort()]

    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    positive = eigvals > 1e-9
    dims = int(min(2, np.sum(positive)))
    if dims == 0:
        dims = 1
        positive[:1] = True

    lambdas = np.sqrt(np.clip(eigvals[:dims], 0.0, None))
    coords = eigvecs[:, :dims] * lambdas
    if dims < 2:
        coords = np.pad(coords, ((0, 0), (0, 2 - dims)), mode="constant")
    return coords.astype(float, copy=False)


class BaselineTSPLibSolver(BaseSolver):
    name = "baseline_tsplib"
    family = AlgorithmFamily.HEURISTIC
    supports_directed = False

    def solve(self, graph: np.ndarray, time_limit: float = 5.0) -> AlgorithmResult:
        dist_matrix = np.asarray(graph, dtype=float)
        start_time = current_time()
        n = dist_matrix.shape[0]
        identity_cycle = best_cycle(list(range(n)))

        try:
            enforce_time_budget(start_time, time_limit)
            coords = _recover_coordinates(dist_matrix)
            order, candidate_count = _tour(coords, start_time=start_time, time_limit=time_limit)
            path = best_cycle(order.tolist())
            cost = compute_cycle_cost(dist_matrix, path)
            elapsed = current_time() - start_time
            metadata = {
                "embedding_dim": 2,
                "num_candidates": candidate_count,
            }
            return AlgorithmResult(
                name=self.name,
                path=path,
                cost=cost,
                elapsed=elapsed,
                status="complete",
                metadata=metadata,
            )
        except TimeLimitExpired:
            elapsed = current_time() - start_time
            return AlgorithmResult(
                name=self.name,
                path=identity_cycle,
                cost=compute_cycle_cost(dist_matrix, identity_cycle),
                elapsed=elapsed,
                status="timeout",
                metadata={"fallback": "identity_cycle"},
            )
        except Exception as exc:  # noqa: BLE001
            elapsed = current_time() - start_time
            return AlgorithmResult(
                name=self.name,
                path=identity_cycle,
                cost=compute_cycle_cost(dist_matrix, identity_cycle),
                elapsed=elapsed,
                status="failed",
                metadata={"error": str(exc)},
            )


__all__ = ["BaselineTSPLibSolver"]
