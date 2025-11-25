from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np


@dataclass(frozen=True)
class FeatureVector:
    values: Dict[str, float | int | bool | None]
    elapsed: float


class FeatureExtractor:
    """Lightweight, fast feature extraction for TSP instances."""

    @staticmethod
    def extract(problem_data: dict[str, Any], rng: Optional[np.random.Generator] = None) -> FeatureVector:
        if rng is None:
            rng = np.random.default_rng(42)

        metric_hint = (problem_data.get("metric") or "").lower()
        coords = FeatureExtractor._get_coordinates(problem_data)
        dist_matrix = FeatureExtractor._get_distance_matrix(problem_data)
        n_nodes = FeatureExtractor._infer_n_nodes(coords, dist_matrix)

        start = FeatureExtractor._now()
        features: Dict[str, float | int | bool | None] = {}
        features["n_nodes"] = n_nodes
        features["is_metric"] = FeatureExtractor._is_metric(metric_hint, coords, dist_matrix, rng=rng)

        if coords is not None and len(coords) > 0:
            # Raw geometric descriptors (scale-dependent).
            stds = np.std(coords, axis=0)
            features["std_dev_x"] = float(stds[0])
            features["std_dev_y"] = float(stds[1])
            span = coords.max(axis=0) - coords.min(axis=0)
            bbox_area = float(span[0] * span[1])
            features["bbox_area"] = bbox_area
            centroid = coords.mean(axis=0)
            dispersion = float(np.linalg.norm(coords - centroid, axis=1).mean())
            features["centroid_dispersion"] = dispersion
            features["landmark_10_dist"] = FeatureExtractor._landmark_distance(coords, metric_hint, rng)
            features["nn_probe_cost"] = FeatureExtractor._nn_probe_cost(coords, metric_hint)
            # Scale-normalised descriptors (invariant to uniform rescaling).
            norm_coords = FeatureExtractor._normalise_coords(coords)
            if norm_coords is not None:
                norm_stds = np.std(norm_coords, axis=0)
                features["std_dev_x_norm"] = float(norm_stds[0])
                features["std_dev_y_norm"] = float(norm_stds[1])
                norm_span = norm_coords.max(axis=0) - norm_coords.min(axis=0)
                features["bbox_area_norm"] = float(norm_span[0] * norm_span[1])
                norm_centroid = norm_coords.mean(axis=0)
                features["centroid_dispersion_norm"] = float(
                    np.linalg.norm(norm_coords - norm_centroid, axis=1).mean()
                )
                features["landmark_10_dist_norm"] = FeatureExtractor._landmark_distance(norm_coords, metric_hint, rng)
                features["nn_probe_cost_norm"] = FeatureExtractor._nn_probe_cost(norm_coords, metric_hint)
        elif dist_matrix is not None:
            # Fall back to distance-matrix-only approximations.
            features["std_dev_x"] = None
            features["std_dev_y"] = None
            features["bbox_area"] = None
            features["centroid_dispersion"] = None
            features["landmark_10_dist"] = FeatureExtractor._landmark_distance_from_matrix(dist_matrix, rng)
            features["nn_probe_cost"] = FeatureExtractor._nn_probe_cost_from_matrix(dist_matrix)
            # Without coordinates we cannot derive spatial dispersion, but we can normalise the matrix itself.
            features["std_dev_x_norm"] = None
            features["std_dev_y_norm"] = None
            features["bbox_area_norm"] = None
            features["centroid_dispersion_norm"] = None
            features["landmark_10_dist_norm"] = None
            features["nn_probe_cost_norm"] = None
            if dist_matrix.size > 0:
                max_edge = float(np.max(dist_matrix))
                if np.isfinite(max_edge) and max_edge > 0.0:
                    norm_mat = dist_matrix / max_edge
                    features["landmark_10_dist_norm"] = FeatureExtractor._landmark_distance_from_matrix(norm_mat, rng)
                    features["nn_probe_cost_norm"] = FeatureExtractor._nn_probe_cost_from_matrix(norm_mat)
        else:
            features.update(
                {
                    "std_dev_x": None,
                    "std_dev_y": None,
                    "bbox_area": None,
                    "centroid_dispersion": None,
                    "landmark_10_dist": None,
                    "nn_probe_cost": None,
                    "std_dev_x_norm": None,
                    "std_dev_y_norm": None,
                    "bbox_area_norm": None,
                    "centroid_dispersion_norm": None,
                    "landmark_10_dist_norm": None,
                    "nn_probe_cost_norm": None,
                }
            )

        elapsed = FeatureExtractor._now() - start
        if features.get("nn_probe_cost") is not None and n_nodes > 0:
            try:
                features["nn_probe_cost_per_node"] = float(features["nn_probe_cost"]) / float(n_nodes)
            except Exception:
                features["nn_probe_cost_per_node"] = None
        else:
            features["nn_probe_cost_per_node"] = None
        return FeatureVector(values=features, elapsed=elapsed)

    @staticmethod
    def _now() -> float:
        import time

        return time.perf_counter()

    @staticmethod
    def _get_coordinates(problem_data: dict[str, Any]) -> Optional[np.ndarray]:
        coords = problem_data.get("coordinates")
        if coords is None:
            return None
        arr = np.asarray(coords, dtype=float)
        if arr.ndim != 2 or arr.shape[1] < 2:
            return None
        return arr[:, :2]

    @staticmethod
    def _get_distance_matrix(problem_data: dict[str, Any]) -> Optional[np.ndarray]:
        matrix = problem_data.get("distance_matrix")
        if matrix is None:
            return None
        arr = np.asarray(matrix, dtype=float)
        return arr if arr.ndim == 2 else None

    @staticmethod
    def _infer_n_nodes(coords: Optional[np.ndarray], dist_matrix: Optional[np.ndarray]) -> int:
        if coords is not None:
            return int(coords.shape[0])
        if dist_matrix is not None:
            return int(dist_matrix.shape[0])
        return 0

    @staticmethod
    def _is_metric(
        metric_hint: str, coords: Optional[np.ndarray], dist_matrix: Optional[np.ndarray], rng: np.random.Generator
    ) -> bool:
        if metric_hint:
            metric_hint = metric_hint.lower()
            if any(metric_hint.startswith(prefix) for prefix in ["euc", "geo", "manhattan", "pseudo"]):
                return True
        if dist_matrix is None and coords is None:
            return False
        if dist_matrix is None and coords is not None:
            # Coordinate-based quick check (symmetric Euclidean / Manhattan).
            return True

        assert dist_matrix is not None
        n = dist_matrix.shape[0]
        if n == 0:
            return True
        diag_ok = np.allclose(np.diag(dist_matrix), 0, atol=1e-6)
        if not diag_ok:
            return False
        symmetric_ok = True
        samples = min(10, n * (n - 1) // 2)
        for _ in range(samples):
            i, j = rng.integers(0, n, size=2)
            if abs(float(dist_matrix[i, j] - dist_matrix[j, i])) > 1e-6:
                symmetric_ok = False
                break
        if not symmetric_ok:
            return False

        # Spot-check triangle inequality on a small, fixed number of triples.
        for _ in range(min(20, n**2)):
            a, b, c = rng.integers(0, n, size=3)
            if float(dist_matrix[a, b]) - (float(dist_matrix[a, c]) + float(dist_matrix[c, b])) > 1e-6:
                return False
        return True

    @staticmethod
    def _distance(a: np.ndarray, b: np.ndarray, metric_hint: str) -> np.ndarray:
        if metric_hint == "manhattan":
            return np.abs(a - b).sum(axis=-1)
        return np.linalg.norm(a - b, axis=-1)

    @staticmethod
    def _landmark_distance(coords: np.ndarray, metric_hint: str, rng: np.random.Generator) -> float:
        n = coords.shape[0]
        if n == 0:
            return 0.0
        landmark = int(rng.integers(0, n))
        if n == 1:
            return 0.0
        sample_size = min(10, n - 1)
        choices = [idx for idx in range(n) if idx != landmark]
        sample = rng.choice(choices, size=sample_size, replace=False)
        dists = FeatureExtractor._distance(coords[sample], coords[landmark], metric_hint)
        return float(np.mean(dists))

    @staticmethod
    def _landmark_distance_from_matrix(dist_matrix: np.ndarray, rng: np.random.Generator) -> float:
        n = dist_matrix.shape[0]
        if n == 0:
            return 0.0
        landmark = int(rng.integers(0, n))
        sample_size = min(10, n - 1)
        choices = [idx for idx in range(n) if idx != landmark]
        sample = rng.choice(choices, size=sample_size, replace=False)
        return float(np.mean(dist_matrix[landmark, sample]))

    @staticmethod
    def _nn_probe_cost(coords: np.ndarray, metric_hint: str) -> float:
        n = coords.shape[0]
        if n == 0:
            return 0.0
        probe_size = max(1, int(max(1, 0.1 * n)))
        visited = [0]
        unvisited = set(range(1, min(n, probe_size + 1)))
        cost = 0.0
        while unvisited:
            current = visited[-1]
            remaining_idx = np.fromiter(unvisited, dtype=int)
            dists = FeatureExtractor._distance(coords[remaining_idx], coords[current], metric_hint)
            next_city_idx = int(remaining_idx[int(np.argmin(dists))])
            cost += float(dists.min())
            visited.append(next_city_idx)
            unvisited.remove(next_city_idx)
        cost += float(FeatureExtractor._distance(coords[[visited[0]]], coords[visited[-1]], metric_hint).item())
        return cost

    @staticmethod
    def _nn_probe_cost_from_matrix(dist_matrix: np.ndarray) -> float:
        n = dist_matrix.shape[0]
        if n == 0:
            return 0.0
        probe_size = max(1, int(max(1, 0.1 * n)))
        visited = [0]
        unvisited = set(range(1, min(n, probe_size + 1)))
        cost = 0.0
        while unvisited:
            current = visited[-1]
            remaining_idx = np.fromiter(unvisited, dtype=int)
            dists = dist_matrix[current, remaining_idx]
            idx = int(np.argmin(dists))
            next_city = int(remaining_idx[idx])
            cost += float(dists[idx])
            visited.append(next_city)
            unvisited.remove(next_city)
        cost += float(dist_matrix[visited[-1], visited[0]])
        return cost

    @staticmethod
    def _normalise_coords(coords: np.ndarray) -> Optional[np.ndarray]:
        """Normalise coordinates to the unit square to make geometric features scale-invariant."""
        if coords.ndim != 2 or coords.shape[0] == 0:
            return None
        mins = coords.min(axis=0)
        spans = coords.max(axis=0) - mins
        spans[spans == 0] = 1.0  # avoid division by zero for degenerate axes
        return (coords - mins) / spans


__all__ = ["FeatureExtractor", "FeatureVector"]
