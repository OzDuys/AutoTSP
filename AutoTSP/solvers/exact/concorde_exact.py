from __future__ import annotations

import os
import tempfile
from contextlib import contextmanager
from typing import Optional

import numpy as np

from AutoTSP.solvers.base import (
    AlgorithmResult,
    BaseSolver,
    best_cycle,
    compute_cycle_cost,
    current_time,
)
from AutoTSP.utils.taxonomy import AlgorithmFamily


def _ensure_pyconcorde():
    try:
        from concorde.tsp import TSPSolver  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise ImportError("pyconcorde (concorde) is required for concorde_exact") from exc
    return TSPSolver


class suppress_concorde_output:
    def __enter__(self):
        self.devnull = open(os.devnull, "w")
        self.stdout_fd = os.dup(1)
        self.stderr_fd = os.dup(2)
        try:
            import sys

            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            pass
        os.dup2(self.devnull.fileno(), 1)
        os.dup2(self.devnull.fileno(), 2)
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            os.dup2(self.stdout_fd, 1)
            os.dup2(self.stderr_fd, 2)
        finally:
            os.close(self.stdout_fd)
            os.close(self.stderr_fd)
            self.devnull.close()


@contextmanager
def temporary_concorde_dir():
    """Run Concorde in a temp directory to avoid leaving .res/.sol files behind."""
    orig_cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            os.chdir(tmpdir)
            yield tmpdir
        finally:
            os.chdir(orig_cwd)


def _is_metric(dist_matrix: np.ndarray, atol: float = 1e-6) -> bool:
    n = dist_matrix.shape[0]
    if not np.allclose(dist_matrix, dist_matrix.T, atol=atol):
        return False
    if np.any(dist_matrix < -atol):
        return False
    if not np.allclose(np.diag(dist_matrix), 0, atol=atol):
        return False
    if n <= 200:
        for k in range(n):
            tri = dist_matrix[:, k][:, None] + dist_matrix[k, :][None, :]
            if np.any(dist_matrix - tri > 1e-5):
                return False
    else:
        rng = np.random.default_rng(0)
        for _ in range(2000):
            i, j, k = rng.integers(0, n, size=3)
            if dist_matrix[i, j] - (dist_matrix[i, k] + dist_matrix[k, j]) > 1e-5:
                return False
    return True


def _mds_coordinates(dist_matrix: np.ndarray, dim: int = 2) -> Optional[np.ndarray]:
    n = dist_matrix.shape[0]
    if n <= dim:
        coords = np.zeros((n, dim))
        for i in range(n):
            coords[i, min(i, dim - 1)] = dist_matrix[i, (i + 1) % n]
        return coords

    sq = dist_matrix**2
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ sq @ J
    eigvals, eigvecs = np.linalg.eigh(B)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    positive = eigvals[:dim].clip(min=0)
    if np.allclose(positive, 0):
        return None
    L = np.diag(np.sqrt(positive))
    X = eigvecs[:, :dim] @ L
    return X


class ConcordeExactSolver(BaseSolver):
    name = "concorde_exact"
    family = AlgorithmFamily.EXACT
    supports_directed = False

    def solve(self, graph: np.ndarray, time_limit: float = 5.0) -> AlgorithmResult:
        dist_matrix = np.asarray(graph, dtype=float)
        start_time = current_time()
        n = dist_matrix.shape[0]

        try:
            TSPSolver = _ensure_pyconcorde()
        except ImportError:
            return AlgorithmResult(
                name=self.name,
                path=None,
                cost=None,
                elapsed=current_time() - start_time,
                status="unsupported",
                metadata={"reason": "pyconcorde_not_installed"},
            )

        if n < 3:
            cycle = best_cycle(list(range(n)))
            return AlgorithmResult(
                name=self.name,
                path=cycle,
                cost=compute_cycle_cost(dist_matrix, cycle),
                elapsed=current_time() - start_time,
                status="complete",
                metadata={"solver": "concorde"},
            )

        if n > 1000:
            return AlgorithmResult(
                name=self.name,
                path=None,
                cost=None,
                elapsed=current_time() - start_time,
                status="unsupported",
                metadata={"reason": "too_many_cities_for_concorde", "num_cities": n},
            )

        if not _is_metric(dist_matrix):
            return AlgorithmResult(
                name=self.name,
                path=None,
                cost=None,
                elapsed=current_time() - start_time,
                status="unsupported",
                metadata={"reason": "non_metric_distance_matrix"},
            )

        coords = _mds_coordinates(dist_matrix)
        if coords is None:
            return AlgorithmResult(
                name=self.name,
                path=None,
                cost=None,
                elapsed=current_time() - start_time,
                status="unsupported",
                metadata={"reason": "mds_embedding_failed"},
            )

        try:
            with temporary_concorde_dir() as tmpdir:
                try:
                    solver = TSPSolver.from_data(coords[:, 0], coords[:, 1], norm="EUC_2D", tmp_dir=tmpdir)
                except TypeError:
                    solver = TSPSolver.from_data(coords[:, 0], coords[:, 1], norm="EUC_2D")
                with suppress_concorde_output():
                    solution = solver.solve(time_bound=max(1.0, float(time_limit)), verbose=False)
        except Exception as exc:  # noqa: BLE001
            return AlgorithmResult(
                name=self.name,
                path=None,
                cost=None,
                elapsed=current_time() - start_time,
                status="error",
                metadata={"error": str(exc)},
            )

        elapsed = current_time() - start_time

        if not getattr(solution, "success", False):
            status = "timeout" if elapsed >= time_limit else "failed"
            return AlgorithmResult(
                name=self.name,
                path=None,
                cost=None,
                elapsed=elapsed,
                status=status,
                metadata={"solver": "concorde", "status": getattr(solution, "success", None)},
            )

        tour = list(map(int, getattr(solution, "tour", [])))
        if not tour:
            return AlgorithmResult(
                name=self.name,
                path=None,
                cost=None,
                elapsed=elapsed,
                status="failed",
                metadata={"solver": "concorde", "reason": "empty_tour"},
            )

        cycle = best_cycle(tour)
        cost = compute_cycle_cost(dist_matrix, cycle)
        return AlgorithmResult(
            name=self.name,
            path=cycle,
            cost=cost,
            elapsed=elapsed,
            status="complete",
            metadata={
                "solver": "concorde",
                "header": getattr(solution, "header", None),
                "success": getattr(solution, "success", None),
            },
        )


__all__ = ["ConcordeExactSolver"]
