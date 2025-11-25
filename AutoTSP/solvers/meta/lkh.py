from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List

import numpy as np

from AutoTSP.solvers.base import AlgorithmResult, BaseSolver, compute_cycle_cost, current_time
from AutoTSP.utils.taxonomy import AlgorithmFamily

try:
    import lkh as pylkh  # type: ignore
except Exception:  # noqa: BLE001
    pylkh = None


class LkhSolver(BaseSolver):
    """Wrapper around the LKH TSP heuristic (expects `LKH` binary on PATH or env `LKH_BIN`)."""

    name = "lkh"
    family = AlgorithmFamily.METAHEURISTIC
    supports_directed = True  # Handle symmetric and asymmetric by switching mode internally.

    def __init__(self, lkh_bin: str | None = None):
        import os

        env_bin = os.environ.get("LKH_BIN")
        repo_root = Path(__file__).resolve().parents[3]
        fallback_bins = [
            lkh_bin,
            env_bin,
            shutil.which("LKH"),
            shutil.which("lkh"),
            str(Path(__file__).resolve().parent / "LKH"),
            str(repo_root / "LKH-3.0.13" / "LKH"),
            str(repo_root / "LKH-3.0.6" / "LKH"),
        ]
        self.lkh_bin = next((p for p in fallback_bins if p and Path(p).exists()), None)
        if self.lkh_bin is None:
            raise FileNotFoundError("LKH binary not found on PATH. Set LKH_BIN or install LKH.")

    def _is_asymmetric(self, dist_matrix: np.ndarray) -> bool:
        return not np.allclose(dist_matrix, dist_matrix.T, atol=1e-9, rtol=1e-9)

    def solve(self, graph: np.ndarray, time_limit: float = 5.0) -> AlgorithmResult:
        dist_matrix = np.asarray(graph, dtype=float)
        n = dist_matrix.shape[0]
        start = current_time()
        atsp_mode = self._is_asymmetric(dist_matrix)

        # Prefer pylkh if available.
        if pylkh is not None:
            try:
                if atsp_mode and hasattr(pylkh, "solve_atsp"):
                    tour = pylkh.solve_atsp(dist_matrix.tolist(), runs=1, seed=1)
                else:
                    tour = pylkh.solve_tsp(dist_matrix.tolist(), runs=1, seed=1)
                cost = compute_cycle_cost(dist_matrix, tour)
                return AlgorithmResult(
                    name=self.name,
                    path=tour,
                    cost=cost,
                    elapsed=current_time() - start,
                    status="complete",
                    metadata={"source": "pylkh", "mode": "atsp" if atsp_mode else "tsp"},
                )
            except Exception:  # noqa: BLE001
                pass  # fall back to binary invocation

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            problem_path = tmp / "instance.tsp"
            tour_path = tmp / "instance.tour"
            par_path = tmp / "params.par"

            self._write_tsplib_problem(problem_path, dist_matrix, atsp=atsp_mode)
            self._write_params(par_path, problem_path, tour_path, time_limit)

            try:
                subprocess.run(
                    [self.lkh_bin, str(par_path)],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=time_limit + 5.0,
                )
            except FileNotFoundError as exc:
                return AlgorithmResult(
                    name=self.name,
                    path=None,
                    cost=None,
                    elapsed=current_time() - start,
                    status="failed",
                    metadata={"reason": "lkh_not_found", "error": str(exc)},
                )
            except subprocess.TimeoutExpired:
                return AlgorithmResult(
                    name=self.name,
                    path=None,
                    cost=None,
                    elapsed=current_time() - start,
                    status="timeout",
                    metadata={"reason": "lkh_timeout"},
                )
            except subprocess.CalledProcessError as exc:
                return AlgorithmResult(
                    name=self.name,
                    path=None,
                    cost=None,
                    elapsed=current_time() - start,
                    status="failed",
                    metadata={"reason": "lkh_error", "error": exc.stderr.decode(errors="ignore")},
                )

            if not tour_path.exists():
                return AlgorithmResult(
                    name=self.name,
                    path=None,
                    cost=None,
                    elapsed=current_time() - start,
                    status="failed",
                    metadata={"reason": "tour_not_written"},
                )

            tour = self._read_tour(tour_path, n)
            cost = compute_cycle_cost(dist_matrix, tour)
            return AlgorithmResult(
                name=self.name,
                path=tour,
                cost=cost,
                elapsed=current_time() - start,
                status="complete",
                metadata={"source": "lkh", "mode": "atsp" if atsp_mode else "tsp"},
            )

    def _write_tsplib_problem(self, path: Path, dist_matrix: np.ndarray, atsp: bool) -> None:
        n = dist_matrix.shape[0]
        with path.open("w", encoding="utf-8") as fh:
            fh.write("NAME: lkh_instance\n")
            fh.write("TYPE: ATSP\n" if atsp else "TYPE: TSP\n")
            fh.write(f"DIMENSION: {n}\n")
            fh.write("EDGE_WEIGHT_TYPE: EXPLICIT\n")
            fh.write("EDGE_WEIGHT_FORMAT: FULL_MATRIX\n")
            fh.write("EDGE_WEIGHT_SECTION\n")
            for i in range(n):
                row = " ".join(f"{float(dist_matrix[i, j]):.6f}" for j in range(n))
                fh.write(row + "\n")
            fh.write("EOF\n")

    def _write_params(self, path: Path, problem: Path, tour: Path, time_limit: float) -> None:
        with path.open("w", encoding="utf-8") as fh:
            fh.write(f"PROBLEM_FILE = {problem}\n")
            fh.write(f"OUTPUT_TOUR_FILE = {tour}\n")
            fh.write("RUNS = 1\n")
            fh.write(f"TIME_LIMIT = {max(1, int(time_limit))}\n")
            fh.write("SEED = 1\n")

    def _read_tour(self, path: Path, n: int) -> List[int]:
        nodes: List[int] = []
        with path.open("r", encoding="utf-8", errors="ignore") as fh:
            in_section = False
            for line in fh:
                line = line.strip()
                if line.upper().startswith("TOUR_SECTION"):
                    in_section = True
                    continue
                if not in_section:
                    continue
                if line.startswith("-1") or line.upper().startswith("EOF"):
                    break
                try:
                    node = int(line)
                except ValueError:
                    continue
                # LKH tours are 1-based.
                nodes.append(node - 1)
        if not nodes:
            return list(range(n)) + [0]
        if nodes[0] != nodes[-1]:
            nodes.append(nodes[0])
        return nodes


__all__ = ["LkhSolver"]
