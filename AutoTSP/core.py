from __future__ import annotations

import time
from typing import Any, Dict

import numpy as np

from AutoTSP.features import FeatureExtractor
from AutoTSP.selectors import RuleBasedSelector, get_selector
from AutoTSP.solvers import AlgorithmResult, get_solver


class AutoTSP:
    """End-to-end AutoTSP pipeline: features -> selector -> solver."""

    def __init__(self, selector: object | None = None, selector_name: str = "rule_based", selector_kwargs: Dict | None = None):
        if selector is not None:
            self.selector = selector
        else:
            self.selector = get_selector(selector_name, **(selector_kwargs or {}))

    def solve(self, problem_data: Dict[str, Any], time_budget: float = 5.0) -> AlgorithmResult:
        start_time = time.perf_counter()
        features = FeatureExtractor.extract(problem_data)
        remaining_budget = max(0.0, float(time_budget) - features.elapsed)
        selector_features = dict(features.values)
        selector_features["time_budget"] = float(time_budget)
        selector_features["remaining_budget"] = remaining_budget

        solver_cls = self.selector.predict(selector_features, remaining_budget)
        solver = get_solver(solver_cls.name)
        dist_matrix = self._to_distance_matrix(problem_data)

        result = solver.solve(dist_matrix, time_limit=remaining_budget)
        metadata = dict(result.metadata)
        metadata.update(
            {
                "selected_solver": solver_cls.name,
                "feature_time": features.elapsed,
                "features": features.values,
                "budget_requested": float(time_budget),
                "budget_remaining": remaining_budget,
                "wallclock_total": time.perf_counter() - start_time,
            }
        )
        return AlgorithmResult(
            name=result.name,
            path=result.path,
            cost=result.cost,
            elapsed=result.elapsed,
            status=result.status,
            metadata=metadata,
        )

    def _to_distance_matrix(self, problem_data: Dict[str, Any]) -> np.ndarray:
        if "distance_matrix" in problem_data and problem_data["distance_matrix"] is not None:
            return np.asarray(problem_data["distance_matrix"], dtype=float)
        if "coordinates" not in problem_data or problem_data["coordinates"] is None:
            raise ValueError("Problem data must contain either 'distance_matrix' or 'coordinates'.")

        coords = np.asarray(problem_data["coordinates"], dtype=float)
        metric = (problem_data.get("metric") or "euclidean").lower()
        if metric == "manhattan":
            diff = coords[:, None, :] - coords[None, :, :]
            return np.abs(diff).sum(axis=-1)
        diff = coords[:, None, :] - coords[None, :, :]
        return np.linalg.norm(diff, axis=-1)


__all__ = ["AutoTSP"]
