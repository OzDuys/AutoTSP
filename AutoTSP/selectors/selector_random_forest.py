from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, List, Optional

import numpy as np

from AutoTSP.selectors.base import BaseSelector
from AutoTSP.selectors.selector_rule_based import RuleBasedSelector
from AutoTSP.solvers import SOLVER_REGISTRY


class RandomForestSelector(BaseSelector):
    """
    Phase 3 V2: Random Forest classifier over graph features.

    Expects training samples shaped like:
    {"features": {"n_nodes": 42, ...}, "best_solver": "simulated_annealing"}
    """

    def __init__(self, model: Any | None = None, feature_order: Optional[List[str]] = None):
        try:
            from sklearn.ensemble import RandomForestClassifier  # type: ignore
        except Exception as exc:  # noqa: BLE001
            raise ImportError("scikit-learn is required for RandomForestSelector") from exc

        self._RF = RandomForestClassifier
        self.model = model
        self.feature_order = feature_order
        self.fallback = RuleBasedSelector()

    def fit(self, samples: Iterable[dict[str, Any]]) -> None:
        samples = list(samples)
        if not samples:
            self.model = None
            return
        if self.feature_order is None:
            keys = set()
            for row in samples:
                keys.update(row.get("features", {}).keys())
            self.feature_order = sorted(keys)
        X = np.asarray([self._vectorize(row.get("features", {})) for row in samples], dtype=float)
        y = [row.get("best_solver") for row in samples]
        self.model = self._RF(n_estimators=100, random_state=42).fit(X, y)

    def fit_from_jsonl(self, path: str | Path) -> None:
        path = Path(path)
        rows: list[dict[str, Any]] = []
        if not path.exists():
            raise FileNotFoundError(f"Training file not found: {path}")
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        self.fit(rows)

    def predict(self, features: dict, remaining_budget: float):
        if self.model is None or self.feature_order is None:
            # Safety: fall back to rule-based if not fitted.
            return self.fallback.predict(features, remaining_budget)
        vec = self._vectorize(features)
        solver_name = self.model.predict([vec])[0]
        solver_cls = SOLVER_REGISTRY.get(solver_name)
        if solver_cls is None:
            return self.fallback.predict(features, remaining_budget)
        return solver_cls

    def _vectorize(self, features: dict) -> List[float]:
        def to_float(val: Any) -> float:
            if val is None:
                return 0.0
            if isinstance(val, bool):
                return float(val)
            try:
                return float(val)
            except Exception:
                return 0.0

        return [to_float(features.get(key)) for key in self.feature_order]


__all__ = ["RandomForestSelector"]
