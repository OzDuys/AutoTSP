from __future__ import annotations

from enum import Enum


class AlgorithmFamily(str, Enum):
    EXACT = "exact"
    APPROXIMATION = "approximation"
    HEURISTIC = "heuristic"
    METAHEURISTIC = "metaheuristic"
    AUTO = "autotsp"


__all__ = ["AlgorithmFamily"]
