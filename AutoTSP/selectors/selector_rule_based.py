from __future__ import annotations

from AutoTSP.selectors.base import BaseSelector
from AutoTSP.solvers import (
    BranchAndBoundSolver,
    GeneticAlgorithmSolver,
    HeldKarpSolver,
    IteratedLocalSearchSolver,
    MultiStartNearestNeighborSolver,
    SimpleNearestNeighborSolver,
    SimulatedAnnealingSolver,
    ThreeOptSolver,
)


class RuleBasedSelector(BaseSelector):
    """Phase 3 V1: hard-coded solver selection heuristics."""

    def predict(self, features: dict, remaining_budget: float):
        n = int(features.get("n_nodes") or 0)
        is_metric = bool(features.get("is_metric"))
        budget = float(remaining_budget)

        # Fast-path for very tight budgets.
        if budget < 1.0:
            return MultiStartNearestNeighborSolver
        if budget < 2.0:
            return SimpleNearestNeighborSolver

        if n < 14:
            return HeldKarpSolver
        if n < 25:
            return BranchAndBoundSolver
        if n > 2000:
            return MultiStartNearestNeighborSolver
        if not is_metric:
            return GeneticAlgorithmSolver
        if n < 150:
            return ThreeOptSolver
        return IteratedLocalSearchSolver


__all__ = ["RuleBasedSelector"]
