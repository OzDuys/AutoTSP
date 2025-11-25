from __future__ import annotations

from AutoTSP.selectors.base import BaseSelector
from AutoTSP.solvers import (
    BranchAndBoundSolver,
    GeneticAlgorithmSolver,
    HeldKarpSolver,
    IteratedLocalSearchSolver,
    LkhSolver,
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
        probe_per_node = features.get("nn_probe_cost_per_node")
        probe_norm = features.get("nn_probe_cost_norm")
        dispersion = features.get("centroid_dispersion_norm")

        # Fast-path for very tight budgets.
        if budget < 1.0:
            return MultiStartNearestNeighborSolver
        if budget < 2.0:
            return SimpleNearestNeighborSolver

        # Small instances: prefer exact solvers when budget allows.
        if n <= 12:
            return HeldKarpSolver
        if n <= 30:
            return BranchAndBoundSolver

        # Extremely large instances: fallback to very fast heuristics.
        if n >= 4000:
            return MultiStartNearestNeighborSolver

        # Non-metric or asymmetric cases: avoid solvers that assume Euclidean structure.
        if not is_metric:
            if n >= 800:
                return MultiStartNearestNeighborSolver
            # If the instance looks easy (short probe cost per node), try GA; otherwise LKH (supports directed).
            if probe_per_node is not None and probe_per_node < 2.0:
                return GeneticAlgorithmSolver
            return LkhSolver

        # Metric instances: choose based on size and local difficulty.
        if n <= 150:
            return ThreeOptSolver
        if n <= 500:
            return IteratedLocalSearchSolver

        # Mid-to-large metric instances: prefer LKH if structure looks clustered/tight, otherwise simulated annealing.
        if probe_norm is not None and probe_norm < 0.2:
            return LkhSolver
        if dispersion is not None and dispersion < 0.3:
            return LkhSolver
        return SimulatedAnnealingSolver


__all__ = ["RuleBasedSelector"]
