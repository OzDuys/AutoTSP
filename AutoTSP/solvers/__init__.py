from __future__ import annotations

from AutoTSP.solvers.approx import ChristofidesSolver
from AutoTSP.solvers.base import AlgorithmResult, BaseSolver, SolverSpec
from AutoTSP.solvers.exact import BranchAndBoundSolver, ConcordeExactSolver, CuttingPlaneSolver, HeldKarpSolver
from AutoTSP.solvers.heuristics import (
    MultiStartNearestNeighborSolver,
    ShinkaSpatialHeuristicSolver,
    SimpleNearestNeighborSolver,
    ThreeOptSolver,
    TwoOptSolver,
)
from AutoTSP.solvers.meta import (
    AntColonySolver,
    GeneticAlgorithmSolver,
    IteratedLocalSearchSolver,
    LkhSolver,
    SimulatedAnnealingSolver,
)
from AutoTSP.utils.taxonomy import AlgorithmFamily

SOLVER_SPECS: dict[str, SolverSpec] = {
    BranchAndBoundSolver.name: SolverSpec(
        name=BranchAndBoundSolver.name,
        cls=BranchAndBoundSolver,
        family=BranchAndBoundSolver.family,
        supports_directed=BranchAndBoundSolver.supports_directed,
    ),
    HeldKarpSolver.name: SolverSpec(
        name=HeldKarpSolver.name,
        cls=HeldKarpSolver,
        family=HeldKarpSolver.family,
        supports_directed=HeldKarpSolver.supports_directed,
    ),
    CuttingPlaneSolver.name: SolverSpec(
        name=CuttingPlaneSolver.name,
        cls=CuttingPlaneSolver,
        family=CuttingPlaneSolver.family,
        supports_directed=CuttingPlaneSolver.supports_directed,
    ),
    ConcordeExactSolver.name: SolverSpec(
        name=ConcordeExactSolver.name,
        cls=ConcordeExactSolver,
        family=ConcordeExactSolver.family,
        supports_directed=ConcordeExactSolver.supports_directed,
    ),
    ChristofidesSolver.name: SolverSpec(
        name=ChristofidesSolver.name,
        cls=ChristofidesSolver,
        family=ChristofidesSolver.family,
        supports_directed=ChristofidesSolver.supports_directed,
    ),
    SimpleNearestNeighborSolver.name: SolverSpec(
        name=SimpleNearestNeighborSolver.name,
        cls=SimpleNearestNeighborSolver,
        family=SimpleNearestNeighborSolver.family,
        supports_directed=SimpleNearestNeighborSolver.supports_directed,
    ),
    MultiStartNearestNeighborSolver.name: SolverSpec(
        name=MultiStartNearestNeighborSolver.name,
        cls=MultiStartNearestNeighborSolver,
        family=MultiStartNearestNeighborSolver.family,
        supports_directed=MultiStartNearestNeighborSolver.supports_directed,
    ),
    ThreeOptSolver.name: SolverSpec(
        name=ThreeOptSolver.name,
        cls=ThreeOptSolver,
        family=ThreeOptSolver.family,
        supports_directed=ThreeOptSolver.supports_directed,
    ),
    TwoOptSolver.name: SolverSpec(
        name=TwoOptSolver.name,
        cls=TwoOptSolver,
        family=TwoOptSolver.family,
        supports_directed=TwoOptSolver.supports_directed,
    ),
    ShinkaSpatialHeuristicSolver.name: SolverSpec(
        name=ShinkaSpatialHeuristicSolver.name,
        cls=ShinkaSpatialHeuristicSolver,
        family=ShinkaSpatialHeuristicSolver.family,
        supports_directed=ShinkaSpatialHeuristicSolver.supports_directed,
    ),
    SimulatedAnnealingSolver.name: SolverSpec(
        name=SimulatedAnnealingSolver.name,
        cls=SimulatedAnnealingSolver,
        family=SimulatedAnnealingSolver.family,
        supports_directed=SimulatedAnnealingSolver.supports_directed,
    ),
    GeneticAlgorithmSolver.name: SolverSpec(
        name=GeneticAlgorithmSolver.name,
        cls=GeneticAlgorithmSolver,
        family=GeneticAlgorithmSolver.family,
        supports_directed=GeneticAlgorithmSolver.supports_directed,
    ),
    IteratedLocalSearchSolver.name: SolverSpec(
        name=IteratedLocalSearchSolver.name,
        cls=IteratedLocalSearchSolver,
        family=IteratedLocalSearchSolver.family,
        supports_directed=IteratedLocalSearchSolver.supports_directed,
    ),
    LkhSolver.name: SolverSpec(
        name=LkhSolver.name,
        cls=LkhSolver,
        family=LkhSolver.family,
        supports_directed=LkhSolver.supports_directed,
    ),
    AntColonySolver.name: SolverSpec(
        name=AntColonySolver.name,
        cls=AntColonySolver,
        family=AntColonySolver.family,
        supports_directed=AntColonySolver.supports_directed,
    ),
}

SOLVER_REGISTRY: dict[str, type[BaseSolver]] = {name: spec.cls for name, spec in SOLVER_SPECS.items()}
SOLVER_FAMILIES: dict[str, AlgorithmFamily] = {name: spec.family for name, spec in SOLVER_SPECS.items()}
SOLVER_SUPPORT: dict[str, bool] = {name: spec.supports_directed for name, spec in SOLVER_SPECS.items()}


def get_solver(name: str) -> BaseSolver:
    solver_cls = SOLVER_REGISTRY.get(name)
    if solver_cls is None:
        raise KeyError(f"Unknown solver: {name}")
    return solver_cls()


__all__ = [
    "AlgorithmResult",
    "BaseSolver",
    "SOLVER_FAMILIES",
    "SOLVER_REGISTRY",
    "SOLVER_SPECS",
    "SOLVER_SUPPORT",
    "AlgorithmFamily",
    "get_solver",
    "BranchAndBoundSolver",
    "HeldKarpSolver",
    "CuttingPlaneSolver",
    "ConcordeExactSolver",
    "ChristofidesSolver",
    "SimpleNearestNeighborSolver",
    "MultiStartNearestNeighborSolver",
    "ThreeOptSolver",
    "TwoOptSolver",
    "BaselineTSPLibSolver",
    "SimulatedAnnealingSolver",
    "GeneticAlgorithmSolver",
    "IteratedLocalSearchSolver",
    "AntColonySolver",
]
