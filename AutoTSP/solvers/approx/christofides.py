from __future__ import annotations

import networkx as nx
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


class ChristofidesSolver(BaseSolver):
    name = "christofides"
    family = AlgorithmFamily.APPROXIMATION
    supports_directed = False

    def solve(self, graph: np.ndarray, time_limit: float = 5.0) -> AlgorithmResult:
        dist_matrix = np.asarray(graph, dtype=float)
        start_time = current_time()
        n = dist_matrix.shape[0]
        if n <= 2:
            cycle = best_cycle(list(range(n)))
            cost = compute_cycle_cost(dist_matrix, cycle)
            return AlgorithmResult(
                name=self.name,
                path=cycle,
                cost=cost,
                elapsed=current_time() - start_time,
                status="complete",
                metadata={"odd_vertices": 0},
            )

        try:
            enforce_time_budget(start_time, time_limit)
            graph_nx = nx.Graph()
            for i in range(n):
                for j in range(i + 1, n):
                    weight = float(dist_matrix[i, j])
                    graph_nx.add_edge(i, j, weight=weight)

            enforce_time_budget(start_time, time_limit)
            mst = nx.minimum_spanning_tree(graph_nx, weight="weight")

            enforce_time_budget(start_time, time_limit)
            odd_vertices = [node for node, degree in mst.degree() if degree % 2 == 1]
            if len(odd_vertices) % 2 == 1:
                odd_vertices.pop()

            matching = set()
            if odd_vertices:
                enforce_time_budget(start_time, time_limit)
                induced = graph_nx.subgraph(odd_vertices)
                matching = nx.algorithms.matching.min_weight_matching(induced, weight="weight")

            multigraph = nx.MultiGraph()
            multigraph.add_nodes_from(graph_nx.nodes())
            multigraph.add_edges_from(mst.edges())
            multigraph.add_edges_from(matching)

            enforce_time_budget(start_time, time_limit)
            eulerian_tour = list(nx.eulerian_circuit(multigraph, source=0))
            seen = set()
            path = []
            for u, v in eulerian_tour:
                if u not in seen:
                    path.append(u)
                    seen.add(u)
                if v not in seen:
                    path.append(v)
                    seen.add(v)

            cycle = best_cycle(path)
            cost = compute_cycle_cost(dist_matrix, cycle)
            elapsed = current_time() - start_time
            return AlgorithmResult(
                name=self.name,
                path=cycle,
                cost=cost,
                elapsed=elapsed,
                status="complete",
                metadata={"odd_vertices": len(odd_vertices), "matching_size": len(matching)},
            )
        except TimeLimitExpired:
            return AlgorithmResult(
                name=self.name,
                path=None,
                cost=None,
                elapsed=current_time() - start_time,
                status="timeout",
                metadata={},
            )


__all__ = ["ChristofidesSolver"]
