"""
Best Anchor heuristic for TSP.

Tries single anchor from each vertex and returns best tour found.
Research algorithm for identifying optimal anchor vertices.
"""

from typing import List
import time
from .base import TSPAlgorithm, TourResult, AlgorithmMetadata
from .registry import register_algorithm
from .single_anchor import SingleAnchorAlgorithm


@register_algorithm(
    "best_anchor_exhaustive",
    tags=["anchor", "search"],
    constraints={}
)
class BestAnchorAlgorithm(TSPAlgorithm):
    """Best anchor search by trying all vertices."""

    def solve(self, adjacency_matrix: List[List[float]], **kwargs) -> TourResult:
        """
        Solve TSP by trying each vertex as anchor and returning best.

        Args:
            adjacency_matrix: 2D adjacency matrix
            **kwargs: unused

        Returns:
            TourResult with best tour found
        """
        start_time = time.time()

        num_vertices = len(adjacency_matrix)
        single_anchor_algo = SingleAnchorAlgorithm(random_seed=self.random_seed)

        best_tour = None
        best_weight = float('inf')
        best_anchor = -1
        all_weights = []

        # Try each vertex as anchor
        for anchor_vertex in range(num_vertices):
            result = single_anchor_algo.solve(
                adjacency_matrix,
                anchor_vertex=anchor_vertex
            )

            all_weights.append(result.weight if result.success else float('inf'))

            if result.success and result.weight < best_weight:
                best_weight = result.weight
                best_tour = result.tour
                best_anchor = anchor_vertex

        runtime = time.time() - start_time

        if best_tour is None:
            return self._create_failure_result(
                "No valid tour found with any anchor vertex"
            )

        return TourResult(
            tour=best_tour,
            weight=best_weight,
            runtime=runtime,
            metadata={
                'best_anchor_vertex': best_anchor,
                'all_anchor_weights': [float(w) for w in all_weights],
                'search_time': runtime
            },
            success=True
        )

    def get_metadata(self) -> AlgorithmMetadata:
        """Return algorithm metadata."""
        return AlgorithmMetadata(
            name="best_anchor_exhaustive",
            version="1.0.0",
            description="Best anchor search exhausting all vertices",
            parameters={"anchor_selection": "exhaustive"}
        )
