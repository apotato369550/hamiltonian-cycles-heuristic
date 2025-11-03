"""
Single Anchor heuristic for TSP.

Pre-commits edges from anchor vertex and builds remaining tour greedily.
Research algorithm for anchor-based TSP investigation.
"""

from typing import List
import time
from .base import TSPAlgorithm, TourResult, AlgorithmMetadata
from .registry import register_algorithm


@register_algorithm(
    "single_anchor",
    tags=["anchor", "heuristic"],
    constraints={}
)
class SingleAnchorAlgorithm(TSPAlgorithm):
    """Single anchor heuristic with two cheapest edges pre-committed."""

    def solve(self, adjacency_matrix: List[List[float]], **kwargs) -> TourResult:
        """
        Solve TSP using single anchor heuristic.

        Args:
            adjacency_matrix: 2D adjacency matrix
            anchor_vertex: Which vertex to use as anchor (default 0)
            **kwargs: unused

        Returns:
            TourResult with tour or failure if anchor edges don't lead to valid tour
        """
        start_time = time.time()

        anchor_vertex = kwargs.get('anchor_vertex', 0)
        num_vertices = len(adjacency_matrix)

        if anchor_vertex < 0 or anchor_vertex >= num_vertices:
            return self._create_failure_result(f"Invalid anchor vertex: {anchor_vertex}")

        # Find two cheapest edges from anchor
        edges_from_anchor = [
            (adjacency_matrix[anchor_vertex][v], v)
            for v in range(num_vertices)
            if v != anchor_vertex
        ]
        edges_from_anchor.sort()

        if len(edges_from_anchor) < 2:
            return self._create_failure_result("Not enough neighbors for two edges")

        anchor_edge_1 = edges_from_anchor[0]
        anchor_edge_2 = edges_from_anchor[1]

        # Build tour: anchor -> neighbor1 -> ... -> neighbor2 -> anchor
        try:
            tour = self._build_tour_with_anchors(
                adjacency_matrix,
                anchor_vertex,
                anchor_edge_1[1],  # First neighbor
                anchor_edge_2[1]   # Second neighbor
            )
        except Exception as e:
            return self._create_failure_result(f"Tour construction failed: {str(e)}")

        if not tour or len(tour) != num_vertices:
            return self._create_failure_result("Failed to build Hamiltonian cycle")

        # Validate tour structure
        is_valid, msg = self._validate_tour_structure(tour, num_vertices)
        if not is_valid:
            return self._create_failure_result(msg)

        weight = self._compute_tour_weight(tour, adjacency_matrix)
        runtime = time.time() - start_time

        return TourResult(
            tour=tour,
            weight=weight,
            runtime=runtime,
            metadata={
                'anchor_vertex': anchor_vertex,
                'anchor_neighbors': [anchor_edge_1[1], anchor_edge_2[1]],
                'anchor_edge_weights': [anchor_edge_1[0], anchor_edge_2[0]]
            },
            success=True
        )

    def get_metadata(self) -> AlgorithmMetadata:
        """Return algorithm metadata."""
        return AlgorithmMetadata(
            name="single_anchor",
            version="1.0.0",
            description="Single anchor heuristic with pre-committed edges",
            parameters={"anchor_vertex": "specified"}
        )

    def _build_tour_with_anchors(
        self,
        adjacency_matrix: List[List[float]],
        anchor: int,
        neighbor1: int,
        neighbor2: int
    ) -> List[int]:
        """Build tour with anchor and its two neighbors."""
        num_vertices = len(adjacency_matrix)

        # Tour structure: anchor -> neighbor1 -> ... path ... -> neighbor2 -> anchor
        unvisited = set(range(num_vertices)) - {anchor, neighbor1, neighbor2}

        # Start path from neighbor1
        path = [neighbor1]
        current = neighbor1

        # Build path greedily to neighbor2
        while unvisited:
            # Find nearest unvisited neighbor
            nearest = None
            nearest_dist = float('inf')

            for v in unvisited:
                dist = adjacency_matrix[current][v]
                if dist < nearest_dist:
                    nearest_dist = dist
                    nearest = v
                elif dist == nearest_dist and (nearest is None or v < nearest):
                    nearest = v

            if nearest is None:
                break

            path.append(nearest)
            unvisited.remove(nearest)
            current = nearest

        # Complete tour
        tour = [anchor] + path + [neighbor2]

        return tour
