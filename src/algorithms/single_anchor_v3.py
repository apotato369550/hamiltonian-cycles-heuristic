"""
Single Anchor V3 heuristic for TSP.

Adaptive anchor-based algorithm that extends from both ends of the path
after fixing the two cheapest edges from the anchor vertex.
"""

from typing import List, Set
import time
import numpy as np
from .base import TSPAlgorithm, TourResult, AlgorithmMetadata
from .registry import register_algorithm


@register_algorithm(
    "single_anchor_v3",
    tags=["anchor", "heuristic", "adaptive"],
    constraints={}
)
class SingleAnchorV3Algorithm(TSPAlgorithm):
    """
    Single anchor V3 - Adaptive path building from both ends.

    Fixes the two cheapest edges from anchor vertex first, then builds
    the path by checking both ends and extending from the end with the
    cheaper connection. Similar to Prim's algorithm but ensures a
    Hamiltonian cycle.
    """

    def solve(self, adjacency_matrix: List[List[float]], **kwargs) -> TourResult:
        """
        Solve TSP using single anchor V3 (adaptive both-ends).

        Args:
            adjacency_matrix: 2D adjacency matrix
            anchor_vertex: Which vertex to use as anchor (default: determined by random_seed)
            **kwargs: unused

        Returns:
            TourResult with tour or failure if construction fails
        """
        start_time = time.time()

        num_vertices = len(adjacency_matrix)

        # Determine anchor vertex (use random_seed for consistency)
        if 'anchor_vertex' in kwargs:
            anchor_vertex = kwargs['anchor_vertex']
        else:
            if self.random_seed is not None:
                np.random.seed(self.random_seed)
            anchor_vertex = np.random.randint(0, num_vertices)

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

        neighbor_1 = anchor_edge_1[1]
        neighbor_2 = anchor_edge_2[1]

        # Build tour adaptively from both ends
        try:
            tour = self._build_tour_adaptive(
                adjacency_matrix,
                anchor_vertex,
                neighbor_1,
                neighbor_2
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
                'anchor_vertex': int(anchor_vertex),
                'neighbor_1': neighbor_1,
                'neighbor_2': neighbor_2,
                'anchor_edge_weights': [anchor_edge_1[0], anchor_edge_2[0]]
            },
            success=True
        )

    def get_metadata(self) -> AlgorithmMetadata:
        """Return algorithm metadata."""
        return AlgorithmMetadata(
            name="single_anchor_v3",
            version="1.0.0",
            description="Single anchor V3: adaptive path building from both ends after fixing anchor edges",
            parameters={"anchor_vertex": "random or specified", "strategy": "both_ends"}
        )

    def _build_tour_adaptive(
        self,
        adjacency_matrix: List[List[float]],
        anchor: int,
        neighbor_1: int,
        neighbor_2: int
    ) -> List[int]:
        """
        Build tour adaptively from both ends.

        Tour structure: anchor -> neighbor_1 -> ... path ... -> neighbor_2 -> anchor

        The path between neighbor_1 and neighbor_2 is built by:
        1. Starting with path = [neighbor_1, neighbor_2]
        2. At each step, find cheapest edge from left end (neighbor_1 side)
        3. Find cheapest edge from right end (neighbor_2 side)
        4. Extend from the end with cheaper edge

        Args:
            adjacency_matrix: Graph weights
            anchor: Anchor vertex
            neighbor_1: First neighbor (left end)
            neighbor_2: Second neighbor (right end)

        Returns:
            Complete tour as list of vertex indices
        """
        num_vertices = len(adjacency_matrix)

        # Initialize path with the two anchor neighbors
        # Path will be built between these two vertices
        path = [neighbor_1, neighbor_2]
        unvisited: Set[int] = set(range(num_vertices)) - {anchor, neighbor_1, neighbor_2}

        # Build path by extending from both ends
        while unvisited:
            left_end = path[0]   # neighbor_1 side
            right_end = path[-1]  # neighbor_2 side

            # Find best extension from left end
            best_left = None
            best_left_dist = float('inf')
            for vertex in unvisited:
                dist = adjacency_matrix[vertex][left_end]  # Edge TO left end
                if dist < best_left_dist:
                    best_left_dist = dist
                    best_left = vertex
                elif dist == best_left_dist and (best_left is None or vertex < best_left):
                    # Tie-breaking: choose lowest index
                    best_left = vertex

            # Find best extension from right end
            best_right = None
            best_right_dist = float('inf')
            for vertex in unvisited:
                dist = adjacency_matrix[right_end][vertex]  # Edge FROM right end
                if dist < best_right_dist:
                    best_right_dist = dist
                    best_right = vertex
                elif dist == best_right_dist and (best_right is None or vertex < best_right):
                    # Tie-breaking: choose lowest index
                    best_right = vertex

            # Extend from the end with cheaper connection
            if best_left_dist <= best_right_dist:
                # Extend from left (prepend to path)
                path.insert(0, best_left)
                unvisited.remove(best_left)
            else:
                # Extend from right (append to path)
                path.append(best_right)
                unvisited.remove(best_right)

        # Complete the tour: anchor -> path[0] -> ... -> path[-1] -> anchor
        tour = [anchor] + path

        return tour

    def _create_failure_result(self, error_message: str) -> TourResult:
        """Create a failed TourResult."""
        return TourResult(
            tour=[],
            weight=float('inf'),
            runtime=0.0,
            metadata={},
            success=False,
            error_message=error_message
        )
