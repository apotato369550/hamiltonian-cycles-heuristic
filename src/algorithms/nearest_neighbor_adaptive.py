"""
Adaptive Nearest Neighbor algorithm for TSP.

Path-building heuristic that extends from both ends of the current path,
similar to Prim's algorithm but with degree constraints to ensure a Hamiltonian cycle.
"""

from typing import List, Set
import time
import numpy as np
from .base import TSPAlgorithm, TourResult, AlgorithmMetadata
from .registry import register_algorithm


@register_algorithm(
    "nearest_neighbor_adaptive",
    tags=["heuristic", "greedy", "nearest_neighbor", "adaptive"],
    constraints={}
)
class NearestNeighborAdaptive(TSPAlgorithm):
    """
    Adaptive Nearest Neighbor that builds path from both ends.

    Similar to Prim's algorithm but designed to form a Hamiltonian cycle.
    At each step, checks both ends of the current path and extends from
    the end with the cheaper connection.
    """

    def solve(self, adjacency_matrix: List[List[float]], **kwargs) -> TourResult:
        """
        Solve TSP using adaptive nearest neighbor.

        Args:
            adjacency_matrix: 2D adjacency matrix
            **kwargs: unused

        Returns:
            TourResult with tour, weight, runtime, and metadata
        """
        start_time = time.time()

        num_vertices = len(adjacency_matrix)

        # Choose random starting vertex
        if self.random_seed is not None:
            np.random.seed(self.random_seed)

        start_vertex = np.random.randint(0, num_vertices)

        # Build tour adaptively from both ends
        tour = self._adaptive_path_building(adjacency_matrix, start_vertex)

        weight = self._compute_tour_weight(tour, adjacency_matrix)
        runtime = time.time() - start_time

        return TourResult(
            tour=tour,
            weight=weight,
            runtime=runtime,
            metadata={'start_vertex': int(start_vertex)},
            success=True
        )

    def get_metadata(self) -> AlgorithmMetadata:
        """Return algorithm metadata."""
        return AlgorithmMetadata(
            name="nearest_neighbor_adaptive",
            version="1.0.0",
            description="Adaptive nearest neighbor that extends path from both ends",
            parameters={"start_vertex": "random", "strategy": "both_ends"}
        )

    def _adaptive_path_building(
        self,
        adjacency_matrix: List[List[float]],
        start_vertex: int
    ) -> List[int]:
        """
        Build path by extending from both ends.

        At each step:
        1. Find cheapest edge from left end to unvisited vertex
        2. Find cheapest edge from right end to unvisited vertex
        3. Extend from the end with cheaper edge

        Args:
            adjacency_matrix: Graph weights
            start_vertex: Starting vertex

        Returns:
            Complete Hamiltonian path (cycle when you connect ends)
        """
        num_vertices = len(adjacency_matrix)
        unvisited: Set[int] = set(range(num_vertices))

        # Initialize path with starting vertex
        path = [start_vertex]
        unvisited.remove(start_vertex)

        while unvisited:
            left_end = path[0]
            right_end = path[-1]

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

        return path
