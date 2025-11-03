"""
Multi-Anchor heuristics for TSP.

Uses multiple anchor vertices with different selection strategies.
Research algorithms for comparing anchor distribution approaches.
"""

from typing import List, Set
import time
import numpy as np
from .base import TSPAlgorithm, TourResult, AlgorithmMetadata
from .registry import register_algorithm


def _select_random_anchors(num_vertices: int, num_anchors: int, random_seed=None) -> List[int]:
    """Randomly select K anchor vertices."""
    if random_seed is not None:
        np.random.seed(random_seed)
    return list(np.random.choice(num_vertices, min(num_anchors, num_vertices), replace=False))


def _select_distributed_anchors(
    adjacency_matrix: List[List[float]],
    num_anchors: int
) -> List[int]:
    """Greedily select well-distributed anchor vertices using maximum distance."""
    num_vertices = len(adjacency_matrix)
    anchors = []

    # Start with vertex 0
    anchors.append(0)

    # Greedily add vertices that maximize minimum distance to existing anchors
    while len(anchors) < min(num_anchors, num_vertices):
        best_vertex = -1
        best_min_distance = -1

        for v in range(num_vertices):
            if v in anchors:
                continue

            # Find minimum distance to any anchor
            min_dist = float('inf')
            for anchor in anchors:
                dist = adjacency_matrix[v][anchor]
                min_dist = min(min_dist, dist)

            if min_dist > best_min_distance:
                best_min_distance = min_dist
                best_vertex = v

        if best_vertex >= 0:
            anchors.append(best_vertex)

    return anchors


@register_algorithm(
    "multi_anchor_random",
    tags=["anchor", "multi"],
    constraints={}
)
class MultiAnchorRandom(TSPAlgorithm):
    """Multi-anchor heuristic with random selection."""

    def solve(self, adjacency_matrix: List[List[float]], **kwargs) -> TourResult:
        """
        Solve TSP using multiple random anchors.

        Args:
            adjacency_matrix: 2D adjacency matrix
            num_anchors: Number of anchors to use (default 2)
            **kwargs: unused

        Returns:
            TourResult with tour constructed from multiple anchors
        """
        start_time = time.time()

        num_anchors = kwargs.get('num_anchors', 2)
        num_vertices = len(adjacency_matrix)

        if num_anchors < 1 or num_anchors > num_vertices:
            return self._create_failure_result(
                f"Invalid number of anchors: {num_anchors}"
            )

        # Select random anchors
        anchors = _select_random_anchors(num_vertices, num_anchors, self.random_seed)

        # Build tour
        try:
            tour = self._build_multi_anchor_tour(adjacency_matrix, anchors)
        except Exception as e:
            return self._create_failure_result(f"Tour construction failed: {str(e)}")

        if not tour or len(tour) != num_vertices:
            return self._create_failure_result("Failed to build Hamiltonian cycle")

        weight = self._compute_tour_weight(tour, adjacency_matrix)
        runtime = time.time() - start_time

        return TourResult(
            tour=tour,
            weight=weight,
            runtime=runtime,
            metadata={
                'anchor_vertices': anchors,
                'num_anchors': len(anchors),
                'selection_strategy': 'random'
            },
            success=True
        )

    def get_metadata(self) -> AlgorithmMetadata:
        """Return algorithm metadata."""
        return AlgorithmMetadata(
            name="multi_anchor_random",
            version="1.0.0",
            description="Multi-anchor heuristic with random selection",
            parameters={"anchor_selection": "random"}
        )

    def _build_multi_anchor_tour(
        self,
        adjacency_matrix: List[List[float]],
        anchors: List[int]
    ) -> List[int]:
        """Build tour from multiple anchors using greedy nearest neighbor."""
        num_vertices = len(adjacency_matrix)
        unvisited = set(range(num_vertices)) - set(anchors)

        # Start from first anchor
        tour = [anchors[0]]
        current = anchors[0]

        # Add other anchors to tour
        for anchor in anchors[1:]:
            tour.append(anchor)

        # Build path through remaining vertices
        while unvisited:
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

            tour.append(nearest)
            unvisited.remove(nearest)
            current = nearest

        return tour


@register_algorithm(
    "multi_anchor_distributed",
    tags=["anchor", "multi"],
    constraints={}
)
class MultiAnchorDistributed(TSPAlgorithm):
    """Multi-anchor heuristic with distributed selection."""

    def solve(self, adjacency_matrix: List[List[float]], **kwargs) -> TourResult:
        """
        Solve TSP using well-distributed multiple anchors.

        Args:
            adjacency_matrix: 2D adjacency matrix
            num_anchors: Number of anchors to use (default 2)
            **kwargs: unused

        Returns:
            TourResult with tour constructed from distributed anchors
        """
        start_time = time.time()

        num_anchors = kwargs.get('num_anchors', 2)
        num_vertices = len(adjacency_matrix)

        if num_anchors < 1 or num_anchors > num_vertices:
            return self._create_failure_result(
                f"Invalid number of anchors: {num_anchors}"
            )

        # Select distributed anchors
        anchors = _select_distributed_anchors(adjacency_matrix, num_anchors)

        # Build tour
        try:
            tour = self._build_multi_anchor_tour(adjacency_matrix, anchors)
        except Exception as e:
            return self._create_failure_result(f"Tour construction failed: {str(e)}")

        if not tour or len(tour) != num_vertices:
            return self._create_failure_result("Failed to build Hamiltonian cycle")

        weight = self._compute_tour_weight(tour, adjacency_matrix)
        runtime = time.time() - start_time

        return TourResult(
            tour=tour,
            weight=weight,
            runtime=runtime,
            metadata={
                'anchor_vertices': anchors,
                'num_anchors': len(anchors),
                'selection_strategy': 'distributed'
            },
            success=True
        )

    def get_metadata(self) -> AlgorithmMetadata:
        """Return algorithm metadata."""
        return AlgorithmMetadata(
            name="multi_anchor_distributed",
            version="1.0.0",
            description="Multi-anchor heuristic with distributed selection",
            parameters={"anchor_selection": "distributed"}
        )

    def _build_multi_anchor_tour(
        self,
        adjacency_matrix: List[List[float]],
        anchors: List[int]
    ) -> List[int]:
        """Build tour from multiple anchors using greedy nearest neighbor."""
        num_vertices = len(adjacency_matrix)
        unvisited = set(range(num_vertices)) - set(anchors)

        # Start from first anchor
        tour = [anchors[0]]
        current = anchors[0]

        # Add other anchors to tour
        for anchor in anchors[1:]:
            tour.append(anchor)

        # Build path through remaining vertices
        while unvisited:
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

            tour.append(nearest)
            unvisited.remove(nearest)
            current = nearest

        return tour
