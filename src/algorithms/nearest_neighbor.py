"""
Nearest Neighbor algorithm for TSP.

Implements the nearest neighbor heuristic with variants for different starting vertex strategies.
"""

from typing import List, Optional
import time
import numpy as np
from .base import TSPAlgorithm, TourResult, AlgorithmMetadata
from .registry import register_algorithm


@register_algorithm(
    "nearest_neighbor_random",
    tags=["baseline", "greedy", "nearest_neighbor"],
    constraints={}
)
class NearestNeighborRandom(TSPAlgorithm):
    """Nearest Neighbor starting from a random vertex."""

    def solve(self, adjacency_matrix: List[List[float]], **kwargs) -> TourResult:
        """
        Solve TSP using nearest neighbor from random start.

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

        # Run nearest neighbor from this vertex
        tour = self._nearest_neighbor_from_vertex(adjacency_matrix, start_vertex)

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
            name="nearest_neighbor_random",
            version="1.0.0",
            description="Nearest neighbor starting from random vertex",
            parameters={"start_vertex": "random"}
        )

    def _nearest_neighbor_from_vertex(
        self,
        adjacency_matrix: List[List[float]],
        start_vertex: int
    ) -> List[int]:
        """Run nearest neighbor from a specific start vertex."""
        num_vertices = len(adjacency_matrix)
        unvisited = set(range(num_vertices))
        tour = [start_vertex]
        unvisited.remove(start_vertex)

        current = start_vertex

        while unvisited:
            # Find nearest unvisited neighbor
            nearest = None
            nearest_dist = float('inf')

            for neighbor in unvisited:
                dist = adjacency_matrix[current][neighbor]
                if dist < nearest_dist:
                    nearest_dist = dist
                    nearest = neighbor
                elif dist == nearest_dist and (nearest is None or neighbor < nearest):
                    # Tie-breaking: choose lowest index
                    nearest = neighbor

            tour.append(nearest)
            unvisited.remove(nearest)
            current = nearest

        return tour


@register_algorithm(
    "nearest_neighbor_best",
    tags=["baseline", "greedy", "nearest_neighbor"],
    constraints={}
)
class NearestNeighborBest(TSPAlgorithm):
    """Nearest Neighbor trying all starting vertices and returning best."""

    def solve(self, adjacency_matrix: List[List[float]], **kwargs) -> TourResult:
        """
        Solve TSP using nearest neighbor from best starting vertex.

        Args:
            adjacency_matrix: 2D adjacency matrix
            **kwargs: unused

        Returns:
            TourResult with best tour found
        """
        start_time = time.time()

        num_vertices = len(adjacency_matrix)
        best_tour = None
        best_weight = float('inf')
        best_start = -1

        # Try from each vertex
        for start_vertex in range(num_vertices):
            tour = self._nearest_neighbor_from_vertex(adjacency_matrix, start_vertex)
            weight = self._compute_tour_weight(tour, adjacency_matrix)

            if weight < best_weight:
                best_weight = weight
                best_tour = tour
                best_start = start_vertex

        runtime = time.time() - start_time

        return TourResult(
            tour=best_tour,
            weight=best_weight,
            runtime=runtime,
            metadata={'best_start_vertex': best_start},
            success=True
        )

    def get_metadata(self) -> AlgorithmMetadata:
        """Return algorithm metadata."""
        return AlgorithmMetadata(
            name="nearest_neighbor_best",
            version="1.0.0",
            description="Nearest neighbor trying all starting vertices",
            parameters={"start_vertex": "best"}
        )

    def _nearest_neighbor_from_vertex(
        self,
        adjacency_matrix: List[List[float]],
        start_vertex: int
    ) -> List[int]:
        """Run nearest neighbor from a specific start vertex."""
        num_vertices = len(adjacency_matrix)
        unvisited = set(range(num_vertices))
        tour = [start_vertex]
        unvisited.remove(start_vertex)

        current = start_vertex

        while unvisited:
            # Find nearest unvisited neighbor
            nearest = None
            nearest_dist = float('inf')

            for neighbor in unvisited:
                dist = adjacency_matrix[current][neighbor]
                if dist < nearest_dist:
                    nearest_dist = dist
                    nearest = neighbor
                elif dist == nearest_dist and (nearest is None or neighbor < nearest):
                    # Tie-breaking: choose lowest index
                    nearest = neighbor

            tour.append(nearest)
            unvisited.remove(nearest)
            current = nearest

        return tour
