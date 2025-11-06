"""
Single Anchor heuristic for TSP.

Pre-commits edges from anchor vertex and builds remaining tour greedily.
Research algorithm for anchor-based TSP investigation.

Two variants:
- v1: Single direction (entrance -> exit)
- v2: Bidirectional (tries both directions, returns best)
"""

from typing import List
import time
import numpy as np
from .base import TSPAlgorithm, TourResult, AlgorithmMetadata
from .registry import register_algorithm


@register_algorithm(
    "single_anchor_v1",
    tags=["anchor", "heuristic"],
    constraints={}
)
class SingleAnchorV1Algorithm(TSPAlgorithm):
    """
    Single anchor heuristic V1.

    Picks an arbitrary vertex and two of its lowest cost edges.
    Forms a greedy path from one (entrance vertex) to the other (exit vertex)
    using the nearest neighbor heuristic method.
    """

    def solve(self, adjacency_matrix: List[List[float]], **kwargs) -> TourResult:
        """
        Solve TSP using single anchor heuristic V1.

        Args:
            adjacency_matrix: 2D adjacency matrix
            anchor_vertex: Which vertex to use as anchor (default: determined by random_seed)
            **kwargs: unused

        Returns:
            TourResult with tour or failure if anchor edges don't lead to valid tour
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

        entrance_vertex = anchor_edge_1[1]
        exit_vertex = anchor_edge_2[1]

        # Build tour: anchor -> entrance -> ... -> exit -> anchor
        try:
            tour = self._build_tour_single_direction(
                adjacency_matrix,
                anchor_vertex,
                entrance_vertex,
                exit_vertex
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
                'entrance_vertex': entrance_vertex,
                'exit_vertex': exit_vertex,
                'anchor_edge_weights': [anchor_edge_1[0], anchor_edge_2[0]]
            },
            success=True
        )

    def get_metadata(self) -> AlgorithmMetadata:
        """Return algorithm metadata."""
        return AlgorithmMetadata(
            name="single_anchor_v1",
            version="1.0.0",
            description="Single anchor V1: entrance -> exit path with nearest neighbor",
            parameters={"anchor_vertex": "random or specified"}
        )

    def _build_tour_single_direction(
        self,
        adjacency_matrix: List[List[float]],
        anchor: int,
        entrance: int,
        exit: int
    ) -> List[int]:
        """Build tour in single direction: anchor -> entrance -> ... -> exit -> anchor."""
        num_vertices = len(adjacency_matrix)

        # Tour structure: anchor -> entrance -> ... path ... -> exit -> anchor
        unvisited = set(range(num_vertices)) - {anchor, entrance, exit}

        # Start path from entrance
        path = [entrance]
        current = entrance

        # Build path greedily using nearest neighbor to exit
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
                    # Tie-breaking: choose lowest index
                    nearest = v

            if nearest is None:
                break

            path.append(nearest)
            unvisited.remove(nearest)
            current = nearest

        # Complete tour: anchor -> entrance -> path -> exit -> anchor
        tour = [anchor] + path + [exit]

        return tour


@register_algorithm(
    "single_anchor_v2",
    tags=["anchor", "heuristic", "bidirectional"],
    constraints={}
)
class SingleAnchorV2Algorithm(TSPAlgorithm):
    """
    Single anchor heuristic V2.

    Picks an arbitrary vertex and two of its lowest cost edges.
    Tries both directions:
    1. edge1 as entrance, edge2 as exit
    2. edge2 as entrance, edge1 as exit
    Returns the tour with the lowest cost.
    """

    def solve(self, adjacency_matrix: List[List[float]], **kwargs) -> TourResult:
        """
        Solve TSP using single anchor heuristic V2.

        Args:
            adjacency_matrix: 2D adjacency matrix
            anchor_vertex: Which vertex to use as anchor (default: determined by random_seed)
            **kwargs: unused

        Returns:
            TourResult with best tour from both directions
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

        # Try direction 1: edge1 as entrance, edge2 as exit
        try:
            tour1 = self._build_tour_single_direction(
                adjacency_matrix,
                anchor_vertex,
                anchor_edge_1[1],  # entrance
                anchor_edge_2[1]   # exit
            )
            weight1 = self._compute_tour_weight(tour1, adjacency_matrix) if tour1 else float('inf')
        except Exception:
            tour1 = None
            weight1 = float('inf')

        # Try direction 2: edge2 as entrance, edge1 as exit
        try:
            tour2 = self._build_tour_single_direction(
                adjacency_matrix,
                anchor_vertex,
                anchor_edge_2[1],  # entrance
                anchor_edge_1[1]   # exit
            )
            weight2 = self._compute_tour_weight(tour2, adjacency_matrix) if tour2 else float('inf')
        except Exception:
            tour2 = None
            weight2 = float('inf')

        # Select best tour
        if weight1 <= weight2 and tour1 is not None:
            best_tour = tour1
            best_weight = weight1
            best_direction = 1
            entrance = anchor_edge_1[1]
            exit = anchor_edge_2[1]
        elif tour2 is not None:
            best_tour = tour2
            best_weight = weight2
            best_direction = 2
            entrance = anchor_edge_2[1]
            exit = anchor_edge_1[1]
        else:
            return self._create_failure_result("Failed to build valid tour in either direction")

        # Validate tour structure
        is_valid, msg = self._validate_tour_structure(best_tour, num_vertices)
        if not is_valid:
            return self._create_failure_result(msg)

        runtime = time.time() - start_time

        return TourResult(
            tour=best_tour,
            weight=best_weight,
            runtime=runtime,
            metadata={
                'anchor_vertex': int(anchor_vertex),
                'entrance_vertex': entrance,
                'exit_vertex': exit,
                'anchor_edge_weights': [anchor_edge_1[0], anchor_edge_2[0]],
                'direction_tried': best_direction,
                'weight_dir1': weight1,
                'weight_dir2': weight2
            },
            success=True
        )

    def get_metadata(self) -> AlgorithmMetadata:
        """Return algorithm metadata."""
        return AlgorithmMetadata(
            name="single_anchor_v2",
            version="1.0.0",
            description="Single anchor V2: tries both directions, returns best",
            parameters={"anchor_vertex": "random or specified"}
        )

    def _build_tour_single_direction(
        self,
        adjacency_matrix: List[List[float]],
        anchor: int,
        entrance: int,
        exit: int
    ) -> List[int]:
        """Build tour in single direction: anchor -> entrance -> ... -> exit -> anchor."""
        num_vertices = len(adjacency_matrix)

        # Tour structure: anchor -> entrance -> ... path ... -> exit -> anchor
        unvisited = set(range(num_vertices)) - {anchor, entrance, exit}

        # Start path from entrance
        path = [entrance]
        current = entrance

        # Build path greedily using nearest neighbor to exit
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
                    # Tie-breaking: choose lowest index
                    nearest = v

            if nearest is None:
                break

            path.append(nearest)
            unvisited.remove(nearest)
            current = nearest

        # Complete tour: anchor -> entrance -> path -> exit -> anchor
        tour = [anchor] + path + [exit]

        return tour


