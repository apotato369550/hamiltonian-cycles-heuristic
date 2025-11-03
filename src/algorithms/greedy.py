"""
Greedy Edge-Picking algorithm for TSP.

Implements the classic greedy edge-picking algorithm which iteratively
adds the cheapest edges that maintain valid Hamiltonian cycle structure.
"""

from typing import List, Tuple
import time
from .base import TSPAlgorithm, TourResult, AlgorithmMetadata
from .registry import register_algorithm


class UnionFind:
    """Union-Find (Disjoint Set Union) data structure for cycle detection."""

    def __init__(self, n: int):
        """Initialize union-find with n elements."""
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        """Find the root of element x with path compression."""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: int, y: int) -> bool:
        """
        Union two sets. Return True if union happened, False if already same set.
        """
        root_x = self.find(x)
        root_y = self.find(y)

        if root_x == root_y:
            return False

        # Union by rank
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        else:
            self.parent[root_y] = root_x
            self.rank[root_x] += 1

        return True

    def same_set(self, x: int, y: int) -> bool:
        """Check if x and y are in the same set."""
        return self.find(x) == self.find(y)


@register_algorithm(
    "greedy_edge",
    tags=["baseline", "greedy"],
    constraints={}
)
class GreedyEdgeAlgorithm(TSPAlgorithm):
    """Greedy edge-picking algorithm for TSP."""

    def solve(self, adjacency_matrix: List[List[float]], **kwargs) -> TourResult:
        """
        Solve TSP using greedy edge-picking algorithm.

        Args:
            adjacency_matrix: 2D adjacency matrix
            **kwargs: unused

        Returns:
            TourResult with tour found or failure
        """
        start_time = time.time()

        num_vertices = len(adjacency_matrix)

        # Collect all edges with their weights
        edges = []
        for i in range(num_vertices):
            for j in range(i + 1, num_vertices):
                # Add both directions
                edges.append((adjacency_matrix[i][j], i, j))
                if adjacency_matrix[i][j] != adjacency_matrix[j][i]:
                    edges.append((adjacency_matrix[j][i], j, i))

        # Sort edges by weight
        edges.sort()

        # Build tour using greedy approach with union-find for cycle detection
        degree = [0] * num_vertices  # Track vertex degrees
        edge_list = [[] for _ in range(num_vertices)]  # Adjacency list for result
        uf = UnionFind(num_vertices)

        edges_added = 0
        target_edges = num_vertices  # We need exactly n edges for a Hamiltonian cycle

        for weight, u, v in edges:
            # Don't add if either vertex already has degree 2
            if degree[u] >= 2 or degree[v] >= 2:
                continue

            # Don't add if it would create a cycle before we have all vertices
            if uf.same_set(u, v) and edges_added < target_edges - 1:
                continue

            # Don't add if it would create a cycle with a degree-2 vertex not connected to complete it
            if uf.same_set(u, v) and edges_added < target_edges:
                # Check if this would be the final edge
                if not self._would_complete_tour(edge_list, u, v, num_vertices):
                    continue

            # Add this edge
            edge_list[u].append(v)
            edge_list[v].append(u)
            degree[u] += 1
            degree[v] += 1
            uf.union(u, v)
            edges_added += 1

            if edges_added == target_edges:
                break

        runtime = time.time() - start_time

        # Check if we successfully built a Hamiltonian cycle
        if edges_added < target_edges:
            return self._create_failure_result(
                f"Could not build Hamiltonian cycle: only added {edges_added}/{target_edges} edges"
            )

        # Extract tour from edge list
        tour = self._extract_tour(edge_list)

        if not tour or len(tour) != num_vertices:
            return self._create_failure_result("Failed to extract valid tour")

        weight = self._compute_tour_weight(tour, adjacency_matrix)

        return TourResult(
            tour=tour,
            weight=weight,
            runtime=runtime,
            metadata={'edges_added': edges_added},
            success=True
        )

    def get_metadata(self) -> AlgorithmMetadata:
        """Return algorithm metadata."""
        return AlgorithmMetadata(
            name="greedy_edge",
            version="1.0.0",
            description="Greedy edge-picking algorithm"
        )

    def _would_complete_tour(
        self,
        edge_list: List[List[int]],
        u: int,
        v: int,
        num_vertices: int
    ) -> bool:
        """Check if adding edge (u,v) would complete a valid tour."""
        # Check if this edge would give u and v degree 2 and form a cycle
        # that includes all vertices

        if len(edge_list[u]) == 0 or len(edge_list[v]) == 0:
            # If either vertex has no edges yet, can't complete
            return False

        # Both vertices should end up with degree 2
        if len(edge_list[u]) >= 2 or len(edge_list[v]) >= 2:
            return False

        # Do a traversal to see if we'd have all vertices
        visited = set()
        current = u
        visited.add(current)

        # Traverse through existing edges
        while True:
            neighbors = edge_list[current]
            next_vertex = None

            for neighbor in neighbors:
                if neighbor not in visited:
                    next_vertex = neighbor
                    break

            if next_vertex is None:
                # Check if adding (u, v) would let us continue
                if current == u and v not in visited:
                    next_vertex = v
                elif current == v and u not in visited:
                    next_vertex = u
                else:
                    break

            if next_vertex is None:
                break

            visited.add(next_vertex)
            current = next_vertex

        return len(visited) == num_vertices

    def _extract_tour(self, edge_list: List[List[int]]) -> List[int]:
        """Extract Hamiltonian cycle from edge list."""
        if not edge_list:
            return []

        num_vertices = len(edge_list)

        # Start from vertex 0
        tour = [0]
        current = 0
        previous = -1

        while len(tour) < num_vertices:
            # Find next unvisited neighbor
            neighbors = edge_list[current]

            if not neighbors:
                break

            next_vertex = None
            for neighbor in neighbors:
                if neighbor != previous:
                    next_vertex = neighbor
                    break

            if next_vertex is None:
                break

            if next_vertex in tour and len(tour) < num_vertices:
                break

            tour.append(next_vertex)
            previous = current
            current = next_vertex

        # Verify tour is valid
        if len(tour) == num_vertices and tour[-1] in edge_list[tour[0]]:
            return tour
        else:
            return []
