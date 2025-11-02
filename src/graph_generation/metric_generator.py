"""
Metric graph generator for non-Euclidean TSP instances.

Generates graphs that satisfy the triangle inequality but are not
necessarily embeddable in Euclidean space.
"""

import random
import heapq
from typing import List, Tuple, Optional, Literal
import numpy as np


StrategyType = Literal['mst', 'completion']


class MetricGraphGenerator:
    """
    Generator for metric TSP graphs (non-Euclidean).

    Creates graphs that satisfy the triangle inequality using
    either MST-based construction or distance matrix completion.
    """

    def __init__(self, random_seed: Optional[int] = None):
        """
        Initialize the generator.

        Args:
            random_seed: Random seed for reproducibility
        """
        self.random_seed = random_seed
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)

    def generate(
        self,
        num_vertices: int,
        weight_range: Tuple[float, float] = (1.0, 100.0),
        strategy: StrategyType = 'mst',
        metric_strictness: float = 1.0,
        is_symmetric: bool = True,
        distribution: str = 'uniform'
    ) -> List[List[float]]:
        """
        Generate a metric graph.

        Args:
            num_vertices: Number of vertices
            weight_range: (min, max) edge weights
            strategy: 'mst' or 'completion'
            metric_strictness: How tightly triangle inequality is satisfied (0.0 to 1.0)
            is_symmetric: Whether graph should be symmetric
            distribution: Weight distribution ('uniform', 'normal', 'exponential')

        Returns:
            Adjacency matrix
        """
        if num_vertices < 1:
            raise ValueError("Number of vertices must be at least 1")

        if not (0.0 <= metric_strictness <= 1.0):
            raise ValueError("Metric strictness must be between 0 and 1")

        if strategy == 'mst':
            return self._mst_based_generation(
                num_vertices, weight_range, metric_strictness,
                is_symmetric, distribution
            )
        elif strategy == 'completion':
            return self._completion_based_generation(
                num_vertices, weight_range, metric_strictness,
                is_symmetric, distribution
            )
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def _mst_based_generation(
        self,
        num_vertices: int,
        weight_range: Tuple[float, float],
        metric_strictness: float,
        is_symmetric: bool,
        distribution: str
    ) -> List[List[float]]:
        """
        Generate metric graph using MST-based construction.

        Strategy:
        1. Build a minimum spanning tree
        2. Assign weights to tree edges from distribution
        3. For non-tree edges, use path distance through tree
        4. Optionally perturb while maintaining metricity
        """
        min_weight, max_weight = weight_range

        # Step 1: Build MST using Prim's algorithm with random weights
        tree_edges = self._build_random_mst(num_vertices, weight_range, distribution)

        # Step 2: Initialize adjacency matrix with infinity
        INF = float('inf')
        matrix = [[INF] * num_vertices for _ in range(num_vertices)]

        # Set diagonal to 0
        for i in range(num_vertices):
            matrix[i][i] = 0.0

        # Step 3: Add tree edges to matrix
        for u, v, weight in tree_edges:
            matrix[u][v] = weight
            if is_symmetric:
                matrix[v][u] = weight

        # Step 4: Compute all-pairs shortest paths (Floyd-Warshall)
        # This gives us the metric completion
        for k in range(num_vertices):
            for i in range(num_vertices):
                for j in range(num_vertices):
                    if matrix[i][k] + matrix[k][j] < matrix[i][j]:
                        matrix[i][j] = matrix[i][k] + matrix[k][j]

        # Step 5: Perturb non-tree edges based on metric_strictness
        # Lower strictness = more perturbation while maintaining metricity
        if metric_strictness < 1.0:
            matrix = self._perturb_metric_weights(
                matrix, tree_edges, metric_strictness,
                weight_range, is_symmetric
            )

        return matrix

    def _build_random_mst(
        self,
        num_vertices: int,
        weight_range: Tuple[float, float],
        distribution: str
    ) -> List[Tuple[int, int, float]]:
        """Build a random minimum spanning tree."""
        min_weight, max_weight = weight_range

        # Prim's algorithm with random edge selection
        in_tree = [False] * num_vertices
        edges = []

        # Start with vertex 0
        in_tree[0] = True
        available_edges = []

        # Add edges from vertex 0 to all others
        for v in range(1, num_vertices):
            weight = self._sample_weight(min_weight, max_weight, distribution)
            heapq.heappush(available_edges, (weight, 0, v))

        # Build tree
        while len(edges) < num_vertices - 1:
            if not available_edges:
                break

            weight, u, v = heapq.heappop(available_edges)

            if in_tree[v]:
                continue

            # Add edge to tree
            edges.append((u, v, weight))
            in_tree[v] = True

            # Add new edges from v
            for w in range(num_vertices):
                if not in_tree[w]:
                    new_weight = self._sample_weight(min_weight, max_weight, distribution)
                    heapq.heappush(available_edges, (new_weight, v, w))

        return edges

    def _perturb_metric_weights(
        self,
        matrix: List[List[float]],
        tree_edges: List[Tuple[int, int, float]],
        metric_strictness: float,
        weight_range: Tuple[float, float],
        is_symmetric: bool
    ) -> List[List[float]]:
        """
        Perturb non-tree edge weights while maintaining metricity.

        Higher metric_strictness means less perturbation.
        """
        n = len(matrix)
        tree_edge_set = {(min(u, v), max(u, v)) for u, v, _ in tree_edges}

        # Calculate maximum perturbation factor
        max_perturbation = 1.0 - metric_strictness

        for i in range(n):
            for j in range(i + 1, n):
                # Skip tree edges
                if (i, j) in tree_edge_set:
                    continue

                current_weight = matrix[i][j]

                # Find the tightest constraint from triangle inequality
                # The weight can be at most the shortest path through any other vertex
                min_constraint = current_weight
                for k in range(n):
                    if k != i and k != j:
                        path_weight = matrix[i][k] + matrix[k][j]
                        min_constraint = min(min_constraint, path_weight)

                # Perturb within safe bounds
                # Can reduce the weight but not below 0
                min_safe = max(weight_range[0], current_weight * (1 - max_perturbation))
                max_safe = min(min_constraint, current_weight * (1 + max_perturbation * 0.1))

                if min_safe < max_safe:
                    new_weight = random.uniform(min_safe, max_safe)
                    matrix[i][j] = new_weight
                    if is_symmetric:
                        matrix[j][i] = new_weight

        return matrix

    def _completion_based_generation(
        self,
        num_vertices: int,
        weight_range: Tuple[float, float],
        metric_strictness: float,
        is_symmetric: bool,
        distribution: str
    ) -> List[List[float]]:
        """
        Generate metric graph using distance matrix completion.

        This approach is more computationally expensive but produces
        more random-looking metric graphs.
        """
        min_weight, max_weight = weight_range

        # Start with random initial weights
        matrix = [[0.0] * num_vertices for _ in range(num_vertices)]

        for i in range(num_vertices):
            for j in range(i + 1, num_vertices):
                weight = self._sample_weight(min_weight, max_weight, distribution)
                matrix[i][j] = weight
                if is_symmetric:
                    matrix[j][i] = weight

        # Apply Floyd-Warshall to enforce metric property
        # This reduces weights to satisfy triangle inequality
        for k in range(num_vertices):
            for i in range(num_vertices):
                for j in range(num_vertices):
                    if i != j and matrix[i][j] > matrix[i][k] + matrix[k][j]:
                        matrix[i][j] = matrix[i][k] + matrix[k][j]
                        if is_symmetric:
                            matrix[j][i] = matrix[i][j]

        return matrix

    def _sample_weight(
        self,
        min_weight: float,
        max_weight: float,
        distribution: str
    ) -> float:
        """Sample a weight from the specified distribution."""
        if distribution == 'uniform':
            return random.uniform(min_weight, max_weight)

        elif distribution == 'normal':
            # Use normal distribution centered in range
            mean = (min_weight + max_weight) / 2
            std = (max_weight - min_weight) / 6  # ~99% within range
            value = np.random.normal(mean, std)
            return np.clip(value, min_weight, max_weight)

        elif distribution == 'exponential':
            # Exponential distribution scaled to range
            scale = (max_weight - min_weight) / 3
            value = min_weight + np.random.exponential(scale)
            return min(value, max_weight)

        else:
            raise ValueError(f"Unknown distribution: {distribution}")


class QuasiMetricGraphGenerator:
    """
    Generator for quasi-metric (asymmetric metric) graphs.

    These graphs have directional edge weights but still satisfy
    a form of triangle inequality.
    """

    def __init__(self, random_seed: Optional[int] = None):
        """Initialize the generator."""
        self.random_seed = random_seed
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)

    def generate(
        self,
        num_vertices: int,
        weight_range: Tuple[float, float] = (1.0, 100.0),
        asymmetry_factor: float = 0.2
    ) -> List[List[float]]:
        """
        Generate a quasi-metric graph.

        Args:
            num_vertices: Number of vertices
            weight_range: (min, max) edge weights
            asymmetry_factor: How much forward/backward weights can differ (0.0 to 1.0)

        Returns:
            Asymmetric adjacency matrix
        """
        # Start with a symmetric metric graph
        metric_gen = MetricGraphGenerator(random_seed=self.random_seed)
        base_matrix = metric_gen.generate(
            num_vertices=num_vertices,
            weight_range=weight_range,
            strategy='mst',
            is_symmetric=True
        )

        # Add controlled asymmetry
        # Apply perturbations to pairs of edges (i,j) and (j,i) separately
        matrix = [[0.0] * num_vertices for _ in range(num_vertices)]

        for i in range(num_vertices):
            for j in range(i + 1, num_vertices):
                # Get base symmetric weight
                base_weight = base_matrix[i][j]

                # Apply independent perturbations for forward and backward directions
                perturb_forward = random.uniform(-asymmetry_factor, asymmetry_factor)
                perturb_backward = random.uniform(-asymmetry_factor, asymmetry_factor)

                # Calculate new weights and clamp to range
                weight_ij = base_weight * (1 + perturb_forward)
                weight_ji = base_weight * (1 + perturb_backward)

                matrix[i][j] = max(weight_range[0], min(weight_range[1], weight_ij))
                matrix[j][i] = max(weight_range[0], min(weight_range[1], weight_ji))

        # Ensure triangle inequality holds after perturbations
        # Run this multiple times if needed to ensure convergence
        # because clamping before Floyd-Warshall can create issues
        for _ in range(2):  # Run twice to ensure full enforcement
            self._enforce_triangle_inequality(matrix)

        return matrix

    def _enforce_triangle_inequality(self, matrix: List[List[float]]) -> None:
        """Enforce triangle inequality in-place using Floyd-Warshall."""
        n = len(matrix)
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if matrix[i][j] > matrix[i][k] + matrix[k][j]:
                        matrix[i][j] = matrix[i][k] + matrix[k][j]


def generate_metric_graph(
    num_vertices: int,
    weight_range: Tuple[float, float] = (1.0, 100.0),
    strategy: StrategyType = 'mst',
    metric_strictness: float = 1.0,
    is_symmetric: bool = True,
    distribution: str = 'uniform',
    random_seed: Optional[int] = None
) -> List[List[float]]:
    """
    Convenience function to generate a metric graph.

    Args:
        num_vertices: Number of vertices
        weight_range: (min, max) edge weights
        strategy: Generation strategy ('mst' or 'completion')
        metric_strictness: How tightly triangle inequality is satisfied
        is_symmetric: Whether graph should be symmetric
        distribution: Weight distribution
        random_seed: Random seed for reproducibility

    Returns:
        Adjacency matrix
    """
    generator = MetricGraphGenerator(random_seed=random_seed)
    return generator.generate(
        num_vertices=num_vertices,
        weight_range=weight_range,
        strategy=strategy,
        metric_strictness=metric_strictness,
        is_symmetric=is_symmetric,
        distribution=distribution
    )


def generate_quasi_metric_graph(
    num_vertices: int,
    weight_range: Tuple[float, float] = (1.0, 100.0),
    asymmetry_factor: float = 0.2,
    random_seed: Optional[int] = None
) -> List[List[float]]:
    """
    Convenience function to generate a quasi-metric graph.

    Args:
        num_vertices: Number of vertices
        weight_range: (min, max) edge weights
        asymmetry_factor: Degree of asymmetry
        random_seed: Random seed for reproducibility

    Returns:
        Asymmetric adjacency matrix
    """
    generator = QuasiMetricGraphGenerator(random_seed=random_seed)
    return generator.generate(
        num_vertices=num_vertices,
        weight_range=weight_range,
        asymmetry_factor=asymmetry_factor
    )
