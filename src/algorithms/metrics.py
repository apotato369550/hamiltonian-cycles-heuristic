"""
Quality metrics computation for TSP solutions.

Provides functions to compute tour weights, statistical metrics,
optimality gaps, and comparative performance metrics.
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np


@dataclass
class TourStatistics:
    """Statistics about a set of tours."""
    mean_weight: float
    median_weight: float
    std_weight: float
    min_weight: float
    max_weight: float
    weight_range: Tuple[float, float]
    count: int


@dataclass
class PerformanceComparison:
    """Comparison between two algorithms' performance."""
    algorithm_a: str
    algorithm_b: str
    wins_a: int
    wins_b: int
    ties: int
    mean_improvement_a: float  # Percentage improvement of A over B
    median_improvement_a: float


def compute_tour_weight(tour: List[int], adjacency_matrix: List[List[float]]) -> float:
    """
    Compute total weight of a tour.

    Args:
        tour: List of vertex indices forming a Hamiltonian cycle
        adjacency_matrix: 2D adjacency matrix

    Returns:
        Sum of edge weights in the tour
    """
    if len(tour) < 2:
        return 0.0

    total_weight = 0.0
    for i in range(len(tour)):
        current = tour[i]
        next_vertex = tour[(i + 1) % len(tour)]
        total_weight += adjacency_matrix[current][next_vertex]

    return float(total_weight)


def compute_tour_statistics(weights: List[float]) -> TourStatistics:
    """
    Compute statistical summary of tour weights.

    Args:
        weights: List of tour weights

    Returns:
        TourStatistics object with computed metrics
    """
    if not weights:
        return TourStatistics(
            mean_weight=0.0,
            median_weight=0.0,
            std_weight=0.0,
            min_weight=0.0,
            max_weight=0.0,
            weight_range=(0.0, 0.0),
            count=0
        )

    weights_array = np.array(weights)

    return TourStatistics(
        mean_weight=float(np.mean(weights_array)),
        median_weight=float(np.median(weights_array)),
        std_weight=float(np.std(weights_array)),
        min_weight=float(np.min(weights_array)),
        max_weight=float(np.max(weights_array)),
        weight_range=(float(np.min(weights_array)), float(np.max(weights_array))),
        count=len(weights)
    )


def compute_optimality_gap(
    heuristic_weight: float,
    optimal_weight: float,
    epsilon: float = 1e-9
) -> float:
    """
    Compute optimality gap as percentage.

    Args:
        heuristic_weight: Weight found by heuristic algorithm
        optimal_weight: Known optimal weight
        epsilon: Small value to prevent division by zero

    Returns:
        Optimality gap as percentage (0 = optimal, >0 = suboptimal)
    """
    if optimal_weight == 0:
        if heuristic_weight == 0:
            return 0.0
        else:
            return float('inf')

    gap = (heuristic_weight - optimal_weight) / (optimal_weight + epsilon)
    return max(0.0, gap * 100.0)


def compute_approximation_ratio(
    heuristic_weight: float,
    optimal_weight: float,
    epsilon: float = 1e-9
) -> float:
    """
    Compute approximation ratio (heuristic / optimal).

    Args:
        heuristic_weight: Weight found by heuristic
        optimal_weight: Known optimal weight
        epsilon: Small value to prevent division by zero

    Returns:
        Approximation ratio (1.0 = optimal, >1.0 = suboptimal)
    """
    if optimal_weight == 0:
        if heuristic_weight == 0:
            return 1.0
        else:
            return float('inf')

    return heuristic_weight / (optimal_weight + epsilon)


def compute_relative_performance(
    algorithm_weights: Dict[str, List[float]],
    baseline_name: str = "nearest_neighbor"
) -> Dict[str, Dict[str, float]]:
    """
    Compute performance of all algorithms relative to a baseline.

    Args:
        algorithm_weights: Dict mapping algorithm names to lists of tour weights
        baseline_name: Name of baseline algorithm to compare against

    Returns:
        Dict mapping algorithm names to relative performance metrics
    """
    if baseline_name not in algorithm_weights:
        raise ValueError(f"Baseline algorithm '{baseline_name}' not found")

    baseline_weights = np.array(algorithm_weights[baseline_name])
    results = {}

    for algo_name, weights in algorithm_weights.items():
        weights_array = np.array(weights)

        # Compute improvements (negative = worse, positive = better)
        improvements = (baseline_weights - weights_array) / baseline_weights * 100

        results[algo_name] = {
            'mean_improvement_percent': float(np.mean(improvements)),
            'median_improvement_percent': float(np.median(improvements)),
            'std_improvement_percent': float(np.std(improvements)),
            'min_improvement_percent': float(np.min(improvements)),
            'max_improvement_percent': float(np.max(improvements)),
            'wins': int(np.sum(weights_array < baseline_weights)),
            'losses': int(np.sum(weights_array > baseline_weights)),
            'ties': int(np.sum(np.abs(weights_array - baseline_weights) < 1e-9)),
        }

    return results


def compute_tour_properties(
    tour: List[int],
    adjacency_matrix: List[List[float]]
) -> Dict[str, Any]:
    """
    Compute additional tour properties beyond just weight.

    Args:
        tour: Tour (list of vertex indices)
        adjacency_matrix: Adjacency matrix

    Returns:
        Dict of computed properties
    """
    if len(tour) < 2:
        return {}

    edges = []
    for i in range(len(tour)):
        current = tour[i]
        next_vertex = tour[(i + 1) % len(tour)]
        weight = adjacency_matrix[current][next_vertex]
        edges.append(weight)

    edges_array = np.array(edges)

    properties = {
        'total_weight': float(np.sum(edges_array)),
        'num_edges': len(edges),
        'max_edge_weight': float(np.max(edges_array)),
        'min_edge_weight': float(np.min(edges_array)),
        'mean_edge_weight': float(np.mean(edges_array)),
        'std_edge_weight': float(np.std(edges_array)),
        'edge_weight_range': (float(np.min(edges_array)), float(np.max(edges_array))),
        # "Smoothness" = inverse of coefficient of variation
        'smoothness': float(np.mean(edges_array) / (np.std(edges_array) + 1e-9)),
        # Number of edges above median weight
        'num_above_median': int(np.sum(edges_array > np.median(edges_array))),
    }

    return properties


class MetricsCalculator:
    """Class for computing and caching expensive metrics."""

    def __init__(self, cache_results: bool = True):
        """
        Initialize metrics calculator.

        Args:
            cache_results: Whether to cache expensive computations
        """
        self.cache_results = cache_results
        self._weight_cache: Dict = {}
        self._properties_cache: Dict = {}

    def compute_weight(
        self,
        tour: List[int],
        adjacency_matrix: List[List[float]],
        use_cache: bool = True
    ) -> float:
        """
        Compute tour weight, using cache if enabled.

        Args:
            tour: Tour
            adjacency_matrix: Adjacency matrix
            use_cache: Whether to use cache

        Returns:
            Tour weight
        """
        if self.cache_results and use_cache:
            tour_key = tuple(tour)
            if tour_key in self._weight_cache:
                return self._weight_cache[tour_key]

        weight = compute_tour_weight(tour, adjacency_matrix)

        if self.cache_results:
            self._weight_cache[tuple(tour)] = weight

        return weight

    def compute_properties(
        self,
        tour: List[int],
        adjacency_matrix: List[List[float]],
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        Compute tour properties, using cache if enabled.

        Args:
            tour: Tour
            adjacency_matrix: Adjacency matrix
            use_cache: Whether to use cache

        Returns:
            Properties dict
        """
        if self.cache_results and use_cache:
            tour_key = tuple(tour)
            if tour_key in self._properties_cache:
                return self._properties_cache[tour_key]

        props = compute_tour_properties(tour, adjacency_matrix)

        if self.cache_results:
            self._properties_cache[tuple(tour)] = props

        return props

    def clear_cache(self) -> None:
        """Clear all caches."""
        self._weight_cache = {}
        self._properties_cache = {}
