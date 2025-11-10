"""
Anchor Quality Labeling System for TSP Vertex Classification.

This module provides systems for assigning quality scores to vertices based
on their performance as TSP tour anchor points. These labels serve as the
target variable for machine learning models.

Implements multiple labeling strategies:
- Absolute quality (based on tour weights)
- Rank-based percentiles
- Binary classification (good/bad anchors)
- Multi-class classification (excellent/good/mediocre/poor)
- Relative to optimal (ratio to known optimal solution)
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import time

# Import Phase 2 algorithm infrastructure
from src.algorithms.registry import AlgorithmRegistry
from src.algorithms.base import TourResult


class LabelingStrategy(Enum):
    """Strategies for assigning quality scores to anchor vertices."""
    ABSOLUTE_QUALITY = "absolute"      # Score = 1 / tour_weight
    RANK_BASED = "rank"                # Percentile ranks (0-100)
    BINARY = "binary"                  # Good (1) vs bad (0) anchors
    MULTICLASS = "multiclass"          # 4 classes: excellent/good/mediocre/poor
    RELATIVE_TO_OPTIMAL = "optimal"    # Ratio to optimal tour weight


@dataclass
class LabelingResult:
    """
    Result of anchor quality labeling.

    Attributes:
        labels: Quality scores for each vertex (shape: n_vertices)
        tour_weights: Tour weights for each vertex as anchor (shape: n_vertices)
        successful_vertices: Indices of vertices with successful tour construction
        failed_vertices: Indices of vertices where tour construction failed
        metadata: Additional information about labeling process
    """
    labels: np.ndarray
    tour_weights: np.ndarray
    successful_vertices: np.ndarray
    failed_vertices: np.ndarray
    metadata: Dict[str, Any]

    def __post_init__(self):
        """Validate result consistency."""
        if len(self.labels) != len(self.tour_weights):
            raise ValueError(
                f"Label count ({len(self.labels)}) must match "
                f"tour weight count ({len(self.tour_weights)})"
            )


class AnchorQualityLabeler:
    """
    Assigns quality scores to vertices based on anchor-based algorithm performance.

    The labeler runs a specified TSP algorithm (typically single_anchor) from each
    vertex and assigns quality scores based on resulting tour weights.

    Different labeling strategies support different ML formulations:
    - Regression: Use ABSOLUTE_QUALITY or RANK_BASED
    - Binary classification: Use BINARY
    - Multi-class: Use MULTICLASS
    - Optimality-aware: Use RELATIVE_TO_OPTIMAL (requires known optimal)
    """

    def __init__(
        self,
        strategy: LabelingStrategy,
        algorithm_name: str = 'single_anchor',
        top_k_percent: float = 10.0,
        random_seed: Optional[int] = None
    ):
        """
        Initialize the labeler.

        Args:
            strategy: Which labeling strategy to use
            algorithm_name: Name of algorithm in registry (must accept anchor_vertex param)
            top_k_percent: For BINARY strategy, percentage of vertices labeled as positive
            random_seed: Random seed for reproducibility
        """
        self.strategy = strategy
        self.algorithm_name = algorithm_name
        self.top_k_percent = top_k_percent
        self.random_seed = random_seed

        # Validate algorithm exists
        if not AlgorithmRegistry.is_registered(algorithm_name):
            raise ValueError(f"Algorithm '{algorithm_name}' not registered")

    def label_vertices(
        self,
        graph: np.ndarray,
        optimal_weight: Optional[float] = None,
        timeout_per_vertex: Optional[float] = None
    ) -> LabelingResult:
        """
        Assign quality labels to all vertices in the graph.

        Args:
            graph: NxN adjacency matrix
            optimal_weight: Known optimal tour weight (required for RELATIVE_TO_OPTIMAL)
            timeout_per_vertex: Maximum time per vertex (seconds)

        Returns:
            LabelingResult with labels and metadata

        Raises:
            ValueError: If optimal_weight required but not provided
        """
        n = len(graph)
        start_time = time.time()

        # Run algorithm from each vertex as anchor
        tour_weights = np.full(n, np.inf, dtype=float)
        successful_vertices = []
        failed_vertices = []

        for vertex in range(n):
            try:
                # Get algorithm instance
                algo = AlgorithmRegistry.get_algorithm(
                    self.algorithm_name,
                    random_seed=self.random_seed
                )

                # Run with this vertex as anchor
                result: TourResult = algo.solve(graph, anchor_vertex=vertex)

                if result.success:
                    tour_weights[vertex] = result.weight
                    successful_vertices.append(vertex)
                else:
                    failed_vertices.append(vertex)

            except Exception as e:
                # Handle failures gracefully
                failed_vertices.append(vertex)
                continue

        successful_vertices = np.array(successful_vertices, dtype=int)
        failed_vertices = np.array(failed_vertices, dtype=int)

        # Compute labels based on strategy
        labels = self._compute_labels(
            tour_weights,
            successful_vertices,
            optimal_weight
        )

        # Build metadata
        metadata = {
            'strategy': self.strategy.value,
            'algorithm': self.algorithm_name,
            'labeling_time': time.time() - start_time,
            'successful_count': len(successful_vertices),
            'failed_count': len(failed_vertices),
            'best_vertex': int(np.argmin(tour_weights)) if len(successful_vertices) > 0 else None,
            'best_tour_weight': float(np.min(tour_weights)) if len(successful_vertices) > 0 else None,
            'worst_tour_weight': float(np.max(tour_weights[successful_vertices])) if len(successful_vertices) > 0 else None,
        }

        return LabelingResult(
            labels=labels,
            tour_weights=tour_weights,
            successful_vertices=successful_vertices,
            failed_vertices=failed_vertices,
            metadata=metadata
        )

    def _compute_labels(
        self,
        tour_weights: np.ndarray,
        successful_vertices: np.ndarray,
        optimal_weight: Optional[float]
    ) -> np.ndarray:
        """
        Compute labels from tour weights based on strategy.

        Args:
            tour_weights: Tour weight for each vertex (inf for failed)
            successful_vertices: Indices of successful vertices
            optimal_weight: Known optimal (if available)

        Returns:
            Label array (shape: n_vertices)
        """
        n = len(tour_weights)
        labels = np.zeros(n, dtype=float)

        if len(successful_vertices) == 0:
            # All vertices failed - return zeros
            return labels

        if self.strategy == LabelingStrategy.ABSOLUTE_QUALITY:
            return self._absolute_quality(tour_weights, successful_vertices)

        elif self.strategy == LabelingStrategy.RANK_BASED:
            return self._rank_based(tour_weights, successful_vertices)

        elif self.strategy == LabelingStrategy.BINARY:
            return self._binary_classification(tour_weights, successful_vertices)

        elif self.strategy == LabelingStrategy.MULTICLASS:
            return self._multiclass_classification(tour_weights, successful_vertices)

        elif self.strategy == LabelingStrategy.RELATIVE_TO_OPTIMAL:
            if optimal_weight is None:
                raise ValueError("optimal_weight required for RELATIVE_TO_OPTIMAL strategy")
            return self._relative_to_optimal(tour_weights, successful_vertices, optimal_weight)

        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def _absolute_quality(
        self,
        tour_weights: np.ndarray,
        successful_vertices: np.ndarray
    ) -> np.ndarray:
        """
        Compute absolute quality scores: higher is better.

        Uses formula: score = 1 / tour_weight
        Failed vertices get score of 0.

        Args:
            tour_weights: Tour weights (inf for failed)
            successful_vertices: Successful vertex indices

        Returns:
            Quality scores
        """
        labels = np.zeros(len(tour_weights), dtype=float)

        # Only assign scores to successful vertices
        for v in successful_vertices:
            if tour_weights[v] > 0 and np.isfinite(tour_weights[v]):
                labels[v] = 1.0 / tour_weights[v]

        return labels

    def _rank_based(
        self,
        tour_weights: np.ndarray,
        successful_vertices: np.ndarray
    ) -> np.ndarray:
        """
        Compute rank-based percentile scores (0-100).

        Best vertex gets 100, worst gets 0. Handles ties using average ranks.

        Args:
            tour_weights: Tour weights
            successful_vertices: Successful vertex indices

        Returns:
            Percentile ranks
        """
        labels = np.zeros(len(tour_weights), dtype=float)

        if len(successful_vertices) == 0:
            return labels

        # Extract weights for successful vertices
        weights = tour_weights[successful_vertices]

        # Rank them (lower weight = better = higher rank)
        # Use scipy's rankdata for tie handling
        from scipy.stats import rankdata
        ranks = rankdata(weights, method='average')

        # Convert to percentiles (0-100 scale, higher is better)
        if len(ranks) == 1:
            percentiles = np.array([100.0])
        else:
            # Invert ranks so best gets highest percentile
            percentiles = 100.0 * (len(ranks) - ranks + 1) / len(ranks)

        # Assign percentiles to successful vertices
        for i, v in enumerate(successful_vertices):
            labels[v] = percentiles[i]

        return labels

    def _binary_classification(
        self,
        tour_weights: np.ndarray,
        successful_vertices: np.ndarray
    ) -> np.ndarray:
        """
        Binary labels: top k% are positive (1), rest are negative (0).

        Args:
            tour_weights: Tour weights
            successful_vertices: Successful vertex indices

        Returns:
            Binary labels (0 or 1)
        """
        labels = np.zeros(len(tour_weights), dtype=float)

        if len(successful_vertices) == 0:
            return labels

        # Compute threshold for top k%
        weights = tour_weights[successful_vertices]
        k_count = max(1, int(np.ceil(len(weights) * self.top_k_percent / 100.0)))

        # Find k-th smallest weight (threshold)
        sorted_weights = np.sort(weights)
        threshold = sorted_weights[k_count - 1]

        # Assign positive label to vertices with weight <= threshold
        for v in successful_vertices:
            if tour_weights[v] <= threshold:
                labels[v] = 1.0

        return labels

    def _multiclass_classification(
        self,
        tour_weights: np.ndarray,
        successful_vertices: np.ndarray
    ) -> np.ndarray:
        """
        Multi-class labels:
        - Class 3: Excellent (top 10%)
        - Class 2: Good (10-30%)
        - Class 1: Mediocre (30-70%)
        - Class 0: Poor (70-100%)

        Args:
            tour_weights: Tour weights
            successful_vertices: Successful vertex indices

        Returns:
            Class labels (0-3)
        """
        labels = np.zeros(len(tour_weights), dtype=float)

        if len(successful_vertices) == 0:
            return labels

        # Compute percentile ranks first
        weights = tour_weights[successful_vertices]
        from scipy.stats import rankdata
        ranks = rankdata(weights, method='average')

        # Convert to percentiles (0-100, higher is better)
        if len(ranks) == 1:
            percentiles = np.array([100.0])
        else:
            percentiles = 100.0 * (len(ranks) - ranks + 1) / len(ranks)

        # Assign classes based on percentile thresholds
        for i, v in enumerate(successful_vertices):
            p = percentiles[i]
            if p >= 90:
                labels[v] = 3  # Excellent
            elif p >= 70:
                labels[v] = 2  # Good
            elif p >= 30:
                labels[v] = 1  # Mediocre
            else:
                labels[v] = 0  # Poor

        return labels

    def _relative_to_optimal(
        self,
        tour_weights: np.ndarray,
        successful_vertices: np.ndarray,
        optimal_weight: float
    ) -> np.ndarray:
        """
        Compute quality as ratio to optimal: tour_weight / optimal_weight.

        Perfect anchor has score 1.0, worse anchors have scores > 1.0.
        Failed vertices get very large score.

        Args:
            tour_weights: Tour weights
            successful_vertices: Successful vertex indices
            optimal_weight: Known optimal tour weight

        Returns:
            Optimality ratios
        """
        if optimal_weight <= 0:
            raise ValueError("optimal_weight must be positive")

        labels = np.full(len(tour_weights), 10.0, dtype=float)  # Large default for failed

        # Compute ratios for successful vertices
        for v in successful_vertices:
            if np.isfinite(tour_weights[v]) and tour_weights[v] > 0:
                labels[v] = tour_weights[v] / optimal_weight

        return labels

    def __repr__(self) -> str:
        return (
            f"AnchorQualityLabeler(strategy={self.strategy.value}, "
            f"algorithm={self.algorithm_name})"
        )
