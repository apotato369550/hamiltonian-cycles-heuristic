"""
Heuristic-specific vertex feature extraction.

Extracts features directly inspired by how anchor-based TSP heuristics work.
These features capture properties relevant to anchor selection.
"""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from ..base import VertexFeatureExtractor


class HeuristicFeatureExtractor(VertexFeatureExtractor):
    """
    Extracts features specific to anchor-based heuristics.

    Features include:
    - Anchor edge features (two cheapest edges)
    - Tour construction estimates
    - Constraint features (edge directionality)
    - Baseline comparison features

    Note: Some features use fast heuristics to estimate tour quality,
    which may be "cheating" if used to predict anchor quality. However,
    if these features are cheaper to compute than exhaustive anchor search,
    they're still useful for prediction.
    """

    def __init__(
        self,
        include_tour_estimates: bool = True,
        include_baseline_comparison: bool = True,
        name: str = "heuristic"
    ):
        """
        Initialize heuristic feature extractor.

        Args:
            include_tour_estimates: Compute fast tour quality estimates
            include_baseline_comparison: Compare to baseline algorithms
            name: Extractor name
        """
        super().__init__(name)
        self.include_tour_estimates = include_tour_estimates
        self.include_baseline_comparison = include_baseline_comparison

    def extract(
        self,
        graph: np.ndarray,
        cache: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Extract heuristic-specific features for all vertices.

        Args:
            graph: NxN adjacency matrix
            cache: Optional cache

        Returns:
            features: NxF feature matrix
            feature_names: List of feature names
        """
        n = graph.shape[0]
        features_list = []

        for i in range(n):
            vertex_features = []

            # Anchor edge features
            anchor_features = self._extract_anchor_edge_features(i, graph, n)
            vertex_features.extend(anchor_features)

            # Tour construction estimates
            if self.include_tour_estimates:
                tour_features = self._extract_tour_estimate_features(i, graph, n)
                vertex_features.extend(tour_features)

            # Baseline comparison
            if self.include_baseline_comparison:
                baseline_features = self._extract_baseline_features(i, graph, n)
                vertex_features.extend(baseline_features)

            features_list.append(vertex_features)

        features = np.array(features_list)
        feature_names = self.get_feature_names()

        return features, feature_names

    def _extract_anchor_edge_features(
        self,
        vertex: int,
        graph: np.ndarray,
        n: int
    ) -> List[float]:
        """
        Extract features based on anchor edges (two cheapest edges).

        Args:
            vertex: Vertex index
            graph: Adjacency matrix
            n: Number of vertices

        Returns:
            List of anchor edge features
        """
        # Get distances to all other vertices
        distances = np.concatenate([graph[vertex, :vertex], graph[vertex, vertex+1:]])

        if len(distances) < 2:
            return [0.0] * 8

        # Sort to find cheapest edges
        sorted_distances = np.sort(distances)

        # Two cheapest edges (anchor edges)
        edge1 = sorted_distances[0]
        edge2 = sorted_distances[1]

        # Sum and product
        sum_anchor_edges = edge1 + edge2
        product_anchor_edges = edge1 * edge2

        # Ratio
        ratio_anchor_edges = edge2 / edge1 if edge1 > 0 else 0.0

        # Gap to third cheapest (if exists)
        if len(sorted_distances) >= 3:
            edge3 = sorted_distances[2]
            gap_to_third = edge3 - edge2
        else:
            gap_to_third = 0.0

        # Average of remaining edges (non-anchor)
        if len(sorted_distances) > 2:
            avg_remaining = np.mean(sorted_distances[2:])
        else:
            avg_remaining = 0.0

        # Anchor quality score: ratio of anchor sum to average edge
        avg_all = np.mean(sorted_distances)
        anchor_quality = sum_anchor_edges / avg_all if avg_all > 0 else 0.0

        # Relative gap: gap normalized by edge2
        relative_gap = gap_to_third / edge2 if edge2 > 0 else 0.0

        return [
            edge1,
            edge2,
            sum_anchor_edges,
            product_anchor_edges,
            ratio_anchor_edges,
            gap_to_third,
            anchor_quality,
            relative_gap
        ]

    def _extract_tour_estimate_features(
        self,
        vertex: int,
        graph: np.ndarray,
        n: int
    ) -> List[float]:
        """
        Extract fast tour quality estimates.

        Uses greedy construction heuristics to estimate tour quality
        if starting from this vertex as anchor.

        Args:
            vertex: Vertex index
            graph: Adjacency matrix
            n: Number of vertices

        Returns:
            List of tour estimate features
        """
        # Estimate 1: Greedy nearest neighbor from this vertex
        nn_estimate = self._greedy_nearest_neighbor_estimate(vertex, graph, n)

        # Estimate 2: Simple lower bound based on anchor edges
        # Lower bound: sum of all cheapest edges + small penalty
        distances = graph[vertex, :]
        sorted_distances = np.sort(distances)
        if len(sorted_distances) >= 2:
            # Use sum of two cheapest as starting point
            anchor_sum = sorted_distances[1] + sorted_distances[2]  # Skip self-loop at 0
            # Add average of remaining as rough estimate
            remaining_avg = np.mean(sorted_distances[3:]) if len(sorted_distances) > 3 else 0
            lower_bound = anchor_sum + (n - 2) * remaining_avg
        else:
            lower_bound = 0.0

        # Estimate 3: Ratio of estimate to graph average (normalized quality)
        avg_edge = np.mean(graph[graph > 0])
        if avg_edge > 0 and nn_estimate > 0:
            normalized_estimate = nn_estimate / (n * avg_edge)
        else:
            normalized_estimate = 0.0

        return [nn_estimate, lower_bound, normalized_estimate]

    def _greedy_nearest_neighbor_estimate(
        self,
        start: int,
        graph: np.ndarray,
        n: int
    ) -> float:
        """
        Estimate tour cost using greedy nearest neighbor from start vertex.

        This is a very fast O(n²) heuristic.

        Args:
            start: Starting vertex
            graph: Adjacency matrix
            n: Number of vertices

        Returns:
            Estimated tour cost
        """
        if n <= 1:
            return 0.0

        visited = {start}
        current = start
        total_cost = 0.0

        # Build tour greedily
        for _ in range(n - 1):
            # Find nearest unvisited neighbor
            best_next = None
            best_cost = float('inf')

            for v in range(n):
                if v not in visited and graph[current, v] < best_cost:
                    best_next = v
                    best_cost = graph[current, v]

            if best_next is None:
                break

            total_cost += best_cost
            visited.add(best_next)
            current = best_next

        # Return to start
        total_cost += graph[current, start]

        return total_cost

    def _extract_baseline_features(
        self,
        vertex: int,
        graph: np.ndarray,
        n: int
    ) -> List[float]:
        """
        Extract baseline comparison features.

        Compare anchor-based approach to simple baselines.

        Args:
            vertex: Vertex index
            graph: Adjacency matrix
            n: Number of vertices

        Returns:
            List of baseline features
        """
        # Baseline: Nearest neighbor tour from this vertex (no anchor constraint)
        nn_tour_cost = self._greedy_nearest_neighbor_estimate(vertex, graph, n)

        # Estimate with anchor constraint (forces using two cheapest edges)
        anchor_tour_estimate = self._anchor_constrained_estimate(vertex, graph, n)

        # Difference: is anchor helping or hurting?
        anchor_benefit = nn_tour_cost - anchor_tour_estimate

        # Ratio
        if nn_tour_cost > 0:
            anchor_ratio = anchor_tour_estimate / nn_tour_cost
        else:
            anchor_ratio = 1.0

        return [nn_tour_cost, anchor_tour_estimate, anchor_benefit, anchor_ratio]

    def _anchor_constrained_estimate(
        self,
        vertex: int,
        graph: np.ndarray,
        n: int
    ) -> float:
        """
        Estimate tour cost with anchor constraint.

        Forces tour to start with two cheapest edges from vertex,
        then continues greedily.

        Args:
            vertex: Vertex index
            graph: Adjacency matrix
            n: Number of vertices

        Returns:
            Estimated tour cost
        """
        if n <= 2:
            return 0.0

        # Find two cheapest edges
        distances = graph[vertex, :]
        sorted_indices = np.argsort(distances)

        # Skip self-loop (index where distances[i] == 0 or distances[i] == distances[vertex])
        candidates = [i for i in sorted_indices if i != vertex]

        if len(candidates) < 2:
            return self._greedy_nearest_neighbor_estimate(vertex, graph, n)

        anchor1 = candidates[0]
        anchor2 = candidates[1]

        # Start tour with anchor edges
        total_cost = graph[vertex, anchor1] + graph[vertex, anchor2]

        # Now build rest of tour greedily from both endpoints
        # Simplified: continue from anchor2
        visited = {vertex, anchor1, anchor2}
        current = anchor2

        for _ in range(n - 3):
            best_next = None
            best_cost = float('inf')

            for v in range(n):
                if v not in visited and graph[current, v] < best_cost:
                    best_next = v
                    best_cost = graph[current, v]

            if best_next is None:
                break

            total_cost += best_cost
            visited.add(best_next)
            current = best_next

        # Close tour: current -> anchor1 -> vertex
        total_cost += graph[current, anchor1]

        return total_cost

    def get_feature_names(self) -> List[str]:
        """
        Get all feature names this extractor produces.

        Returns:
            List of feature names
        """
        names = [
            "anchor_edge_1",
            "anchor_edge_2",
            "anchor_sum",
            "anchor_product",
            "anchor_ratio",
            "anchor_gap_to_third",
            "anchor_quality_score",
            "anchor_relative_gap"
        ]

        if self.include_tour_estimates:
            names.extend([
                "tour_estimate_nn",
                "tour_estimate_lower_bound",
                "tour_estimate_normalized"
            ])

        if self.include_baseline_comparison:
            names.extend([
                "baseline_nn_cost",
                "baseline_anchor_cost",
                "baseline_anchor_benefit",
                "baseline_anchor_ratio"
            ])

        return names
