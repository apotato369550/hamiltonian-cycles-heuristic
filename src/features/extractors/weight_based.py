"""
Weight-based vertex feature extraction.

Extracts features derived from the edge weights incident to each vertex,
including basic statistics, distribution features, and relative measures.
"""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from scipy import stats
from ..base import VertexFeatureExtractor


class WeightFeatureExtractor(VertexFeatureExtractor):
    """
    Extracts features based on edge weight statistics for each vertex.

    For each vertex, computes statistics over its incident edge weights:
    - Basic statistics: sum, mean, median, std, variance
    - Distribution features: min, max, quantiles, skewness, kurtosis
    - Relative features: ranks, comparisons to graph statistics

    For asymmetric graphs, can compute separate features for outgoing
    and incoming edges.
    """

    def __init__(
        self,
        include_asymmetric_features: bool = True,
        name: str = "weight_based"
    ):
        """
        Initialize weight-based feature extractor.

        Args:
            include_asymmetric_features: If True and graph is asymmetric,
                compute separate features for outgoing/incoming edges
            name: Extractor name
        """
        super().__init__(name)
        self.include_asymmetric_features = include_asymmetric_features

    def extract(
        self,
        graph: np.ndarray,
        cache: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Extract weight-based features for all vertices.

        Args:
            graph: NxN adjacency matrix
            cache: Optional cache (not used by this extractor)

        Returns:
            features: NxF feature matrix
            feature_names: List of feature names
        """
        n = graph.shape[0]

        # Check if graph is symmetric
        is_symmetric = np.allclose(graph, graph.T)

        if is_symmetric or not self.include_asymmetric_features:
            # Use outgoing edges only (or average for symmetric)
            features = self._extract_directed_features(graph, n, "")
            feature_names = self._get_directed_feature_names("")
        else:
            # Extract features for both directions
            outgoing_features = self._extract_directed_features(graph, n, "out_")
            incoming_features = self._extract_directed_features(graph.T, n, "in_")

            # Compute asymmetry metrics
            asymmetry_features = self._compute_asymmetry_features(
                graph, n, outgoing_features, incoming_features
            )

            # Combine all features
            features = np.hstack([
                outgoing_features,
                incoming_features,
                asymmetry_features
            ])

            feature_names = (
                self._get_directed_feature_names("out_") +
                self._get_directed_feature_names("in_") +
                self._get_asymmetry_feature_names()
            )

        return features, feature_names

    def _extract_directed_features(
        self,
        graph: np.ndarray,
        n: int,
        prefix: str
    ) -> np.ndarray:
        """
        Extract features for one direction of edges.

        Args:
            graph: NxN adjacency matrix (rows are source vertices)
            n: Number of vertices
            prefix: Prefix for feature names (e.g., "out_", "in_", "")

        Returns:
            NxF feature matrix
        """
        features_list = []

        for i in range(n):
            # Get edge weights from vertex i (excluding self-loop)
            weights = np.concatenate([graph[i, :i], graph[i, i+1:]])

            # Handle edge case: single vertex graph (no edges)
            if len(weights) == 0:
                vertex_features = [0.0] * 14
                features_list.append(vertex_features)
                continue

            # Basic statistics
            total_weight = np.sum(weights)
            mean_weight = np.mean(weights) if len(weights) > 0 else 0.0
            median_weight = np.median(weights) if len(weights) > 0 else 0.0
            std_weight = np.std(weights) if len(weights) > 1 else 0.0
            var_weight = np.var(weights) if len(weights) > 1 else 0.0

            # Distribution features
            min_weight = np.min(weights) if len(weights) > 0 else 0.0
            max_weight = np.max(weights) if len(weights) > 0 else 0.0
            min_max_ratio = min_weight / max_weight if max_weight > 0 else 0.0

            # Quantiles
            q25 = np.percentile(weights, 25) if len(weights) > 0 else 0.0
            q50 = np.percentile(weights, 50) if len(weights) > 0 else 0.0
            q75 = np.percentile(weights, 75) if len(weights) > 0 else 0.0
            iqr = q75 - q25

            # Higher-order statistics (handle uniform weights)
            if len(weights) > 1 and std_weight > 1e-10:
                skewness = stats.skew(weights)
                kurtosis_val = stats.kurtosis(weights)
            else:
                skewness = 0.0
                kurtosis_val = 0.0

            # Relative features (computed later with graph context)
            vertex_features = [
                total_weight,
                mean_weight,
                median_weight,
                std_weight,
                var_weight,
                min_weight,
                max_weight,
                min_max_ratio,
                q25,
                q50,
                q75,
                iqr,
                skewness,
                kurtosis_val,
            ]

            features_list.append(vertex_features)

        # Convert to numpy array
        features = np.array(features_list)

        # Add relative features (graph-level context)
        relative_features = self._compute_relative_features(graph, features)
        features = np.hstack([features, relative_features])

        return features

    def _compute_relative_features(
        self,
        graph: np.ndarray,
        basic_features: np.ndarray
    ) -> np.ndarray:
        """
        Compute features relative to graph-level statistics.

        Args:
            graph: NxN adjacency matrix
            basic_features: Nx14 array of basic features

        Returns:
            NxF array of relative features
        """
        n = graph.shape[0]
        relative_list = []

        # Extract mean weights (column 1 of basic features)
        mean_weights = basic_features[:, 1]

        # Extract min weights (column 5)
        min_weights = basic_features[:, 5]

        # Graph-level statistics
        graph_mean_weight = np.mean(mean_weights)
        graph_std_weight = np.std(mean_weights)
        graph_median_weight = np.median(mean_weights)

        # All edge weights in graph (excluding diagonal)
        all_weights = []
        for i in range(n):
            all_weights.extend(np.concatenate([graph[i, :i], graph[i, i+1:]]))
        all_weights = np.array(all_weights)
        global_median = np.median(all_weights)

        for i in range(n):
            # Z-score of mean weight
            z_score_mean = (
                (mean_weights[i] - graph_mean_weight) / graph_std_weight
                if graph_std_weight > 0 else 0
            )

            # Percentile rank of mean weight
            percentile_rank_mean = stats.percentileofscore(mean_weights, mean_weights[i])

            # Distance from graph median
            distance_from_median = mean_weights[i] - graph_median_weight

            # Rank of cheapest edge
            rank_cheapest = stats.rankdata(min_weights, method='average')[i]

            # Proportion of edges below graph median
            vertex_weights = np.concatenate([graph[i, :i], graph[i, i+1:]])
            prop_below_median = np.mean(vertex_weights < global_median)

            # Distance to centroid (vertex with minimum sum of distances)
            # We approximate centroid as vertex with minimum mean weight
            centroid_idx = np.argmin(mean_weights)
            distance_to_centroid = graph[i, centroid_idx] if i != centroid_idx else 0

            relative_features = [
                z_score_mean,
                percentile_rank_mean,
                distance_from_median,
                rank_cheapest,
                prop_below_median,
                distance_to_centroid,
            ]

            relative_list.append(relative_features)

        return np.array(relative_list)

    def _compute_asymmetry_features(
        self,
        graph: np.ndarray,
        n: int,
        outgoing_features: np.ndarray,
        incoming_features: np.ndarray
    ) -> np.ndarray:
        """
        Compute asymmetry metrics between outgoing and incoming edges.

        Args:
            graph: NxN adjacency matrix
            n: Number of vertices
            outgoing_features: Features for outgoing edges
            incoming_features: Features for incoming edges

        Returns:
            NxF array of asymmetry features
        """
        asymmetry_list = []

        # Indices of key features (from _extract_directed_features)
        # 0: total_weight, 1: mean_weight, 5: min_weight, 6: max_weight
        total_idx, mean_idx, min_idx, max_idx = 0, 1, 5, 6

        for i in range(n):
            # Difference metrics
            total_diff = outgoing_features[i, total_idx] - incoming_features[i, total_idx]
            mean_diff = outgoing_features[i, mean_idx] - incoming_features[i, mean_idx]
            min_diff = outgoing_features[i, min_idx] - incoming_features[i, min_idx]
            max_diff = outgoing_features[i, max_idx] - incoming_features[i, max_idx]

            # Ratio metrics (avoid division by zero)
            total_ratio = (
                outgoing_features[i, total_idx] / incoming_features[i, total_idx]
                if incoming_features[i, total_idx] > 0 else 0
            )
            mean_ratio = (
                outgoing_features[i, mean_idx] / incoming_features[i, mean_idx]
                if incoming_features[i, mean_idx] > 0 else 0
            )

            asymmetry_features = [
                total_diff,
                mean_diff,
                min_diff,
                max_diff,
                total_ratio,
                mean_ratio,
            ]

            asymmetry_list.append(asymmetry_features)

        return np.array(asymmetry_list)

    def _get_directed_feature_names(self, prefix: str) -> List[str]:
        """
        Get feature names for directed features.

        Args:
            prefix: Prefix to add (e.g., "out_", "in_", "")

        Returns:
            List of feature names
        """
        basic_names = [
            f"{prefix}total_weight",
            f"{prefix}mean_weight",
            f"{prefix}median_weight",
            f"{prefix}std_weight",
            f"{prefix}var_weight",
            f"{prefix}min_weight",
            f"{prefix}max_weight",
            f"{prefix}min_max_ratio",
            f"{prefix}q25_weight",
            f"{prefix}q50_weight",
            f"{prefix}q75_weight",
            f"{prefix}iqr_weight",
            f"{prefix}skewness",
            f"{prefix}kurtosis",
        ]

        relative_names = [
            f"{prefix}z_score_mean",
            f"{prefix}percentile_rank_mean",
            f"{prefix}distance_from_median",
            f"{prefix}rank_cheapest",
            f"{prefix}prop_below_median",
            f"{prefix}distance_to_centroid",
        ]

        return basic_names + relative_names

    def _get_asymmetry_feature_names(self) -> List[str]:
        """
        Get feature names for asymmetry features.

        Returns:
            List of feature names
        """
        return [
            "asym_total_diff",
            "asym_mean_diff",
            "asym_min_diff",
            "asym_max_diff",
            "asym_total_ratio",
            "asym_mean_ratio",
        ]

    def get_feature_names(self) -> List[str]:
        """
        Get all feature names this extractor produces.

        Returns:
            List of feature names

        Note:
            Actual names depend on whether graph is symmetric and
            whether asymmetric features are enabled. This returns
            the maximum set for documentation purposes.
        """
        # Return symmetric case by default
        return self._get_directed_feature_names("")
