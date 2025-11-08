"""
Graph-level context feature extraction.

Adds graph-level properties and normalized versions of vertex features
to provide context about the graph structure and relative vertex importance.
"""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from scipy import stats
from scipy.sparse.csgraph import shortest_path
from ..base import VertexFeatureExtractor, CachedComputation


class GraphContextFeatureExtractor(VertexFeatureExtractor):
    """
    Extracts graph-level context features.

    These features help ML models understand:
    - Graph type and structure (size, density, metricity)
    - Vertex position relative to graph statistics
    - Normalized importance metrics

    Many of these features are constant across all vertices in a graph
    but help distinguish different graph types during training.
    """

    def __init__(
        self,
        include_graph_properties: bool = True,
        include_normalized_importance: bool = True,
        name: str = "graph_context"
    ):
        """
        Initialize graph context feature extractor.

        Args:
            include_graph_properties: Include constant graph-level features
            include_normalized_importance: Include relative importance metrics
            name: Extractor name
        """
        super().__init__(name)
        self.include_graph_properties = include_graph_properties
        self.include_normalized_importance = include_normalized_importance

    def extract(
        self,
        graph: np.ndarray,
        cache: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Extract graph context features for all vertices.

        Args:
            graph: NxN adjacency matrix
            cache: Optional cache

        Returns:
            features: NxF feature matrix
            feature_names: List of feature names
        """
        if cache is None:
            cache = {}

        n = graph.shape[0]
        features_list = []

        # Compute graph-level statistics once
        graph_stats = self._compute_graph_statistics(graph, n)

        # Compute importance metrics if needed
        if self.include_normalized_importance:
            importance_metrics = self._compute_importance_metrics(graph, n, cache)
        else:
            importance_metrics = None

        for i in range(n):
            vertex_features = []

            # Graph properties (constant for all vertices)
            if self.include_graph_properties:
                graph_features = [
                    graph_stats['size'],
                    graph_stats['density'],
                    graph_stats['metricity_score'],
                    graph_stats['weight_mean'],
                    graph_stats['weight_std'],
                    graph_stats['weight_skewness'],
                    graph_stats['weight_kurtosis'],
                    graph_stats['diameter'],
                    graph_stats['avg_path_length']
                ]
                vertex_features.extend(graph_features)

            # Normalized importance
            if self.include_normalized_importance:
                importance_features = [
                    importance_metrics['closeness_normalized'][i],
                    importance_metrics['degree_normalized'][i],
                    importance_metrics['weight_normalized'][i]
                ]
                vertex_features.extend(importance_features)

            features_list.append(vertex_features)

        features = np.array(features_list)
        feature_names = self.get_feature_names()

        return features, feature_names

    def _compute_graph_statistics(
        self,
        graph: np.ndarray,
        n: int
    ) -> Dict[str, float]:
        """
        Compute graph-level statistics.

        Args:
            graph: Adjacency matrix
            n: Number of vertices

        Returns:
            Dictionary of graph statistics
        """
        stats_dict = {}

        # Size
        stats_dict['size'] = float(n)

        # Density (for sparse graphs, this would be < 1)
        # For complete graphs, this is 1.0
        max_edges = n * (n - 1)
        actual_edges = np.sum(graph > 0)
        stats_dict['density'] = actual_edges / max_edges if max_edges > 0 else 0.0

        # Metricity score: percentage of triplets satisfying triangle inequality
        stats_dict['metricity_score'] = self._compute_metricity_score(graph, n)

        # Weight distribution statistics
        all_weights = graph[graph > 0]
        if len(all_weights) > 0:
            stats_dict['weight_mean'] = np.mean(all_weights)
            stats_dict['weight_std'] = np.std(all_weights)
            stats_dict['weight_skewness'] = float(stats.skew(all_weights))
            stats_dict['weight_kurtosis'] = float(stats.kurtosis(all_weights))
        else:
            stats_dict['weight_mean'] = 0.0
            stats_dict['weight_std'] = 0.0
            stats_dict['weight_skewness'] = 0.0
            stats_dict['weight_kurtosis'] = 0.0

        # Diameter and average path length
        try:
            dist_matrix = shortest_path(graph, method='D', directed=True)
            dist_matrix[np.isinf(dist_matrix)] = 0

            stats_dict['diameter'] = np.max(dist_matrix)
            # Average path length (excluding self-loops)
            mask = ~np.eye(n, dtype=bool)
            stats_dict['avg_path_length'] = np.mean(dist_matrix[mask])
        except Exception:
            stats_dict['diameter'] = 0.0
            stats_dict['avg_path_length'] = 0.0

        return stats_dict

    def _compute_metricity_score(
        self,
        graph: np.ndarray,
        n: int,
        sample_size: int = 100
    ) -> float:
        """
        Compute metricity score: percentage of triplets satisfying triangle inequality.

        For large graphs, samples triplets rather than checking all.

        Args:
            graph: Adjacency matrix
            n: Number of vertices
            sample_size: Number of triplets to sample (if n > 20)

        Returns:
            Metricity score (0-1)
        """
        if n < 3:
            return 1.0

        # For small graphs, check all triplets
        if n <= 20:
            total = 0
            satisfied = 0

            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    for k in range(n):
                        if k == i or k == j:
                            continue

                        total += 1
                        # Check triangle inequality: d(i,k) <= d(i,j) + d(j,k)
                        if graph[i, k] <= graph[i, j] + graph[j, k] + 1e-9:
                            satisfied += 1

            return satisfied / total if total > 0 else 1.0

        # For large graphs, sample triplets
        total = 0
        satisfied = 0

        for _ in range(sample_size):
            i, j, k = np.random.choice(n, size=3, replace=False)

            total += 1
            if graph[i, k] <= graph[i, j] + graph[j, k] + 1e-9:
                satisfied += 1

        return satisfied / total if total > 0 else 1.0

    def _compute_importance_metrics(
        self,
        graph: np.ndarray,
        n: int,
        cache: Dict[str, Any]
    ) -> Dict[str, np.ndarray]:
        """
        Compute normalized importance metrics for all vertices.

        Args:
            graph: Adjacency matrix
            n: Number of vertices
            cache: Cache dictionary

        Returns:
            Dictionary of importance metrics (each is array of length n)
        """
        metrics = {}

        # Closeness centrality (normalized)
        dist_matrix = CachedComputation.get_or_compute(
            cache,
            'shortest_paths',
            lambda: shortest_path(graph, method='D', directed=True)
        )

        closeness = np.zeros(n)
        for i in range(n):
            total_dist = np.sum(dist_matrix[i])
            if total_dist > 0:
                closeness[i] = (n - 1) / total_dist

        # Normalize to [0, 1]
        max_closeness = np.max(closeness) if np.max(closeness) > 0 else 1.0
        metrics['closeness_normalized'] = closeness / max_closeness

        # Degree (normalized by max possible degree)
        degrees = np.array([n - 1] * n)  # In complete graph, all have degree n-1
        max_degree = np.max(degrees) if np.max(degrees) > 0 else 1.0
        metrics['degree_normalized'] = degrees / max_degree

        # Total weight (normalized by max total weight)
        total_weights = np.sum(graph, axis=1) - np.diag(graph)
        max_weight = np.max(total_weights) if np.max(total_weights) > 0 else 1.0
        metrics['weight_normalized'] = total_weights / max_weight

        return metrics

    def get_feature_names(self) -> List[str]:
        """
        Get all feature names this extractor produces.

        Returns:
            List of feature names
        """
        names = []

        if self.include_graph_properties:
            names.extend([
                "graph_size",
                "graph_density",
                "graph_metricity_score",
                "graph_weight_mean",
                "graph_weight_std",
                "graph_weight_skewness",
                "graph_weight_kurtosis",
                "graph_diameter",
                "graph_avg_path_length"
            ])

        if self.include_normalized_importance:
            names.extend([
                "closeness_normalized",
                "degree_normalized",
                "weight_normalized"
            ])

        return names
