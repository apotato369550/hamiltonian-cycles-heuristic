"""
Topological vertex feature extraction.

Extracts features based on graph topology and centrality measures,
including degree-based features, centrality measures, clustering,
and distance-based features.
"""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path, connected_components
from ..base import VertexFeatureExtractor, CachedComputation


class TopologicalFeatureExtractor(VertexFeatureExtractor):
    """
    Extracts features based on graph topology and centrality.

    Features include:
    - Degree-based: degree, weighted degree
    - Centrality: closeness, betweenness, eigenvector
    - Clustering: local clustering coefficient
    - Distance: eccentricity, average shortest path length

    Note: Some features (betweenness centrality) are expensive O(n^3).
    Use with caution on large graphs or disable via constructor.
    """

    def __init__(
        self,
        include_betweenness: bool = True,
        include_eigenvector: bool = True,
        include_clustering: bool = True,
        name: str = "topological"
    ):
        """
        Initialize topological feature extractor.

        Args:
            include_betweenness: Compute betweenness centrality (expensive)
            include_eigenvector: Compute eigenvector centrality
            include_clustering: Compute clustering coefficient
            name: Extractor name
        """
        super().__init__(name)
        self.include_betweenness = include_betweenness
        self.include_eigenvector = include_eigenvector
        self.include_clustering = include_clustering

    def extract(
        self,
        graph: np.ndarray,
        cache: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Extract topological features for all vertices.

        Args:
            graph: NxN adjacency matrix
            cache: Optional cache for shortest paths

        Returns:
            features: NxF feature matrix
            feature_names: List of feature names
        """
        if cache is None:
            cache = {}

        n = graph.shape[0]

        # Get or compute shortest paths (expensive, should be cached)
        dist_matrix = CachedComputation.get_or_compute(
            cache,
            'shortest_paths',
            lambda: self._compute_shortest_paths(graph)
        )

        features_list = []

        for i in range(n):
            vertex_features = []

            # Degree-based features
            degree = n - 1  # In complete graph, degree is n-1
            weighted_degree = np.sum(graph[i]) - graph[i, i]  # Exclude self-loop
            vertex_features.extend([degree, weighted_degree])

            # Closeness centrality
            closeness = self._compute_closeness_centrality(i, dist_matrix, n)
            vertex_features.append(closeness)

            # Betweenness centrality (expensive)
            if self.include_betweenness:
                # We'll compute all at once and cache
                betweenness_all = CachedComputation.get_or_compute(
                    cache,
                    'betweenness_centrality',
                    lambda: self._compute_betweenness_centrality(graph, dist_matrix)
                )
                betweenness = betweenness_all[i]
                vertex_features.append(betweenness)

            # Eigenvector centrality (expensive)
            if self.include_eigenvector:
                eigenvector_all = CachedComputation.get_or_compute(
                    cache,
                    'eigenvector_centrality',
                    lambda: self._compute_eigenvector_centrality(graph)
                )
                eigenvector = eigenvector_all[i]
                vertex_features.append(eigenvector)

            # Clustering coefficient
            if self.include_clustering:
                clustering = self._compute_clustering_coefficient(i, graph, n)
                vertex_features.append(clustering)

            # Distance-based features
            eccentricity = np.max(dist_matrix[i])
            avg_shortest_path = np.mean(dist_matrix[i])
            vertex_features.extend([eccentricity, avg_shortest_path])

            features_list.append(vertex_features)

        features = np.array(features_list)
        feature_names = self.get_feature_names()

        return features, feature_names

    def _compute_shortest_paths(self, graph: np.ndarray) -> np.ndarray:
        """
        Compute all-pairs shortest paths using Floyd-Warshall or Dijkstra.

        Args:
            graph: NxN adjacency matrix

        Returns:
            NxN distance matrix where dist[i][j] is shortest path from i to j
        """
        # Use scipy's shortest_path with Dijkstra for each source
        # directed=True to handle asymmetric graphs
        dist_matrix = shortest_path(
            csgraph=graph,
            method='D',  # Dijkstra
            directed=True
        )

        # Handle infinite distances (disconnected vertices) - shouldn't happen in complete graph
        # but good to be safe
        dist_matrix[np.isinf(dist_matrix)] = np.max(graph) * graph.shape[0]

        return dist_matrix

    def _compute_closeness_centrality(
        self,
        vertex: int,
        dist_matrix: np.ndarray,
        n: int
    ) -> float:
        """
        Compute closeness centrality for a vertex.

        Closeness = (n-1) / sum of distances to all other vertices
        Higher closeness = more central

        Args:
            vertex: Vertex index
            dist_matrix: NxN shortest path matrix
            n: Number of vertices

        Returns:
            Closeness centrality value
        """
        # Sum of distances to all other vertices
        total_distance = np.sum(dist_matrix[vertex])

        # Closeness is inverse of average distance
        if total_distance > 0:
            closeness = (n - 1) / total_distance
        else:
            closeness = 0

        return closeness

    def _compute_betweenness_centrality(
        self,
        graph: np.ndarray,
        dist_matrix: np.ndarray
    ) -> np.ndarray:
        """
        Compute betweenness centrality for all vertices.

        Betweenness = number of shortest paths passing through vertex
        normalized by total number of shortest paths.

        This is expensive: O(n^3) for naive implementation.

        Args:
            graph: NxN adjacency matrix
            dist_matrix: NxN shortest path matrix

        Returns:
            Array of betweenness centrality values for each vertex
        """
        n = graph.shape[0]
        betweenness = np.zeros(n)

        # For each pair of vertices (s, t)
        for s in range(n):
            for t in range(n):
                if s == t:
                    continue

                # Find all vertices on shortest paths from s to t
                shortest_dist = dist_matrix[s, t]

                # Check each potential intermediate vertex
                for v in range(n):
                    if v == s or v == t:
                        continue

                    # If v is on a shortest path from s to t:
                    # dist(s, v) + dist(v, t) == dist(s, t)
                    if abs(dist_matrix[s, v] + dist_matrix[v, t] - shortest_dist) < 1e-9:
                        betweenness[v] += 1

        # Normalize by number of pairs (excluding self-pairs)
        # For directed graph: n * (n - 1) pairs
        # For undirected: n * (n - 1) / 2 pairs
        # We'll use directed normalization
        max_pairs = n * (n - 1)
        if max_pairs > 0:
            betweenness /= max_pairs

        return betweenness

    def _compute_eigenvector_centrality(
        self,
        graph: np.ndarray,
        max_iter: int = 100,
        tol: float = 1e-6
    ) -> np.ndarray:
        """
        Compute eigenvector centrality using power iteration.

        Eigenvector centrality: importance based on connections to
        other important vertices.

        Args:
            graph: NxN adjacency matrix
            max_iter: Maximum iterations for power method
            tol: Convergence tolerance

        Returns:
            Array of eigenvector centrality values
        """
        n = graph.shape[0]

        # Create adjacency matrix (use 1/weight as adjacency for weighted graphs)
        # This way, stronger connections (lower weights) have higher adjacency
        adj = np.zeros_like(graph)
        max_weight = np.max(graph)
        for i in range(n):
            for j in range(n):
                if i != j and graph[i, j] > 0:
                    adj[i, j] = max_weight / graph[i, j]

        # Power iteration
        x = np.ones(n) / n  # Initial vector
        for _ in range(max_iter):
            x_new = adj @ x
            norm = np.linalg.norm(x_new)
            if norm > 0:
                x_new /= norm

            # Check convergence
            if np.linalg.norm(x_new - x) < tol:
                break

            x = x_new

        return x

    def _compute_clustering_coefficient(
        self,
        vertex: int,
        graph: np.ndarray,
        n: int
    ) -> float:
        """
        Compute local clustering coefficient for a vertex.

        Clustering coefficient = proportion of neighbor pairs that are connected.

        In complete graphs, this is always 1.0.
        In sparse graphs, this measures local density.

        Args:
            vertex: Vertex index
            graph: NxN adjacency matrix
            n: Number of vertices

        Returns:
            Clustering coefficient (0 to 1)
        """
        # Get neighbors (all other vertices in complete graph)
        neighbors = [j for j in range(n) if j != vertex]

        if len(neighbors) < 2:
            return 0.0

        # Count edges between neighbors
        edges_between_neighbors = 0
        for i, u in enumerate(neighbors):
            for v in neighbors[i+1:]:
                # Check if edge exists (weight > 0 or some threshold)
                if graph[u, v] > 0:
                    edges_between_neighbors += 1

        # Maximum possible edges between neighbors
        max_edges = len(neighbors) * (len(neighbors) - 1) / 2

        if max_edges > 0:
            clustering = edges_between_neighbors / max_edges
        else:
            clustering = 0.0

        return clustering

    def get_feature_names(self) -> List[str]:
        """
        Get all feature names this extractor produces.

        Returns:
            List of feature names
        """
        names = [
            "degree",
            "weighted_degree",
            "closeness_centrality",
        ]

        if self.include_betweenness:
            names.append("betweenness_centrality")

        if self.include_eigenvector:
            names.append("eigenvector_centrality")

        if self.include_clustering:
            names.append("clustering_coefficient")

        names.extend([
            "eccentricity",
            "avg_shortest_path_length",
        ])

        return names
