"""
MST-based vertex feature extraction.

Extracts features derived from the minimum spanning tree of the graph.
The MST reveals structural importance and hub vertices.
"""

from typing import Dict, List, Tuple, Optional, Any, Set
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from ..base import VertexFeatureExtractor, CachedComputation


class MSTFeatureExtractor(VertexFeatureExtractor):
    """
    Extracts features based on the minimum spanning tree.

    Features include:
    - MST degree: how many MST edges incident to vertex
    - MST leaf/hub indicators
    - MST edge weight statistics
    - Structural importance metrics
    - Distance to MST center

    The MST is computed once and cached for efficiency.
    """

    def __init__(self, name: str = "mst_based"):
        """
        Initialize MST-based feature extractor.

        Args:
            name: Extractor name
        """
        super().__init__(name)

    def extract(
        self,
        graph: np.ndarray,
        cache: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Extract MST-based features for all vertices.

        Args:
            graph: NxN adjacency matrix
            cache: Optional cache for MST computation

        Returns:
            features: NxF feature matrix
            feature_names: List of feature names
        """
        if cache is None:
            cache = {}

        n = graph.shape[0]

        # Get or compute MST (expensive: O(n^2 log n))
        mst, mst_edges = CachedComputation.get_or_compute(
            cache,
            'mst',
            lambda: self._compute_mst(graph)
        )

        features_list = []

        for i in range(n):
            vertex_features = []

            # MST degree: number of MST edges incident to this vertex
            mst_degree = self._compute_mst_degree(i, mst_edges)
            vertex_features.append(mst_degree)

            # Boolean indicators
            is_leaf = 1.0 if mst_degree == 1 else 0.0
            is_hub = 1.0 if mst_degree >= 3 else 0.0  # Threshold of 3
            vertex_features.extend([is_leaf, is_hub])

            # MST edge weight features
            mst_edge_weights = self._get_mst_edge_weights(i, mst, graph)
            if len(mst_edge_weights) > 0:
                total_mst_weight = np.sum(mst_edge_weights)
                mean_mst_weight = np.mean(mst_edge_weights)
                max_mst_weight = np.max(mst_edge_weights)
            else:
                total_mst_weight = 0.0
                mean_mst_weight = 0.0
                max_mst_weight = 0.0

            vertex_features.extend([
                total_mst_weight,
                mean_mst_weight,
                max_mst_weight
            ])

            # Ratio of MST weight to total weight
            total_vertex_weight = np.sum(graph[i]) - graph[i, i]
            mst_to_total_ratio = (
                total_mst_weight / total_vertex_weight
                if total_vertex_weight > 0 else 0.0
            )
            vertex_features.append(mst_to_total_ratio)

            features_list.append(vertex_features)

        # Compute global MST features (require full MST)
        mst_center_distances = self._compute_mst_center_distances(mst, n)
        removal_impact = self._compute_removal_impact(graph, mst_edges, n)

        # Add global features to each vertex
        features = np.array(features_list)
        features = np.hstack([
            features,
            mst_center_distances.reshape(-1, 1),
            removal_impact.reshape(-1, 1)
        ])

        feature_names = self.get_feature_names()

        return features, feature_names

    def _compute_mst(self, graph: np.ndarray) -> Tuple[np.ndarray, Set[Tuple[int, int]]]:
        """
        Compute minimum spanning tree.

        Args:
            graph: NxN adjacency matrix

        Returns:
            mst: NxN MST adjacency matrix (sparse)
            mst_edges: Set of (u, v) tuples representing MST edges
        """
        # Use scipy's minimum_spanning_tree
        # For undirected graphs, we need to make sure input is symmetric
        # or specify that it's undirected

        # Create sparse matrix for efficiency
        sparse_graph = csr_matrix(graph)

        # Compute MST (returns sparse matrix)
        mst_sparse = minimum_spanning_tree(sparse_graph)

        # Convert to dense for easier manipulation
        mst = mst_sparse.toarray()

        # Make MST symmetric (minimum_spanning_tree returns directed version)
        mst = mst + mst.T

        # Extract edge list
        mst_edges = set()
        n = graph.shape[0]
        for i in range(n):
            for j in range(i + 1, n):
                if mst[i, j] > 0:
                    mst_edges.add((i, j))

        return mst, mst_edges

    def _compute_mst_degree(self, vertex: int, mst_edges: Set[Tuple[int, int]]) -> int:
        """
        Compute MST degree for a vertex.

        Args:
            vertex: Vertex index
            mst_edges: Set of MST edges

        Returns:
            Number of MST edges incident to vertex
        """
        degree = 0
        for u, v in mst_edges:
            if u == vertex or v == vertex:
                degree += 1
        return degree

    def _get_mst_edge_weights(
        self,
        vertex: int,
        mst: np.ndarray,
        graph: np.ndarray
    ) -> List[float]:
        """
        Get weights of MST edges incident to vertex.

        Args:
            vertex: Vertex index
            mst: MST adjacency matrix
            graph: Original graph

        Returns:
            List of edge weights
        """
        weights = []
        n = graph.shape[0]
        for j in range(n):
            if mst[vertex, j] > 0:
                weights.append(graph[vertex, j])
        return weights

    def _compute_mst_center_distances(
        self,
        mst: np.ndarray,
        n: int
    ) -> np.ndarray:
        """
        Compute distance from each vertex to MST center.

        MST center is defined as the vertex that minimizes the maximum
        distance to any other vertex in the MST.

        Args:
            mst: MST adjacency matrix
            n: Number of vertices

        Returns:
            Array of distances to MST center
        """
        # Compute shortest paths in MST using BFS from each vertex
        mst_distances = np.zeros((n, n))

        for source in range(n):
            # BFS from source
            distances = self._bfs_distances(mst, source, n)
            mst_distances[source] = distances

        # Find center: vertex with minimum eccentricity (max distance)
        eccentricities = np.max(mst_distances, axis=1)
        center = np.argmin(eccentricities)

        # Return distances from each vertex to center
        center_distances = mst_distances[:, center]

        return center_distances

    def _bfs_distances(
        self,
        adj_matrix: np.ndarray,
        source: int,
        n: int
    ) -> np.ndarray:
        """
        Compute shortest path distances from source using BFS.

        Args:
            adj_matrix: Adjacency matrix (unweighted for BFS)
            source: Source vertex
            n: Number of vertices

        Returns:
            Array of distances from source
        """
        distances = np.full(n, -1)
        distances[source] = 0
        queue = [source]
        visited = set([source])

        while queue:
            u = queue.pop(0)
            for v in range(n):
                if adj_matrix[u, v] > 0 and v not in visited:
                    distances[v] = distances[u] + 1
                    visited.add(v)
                    queue.append(v)

        # Set unvisited vertices to max distance
        distances[distances == -1] = n

        return distances

    def _compute_removal_impact(
        self,
        graph: np.ndarray,
        mst_edges: Set[Tuple[int, int]],
        n: int
    ) -> np.ndarray:
        """
        Compute structural importance by vertex removal.

        For each vertex, compute how much the MST weight would change
        if that vertex were removed.

        Args:
            graph: Original graph
            mst_edges: MST edges
            n: Number of vertices

        Returns:
            Array of removal impact scores
        """
        # Compute original MST weight
        original_mst_weight = sum(
            graph[u, v] for u, v in mst_edges
        )

        impact = np.zeros(n)

        for vertex in range(n):
            # Create graph without this vertex
            # Set all edges to/from vertex to infinity (very large value)
            modified_graph = graph.copy()
            modified_graph[vertex, :] = np.max(graph) * n * 10
            modified_graph[:, vertex] = np.max(graph) * n * 10

            # Compute MST on modified graph
            try:
                modified_mst, modified_edges = self._compute_mst(modified_graph)
                modified_mst_weight = sum(
                    modified_graph[u, v]
                    for u, v in modified_edges
                    if u != vertex and v != vertex
                )

                # Impact is difference in MST weight
                # Higher impact = more important vertex
                impact[vertex] = modified_mst_weight - original_mst_weight

            except Exception:
                # If MST computation fails (disconnected graph), set high impact
                impact[vertex] = original_mst_weight

        # Normalize by original MST weight
        if original_mst_weight > 0:
            impact /= original_mst_weight

        return impact

    def get_feature_names(self) -> List[str]:
        """
        Get all feature names this extractor produces.

        Returns:
            List of feature names
        """
        return [
            "mst_degree",
            "mst_is_leaf",
            "mst_is_hub",
            "mst_total_weight",
            "mst_mean_weight",
            "mst_max_weight",
            "mst_to_total_ratio",
            "mst_center_distance",
            "mst_removal_impact",
        ]
