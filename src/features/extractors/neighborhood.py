"""
Neighborhood-based vertex feature extraction.

Extracts features based on local neighborhood structure around each vertex,
including k-nearest neighbors, density, and radial patterns.
"""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from ..base import VertexFeatureExtractor


class NeighborhoodFeatureExtractor(VertexFeatureExtractor):
    """
    Extracts features based on local neighborhood structure.

    Features include:
    - K-nearest neighbor statistics (for k=1,2,3,5)
    - Neighborhood density at various radii
    - Radial features (shell-based analysis)
    - Voronoi-like region size

    These features capture local structure around vertices,
    distinguishing between locally central vs globally central vertices.
    """

    def __init__(
        self,
        k_values: List[int] = None,
        density_percentiles: List[float] = None,
        n_shells: int = 3,
        name: str = "neighborhood"
    ):
        """
        Initialize neighborhood feature extractor.

        Args:
            k_values: List of k values for k-NN features (default: [1, 2, 3, 5])
            density_percentiles: Percentiles for neighborhood radius
                                (default: [25, 50, 75] - defines radius as
                                 25th, 50th, 75th percentile of edge weights)
            n_shells: Number of concentric shells for radial features
            name: Extractor name
        """
        super().__init__(name)
        self.k_values = k_values or [1, 2, 3, 5]
        self.density_percentiles = density_percentiles or [25, 50, 75]
        self.n_shells = n_shells

    def extract(
        self,
        graph: np.ndarray,
        cache: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Extract neighborhood features for all vertices.

        Args:
            graph: NxN adjacency matrix
            cache: Optional cache (not used by this extractor)

        Returns:
            features: NxF feature matrix
            feature_names: List of feature names
        """
        n = graph.shape[0]
        features_list = []

        for i in range(n):
            vertex_features = []

            # K-nearest neighbor features
            knn_features = self._extract_knn_features(i, graph, n)
            vertex_features.extend(knn_features)

            # Neighborhood density features
            density_features = self._extract_density_features(i, graph, n)
            vertex_features.extend(density_features)

            # Radial features
            radial_features = self._extract_radial_features(i, graph, n)
            vertex_features.extend(radial_features)

            # Voronoi-like region size
            voronoi_size = self._compute_voronoi_region_size(i, graph, n)
            vertex_features.append(voronoi_size)

            features_list.append(vertex_features)

        features = np.array(features_list)
        feature_names = self.get_feature_names()

        return features, feature_names

    def _extract_knn_features(
        self,
        vertex: int,
        graph: np.ndarray,
        n: int
    ) -> List[float]:
        """
        Extract k-nearest neighbor features.

        For each k in k_values, compute statistics over k nearest neighbors.

        Args:
            vertex: Vertex index
            graph: Adjacency matrix
            n: Number of vertices

        Returns:
            List of k-NN features
        """
        features = []

        # Get distances to all other vertices
        distances = np.concatenate([graph[vertex, :vertex], graph[vertex, vertex+1:]])

        # Sort to get k-nearest
        sorted_distances = np.sort(distances)

        for k in self.k_values:
            if k > len(sorted_distances):
                # Not enough neighbors
                features.extend([0.0, 0.0, 0.0])
                continue

            # Get k nearest distances
            k_nearest = sorted_distances[:k]

            # Mean weight of k-nearest
            mean_k = np.mean(k_nearest)

            # Variance of k-nearest
            var_k = np.var(k_nearest)

            # Spread: difference between k-th and 1st nearest
            spread_k = k_nearest[-1] - k_nearest[0] if k > 1 else 0.0

            features.extend([mean_k, var_k, spread_k])

        return features

    def _extract_density_features(
        self,
        vertex: int,
        graph: np.ndarray,
        n: int
    ) -> List[float]:
        """
        Extract neighborhood density features.

        For each percentile, define neighborhood as vertices within
        that distance, then compute density metrics.

        Args:
            vertex: Vertex index
            graph: Adjacency matrix
            n: Number of vertices

        Returns:
            List of density features
        """
        features = []

        # Get distances to all other vertices
        distances = np.concatenate([graph[vertex, :vertex], graph[vertex, vertex+1:]])

        # Global percentiles for radius definition
        for percentile in self.density_percentiles:
            radius = np.percentile(distances, percentile)

            # Count vertices in neighborhood
            n_in_neighborhood = np.sum(distances <= radius)

            # Total weight within neighborhood
            total_weight_in = np.sum(distances[distances <= radius])

            # Average weight within neighborhood
            if n_in_neighborhood > 0:
                avg_weight_in = total_weight_in / n_in_neighborhood
            else:
                avg_weight_in = 0.0

            # Density: proportion of vertices in neighborhood
            density = n_in_neighborhood / (n - 1) if n > 1 else 0.0

            features.extend([n_in_neighborhood, total_weight_in, avg_weight_in, density])

        return features

    def _extract_radial_features(
        self,
        vertex: int,
        graph: np.ndarray,
        n: int
    ) -> List[float]:
        """
        Extract radial shell features.

        Divide vertices into concentric shells by distance from vertex,
        compute statistics for each shell.

        Args:
            vertex: Vertex index
            graph: Adjacency matrix
            n: Number of vertices

        Returns:
            List of radial features
        """
        features = []

        # Get distances
        distances = np.concatenate([graph[vertex, :vertex], graph[vertex, vertex+1:]])

        if len(distances) == 0:
            return [0.0] * (self.n_shells * 2)

        # Define shell boundaries (equal-width bins)
        min_dist = np.min(distances)
        max_dist = np.max(distances)

        if max_dist <= min_dist:
            # All distances equal
            return [len(distances), np.mean(distances)] + [0.0] * ((self.n_shells - 1) * 2)

        shell_width = (max_dist - min_dist) / self.n_shells

        for shell_idx in range(self.n_shells):
            shell_min = min_dist + shell_idx * shell_width
            shell_max = min_dist + (shell_idx + 1) * shell_width

            # Vertices in this shell
            in_shell = (distances >= shell_min) & (distances < shell_max)

            # For last shell, include upper boundary
            if shell_idx == self.n_shells - 1:
                in_shell = (distances >= shell_min) & (distances <= shell_max)

            n_in_shell = np.sum(in_shell)

            if n_in_shell > 0:
                mean_weight_shell = np.mean(distances[in_shell])
            else:
                mean_weight_shell = 0.0

            features.extend([n_in_shell, mean_weight_shell])

        return features

    def _compute_voronoi_region_size(
        self,
        vertex: int,
        graph: np.ndarray,
        n: int
    ) -> float:
        """
        Compute Voronoi-like region size.

        For each other vertex, determine if it's closer to this vertex
        than to any other vertex. Count how many vertices are closest
        to this vertex.

        Args:
            vertex: Vertex index
            graph: Adjacency matrix
            n: Number of vertices

        Returns:
            Proportion of vertices closest to this vertex
        """
        region_size = 0

        for v in range(n):
            if v == vertex:
                continue

            # Distance from v to vertex
            dist_to_vertex = graph[v, vertex]

            # Distances from v to all other vertices
            dists_from_v = graph[v, :]

            # Is vertex the closest to v?
            # (excluding v itself)
            is_closest = True
            for u in range(n):
                if u == v or u == vertex:
                    continue
                if dists_from_v[u] < dist_to_vertex:
                    is_closest = False
                    break

            if is_closest:
                region_size += 1

        # Return as proportion
        return region_size / (n - 1) if n > 1 else 0.0

    def get_feature_names(self) -> List[str]:
        """
        Get all feature names this extractor produces.

        Returns:
            List of feature names
        """
        names = []

        # K-NN features
        for k in self.k_values:
            names.extend([
                f"knn_{k}_mean",
                f"knn_{k}_var",
                f"knn_{k}_spread"
            ])

        # Density features
        for p in self.density_percentiles:
            names.extend([
                f"density_p{int(p)}_count",
                f"density_p{int(p)}_total_weight",
                f"density_p{int(p)}_avg_weight",
                f"density_p{int(p)}_proportion"
            ])

        # Radial features
        for shell_idx in range(self.n_shells):
            names.extend([
                f"shell_{shell_idx}_count",
                f"shell_{shell_idx}_mean_weight"
            ])

        # Voronoi
        names.append("voronoi_region_size")

        return names
