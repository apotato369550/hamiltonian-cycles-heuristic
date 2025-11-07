"""
Base classes and interfaces for vertex feature extraction.

Defines the abstract interface that all feature extractors must implement,
along with common validation utilities.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any
import numpy as np


class FeatureValidationError(Exception):
    """Raised when feature extraction produces invalid results."""
    pass


class VertexFeatureExtractor(ABC):
    """
    Abstract base class for vertex feature extraction.

    All feature extractors must inherit from this class and implement
    the extract() method. This ensures a consistent interface across
    all feature types.

    The extractor pattern allows:
    - Modular feature development (add new extractors without changing existing code)
    - Easy feature selection (enable/disable extractors)
    - Computation caching (expensive computations shared via cache)
    - Clear feature naming (each extractor documents its features)
    """

    def __init__(self, name: str):
        """
        Initialize the feature extractor.

        Args:
            name: Descriptive name for this extractor (e.g., 'weight_based')
        """
        self.name = name

    @abstractmethod
    def extract(
        self,
        graph: np.ndarray,
        cache: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Extract features for all vertices in the graph.

        Args:
            graph: NxN adjacency matrix where graph[i][j] is edge weight from i to j
            cache: Optional cache dictionary for sharing expensive computations
                   (e.g., MST, shortest paths). Extractors can read from and write to cache.

        Returns:
            features: NxF numpy array where N is number of vertices, F is number of features
                     features[i][j] is the j-th feature value for vertex i
            feature_names: List of F strings naming each feature
                          Example: ['total_weight', 'mean_weight', 'min_edge_weight']

        Raises:
            FeatureValidationError: If extraction produces invalid values
        """
        pass

    @abstractmethod
    def get_feature_names(self) -> List[str]:
        """
        Return the names of features this extractor produces.

        Returns:
            List of descriptive feature names

        Note:
            Feature names should be self-documenting:
            - Good: 'mst_degree', 'betweenness_centrality', 'mean_edge_weight'
            - Bad: 'feat1', 'x', 'metric_a'
        """
        pass

    def validate_features(
        self,
        features: np.ndarray,
        feature_names: List[str],
        n_vertices: int
    ) -> None:
        """
        Validate extracted features for common issues.

        Args:
            features: The extracted feature matrix
            feature_names: List of feature names
            n_vertices: Expected number of vertices

        Raises:
            FeatureValidationError: If validation fails
        """
        # Check shape
        if features.shape[0] != n_vertices:
            raise FeatureValidationError(
                f"{self.name}: Expected {n_vertices} rows, got {features.shape[0]}"
            )

        if features.shape[1] != len(feature_names):
            raise FeatureValidationError(
                f"{self.name}: Feature count mismatch. "
                f"Matrix has {features.shape[1]} columns but {len(feature_names)} names provided"
            )

        # Check for NaN values
        if np.any(np.isnan(features)):
            nan_features = [
                feature_names[j]
                for j in range(features.shape[1])
                if np.any(np.isnan(features[:, j]))
            ]
            raise FeatureValidationError(
                f"{self.name}: NaN values found in features: {nan_features}"
            )

        # Check for infinite values
        if np.any(np.isinf(features)):
            inf_features = [
                feature_names[j]
                for j in range(features.shape[1])
                if np.any(np.isinf(features[:, j]))
            ]
            raise FeatureValidationError(
                f"{self.name}: Infinite values found in features: {inf_features}"
            )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"


class CachedComputation:
    """
    Helper for managing cached expensive computations.

    Usage:
        cache = {}
        mst = CachedComputation.get_or_compute(
            cache, 'mst', lambda: compute_mst(graph)
        )
    """

    @staticmethod
    def get_or_compute(
        cache: Dict[str, Any],
        key: str,
        compute_fn: callable
    ) -> Any:
        """
        Get value from cache or compute and store it.

        Args:
            cache: Cache dictionary
            key: Cache key
            compute_fn: Function to call if key not in cache

        Returns:
            Cached or newly computed value
        """
        if key not in cache:
            cache[key] = compute_fn()
        return cache[key]
