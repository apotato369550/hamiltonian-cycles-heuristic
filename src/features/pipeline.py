"""
Feature extraction pipeline orchestration.

The FeatureExtractorPipeline runs multiple feature extractors,
manages caching, combines results, and validates output.
"""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from .base import VertexFeatureExtractor, FeatureValidationError


class FeatureExtractorPipeline:
    """
    Orchestrates multiple feature extractors to produce a unified feature matrix.

    The pipeline:
    1. Accepts a graph and a list of enabled extractors
    2. Creates a shared cache for expensive computations
    3. Runs each extractor sequentially
    4. Combines results into a single feature matrix
    5. Validates the final output

    Example:
        from features.extractors import WeightFeatureExtractor, MSTFeatureExtractor

        pipeline = FeatureExtractorPipeline()
        pipeline.add_extractor(WeightFeatureExtractor())
        pipeline.add_extractor(MSTFeatureExtractor())

        features, names = pipeline.extract_features(graph)
    """

    def __init__(self):
        """Initialize an empty pipeline."""
        self.extractors: List[VertexFeatureExtractor] = []

    def add_extractor(self, extractor: VertexFeatureExtractor) -> None:
        """
        Add a feature extractor to the pipeline.

        Args:
            extractor: A VertexFeatureExtractor instance

        Raises:
            ValueError: If extractor name conflicts with existing extractor
        """
        # Check for name conflicts
        for existing in self.extractors:
            if existing.name == extractor.name:
                raise ValueError(
                    f"Extractor with name '{extractor.name}' already exists in pipeline"
                )

        self.extractors.append(extractor)

    def remove_extractor(self, name: str) -> None:
        """
        Remove an extractor by name.

        Args:
            name: Name of the extractor to remove

        Raises:
            ValueError: If no extractor with that name exists
        """
        for i, extractor in enumerate(self.extractors):
            if extractor.name == name:
                self.extractors.pop(i)
                return

        raise ValueError(f"No extractor with name '{name}' found in pipeline")

    def clear_extractors(self) -> None:
        """Remove all extractors from the pipeline."""
        self.extractors.clear()

    def extract_features(
        self,
        graph: np.ndarray,
        cache: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Extract features from all enabled extractors.

        Args:
            graph: NxN adjacency matrix
            cache: Optional pre-populated cache. If None, creates new cache.

        Returns:
            features: NxF feature matrix where F is total number of features
            feature_names: List of F feature names (prefixed with extractor name)

        Raises:
            ValueError: If no extractors are enabled
            FeatureValidationError: If any extractor produces invalid features
        """
        if not self.extractors:
            raise ValueError("No extractors in pipeline. Add at least one extractor.")

        n_vertices = graph.shape[0]

        # Validate graph shape
        if graph.ndim != 2:
            raise ValueError(f"Graph must be 2D array, got shape {graph.shape}")
        if graph.shape[0] != graph.shape[1]:
            raise ValueError(
                f"Graph must be square, got shape {graph.shape[0]}x{graph.shape[1]}"
            )

        # Create cache if not provided
        if cache is None:
            cache = {}

        # Extract features from each extractor
        all_features: List[np.ndarray] = []
        all_names: List[str] = []

        for extractor in self.extractors:
            # Extract features
            features, names = extractor.extract(graph, cache)

            # Validate
            extractor.validate_features(features, names, n_vertices)

            # Prefix feature names with extractor name for clarity
            # Example: 'weight_based.total_weight'
            prefixed_names = [f"{extractor.name}.{name}" for name in names]

            all_features.append(features)
            all_names.extend(prefixed_names)

        # Combine all features horizontally
        combined_features = np.hstack(all_features)

        # Final validation
        self._validate_combined_features(combined_features, all_names, n_vertices)

        return combined_features, all_names

    def _validate_combined_features(
        self,
        features: np.ndarray,
        feature_names: List[str],
        n_vertices: int
    ) -> None:
        """
        Validate the final combined feature matrix.

        Args:
            features: Combined feature matrix
            feature_names: Combined feature names
            n_vertices: Expected number of vertices

        Raises:
            FeatureValidationError: If validation fails
        """
        # Check shape consistency
        if features.shape[0] != n_vertices:
            raise FeatureValidationError(
                f"Combined features have {features.shape[0]} rows, expected {n_vertices}"
            )

        if features.shape[1] != len(feature_names):
            raise FeatureValidationError(
                f"Feature count mismatch: {features.shape[1]} columns, "
                f"{len(feature_names)} names"
            )

        # Check for duplicate feature names
        unique_names = set(feature_names)
        if len(unique_names) != len(feature_names):
            duplicates = [
                name for name in unique_names
                if feature_names.count(name) > 1
            ]
            raise FeatureValidationError(
                f"Duplicate feature names found: {duplicates}"
            )

        # Check for constant features (zero variance)
        variances = np.var(features, axis=0)
        constant_features = [
            feature_names[i]
            for i in range(len(feature_names))
            if variances[i] == 0
        ]

        # Note: We don't raise an error for constant features, just warn
        # Some features might legitimately be constant (e.g., graph size)
        if constant_features:
            # Store in cache for later inspection if needed
            pass

    def get_extractor_names(self) -> List[str]:
        """
        Get names of all extractors in the pipeline.

        Returns:
            List of extractor names
        """
        return [extractor.name for extractor in self.extractors]

    def get_feature_count(self) -> int:
        """
        Get total number of features that will be extracted.

        Returns:
            Total feature count across all extractors
        """
        return sum(len(ext.get_feature_names()) for ext in self.extractors)

    def __repr__(self) -> str:
        extractor_names = self.get_extractor_names()
        return (
            f"FeatureExtractorPipeline("
            f"extractors={extractor_names}, "
            f"total_features={self.get_feature_count()})"
        )
