"""
Feature Engineering System for TSP Anchor Prediction

This package provides modular feature extraction from graphs and vertices
to support machine learning prediction of optimal anchor vertices.

Core Components:
- base: Abstract base classes and interfaces
- extractors: Modular feature extraction implementations
- pipeline: Orchestration and caching
- validation: Feature quality checks

Usage:
    from features import FeatureExtractorPipeline

    pipeline = FeatureExtractorPipeline()
    feature_matrix, feature_names = pipeline.extract_features(graph)
"""

from .base import VertexFeatureExtractor, FeatureValidationError
from .pipeline import FeatureExtractorPipeline

__all__ = [
    'VertexFeatureExtractor',
    'FeatureValidationError',
    'FeatureExtractorPipeline',
]
