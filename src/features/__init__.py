"""
Feature Engineering System for TSP Anchor Prediction

This package provides modular feature extraction from graphs and vertices
to support machine learning prediction of optimal anchor vertices.

Core Components:
- base: Abstract base classes and interfaces
- extractors: Modular feature extraction implementations
- pipeline: Orchestration and caching
- analysis: Feature validation and analysis tools

Usage:
    from features import FeatureExtractorPipeline, FeatureAnalyzer

    pipeline = FeatureExtractorPipeline()
    feature_matrix, feature_names = pipeline.extract_features(graph)

    analyzer = FeatureAnalyzer(feature_matrix, feature_names)
    report = analyzer.summary_report()
"""

from .base import VertexFeatureExtractor, FeatureValidationError
from .pipeline import FeatureExtractorPipeline
from .analysis import FeatureAnalyzer

__all__ = [
    'VertexFeatureExtractor',
    'FeatureValidationError',
    'FeatureExtractorPipeline',
    'FeatureAnalyzer',
]
