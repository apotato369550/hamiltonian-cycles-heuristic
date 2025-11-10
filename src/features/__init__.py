"""
Feature Engineering System for TSP Anchor Prediction

This package provides modular feature extraction from graphs and vertices
to support machine learning prediction of optimal anchor vertices.

Core Components:
- base: Abstract base classes and interfaces
- extractors: Modular feature extraction implementations
- pipeline: Orchestration and caching
- analysis: Feature validation and analysis tools
- labeling: Anchor quality labeling system
- dataset_pipeline: End-to-end dataset generation
- selection: Feature selection utilities
- transformation: Feature transformation and engineering

Usage:
    from src.features import FeatureExtractorPipeline, FeatureAnalyzer

    pipeline = FeatureExtractorPipeline()
    feature_matrix, feature_names = pipeline.extract_features(graph)

    analyzer = FeatureAnalyzer(feature_matrix, feature_names)
    report = analyzer.summary_report()
"""

# Core imports (no pandas/sklearn dependencies)
from .base import VertexFeatureExtractor, FeatureValidationError
from .pipeline import FeatureExtractorPipeline
from .analysis import FeatureAnalyzer
from .labeling import AnchorQualityLabeler, LabelingStrategy, LabelingResult

__all__ = [
    # Base infrastructure
    'VertexFeatureExtractor',
    'FeatureValidationError',
    'FeatureExtractorPipeline',
    'FeatureAnalyzer',

    # Labeling (Prompt 9)
    'AnchorQualityLabeler',
    'LabelingStrategy',
    'LabelingResult',
]

# Optional imports (require pandas/sklearn)
try:
    from .dataset_pipeline import FeatureDatasetPipeline, DatasetConfig, DatasetResult
    __all__.extend(['FeatureDatasetPipeline', 'DatasetConfig', 'DatasetResult'])
except ImportError:
    pass

try:
    from .selection import FeatureSelector, SelectionMethod, SelectionResult
    __all__.extend(['FeatureSelector', 'SelectionMethod', 'SelectionResult'])
except ImportError:
    pass

try:
    from .transformation import FeatureTransformer, TransformationType, TransformationConfig
    __all__.extend(['FeatureTransformer', 'TransformationType', 'TransformationConfig'])
except ImportError:
    pass
