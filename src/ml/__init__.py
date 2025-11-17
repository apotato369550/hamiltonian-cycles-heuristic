"""
Machine Learning Component for TSP Anchor Prediction (Phase 4).

This package provides ML models to predict which vertices make good TSP tour
starting points (anchors) based on structural graph features.

Key modules:
- dataset: Dataset preparation and train/test splitting
- models: Linear regression and tree-based models
- evaluation: Model evaluation and comparison
- interpretation: Model interpretation and feature importance
"""

from .dataset import (
    MLProblemType,
    DatasetPreparator,
    SplitStrategy,
    TrainTestSplitter,
    DatasetSplit
)

from .models import (
    LinearRegressionModel,
    TreeBasedModel,
    ModelType
)

__all__ = [
    # Dataset
    'MLProblemType',
    'DatasetPreparator',
    'SplitStrategy',
    'TrainTestSplitter',
    'DatasetSplit',

    # Models
    'LinearRegressionModel',
    'TreeBasedModel',
    'ModelType',
]
