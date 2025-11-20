"""
Machine Learning Component for TSP Anchor Prediction (Phase 4).

This package provides ML models to predict which vertices make good TSP tour
starting points (anchors) based on structural graph features.

Key modules:
- dataset: Dataset preparation and train/test splitting (Prompts 1-2)
- models: Linear regression and tree-based models (Prompts 3-4)
- evaluation: Model evaluation and comparison (Prompt 5)
- cross_validation: Cross-validation strategies (Prompt 6)
- tuning: Hyperparameter tuning (Prompt 7)
- feature_engineering: Feature scaling, transformations, interactions (Prompt 8)
- interpretation: Model interpretation and explanation (Prompt 9)
- pipeline: Prediction-to-algorithm pipeline (Prompt 10)
- generalization: Model generalization testing (Prompt 11)
- online_learning: Online learning and model updates (Prompt 12)
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
    ModelType,
    ModelResult
)

from .evaluation import (
    ModelEvaluator,
    ModelComparator,
    PerformanceMatrix,
    PerformanceMetrics,
    AlgorithmPerformanceMetrics,
    ComparisonResult
)

from .cross_validation import (
    CrossValidator,
    NestedCrossValidator,
    CVStrategy,
    CVResult,
    CVFold
)

from .tuning import (
    HyperparameterTuner,
    ModelSpecificTuner,
    TuningStrategy,
    TuningResult
)

from .feature_engineering import (
    FeatureScaler,
    NonLinearTransformer,
    FeatureInteractionGenerator,
    PCAReducer,
    AdvancedFeatureSelector,
    ScalingStrategy,
    TransformationType
)

from .interpretation import (
    LinearModelInterpreter,
    TreeModelInterpreter,
    ModelInterpreter,
    CoefficientAnalysis,
    FeatureContribution,
    PartialDependenceResult,
    CaseStudyAnalyzer
)

from .pipeline import (
    MLPipeline,
    PredictionResult,
    AlgorithmExecutionResult,
    BatchPredictionResult,
    ErrorAnalyzer
)

from .generalization import (
    GeneralizationTester,
    GeneralizationResult,
    GeneralizationType,
    FailureModeAnalyzer,
    ConsistencyAnalyzer
)

from .online_learning import (
    IncrementalLearner,
    ModelVersionManager,
    ActiveLearner,
    ModelEnsemble,
    ModelVersion,
    UpdateStrategy,
    LearningCurvePoint
)

__all__ = [
    # Dataset (Prompts 1-2)
    'MLProblemType',
    'DatasetPreparator',
    'SplitStrategy',
    'TrainTestSplitter',
    'DatasetSplit',

    # Models (Prompts 3-4)
    'LinearRegressionModel',
    'TreeBasedModel',
    'ModelType',
    'ModelResult',

    # Evaluation (Prompt 5)
    'ModelEvaluator',
    'ModelComparator',
    'PerformanceMatrix',
    'PerformanceMetrics',
    'AlgorithmPerformanceMetrics',
    'ComparisonResult',

    # Cross-Validation (Prompt 6)
    'CrossValidator',
    'NestedCrossValidator',
    'CVStrategy',
    'CVResult',
    'CVFold',

    # Tuning (Prompt 7)
    'HyperparameterTuner',
    'ModelSpecificTuner',
    'TuningStrategy',
    'TuningResult',

    # Feature Engineering (Prompt 8)
    'FeatureScaler',
    'NonLinearTransformer',
    'FeatureInteractionGenerator',
    'PCAReducer',
    'AdvancedFeatureSelector',
    'ScalingStrategy',
    'TransformationType',

    # Interpretation (Prompt 9)
    'LinearModelInterpreter',
    'TreeModelInterpreter',
    'ModelInterpreter',
    'CoefficientAnalysis',
    'FeatureContribution',
    'PartialDependenceResult',
    'CaseStudyAnalyzer',

    # Pipeline (Prompt 10)
    'MLPipeline',
    'PredictionResult',
    'AlgorithmExecutionResult',
    'BatchPredictionResult',
    'ErrorAnalyzer',

    # Generalization (Prompt 11)
    'GeneralizationTester',
    'GeneralizationResult',
    'GeneralizationType',
    'FailureModeAnalyzer',
    'ConsistencyAnalyzer',

    # Online Learning (Prompt 12)
    'IncrementalLearner',
    'ModelVersionManager',
    'ActiveLearner',
    'ModelEnsemble',
    'ModelVersion',
    'UpdateStrategy',
    'LearningCurvePoint',
]
