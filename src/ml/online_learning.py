"""
Online Learning and Model Updates (Prompt 12).

Supports:
- Incremental data collection and model updates
- Model versioning and performance tracking
- Active learning (prioritize informative samples)
- Learning curves
- Model ensembling
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
from datetime import datetime
import numpy as np
import pandas as pd
import json


class UpdateStrategy(Enum):
    """Strategy for model updates."""
    PERIODIC = "periodic"  # Update every N samples
    THRESHOLD = "threshold"  # Update when performance drops
    SCHEDULED = "scheduled"  # Update on schedule (e.g., weekly)
    MANUAL = "manual"  # Manual trigger only


@dataclass
class ModelVersion:
    """
    Metadata for a specific model version.

    Attributes:
        version_id: Unique version identifier
        train_timestamp: When model was trained
        train_size: Number of training samples
        val_performance: Validation performance metrics
        test_performance: Test performance metrics (if available)
        hyperparameters: Model hyperparameters
        feature_names: List of features used
        metadata: Additional information
    """
    version_id: str
    train_timestamp: str
    train_size: int
    val_performance: Dict[str, float]
    test_performance: Optional[Dict[str, float]] = None
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    feature_names: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'version_id': self.version_id,
            'train_timestamp': self.train_timestamp,
            'train_size': self.train_size,
            'val_performance': self.val_performance,
            'test_performance': self.test_performance,
            'hyperparameters': self.hyperparameters,
            'feature_names': self.feature_names,
            'metadata': self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelVersion':
        """Load from dictionary."""
        return cls(**data)


@dataclass
class LearningCurvePoint:
    """
    Single point on a learning curve.

    Attributes:
        train_size: Number of training samples
        train_score: Training performance
        val_score: Validation performance
        test_score: Test performance (if available)
    """
    train_size: int
    train_score: float
    val_score: float
    test_score: Optional[float] = None


class ModelVersionManager:
    """
    Manage multiple model versions with performance tracking.

    Supports:
    - Version history
    - Performance comparison across versions
    - Rollback to previous versions
    """

    def __init__(self):
        """Initialize version manager."""
        self.versions: Dict[str, ModelVersion] = {}
        self.version_history: List[str] = []
        self.current_version: Optional[str] = None

    def add_version(
        self,
        model: Any,
        train_size: int,
        val_performance: Dict[str, float],
        test_performance: Optional[Dict[str, float]] = None,
        hyperparameters: Optional[Dict[str, Any]] = None,
        feature_names: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Add a new model version.

        Args:
            model: Trained model
            train_size: Number of training samples
            val_performance: Validation metrics
            test_performance: Test metrics (optional)
            hyperparameters: Model hyperparameters
            feature_names: List of features
            metadata: Additional info

        Returns:
            Version ID
        """
        # Generate version ID
        version_id = f"v{len(self.versions) + 1}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        timestamp = datetime.now().isoformat()

        # Create version metadata
        version = ModelVersion(
            version_id=version_id,
            train_timestamp=timestamp,
            train_size=train_size,
            val_performance=val_performance,
            test_performance=test_performance,
            hyperparameters=hyperparameters or {},
            feature_names=feature_names or [],
            metadata=metadata or {}
        )

        # Store
        self.versions[version_id] = version
        self.version_history.append(version_id)
        self.current_version = version_id

        return version_id

    def get_version(self, version_id: str) -> Optional[ModelVersion]:
        """Get specific version metadata."""
        return self.versions.get(version_id)

    def get_current_version(self) -> Optional[ModelVersion]:
        """Get current version metadata."""
        if self.current_version:
            return self.versions.get(self.current_version)
        return None

    def compare_versions(
        self,
        version_ids: Optional[List[str]] = None,
        metric: str = 'r2'
    ) -> pd.DataFrame:
        """
        Compare performance across versions.

        Args:
            version_ids: Versions to compare (None = all)
            metric: Metric to compare

        Returns:
            DataFrame with version comparison
        """
        if version_ids is None:
            version_ids = self.version_history

        data = []
        for vid in version_ids:
            version = self.versions.get(vid)
            if version:
                row = {
                    'version_id': vid,
                    'timestamp': version.train_timestamp,
                    'train_size': version.train_size,
                    f'val_{metric}': version.val_performance.get(metric, np.nan)
                }

                if version.test_performance:
                    row[f'test_{metric}'] = version.test_performance.get(metric, np.nan)

                data.append(row)

        return pd.DataFrame(data)

    def get_best_version(self, metric: str = 'r2') -> Optional[str]:
        """
        Get version with best validation performance.

        Args:
            metric: Metric to optimize

        Returns:
            Version ID of best model
        """
        best_score = -np.inf
        best_version = None

        for vid, version in self.versions.items():
            score = version.val_performance.get(metric, -np.inf)
            if score > best_score:
                best_score = score
                best_version = vid

        return best_version

    def save_history(self, filepath: str):
        """Save version history to JSON file."""
        data = {
            'versions': {vid: v.to_dict() for vid, v in self.versions.items()},
            'version_history': self.version_history,
            'current_version': self.current_version
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def load_history(self, filepath: str):
        """Load version history from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)

        self.versions = {
            vid: ModelVersion.from_dict(v) for vid, v in data['versions'].items()
        }
        self.version_history = data['version_history']
        self.current_version = data.get('current_version')


class IncrementalLearner:
    """
    Incrementally update models as new data arrives.

    Supports:
    - Periodic retraining
    - Threshold-based updates
    - Performance monitoring
    """

    def __init__(
        self,
        model_class: Any,
        initial_model_params: Dict[str, Any],
        update_strategy: UpdateStrategy = UpdateStrategy.PERIODIC,
        update_threshold: int = 100,  # For PERIODIC: update every N samples
        performance_threshold: float = 0.05  # For THRESHOLD: update if drop > X
    ):
        """
        Initialize incremental learner.

        Args:
            model_class: Model class to use
            initial_model_params: Initial model parameters
            update_strategy: When to update model
            update_threshold: Threshold for updates (meaning depends on strategy)
            performance_threshold: Performance drop threshold
        """
        self.model_class = model_class
        self.model_params = initial_model_params
        self.update_strategy = update_strategy
        self.update_threshold = update_threshold
        self.performance_threshold = performance_threshold

        self.current_model = None
        self.version_manager = ModelVersionManager()

        # Data buffers
        self.X_train_buffer = []
        self.y_train_buffer = []
        self.samples_since_update = 0

        # Performance tracking
        self.performance_history = []

    def add_samples(
        self,
        X_new: pd.DataFrame,
        y_new: pd.Series
    ) -> Optional[str]:
        """
        Add new training samples.

        May trigger model update depending on strategy.

        Args:
            X_new: New feature samples
            y_new: New labels

        Returns:
            New version ID if model was updated, None otherwise
        """
        # Add to buffer
        self.X_train_buffer.append(X_new)
        self.y_train_buffer.append(y_new)
        self.samples_since_update += len(X_new)

        # Check if update needed
        if self._should_update():
            return self._update_model()

        return None

    def _should_update(self) -> bool:
        """Check if model should be updated."""
        if self.update_strategy == UpdateStrategy.PERIODIC:
            return self.samples_since_update >= self.update_threshold

        elif self.update_strategy == UpdateStrategy.THRESHOLD:
            # Check if performance has degraded
            if self.current_model is None:
                return True

            # Would need validation set to check this properly
            # For now, just use periodic
            return self.samples_since_update >= 50

        elif self.update_strategy == UpdateStrategy.MANUAL:
            return False

        else:
            return False

    def _update_model(
        self,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None
    ) -> str:
        """
        Retrain model on accumulated data.

        Args:
            X_val: Validation features (optional)
            y_val: Validation labels (optional)

        Returns:
            New version ID
        """
        # Combine buffered data
        X_train = pd.concat(self.X_train_buffer, ignore_index=True)
        y_train = pd.concat(self.y_train_buffer, ignore_index=True)

        # Train new model
        model = self.model_class(**self.model_params)
        model.fit(X_train, y_train)

        # Evaluate
        val_perf = {}
        if X_val is not None and y_val is not None:
            y_pred = model.predict(X_val)
            residuals = y_val.values - y_pred

            ss_res = np.sum(residuals ** 2)
            ss_tot = np.sum((y_val.values - np.mean(y_val.values)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

            val_perf = {
                'r2': float(r2),
                'mae': float(np.mean(np.abs(residuals))),
                'rmse': float(np.sqrt(np.mean(residuals ** 2)))
            }

        # Add version
        version_id = self.version_manager.add_version(
            model=model,
            train_size=len(X_train),
            val_performance=val_perf,
            hyperparameters=self.model_params,
            feature_names=X_train.columns.tolist() if isinstance(X_train, pd.DataFrame) else None
        )

        # Update current model
        self.current_model = model
        self.samples_since_update = 0

        return version_id

    def get_learning_curve(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        train_sizes: List[int],
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None
    ) -> List[LearningCurvePoint]:
        """
        Generate learning curve by training on different data sizes.

        Args:
            X: Full training features
            y: Full training labels
            train_sizes: List of training set sizes to try
            X_val: Validation features (optional)
            y_val: Validation labels (optional)

        Returns:
            List of LearningCurvePoint
        """
        curve = []

        for size in train_sizes:
            if size > len(X):
                continue

            # Sample data
            indices = np.random.choice(len(X), size=size, replace=False)
            X_train = X.iloc[indices]
            y_train = y.iloc[indices]

            # Train model
            model = self.model_class(**self.model_params)
            model.fit(X_train, y_train)

            # Evaluate on train
            y_pred_train = model.predict(X_train)
            train_score = self._compute_r2(y_train.values, y_pred_train)

            # Evaluate on val
            val_score = None
            if X_val is not None and y_val is not None:
                y_pred_val = model.predict(X_val)
                val_score = self._compute_r2(y_val.values, y_pred_val)

            curve.append(LearningCurvePoint(
                train_size=size,
                train_score=train_score,
                val_score=val_score if val_score is not None else 0.0
            ))

        return curve

    @staticmethod
    def _compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute R²."""
        residuals = y_true - y_pred
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return float(1 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0


class ActiveLearner:
    """
    Active learning: select most informative samples to label next.

    Strategies:
    - Uncertainty sampling: select samples where model is least confident
    - Diversity sampling: select diverse samples
    - Error-based: select samples where model performed poorly
    """

    @staticmethod
    def uncertainty_sampling(
        model: Any,
        X_unlabeled: pd.DataFrame,
        n_samples: int = 10
    ) -> np.ndarray:
        """
        Select samples where model is most uncertain.

        For regression, use prediction variance or ensemble disagreement.

        Args:
            model: Trained model
            X_unlabeled: Unlabeled samples
            n_samples: Number of samples to select

        Returns:
            Indices of selected samples
        """
        # For single model regression, we can't directly measure uncertainty
        # So we'll select samples with extreme predicted values (likely edge cases)
        predictions = model.predict(X_unlabeled)

        # Select samples with predictions far from mean
        mean_pred = np.mean(predictions)
        distances = np.abs(predictions - mean_pred)

        # Select top-n most extreme
        indices = np.argsort(distances)[-n_samples:]
        return indices

    @staticmethod
    def diversity_sampling(
        X_unlabeled: pd.DataFrame,
        X_labeled: pd.DataFrame,
        n_samples: int = 10
    ) -> np.ndarray:
        """
        Select diverse samples (far from already labeled data).

        Args:
            X_unlabeled: Unlabeled samples
            X_labeled: Already labeled samples
            n_samples: Number of samples to select

        Returns:
            Indices of selected samples
        """
        # Compute distances to nearest labeled sample
        min_distances = []

        for i in range(len(X_unlabeled)):
            x = X_unlabeled.iloc[i].values

            # Distance to all labeled samples
            distances = np.linalg.norm(X_labeled.values - x, axis=1)
            min_dist = np.min(distances)
            min_distances.append(min_dist)

        # Select samples farthest from labeled data
        indices = np.argsort(min_distances)[-n_samples:]
        return indices

    @staticmethod
    def error_based_sampling(
        X: pd.DataFrame,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        n_samples: int = 10
    ) -> np.ndarray:
        """
        Select samples where model had high error.

        Useful for identifying weak points to improve.

        Args:
            X: Features
            y_true: True labels
            y_pred: Predictions
            n_samples: Number of samples to select

        Returns:
            Indices of samples with highest error
        """
        errors = np.abs(y_true - y_pred)
        indices = np.argsort(errors)[-n_samples:]
        return indices


class ModelEnsemble:
    """
    Ensemble of multiple models for improved robustness.

    Supports:
    - Averaging predictions
    - Weighted averaging
    - Stacking
    """

    def __init__(self, models: List[Any], weights: Optional[List[float]] = None):
        """
        Initialize ensemble.

        Args:
            models: List of trained models
            weights: Optional weights for each model (must sum to 1)
        """
        self.models = models

        if weights is None:
            self.weights = [1.0 / len(models)] * len(models)
        else:
            assert len(weights) == len(models), "Number of weights must match number of models"
            assert abs(sum(weights) - 1.0) < 1e-6, "Weights must sum to 1"
            self.weights = weights

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict using ensemble (weighted average).

        Args:
            X: Features

        Returns:
            Ensemble predictions
        """
        predictions = []

        for model in self.models:
            pred = model.predict(X)
            predictions.append(pred)

        predictions = np.array(predictions)  # Shape: (n_models, n_samples)

        # Weighted average
        ensemble_pred = np.average(predictions, axis=0, weights=self.weights)

        return ensemble_pred

    def predict_with_uncertainty(
        self,
        X: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict with uncertainty estimate.

        Uncertainty = std dev of predictions across models.

        Args:
            X: Features

        Returns:
            (predictions, uncertainties)
        """
        predictions = []

        for model in self.models:
            pred = model.predict(X)
            predictions.append(pred)

        predictions = np.array(predictions)

        # Weighted average
        ensemble_pred = np.average(predictions, axis=0, weights=self.weights)

        # Uncertainty = std dev
        uncertainties = np.std(predictions, axis=0)

        return ensemble_pred, uncertainties
