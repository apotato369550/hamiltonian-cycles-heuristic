"""
Cross-Validation Strategy (Prompt 6).

Implements:
- K-fold cross-validation
- Stratified k-fold (by graph type)
- Group k-fold (graph-level splits)
- Nested cross-validation (hyperparameter tuning + evaluation)
- Leave-one-graph-out
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable, Tuple
from enum import Enum
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold


class CVStrategy(Enum):
    """Cross-validation strategy."""
    K_FOLD = "k_fold"
    STRATIFIED_K_FOLD = "stratified_k_fold"
    GROUP_K_FOLD = "group_k_fold"
    LEAVE_ONE_GROUP_OUT = "leave_one_group_out"
    NESTED = "nested"


@dataclass
class CVFold:
    """
    Single fold of cross-validation.

    Attributes:
        fold_id: Fold number (0-indexed)
        train_indices: Indices for training
        val_indices: Indices for validation
        train_groups: Graph IDs in training set (if applicable)
        val_groups: Graph IDs in validation set (if applicable)
    """
    fold_id: int
    train_indices: np.ndarray
    val_indices: np.ndarray
    train_groups: Optional[List[Any]] = None
    val_groups: Optional[List[Any]] = None


@dataclass
class CVResult:
    """
    Result of cross-validation.

    Attributes:
        strategy: CV strategy used
        n_folds: Number of folds
        fold_metrics: List of metrics for each fold
        mean_metrics: Mean of metrics across folds
        std_metrics: Std dev of metrics across folds
        best_fold: Fold with best performance
        worst_fold: Fold with worst performance
        metadata: Additional information
    """
    strategy: CVStrategy
    n_folds: int
    fold_metrics: List[Dict[str, float]]
    mean_metrics: Dict[str, float]
    std_metrics: Dict[str, float]
    best_fold: int
    worst_fold: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_summary(self) -> str:
        """Get human-readable summary."""
        summary = []
        summary.append(f"Cross-Validation ({self.strategy.value}):")
        summary.append(f"  Folds: {self.n_folds}")

        for metric, mean_val in self.mean_metrics.items():
            std_val = self.std_metrics[metric]
            summary.append(f"  {metric}: {mean_val:.4f} ± {std_val:.4f}")

        return "\n".join(summary)


class CrossValidator:
    """
    Flexible cross-validation framework.

    Supports multiple CV strategies with consistent API.
    """

    def __init__(
        self,
        strategy: CVStrategy = CVStrategy.GROUP_K_FOLD,
        n_folds: int = 5,
        random_seed: Optional[int] = None,
        shuffle: bool = True
    ):
        """
        Initialize cross-validator.

        Args:
            strategy: CV strategy to use
            n_folds: Number of folds
            random_seed: Random seed for reproducibility
            shuffle: Whether to shuffle data before splitting
        """
        self.strategy = strategy
        self.n_folds = n_folds
        self.random_seed = random_seed
        self.shuffle = shuffle

    def split(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None
    ) -> List[CVFold]:
        """
        Generate train/validation splits.

        Args:
            X: Feature matrix
            y: Labels (required for stratified)
            groups: Group labels (required for group-based)

        Returns:
            List of CVFold objects
        """
        n_samples = len(X)

        if self.strategy == CVStrategy.K_FOLD:
            return self._k_fold_split(n_samples)

        elif self.strategy == CVStrategy.STRATIFIED_K_FOLD:
            if y is None:
                raise ValueError("y required for stratified k-fold")
            return self._stratified_k_fold_split(y)

        elif self.strategy == CVStrategy.GROUP_K_FOLD:
            if groups is None:
                raise ValueError("groups required for group k-fold")
            return self._group_k_fold_split(n_samples, groups)

        elif self.strategy == CVStrategy.LEAVE_ONE_GROUP_OUT:
            if groups is None:
                raise ValueError("groups required for leave-one-group-out")
            return self._leave_one_group_out_split(n_samples, groups)

        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def _k_fold_split(self, n_samples: int) -> List[CVFold]:
        """Standard k-fold split."""
        kf = KFold(
            n_splits=self.n_folds,
            shuffle=self.shuffle,
            random_state=self.random_seed
        )

        folds = []
        for fold_id, (train_idx, val_idx) in enumerate(kf.split(np.arange(n_samples))):
            folds.append(CVFold(
                fold_id=fold_id,
                train_indices=train_idx,
                val_indices=val_idx
            ))

        return folds

    def _stratified_k_fold_split(self, y: np.ndarray) -> List[CVFold]:
        """Stratified k-fold split (balanced classes)."""
        skf = StratifiedKFold(
            n_splits=self.n_folds,
            shuffle=self.shuffle,
            random_state=self.random_seed
        )

        folds = []
        for fold_id, (train_idx, val_idx) in enumerate(skf.split(np.arange(len(y)), y)):
            folds.append(CVFold(
                fold_id=fold_id,
                train_indices=train_idx,
                val_indices=val_idx
            ))

        return folds

    def _group_k_fold_split(
        self,
        n_samples: int,
        groups: np.ndarray
    ) -> List[CVFold]:
        """Group k-fold split (entire groups in single fold)."""
        gkf = GroupKFold(n_splits=self.n_folds)

        folds = []
        for fold_id, (train_idx, val_idx) in enumerate(gkf.split(
            np.arange(n_samples),
            groups=groups
        )):
            # Extract group IDs
            train_groups = list(set(groups[train_idx]))
            val_groups = list(set(groups[val_idx]))

            folds.append(CVFold(
                fold_id=fold_id,
                train_indices=train_idx,
                val_indices=val_idx,
                train_groups=train_groups,
                val_groups=val_groups
            ))

        return folds

    def _leave_one_group_out_split(
        self,
        n_samples: int,
        groups: np.ndarray
    ) -> List[CVFold]:
        """Leave-one-group-out cross-validation."""
        unique_groups = np.unique(groups)
        folds = []

        for fold_id, val_group in enumerate(unique_groups):
            val_idx = np.where(groups == val_group)[0]
            train_idx = np.where(groups != val_group)[0]

            folds.append(CVFold(
                fold_id=fold_id,
                train_indices=train_idx,
                val_indices=val_idx,
                train_groups=list(unique_groups[unique_groups != val_group]),
                val_groups=[val_group]
            ))

        return folds

    def cross_validate(
        self,
        model_class: Any,
        model_params: Dict[str, Any],
        X: np.ndarray,
        y: np.ndarray,
        groups: Optional[np.ndarray] = None,
        metric_fn: Optional[Callable] = None
    ) -> CVResult:
        """
        Perform cross-validation with a model.

        Args:
            model_class: Model class to instantiate
            model_params: Parameters for model initialization
            X: Feature matrix
            y: Labels
            groups: Group labels (for group-based CV)
            metric_fn: Function(y_true, y_pred) -> dict of metrics

        Returns:
            CVResult with metrics for all folds
        """
        # Default metric function
        if metric_fn is None:
            metric_fn = self._default_metric_fn

        # Generate folds
        folds = self.split(X, y, groups)

        # Train and evaluate on each fold
        fold_metrics = []

        for fold in folds:
            # Split data
            X_train = X[fold.train_indices]
            y_train = y[fold.train_indices]
            X_val = X[fold.val_indices]
            y_val = y[fold.val_indices]

            # Train model
            model = model_class(**model_params)
            model.fit(X_train, y_train)

            # Predict
            y_pred = model.predict(X_val)

            # Compute metrics
            metrics = metric_fn(y_val, y_pred)
            fold_metrics.append(metrics)

        # Aggregate metrics
        mean_metrics, std_metrics = self._aggregate_metrics(fold_metrics)

        # Find best/worst folds (by R² or first metric)
        primary_metric = list(mean_metrics.keys())[0]
        fold_scores = [m[primary_metric] for m in fold_metrics]
        best_fold = int(np.argmax(fold_scores))
        worst_fold = int(np.argmin(fold_scores))

        return CVResult(
            strategy=self.strategy,
            n_folds=len(folds),
            fold_metrics=fold_metrics,
            mean_metrics=mean_metrics,
            std_metrics=std_metrics,
            best_fold=best_fold,
            worst_fold=worst_fold,
            metadata={
                'random_seed': self.random_seed,
                'shuffle': self.shuffle
            }
        )

    @staticmethod
    def _default_metric_fn(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Default metric function (R², MAE, RMSE)."""
        residuals = y_true - y_pred
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        return {
            'r2': float(r2),
            'mae': float(np.mean(np.abs(residuals))),
            'rmse': float(np.sqrt(np.mean(residuals ** 2)))
        }

    @staticmethod
    def _aggregate_metrics(
        fold_metrics: List[Dict[str, float]]
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Compute mean and std of metrics across folds."""
        metric_names = fold_metrics[0].keys()

        mean_metrics = {}
        std_metrics = {}

        for metric in metric_names:
            values = [m[metric] for m in fold_metrics]
            mean_metrics[metric] = float(np.mean(values))
            std_metrics[metric] = float(np.std(values))

        return mean_metrics, std_metrics


class NestedCrossValidator:
    """
    Nested cross-validation for hyperparameter tuning + performance estimation.

    Outer loop: Performance estimation
    Inner loop: Hyperparameter tuning
    """

    def __init__(
        self,
        outer_cv: CrossValidator,
        inner_cv: CrossValidator
    ):
        """
        Initialize nested CV.

        Args:
            outer_cv: Cross-validator for outer loop (performance estimation)
            inner_cv: Cross-validator for inner loop (hyperparameter tuning)
        """
        self.outer_cv = outer_cv
        self.inner_cv = inner_cv

    def nested_cross_validate(
        self,
        model_class: Any,
        param_grid: List[Dict[str, Any]],
        X: np.ndarray,
        y: np.ndarray,
        groups: Optional[np.ndarray] = None,
        metric_fn: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Perform nested cross-validation.

        Args:
            model_class: Model class to instantiate
            param_grid: List of parameter dictionaries to try
            X: Feature matrix
            y: Labels
            groups: Group labels
            metric_fn: Metric function

        Returns:
            Dictionary with:
            - outer_fold_metrics: Metrics for each outer fold
            - best_params_per_fold: Best params found in each outer fold
            - mean_outer_metrics: Mean performance across outer folds
            - std_outer_metrics: Std dev across outer folds
        """
        if metric_fn is None:
            metric_fn = CrossValidator._default_metric_fn

        # Outer folds
        outer_folds = self.outer_cv.split(X, y, groups)

        outer_fold_metrics = []
        best_params_per_fold = []

        for outer_fold in outer_folds:
            # Split data for outer fold
            X_train_outer = X[outer_fold.train_indices]
            y_train_outer = y[outer_fold.train_indices]
            X_test_outer = X[outer_fold.val_indices]
            y_test_outer = y[outer_fold.val_indices]

            groups_train_outer = None
            if groups is not None:
                groups_train_outer = groups[outer_fold.train_indices]

            # Inner cross-validation for hyperparameter tuning
            best_params = None
            best_score = -np.inf

            for params in param_grid:
                # Evaluate params using inner CV
                inner_result = self.inner_cv.cross_validate(
                    model_class=model_class,
                    model_params=params,
                    X=X_train_outer,
                    y=y_train_outer,
                    groups=groups_train_outer,
                    metric_fn=metric_fn
                )

                # Use R² as primary metric for comparison
                score = inner_result.mean_metrics.get('r2', 0.0)

                if score > best_score:
                    best_score = score
                    best_params = params

            # Train final model with best params on full outer training set
            model = model_class(**best_params)
            model.fit(X_train_outer, y_train_outer)

            # Evaluate on outer test set
            y_pred_outer = model.predict(X_test_outer)
            outer_metrics = metric_fn(y_test_outer, y_pred_outer)

            outer_fold_metrics.append(outer_metrics)
            best_params_per_fold.append(best_params)

        # Aggregate outer fold metrics
        mean_outer, std_outer = CrossValidator._aggregate_metrics(outer_fold_metrics)

        return {
            'outer_fold_metrics': outer_fold_metrics,
            'best_params_per_fold': best_params_per_fold,
            'mean_outer_metrics': mean_outer,
            'std_outer_metrics': std_outer,
            'n_outer_folds': len(outer_folds),
            'n_inner_folds': self.inner_cv.n_folds
        }
