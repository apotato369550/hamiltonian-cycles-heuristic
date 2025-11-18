"""
Hyperparameter Tuning (Prompt 7).

Implements:
- Grid search
- Random search
- Hyperparameter search spaces for each model type
- Tuning with cross-validation
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable
from enum import Enum
import numpy as np
import itertools


class TuningStrategy(Enum):
    """Hyperparameter tuning strategy."""
    GRID_SEARCH = "grid_search"
    RANDOM_SEARCH = "random_search"


@dataclass
class HyperparameterConfig:
    """
    Configuration for a single hyperparameter.

    Attributes:
        name: Parameter name
        values: List of values (for grid search)
        distribution: Distribution type (for random search): 'uniform', 'loguniform', 'choice'
        range: Range for continuous distributions (min, max)
    """
    name: str
    values: Optional[List[Any]] = None
    distribution: Optional[str] = None
    range: Optional[tuple] = None


@dataclass
class TuningResult:
    """
    Result of hyperparameter tuning.

    Attributes:
        best_params: Best hyperparameters found
        best_score: Best validation score
        all_params: All parameter combinations tried
        all_scores: Validation scores for all combinations
        n_trials: Number of trials performed
        cv_results: Cross-validation results for best params
    """
    best_params: Dict[str, Any]
    best_score: float
    all_params: List[Dict[str, Any]]
    all_scores: List[float]
    n_trials: int
    cv_results: Optional[Dict[str, Any]] = None

    def get_summary(self) -> str:
        """Get human-readable summary."""
        summary = []
        summary.append(f"Hyperparameter Tuning Results:")
        summary.append(f"  Trials: {self.n_trials}")
        summary.append(f"  Best score: {self.best_score:.4f}")
        summary.append(f"  Best params:")
        for param, value in self.best_params.items():
            summary.append(f"    {param}: {value}")
        return "\n".join(summary)


class HyperparameterTuner:
    """
    Hyperparameter tuning framework.

    Supports grid search and random search with cross-validation.
    """

    def __init__(
        self,
        strategy: TuningStrategy = TuningStrategy.GRID_SEARCH,
        n_trials: Optional[int] = None,  # For random search
        random_seed: Optional[int] = None,
        scoring_metric: str = 'r2'  # Metric to optimize
    ):
        """
        Initialize tuner.

        Args:
            strategy: Tuning strategy
            n_trials: Number of trials (for random search)
            random_seed: Random seed for reproducibility
            scoring_metric: Metric to optimize ('r2', 'mae', 'rmse')
        """
        self.strategy = strategy
        self.n_trials = n_trials
        self.random_seed = random_seed
        self.scoring_metric = scoring_metric

        if random_seed is not None:
            np.random.seed(random_seed)

    def tune(
        self,
        model_class: Any,
        param_grid: Dict[str, List[Any]],
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        metric_fn: Optional[Callable] = None
    ) -> TuningResult:
        """
        Tune hyperparameters on validation set.

        Args:
            model_class: Model class to instantiate
            param_grid: Dictionary mapping param_name -> list of values
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            metric_fn: Function(y_true, y_pred) -> dict of metrics

        Returns:
            TuningResult
        """
        if metric_fn is None:
            metric_fn = self._default_metric_fn

        if self.strategy == TuningStrategy.GRID_SEARCH:
            return self._grid_search(
                model_class, param_grid, X_train, y_train, X_val, y_val, metric_fn
            )
        elif self.strategy == TuningStrategy.RANDOM_SEARCH:
            return self._random_search(
                model_class, param_grid, X_train, y_train, X_val, y_val, metric_fn
            )
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def _grid_search(
        self,
        model_class: Any,
        param_grid: Dict[str, List[Any]],
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        metric_fn: Callable
    ) -> TuningResult:
        """Exhaustive grid search."""
        # Generate all combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        all_combinations = list(itertools.product(*param_values))

        all_params = []
        all_scores = []

        best_params = None
        best_score = -np.inf

        for combination in all_combinations:
            # Create param dict
            params = dict(zip(param_names, combination))

            # Train model
            model = model_class(**params)
            model.fit(X_train, y_train)

            # Evaluate
            y_pred = model.predict(X_val)
            metrics = metric_fn(y_val, y_pred)
            score = metrics[self.scoring_metric]

            all_params.append(params)
            all_scores.append(score)

            # Update best
            if score > best_score:
                best_score = score
                best_params = params

        return TuningResult(
            best_params=best_params,
            best_score=float(best_score),
            all_params=all_params,
            all_scores=all_scores,
            n_trials=len(all_combinations)
        )

    def _random_search(
        self,
        model_class: Any,
        param_distributions: Dict[str, List[Any]],
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        metric_fn: Callable
    ) -> TuningResult:
        """Random search over parameter distributions."""
        if self.n_trials is None:
            raise ValueError("n_trials required for random search")

        all_params = []
        all_scores = []

        best_params = None
        best_score = -np.inf

        for _ in range(self.n_trials):
            # Sample random parameters
            params = {}
            for param_name, values in param_distributions.items():
                params[param_name] = np.random.choice(values)

            # Train model
            model = model_class(**params)
            model.fit(X_train, y_train)

            # Evaluate
            y_pred = model.predict(X_val)
            metrics = metric_fn(y_val, y_pred)
            score = metrics[self.scoring_metric]

            all_params.append(params)
            all_scores.append(score)

            # Update best
            if score > best_score:
                best_score = score
                best_params = params

        return TuningResult(
            best_params=best_params,
            best_score=float(best_score),
            all_params=all_params,
            all_scores=all_scores,
            n_trials=self.n_trials
        )

    @staticmethod
    def _default_metric_fn(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Default metric function."""
        residuals = y_true - y_pred
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        return {
            'r2': float(r2),
            'mae': float(np.mean(np.abs(residuals))),
            'rmse': float(np.sqrt(np.mean(residuals ** 2)))
        }


class ModelSpecificTuner:
    """
    Pre-configured tuning for specific model types.

    Provides reasonable search spaces for each model.
    """

    @staticmethod
    def get_linear_param_grid(model_type: str = "ridge") -> Dict[str, List[Any]]:
        """
        Get parameter grid for linear models.

        Args:
            model_type: 'ridge', 'lasso', 'elasticnet'

        Returns:
            Parameter grid
        """
        if model_type == "ridge":
            return {
                'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]
            }
        elif model_type == "lasso":
            return {
                'alpha': [0.001, 0.01, 0.1, 1.0, 10.0]
            }
        elif model_type == "elasticnet":
            return {
                'alpha': [0.01, 0.1, 1.0, 10.0],
                'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
            }
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    @staticmethod
    def get_tree_param_grid(model_type: str = "random_forest") -> Dict[str, List[Any]]:
        """
        Get parameter grid for tree models.

        Args:
            model_type: 'decision_tree', 'random_forest', 'gradient_boosting'

        Returns:
            Parameter grid
        """
        if model_type == "decision_tree":
            return {
                'max_depth': [3, 5, 7, 10, None],
                'min_samples_leaf': [1, 5, 10, 20],
                'min_samples_split': [2, 5, 10, 20]
            }
        elif model_type == "random_forest":
            return {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_leaf': [1, 5, 10],
                'max_features': ['sqrt', 'log2']
            }
        elif model_type == "gradient_boosting":
            return {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'min_samples_leaf': [1, 5, 10]
            }
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    @staticmethod
    def get_coarse_grid(model_type: str) -> Dict[str, List[Any]]:
        """
        Get coarse grid for fast initial search.

        Args:
            model_type: Model type

        Returns:
            Coarse parameter grid
        """
        if model_type == "ridge":
            return {'alpha': [0.1, 1.0, 10.0]}
        elif model_type == "lasso":
            return {'alpha': [0.01, 0.1, 1.0]}
        elif model_type == "random_forest":
            return {
                'n_estimators': [50, 100],
                'max_depth': [5, 10],
                'min_samples_leaf': [1, 5]
            }
        elif model_type == "gradient_boosting":
            return {
                'n_estimators': [50, 100],
                'learning_rate': [0.05, 0.1],
                'max_depth': [3, 5]
            }
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    @staticmethod
    def get_fine_grid(model_type: str, best_params: Dict[str, Any]) -> Dict[str, List[Any]]:
        """
        Get fine grid around best parameters from coarse search.

        Args:
            model_type: Model type
            best_params: Best parameters from coarse search

        Returns:
            Fine-grained parameter grid
        """
        fine_grid = {}

        if model_type in ["ridge", "lasso"]:
            # Refine alpha
            best_alpha = best_params.get('alpha', 1.0)
            fine_grid['alpha'] = [
                best_alpha * 0.1,
                best_alpha * 0.5,
                best_alpha,
                best_alpha * 2.0,
                best_alpha * 10.0
            ]

        elif model_type == "random_forest":
            # Refine n_estimators
            best_n = best_params.get('n_estimators', 100)
            fine_grid['n_estimators'] = [max(10, best_n - 50), best_n, best_n + 50]

            # Refine max_depth
            best_depth = best_params.get('max_depth', 10)
            if best_depth is not None:
                fine_grid['max_depth'] = [max(2, best_depth - 2), best_depth, best_depth + 2]
            else:
                fine_grid['max_depth'] = [10, 15, None]

            # Refine min_samples_leaf
            best_leaf = best_params.get('min_samples_leaf', 1)
            fine_grid['min_samples_leaf'] = [max(1, best_leaf - 2), best_leaf, best_leaf + 2]

        elif model_type == "gradient_boosting":
            # Similar refinement for GB
            best_n = best_params.get('n_estimators', 100)
            fine_grid['n_estimators'] = [max(10, best_n - 50), best_n, best_n + 50]

            best_lr = best_params.get('learning_rate', 0.1)
            fine_grid['learning_rate'] = [best_lr * 0.5, best_lr, best_lr * 2.0]

            best_depth = best_params.get('max_depth', 3)
            fine_grid['max_depth'] = [max(2, best_depth - 1), best_depth, best_depth + 1]

        return fine_grid
