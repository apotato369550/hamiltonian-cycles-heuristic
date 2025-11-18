"""
Model Evaluation and Comparison Framework (Prompt 5).

Provides:
- Comprehensive performance metrics (regression and classification)
- Algorithm performance metrics (predicted anchor vs baselines)
- Comparative analysis (paired tests, effect sizes)
- Per-graph-type performance breakdown
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class PerformanceMetrics:
    """
    Container for model performance metrics.

    Attributes:
        r2: Coefficient of determination (0-1, higher better)
        mae: Mean absolute error (lower better)
        rmse: Root mean squared error (lower better)
        mse: Mean squared error (lower better)
        median_ae: Median absolute error (robust to outliers)
        mean_residual: Mean of residuals (should be ~0)
        std_residual: Std dev of residuals
    """
    r2: float
    mae: float
    rmse: float
    mse: float
    median_ae: float
    mean_residual: float
    std_residual: float

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            'r2': self.r2,
            'mae': self.mae,
            'rmse': self.rmse,
            'mse': self.mse,
            'median_ae': self.median_ae,
            'mean_residual': self.mean_residual,
            'std_residual': self.std_residual
        }

    def get_summary(self) -> str:
        """Get human-readable summary."""
        return f"R²={self.r2:.3f}, MAE={self.mae:.3f}, RMSE={self.rmse:.3f}"


@dataclass
class AlgorithmPerformanceMetrics:
    """
    Metrics for actual algorithm performance (predicted anchor quality).

    Measures how well predicted anchors perform when used in TSP algorithms.

    Attributes:
        mean_tour_quality: Mean tour weight from predicted anchors
        best_anchor_quality: Mean tour weight from best anchors
        random_anchor_quality: Mean tour weight from random anchors
        beat_random_rate: Fraction of graphs where predicted beats random
        optimality_gap: Mean gap between predicted and best (%)
        success_rate: Fraction where predicted within X% of best
    """
    mean_tour_quality: float
    best_anchor_quality: float
    random_anchor_quality: float
    beat_random_rate: float
    optimality_gap: float
    success_rate: float

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            'mean_tour_quality': self.mean_tour_quality,
            'best_anchor_quality': self.best_anchor_quality,
            'random_anchor_quality': self.random_anchor_quality,
            'beat_random_rate': self.beat_random_rate,
            'optimality_gap': self.optimality_gap,
            'success_rate': self.success_rate
        }


@dataclass
class ComparisonResult:
    """
    Result of pairwise model comparison.

    Attributes:
        model_a_name: Name of first model
        model_b_name: Name of second model
        metric_name: Name of metric being compared
        mean_diff: Mean difference (A - B)
        median_diff: Median difference
        std_diff: Standard deviation of differences
        p_value: P-value from statistical test
        effect_size: Cohen's d effect size
        significant: Whether difference is statistically significant
        test_type: Type of statistical test used
    """
    model_a_name: str
    model_b_name: str
    metric_name: str
    mean_diff: float
    median_diff: float
    std_diff: float
    p_value: float
    effect_size: float
    significant: bool
    test_type: str

    def get_summary(self) -> str:
        """Get human-readable summary."""
        sig_str = "significant" if self.significant else "not significant"
        better = self.model_a_name if self.mean_diff > 0 else self.model_b_name
        return (f"{self.model_a_name} vs {self.model_b_name} on {self.metric_name}: "
                f"diff={self.mean_diff:.3f}, p={self.p_value:.4f} ({sig_str}), "
                f"effect_size={self.effect_size:.3f}, better={better}")


class ModelEvaluator:
    """
    Comprehensive model evaluation framework.

    Computes:
    - Regression metrics (R², MAE, RMSE, etc.)
    - Residual statistics
    - Per-graph-type performance
    """

    @staticmethod
    def compute_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> PerformanceMetrics:
        """
        Compute comprehensive performance metrics.

        Args:
            y_true: True values
            y_pred: Predicted values

        Returns:
            PerformanceMetrics object
        """
        # Residuals
        residuals = y_true - y_pred

        # R² (coefficient of determination)
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        # Error metrics
        mae = np.mean(np.abs(residuals))
        mse = np.mean(residuals ** 2)
        rmse = np.sqrt(mse)
        median_ae = np.median(np.abs(residuals))

        # Residual statistics
        mean_residual = np.mean(residuals)
        std_residual = np.std(residuals)

        return PerformanceMetrics(
            r2=float(r2),
            mae=float(mae),
            rmse=float(rmse),
            mse=float(mse),
            median_ae=float(median_ae),
            mean_residual=float(mean_residual),
            std_residual=float(std_residual)
        )

    @staticmethod
    def compute_per_group_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        groups: np.ndarray
    ) -> Dict[Any, PerformanceMetrics]:
        """
        Compute metrics broken down by group (e.g., graph type).

        Args:
            y_true: True values
            y_pred: Predicted values
            groups: Group labels for each sample

        Returns:
            Dictionary mapping group -> metrics
        """
        results = {}

        for group in np.unique(groups):
            mask = groups == group
            if np.sum(mask) > 0:
                results[group] = ModelEvaluator.compute_metrics(
                    y_true[mask],
                    y_pred[mask]
                )

        return results


class ModelComparator:
    """
    Compare multiple models using statistical tests.

    Supports:
    - Paired t-test
    - Wilcoxon signed-rank test
    - Effect size calculation (Cohen's d)
    """

    def __init__(
        self,
        significance_level: float = 0.05,
        use_parametric: bool = True
    ):
        """
        Initialize comparator.

        Args:
            significance_level: Threshold for statistical significance
            use_parametric: If True, use t-test; if False, use Wilcoxon
        """
        self.significance_level = significance_level
        self.use_parametric = use_parametric

    def compare_predictions(
        self,
        y_true: np.ndarray,
        y_pred_a: np.ndarray,
        y_pred_b: np.ndarray,
        model_a_name: str = "Model A",
        model_b_name: str = "Model B",
        metric: str = "squared_error"
    ) -> ComparisonResult:
        """
        Compare two models on same test set using paired test.

        Args:
            y_true: True values
            y_pred_a: Predictions from model A
            y_pred_b: Predictions from model B
            model_a_name: Name of model A
            model_b_name: Name of model B
            metric: Metric to compare ('squared_error', 'absolute_error')

        Returns:
            ComparisonResult
        """
        # Compute errors
        if metric == "squared_error":
            errors_a = (y_true - y_pred_a) ** 2
            errors_b = (y_true - y_pred_b) ** 2
        elif metric == "absolute_error":
            errors_a = np.abs(y_true - y_pred_a)
            errors_b = np.abs(y_true - y_pred_b)
        else:
            raise ValueError(f"Unknown metric: {metric}")

        # Differences (negative = A is better)
        differences = errors_a - errors_b

        # Summary statistics
        mean_diff = np.mean(differences)
        median_diff = np.median(differences)
        std_diff = np.std(differences)

        # Statistical test
        if self.use_parametric:
            # Paired t-test
            statistic, p_value = stats.ttest_rel(errors_a, errors_b)
            test_type = "paired_t_test"
        else:
            # Wilcoxon signed-rank test
            statistic, p_value = stats.wilcoxon(errors_a, errors_b)
            test_type = "wilcoxon"

        # Effect size (Cohen's d)
        effect_size = self._cohens_d(errors_a, errors_b)

        # Significance
        significant = p_value < self.significance_level

        return ComparisonResult(
            model_a_name=model_a_name,
            model_b_name=model_b_name,
            metric_name=metric,
            mean_diff=float(mean_diff),
            median_diff=float(median_diff),
            std_diff=float(std_diff),
            p_value=float(p_value),
            effect_size=float(effect_size),
            significant=significant,
            test_type=test_type
        )

    @staticmethod
    def _cohens_d(
        a: np.ndarray,
        b: np.ndarray
    ) -> float:
        """
        Compute Cohen's d effect size.

        Cohen's d = (mean_a - mean_b) / pooled_std

        Interpretation:
        - 0.2: Small effect
        - 0.5: Medium effect
        - 0.8: Large effect
        """
        mean_a = np.mean(a)
        mean_b = np.mean(b)
        std_a = np.std(a, ddof=1)
        std_b = np.std(b, ddof=1)

        n_a = len(a)
        n_b = len(b)

        # Pooled standard deviation
        pooled_std = np.sqrt(((n_a - 1) * std_a ** 2 + (n_b - 1) * std_b ** 2) / (n_a + n_b - 2))

        if pooled_std == 0:
            return 0.0

        return (mean_a - mean_b) / pooled_std

    def compare_multiple_models(
        self,
        y_true: np.ndarray,
        predictions: Dict[str, np.ndarray],
        metric: str = "squared_error"
    ) -> List[ComparisonResult]:
        """
        Perform pairwise comparisons between all models.

        Args:
            y_true: True values
            predictions: Dictionary mapping model_name -> predictions
            metric: Metric to compare

        Returns:
            List of ComparisonResult for all pairs
        """
        results = []
        model_names = list(predictions.keys())

        for i in range(len(model_names)):
            for j in range(i + 1, len(model_names)):
                name_a = model_names[i]
                name_b = model_names[j]

                result = self.compare_predictions(
                    y_true=y_true,
                    y_pred_a=predictions[name_a],
                    y_pred_b=predictions[name_b],
                    model_a_name=name_a,
                    model_b_name=name_b,
                    metric=metric
                )
                results.append(result)

        return results


class PerformanceMatrix:
    """
    Create performance matrices for visualization.

    Rows = models
    Columns = graph types (or other groupings)
    Cells = metric values
    """

    @staticmethod
    def create_matrix(
        model_predictions: Dict[str, np.ndarray],
        y_true: np.ndarray,
        groups: np.ndarray,
        metric_name: str = "r2"
    ) -> pd.DataFrame:
        """
        Create performance matrix.

        Args:
            model_predictions: Dict mapping model_name -> predictions
            y_true: True values
            groups: Group labels (e.g., graph types)
            metric_name: Metric to compute ('r2', 'mae', 'rmse')

        Returns:
            DataFrame with models as rows, groups as columns
        """
        unique_groups = np.unique(groups)
        matrix_data = {}

        for model_name, y_pred in model_predictions.items():
            row_data = {}

            for group in unique_groups:
                mask = groups == group
                if np.sum(mask) > 0:
                    metrics = ModelEvaluator.compute_metrics(
                        y_true[mask],
                        y_pred[mask]
                    )
                    # Extract requested metric
                    if metric_name == "r2":
                        row_data[group] = metrics.r2
                    elif metric_name == "mae":
                        row_data[group] = metrics.mae
                    elif metric_name == "rmse":
                        row_data[group] = metrics.rmse
                    else:
                        row_data[group] = getattr(metrics, metric_name)

            matrix_data[model_name] = row_data

        return pd.DataFrame(matrix_data).T  # Transpose so models are rows
