"""
Model Generalization Testing (Prompt 11).

Rigorous tests of model generalization to:
- Different graph types
- Different graph sizes
- Different weight distributions
- Adversarial/pathological cases

Helps answer:
- Do learned patterns transfer across graph types?
- Do patterns scale to larger graphs?
- Where does the model fail?
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
import numpy as np
import pandas as pd


class GeneralizationType(Enum):
    """Type of generalization test."""
    CROSS_GRAPH_TYPE = "cross_graph_type"
    CROSS_SIZE = "cross_size"
    CROSS_DISTRIBUTION = "cross_distribution"
    ADVERSARIAL = "adversarial"


@dataclass
class GeneralizationResult:
    """
    Result of a generalization test.

    Attributes:
        test_type: Type of generalization test
        train_performance: Performance metrics on training set
        test_performance: Performance metrics on test set
        performance_degradation: test - train (negative = worse on test)
        degradation_pct: Percentage degradation
        metadata: Additional information about the test
    """
    test_type: GeneralizationType
    train_performance: Dict[str, float]
    test_performance: Dict[str, float]
    performance_degradation: Dict[str, float]
    degradation_pct: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_summary(self) -> str:
        """Get human-readable summary."""
        summary = []
        summary.append(f"Generalization Test: {self.test_type.value}")

        if 'train_types' in self.metadata:
            summary.append(f"  Train types: {self.metadata['train_types']}")
        if 'test_types' in self.metadata:
            summary.append(f"  Test types: {self.metadata['test_types']}")

        summary.append("\nPerformance:")
        for metric in self.train_performance.keys():
            train_val = self.train_performance.get(metric, 0.0)
            test_val = self.test_performance.get(metric, 0.0)
            deg_pct = self.degradation_pct.get(metric, 0.0)

            summary.append(f"  {metric}:")
            summary.append(f"    Train: {train_val:.4f}")
            summary.append(f"    Test: {test_val:.4f}")
            summary.append(f"    Degradation: {deg_pct:.1f}%")

        return "\n".join(summary)


class GeneralizationTester:
    """
    Test model generalization across different domains.

    Supports:
    - Cross-graph-type generalization
    - Cross-size generalization
    - Cross-distribution generalization
    - Adversarial testing
    """

    def __init__(self, model_class: Any, model_params: Dict[str, Any]):
        """
        Initialize tester.

        Args:
            model_class: Model class to test
            model_params: Parameters for model initialization
        """
        self.model_class = model_class
        self.model_params = model_params

    def test_cross_graph_type(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        graph_types: pd.Series,
        train_types: List[str],
        test_types: List[str]
    ) -> GeneralizationResult:
        """
        Test generalization across graph types.

        Train on some graph types, test on others.

        Args:
            X: Features
            y: Labels
            graph_types: Graph type for each sample
            train_types: Graph types to train on
            test_types: Graph types to test on (must be disjoint from train)

        Returns:
            GeneralizationResult
        """
        # Split by type
        train_mask = graph_types.isin(train_types)
        test_mask = graph_types.isin(test_types)

        X_train = X[train_mask]
        y_train = y[train_mask]
        X_test = X[test_mask]
        y_test = y[test_mask]

        # Train model
        model = self.model_class(**self.model_params)
        model.fit(X_train, y_train)

        # Evaluate on both sets
        train_perf = self._evaluate(model, X_train, y_train)
        test_perf = self._evaluate(model, X_test, y_test)

        # Compute degradation
        degradation, degradation_pct = self._compute_degradation(train_perf, test_perf)

        return GeneralizationResult(
            test_type=GeneralizationType.CROSS_GRAPH_TYPE,
            train_performance=train_perf,
            test_performance=test_perf,
            performance_degradation=degradation,
            degradation_pct=degradation_pct,
            metadata={
                'train_types': train_types,
                'test_types': test_types,
                'train_size': len(X_train),
                'test_size': len(X_test)
            }
        )

    def test_cross_size(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        graph_sizes: pd.Series,
        size_threshold: int
    ) -> GeneralizationResult:
        """
        Test generalization to different graph sizes.

        Train on small graphs, test on large graphs.

        Args:
            X: Features
            y: Labels
            graph_sizes: Graph size (number of vertices) for each sample
            size_threshold: Size threshold (train < threshold, test >= threshold)

        Returns:
            GeneralizationResult
        """
        # Split by size
        train_mask = graph_sizes < size_threshold
        test_mask = graph_sizes >= size_threshold

        X_train = X[train_mask]
        y_train = y[train_mask]
        X_test = X[test_mask]
        y_test = y[test_mask]

        # Train model
        model = self.model_class(**self.model_params)
        model.fit(X_train, y_train)

        # Evaluate
        train_perf = self._evaluate(model, X_train, y_train)
        test_perf = self._evaluate(model, X_test, y_test)

        # Degradation
        degradation, degradation_pct = self._compute_degradation(train_perf, test_perf)

        return GeneralizationResult(
            test_type=GeneralizationType.CROSS_SIZE,
            train_performance=train_perf,
            test_performance=test_perf,
            performance_degradation=degradation,
            degradation_pct=degradation_pct,
            metadata={
                'size_threshold': size_threshold,
                'train_size_range': (int(graph_sizes[train_mask].min()), int(graph_sizes[train_mask].max())),
                'test_size_range': (int(graph_sizes[test_mask].min()), int(graph_sizes[test_mask].max())),
                'train_samples': len(X_train),
                'test_samples': len(X_test)
            }
        )

    def test_cross_distribution(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        train_distribution: str = "uniform",
        test_distribution: str = "skewed"
    ) -> GeneralizationResult:
        """
        Test generalization across weight distributions.

        Args:
            X_train: Training features (from one distribution)
            y_train: Training labels
            X_test: Test features (from different distribution)
            y_test: Test labels
            train_distribution: Description of train distribution
            test_distribution: Description of test distribution

        Returns:
            GeneralizationResult
        """
        # Train model
        model = self.model_class(**self.model_params)
        model.fit(X_train, y_train)

        # Evaluate
        train_perf = self._evaluate(model, X_train, y_train)
        test_perf = self._evaluate(model, X_test, y_test)

        # Degradation
        degradation, degradation_pct = self._compute_degradation(train_perf, test_perf)

        return GeneralizationResult(
            test_type=GeneralizationType.CROSS_DISTRIBUTION,
            train_performance=train_perf,
            test_performance=test_perf,
            performance_degradation=degradation,
            degradation_pct=degradation_pct,
            metadata={
                'train_distribution': train_distribution,
                'test_distribution': test_distribution,
                'train_size': len(X_train),
                'test_size': len(X_test)
            }
        )

    def test_adversarial(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_adversarial: pd.DataFrame,
        y_adversarial: pd.Series,
        adversarial_description: str = "pathological_graphs"
    ) -> GeneralizationResult:
        """
        Test on adversarial/pathological cases.

        Args:
            X_train: Training features (normal graphs)
            y_train: Training labels
            X_adversarial: Adversarial features (pathological cases)
            y_adversarial: Adversarial labels
            adversarial_description: Description of adversarial cases

        Returns:
            GeneralizationResult
        """
        # Train model
        model = self.model_class(**self.model_params)
        model.fit(X_train, y_train)

        # Evaluate
        train_perf = self._evaluate(model, X_train, y_train)
        test_perf = self._evaluate(model, X_adversarial, y_adversarial)

        # Degradation
        degradation, degradation_pct = self._compute_degradation(train_perf, test_perf)

        return GeneralizationResult(
            test_type=GeneralizationType.ADVERSARIAL,
            train_performance=train_perf,
            test_performance=test_perf,
            performance_degradation=degradation,
            degradation_pct=degradation_pct,
            metadata={
                'adversarial_type': adversarial_description,
                'train_size': len(X_train),
                'adversarial_size': len(X_adversarial)
            }
        )

    def _evaluate(
        self,
        model: Any,
        X: pd.DataFrame,
        y: pd.Series
    ) -> Dict[str, float]:
        """
        Evaluate model and return metrics.

        Args:
            model: Trained model
            X: Features
            y: Labels

        Returns:
            Dictionary of metrics (r2, mae, rmse)
        """
        y_pred = model.predict(X)
        residuals = y.values - y_pred

        # R²
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y.values - np.mean(y.values)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        return {
            'r2': float(r2),
            'mae': float(np.mean(np.abs(residuals))),
            'rmse': float(np.sqrt(np.mean(residuals ** 2)))
        }

    @staticmethod
    def _compute_degradation(
        train_perf: Dict[str, float],
        test_perf: Dict[str, float]
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Compute performance degradation.

        Args:
            train_perf: Training performance
            test_perf: Test performance

        Returns:
            (degradation_absolute, degradation_percentage)
        """
        degradation = {}
        degradation_pct = {}

        for metric in train_perf.keys():
            train_val = train_perf[metric]
            test_val = test_perf[metric]

            # Absolute degradation (test - train)
            # For R²: negative = worse on test
            # For MAE/RMSE: positive = worse on test
            if metric == 'r2':
                degradation[metric] = test_val - train_val
            else:
                degradation[metric] = test_val - train_val

            # Percentage degradation
            if train_val != 0:
                degradation_pct[metric] = (degradation[metric] / abs(train_val)) * 100.0
            else:
                degradation_pct[metric] = 0.0

        return degradation, degradation_pct


class FailureModeAnalyzer:
    """
    Analyze failure modes to understand where and why models fail.

    Helps identify:
    - Graph properties correlated with high error
    - Feature patterns in failure cases
    - Systematic biases
    """

    @staticmethod
    def identify_failure_cases(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        threshold_percentile: float = 90.0
    ) -> np.ndarray:
        """
        Identify failure cases (high error).

        Args:
            y_true: True labels
            y_pred: Predictions
            threshold_percentile: Percentile threshold for failures

        Returns:
            Boolean mask of failure cases
        """
        errors = np.abs(y_true - y_pred)
        threshold = np.percentile(errors, threshold_percentile)
        return errors >= threshold

    @staticmethod
    def analyze_failure_features(
        X: pd.DataFrame,
        failure_mask: np.ndarray
    ) -> Dict[str, Any]:
        """
        Analyze feature patterns in failure cases.

        Args:
            X: Feature matrix
            failure_mask: Boolean mask of failure cases

        Returns:
            Dictionary with:
            - failure_count: Number of failures
            - feature_means_failures: Mean feature values in failures
            - feature_means_successes: Mean feature values in successes
            - feature_differences: Differences (failures - successes)
        """
        success_mask = ~failure_mask

        failure_count = int(np.sum(failure_mask))
        success_count = int(np.sum(success_mask))

        if failure_count == 0 or success_count == 0:
            return {'failure_count': failure_count, 'success_count': success_count}

        # Compute mean features for each group
        failure_means = X[failure_mask].mean().to_dict()
        success_means = X[success_mask].mean().to_dict()

        # Differences
        differences = {
            feat: failure_means[feat] - success_means[feat]
            for feat in X.columns
        }

        # Sort by absolute difference
        sorted_diffs = sorted(differences.items(), key=lambda x: abs(x[1]), reverse=True)

        return {
            'failure_count': failure_count,
            'success_count': success_count,
            'feature_means_failures': failure_means,
            'feature_means_successes': success_means,
            'feature_differences': dict(sorted_diffs[:20])  # Top 20
        }

    @staticmethod
    def analyze_systematic_bias(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        groups: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Analyze systematic biases in predictions.

        Args:
            y_true: True labels
            y_pred: Predictions
            groups: Optional group labels (e.g., graph types)

        Returns:
            Dictionary with bias analysis
        """
        residuals = y_pred - y_true

        analysis = {
            'mean_bias': float(np.mean(residuals)),
            'median_bias': float(np.median(residuals)),
            'std_bias': float(np.std(residuals)),
            'overestimation_rate': float(np.mean(residuals > 0))
        }

        # Bias by group
        if groups is not None:
            bias_by_group = {}
            for group in np.unique(groups):
                mask = groups == group
                bias_by_group[str(group)] = {
                    'mean_bias': float(np.mean(residuals[mask])),
                    'overestimation_rate': float(np.mean(residuals[mask] > 0)),
                    'count': int(np.sum(mask))
                }
            analysis['bias_by_group'] = bias_by_group

        return analysis


class ConsistencyAnalyzer:
    """
    Analyze consistency of model predictions.

    Tests whether model rankings are consistent even if absolute predictions vary.
    """

    @staticmethod
    def rank_correlation(
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> float:
        """
        Compute Spearman rank correlation.

        Measures whether model ranks vertices correctly even if predictions are off.

        Args:
            y_true: True values
            y_pred: Predicted values

        Returns:
            Spearman correlation coefficient (-1 to 1)
        """
        from scipy.stats import spearmanr
        corr, _ = spearmanr(y_true, y_pred)
        return float(corr)

    @staticmethod
    def top_k_accuracy(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        k: int = 5
    ) -> float:
        """
        Compute top-k accuracy.

        What fraction of top-k predicted vertices are actually in the top-k?

        Args:
            y_true: True values
            y_pred: Predicted values
            k: Number of top elements to consider

        Returns:
            Top-k accuracy (0-1)
        """
        true_top_k = set(np.argsort(y_true)[-k:])
        pred_top_k = set(np.argsort(y_pred)[-k:])

        overlap = len(true_top_k & pred_top_k)
        return overlap / k
