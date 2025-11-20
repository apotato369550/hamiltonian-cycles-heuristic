"""
Model Interpretation and Explanation (Prompt 9).

Provides:
- Coefficient analysis for linear models
- Feature importance visualization
- Partial dependence plots
- Individual prediction explanations
- SHAP-style feature contributions
- Case study tools
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class CoefficientAnalysis:
    """
    Analysis of linear model coefficients.

    Attributes:
        coefficients: Dict mapping feature -> coefficient value
        intercept: Model intercept
        std_coefficients: Standardized coefficients (for comparison)
        confidence_intervals: Dict mapping feature -> (lower, upper) CI
        p_values: Dict mapping feature -> p-value
        significant_features: Features with p-value < threshold
    """
    coefficients: Dict[str, float]
    intercept: float
    std_coefficients: Dict[str, float]
    confidence_intervals: Optional[Dict[str, Tuple[float, float]]] = None
    p_values: Optional[Dict[str, float]] = None
    significant_features: Optional[List[str]] = None

    def get_top_features(self, n: int = 10) -> List[Tuple[str, float]]:
        """
        Get top-n features by absolute coefficient value.

        Args:
            n: Number of features to return

        Returns:
            List of (feature_name, coefficient) sorted by |coef|
        """
        sorted_coefs = sorted(
            self.coefficients.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        return sorted_coefs[:n]

    def get_positive_features(self, n: int = 10) -> List[Tuple[str, float]]:
        """Get top-n features with positive coefficients."""
        positive = [(f, c) for f, c in self.coefficients.items() if c > 0]
        sorted_positive = sorted(positive, key=lambda x: x[1], reverse=True)
        return sorted_positive[:n]

    def get_negative_features(self, n: int = 10) -> List[Tuple[str, float]]:
        """Get top-n features with negative coefficients."""
        negative = [(f, c) for f, c in self.coefficients.items() if c < 0]
        sorted_negative = sorted(negative, key=lambda x: x[1])
        return sorted_negative[:n]

    def get_summary(self) -> str:
        """Get human-readable summary."""
        summary = []
        summary.append("Linear Model Coefficient Analysis:")
        summary.append(f"  Intercept: {self.intercept:.4f}")
        summary.append(f"  Total features: {len(self.coefficients)}")

        if self.significant_features:
            summary.append(f"  Significant features: {len(self.significant_features)}")

        summary.append("\n  Top 5 positive coefficients:")
        for feat, coef in self.get_positive_features(5):
            summary.append(f"    {feat}: {coef:.4f}")

        summary.append("\n  Top 5 negative coefficients:")
        for feat, coef in self.get_negative_features(5):
            summary.append(f"    {feat}: {coef:.4f}")

        return "\n".join(summary)


@dataclass
class FeatureContribution:
    """
    Feature contributions for a single prediction.

    Attributes:
        base_value: Baseline prediction (mean)
        contributions: Dict mapping feature -> contribution to prediction
        feature_values: Dict mapping feature -> actual value for this instance
        predicted_value: Final predicted value
    """
    base_value: float
    contributions: Dict[str, float]
    feature_values: Dict[str, float]
    predicted_value: float

    def get_top_contributors(self, n: int = 10) -> List[Tuple[str, float]]:
        """Get features with largest absolute contributions."""
        sorted_contribs = sorted(
            self.contributions.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        return sorted_contribs[:n]

    def get_summary(self) -> str:
        """Get human-readable summary."""
        summary = []
        summary.append(f"Prediction: {self.predicted_value:.4f}")
        summary.append(f"Base value: {self.base_value:.4f}")
        summary.append("\nTop contributors:")

        for feat, contrib in self.get_top_contributors(10):
            value = self.feature_values.get(feat, 0.0)
            direction = "↑" if contrib > 0 else "↓"
            summary.append(f"  {feat} = {value:.3f} → {direction} {contrib:+.4f}")

        return "\n".join(summary)


@dataclass
class PartialDependenceResult:
    """
    Result of partial dependence analysis.

    Attributes:
        feature_name: Name of feature
        feature_values: Grid of feature values
        average_predictions: Average prediction at each feature value
        std_predictions: Std dev of predictions
        percentile_5: 5th percentile of predictions
        percentile_95: 95th percentile of predictions
    """
    feature_name: str
    feature_values: np.ndarray
    average_predictions: np.ndarray
    std_predictions: np.ndarray
    percentile_5: np.ndarray
    percentile_95: np.ndarray


class LinearModelInterpreter:
    """
    Interpret linear regression models.

    Provides:
    - Coefficient analysis with significance tests
    - Feature importance ranking
    - Individual prediction explanations
    """

    def __init__(self, model: Any, X_train: pd.DataFrame, y_train: pd.Series):
        """
        Initialize interpreter.

        Args:
            model: Trained linear model
            X_train: Training features (for computing std coefficients)
            y_train: Training labels
        """
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.feature_names = X_train.columns.tolist()

    def analyze_coefficients(
        self,
        confidence_level: float = 0.95,
        significance_threshold: float = 0.05
    ) -> CoefficientAnalysis:
        """
        Analyze model coefficients.

        Args:
            confidence_level: Confidence level for intervals
            significance_threshold: P-value threshold

        Returns:
            CoefficientAnalysis object
        """
        # Get raw coefficients
        coefs = self.model.get_coefficients()
        intercept = coefs.get('intercept', 0.0)

        # Remove intercept from feature coefficients
        feature_coefs = {k: v for k, v in coefs.items() if k != 'intercept'}

        # Compute standardized coefficients
        # std_coef = coef * std(X_feature)
        std_coefs = {}
        for feat in self.feature_names:
            if feat in feature_coefs:
                std_coefs[feat] = feature_coefs[feat] * self.X_train[feat].std()

        # Compute confidence intervals and p-values (if OLS)
        # This requires residuals and X'X matrix
        confidence_intervals = None
        p_values = None
        significant_features = None

        if hasattr(self.model, 'model') and hasattr(self.model.model, 'coef_'):
            try:
                confidence_intervals, p_values = self._compute_statistics(
                    confidence_level, significance_threshold
                )
                significant_features = [
                    f for f, p in p_values.items() if p < significance_threshold
                ]
            except:
                # If computation fails, leave as None
                pass

        return CoefficientAnalysis(
            coefficients=feature_coefs,
            intercept=intercept,
            std_coefficients=std_coefs,
            confidence_intervals=confidence_intervals,
            p_values=p_values,
            significant_features=significant_features
        )

    def _compute_statistics(
        self,
        confidence_level: float,
        significance_threshold: float
    ) -> Tuple[Dict[str, Tuple[float, float]], Dict[str, float]]:
        """
        Compute confidence intervals and p-values for OLS.

        Returns:
            (confidence_intervals, p_values)
        """
        # Get residuals
        y_pred = self.model.predict(self.X_train)
        residuals = self.y_train.values - y_pred

        # Residual variance
        n = len(y_pred)
        p = len(self.feature_names)
        residual_var = np.sum(residuals ** 2) / (n - p - 1)

        # Compute (X'X)^-1
        X = self.X_train.values
        X_with_intercept = np.column_stack([np.ones(n), X])

        try:
            XtX_inv = np.linalg.inv(X_with_intercept.T @ X_with_intercept)
        except np.linalg.LinAlgError:
            # Singular matrix, can't compute
            return None, None

        # Standard errors
        se = np.sqrt(np.diag(XtX_inv) * residual_var)

        # T-statistics
        coefs = self.model.get_coefficients()
        coef_values = np.array([coefs.get('intercept', 0.0)] +
                               [coefs.get(f, 0.0) for f in self.feature_names])
        t_stats = coef_values / se

        # P-values (two-tailed)
        df = n - p - 1
        p_vals = 2 * (1 - stats.t.cdf(np.abs(t_stats), df))

        # Confidence intervals
        t_critical = stats.t.ppf((1 + confidence_level) / 2, df)
        ci_lower = coef_values - t_critical * se
        ci_upper = coef_values + t_critical * se

        # Package into dictionaries
        confidence_intervals = {}
        p_values = {}

        for i, feat in enumerate(self.feature_names):
            confidence_intervals[feat] = (ci_lower[i + 1], ci_upper[i + 1])
            p_values[feat] = p_vals[i + 1]

        return confidence_intervals, p_values

    def explain_prediction(
        self,
        x: pd.Series,
        baseline: Optional[float] = None
    ) -> FeatureContribution:
        """
        Explain a single prediction.

        Args:
            x: Feature values for instance
            baseline: Baseline prediction (if None, use mean of training data)

        Returns:
            FeatureContribution object
        """
        if baseline is None:
            baseline = float(self.y_train.mean())

        # Get prediction
        pred = self.model.predict(x.to_frame().T)[0]

        # Get coefficients
        coefs = self.model.get_coefficients()

        # Compute contribution of each feature: coef * value
        contributions = {}
        for feat in self.feature_names:
            if feat in coefs:
                contributions[feat] = coefs[feat] * x[feat]

        # Feature values
        feature_values = {feat: x[feat] for feat in self.feature_names}

        return FeatureContribution(
            base_value=baseline,
            contributions=contributions,
            feature_values=feature_values,
            predicted_value=pred
        )


class TreeModelInterpreter:
    """
    Interpret tree-based models.

    Provides:
    - Feature importance analysis
    - Partial dependence plots
    - Tree structure visualization (for single trees)
    """

    def __init__(self, model: Any, X_train: pd.DataFrame, y_train: pd.Series):
        """
        Initialize interpreter.

        Args:
            model: Trained tree model
            X_train: Training features
            y_train: Training labels
        """
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.feature_names = X_train.columns.tolist()

    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance scores.

        Returns:
            Dict mapping feature -> importance (sum to 1.0)
        """
        return self.model.get_feature_importance()

    def get_top_features(self, n: int = 10) -> List[Tuple[str, float]]:
        """Get top-n most important features."""
        importance = self.get_feature_importance()
        sorted_features = sorted(
            importance.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_features[:n]


class ModelInterpreter:
    """
    Unified interpreter for all model types.

    Automatically selects appropriate interpretation methods.
    """

    @staticmethod
    def create(
        model: Any,
        X_train: pd.DataFrame,
        y_train: pd.Series
    ) -> Any:
        """
        Create appropriate interpreter for model type.

        Args:
            model: Trained model
            X_train: Training features
            y_train: Training labels

        Returns:
            LinearModelInterpreter or TreeModelInterpreter
        """
        # Check model type
        if hasattr(model, 'get_coefficients'):
            # Linear model
            return LinearModelInterpreter(model, X_train, y_train)
        else:
            # Tree model
            return TreeModelInterpreter(model, X_train, y_train)

    @staticmethod
    def partial_dependence(
        model: Any,
        X: pd.DataFrame,
        feature_name: str,
        num_points: int = 50,
        percentile_range: Tuple[float, float] = (5.0, 95.0)
    ) -> PartialDependenceResult:
        """
        Compute partial dependence of prediction on a feature.

        Shows how predictions change as feature varies, averaging over other features.

        Args:
            model: Trained model
            X: Feature matrix
            feature_name: Feature to analyze
            num_points: Number of grid points
            percentile_range: Range of feature values to analyze

        Returns:
            PartialDependenceResult
        """
        # Get feature range
        feat_min = np.percentile(X[feature_name], percentile_range[0])
        feat_max = np.percentile(X[feature_name], percentile_range[1])

        # Create grid
        feature_values = np.linspace(feat_min, feat_max, num_points)

        # For each grid point, predict for all instances with that feature value
        predictions_at_value = []

        for value in feature_values:
            # Create modified X with feature set to value
            X_modified = X.copy()
            X_modified[feature_name] = value

            # Predict
            preds = model.predict(X_modified)
            predictions_at_value.append(preds)

        # Aggregate predictions
        predictions_at_value = np.array(predictions_at_value)  # Shape: (num_points, num_instances)

        average_predictions = np.mean(predictions_at_value, axis=1)
        std_predictions = np.std(predictions_at_value, axis=1)
        percentile_5 = np.percentile(predictions_at_value, 5, axis=1)
        percentile_95 = np.percentile(predictions_at_value, 95, axis=1)

        return PartialDependenceResult(
            feature_name=feature_name,
            feature_values=feature_values,
            average_predictions=average_predictions,
            std_predictions=std_predictions,
            percentile_5=percentile_5,
            percentile_95=percentile_95
        )


class CaseStudyAnalyzer:
    """
    Analyze specific test cases to understand model behavior.

    Useful for:
    - Understanding why model makes certain predictions
    - Identifying failure modes
    - Generating insights for publication
    """

    @staticmethod
    def analyze_best_and_worst(
        model: Any,
        X: pd.DataFrame,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        n_cases: int = 5
    ) -> Dict[str, Any]:
        """
        Analyze best and worst predictions.

        Args:
            model: Trained model
            X: Features
            y_true: True labels
            y_pred: Predictions
            n_cases: Number of cases to analyze

        Returns:
            Dictionary with:
            - best_indices: Indices of best predictions
            - worst_indices: Indices of worst predictions
            - best_errors: Errors for best predictions
            - worst_errors: Errors for worst predictions
        """
        # Compute errors
        errors = np.abs(y_true - y_pred)

        # Find best and worst
        best_indices = np.argsort(errors)[:n_cases]
        worst_indices = np.argsort(errors)[-n_cases:][::-1]

        return {
            'best_indices': best_indices.tolist(),
            'worst_indices': worst_indices.tolist(),
            'best_errors': errors[best_indices].tolist(),
            'worst_errors': errors[worst_indices].tolist(),
            'best_predictions': y_pred[best_indices].tolist(),
            'worst_predictions': y_pred[worst_indices].tolist(),
            'best_true': y_true[best_indices].tolist(),
            'worst_true': y_true[worst_indices].tolist()
        }

    @staticmethod
    def analyze_feature_values(
        X: pd.DataFrame,
        indices: List[int],
        feature_names: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Extract feature values for specific instances.

        Args:
            X: Feature matrix
            indices: Instance indices to analyze
            feature_names: Features to include (None = all)

        Returns:
            DataFrame with feature values for selected instances
        """
        if feature_names is None:
            feature_names = X.columns.tolist()

        return X.iloc[indices][feature_names]

    @staticmethod
    def compare_feature_distributions(
        X: pd.DataFrame,
        good_indices: List[int],
        bad_indices: List[int]
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare feature distributions between good and bad predictions.

        Args:
            X: Feature matrix
            good_indices: Indices of good predictions
            bad_indices: Indices of bad predictions

        Returns:
            Dictionary mapping feature -> {'good_mean', 'bad_mean', 'difference'}
        """
        comparisons = {}

        for feat in X.columns:
            good_mean = X.iloc[good_indices][feat].mean()
            bad_mean = X.iloc[bad_indices][feat].mean()

            comparisons[feat] = {
                'good_mean': float(good_mean),
                'bad_mean': float(bad_mean),
                'difference': float(bad_mean - good_mean),
                'abs_difference': float(abs(bad_mean - good_mean))
            }

        return comparisons
