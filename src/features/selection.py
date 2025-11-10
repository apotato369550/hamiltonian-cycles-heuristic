"""
Feature Selection Utilities for Dimensionality Reduction.

Provides multiple feature selection methods to identify the most predictive
features for anchor quality prediction. Supports univariate, recursive,
model-based, and regularization-based selection strategies.
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Optional, Any, Tuple
import numpy as np
from sklearn.feature_selection import (
    SelectKBest,
    f_regression,
    mutual_info_regression,
    RFE,
    VarianceThreshold
)
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LassoCV
from sklearn.preprocessing import StandardScaler


class SelectionMethod(Enum):
    """Feature selection methods."""
    CORRELATION = "correlation"           # Univariate correlation
    F_TEST = "f_test"                     # F-statistic
    MUTUAL_INFO = "mutual_info"           # Mutual information
    RFE = "rfe"                           # Recursive feature elimination
    MODEL_BASED = "model_based"           # Random forest importance
    L1_REGULARIZATION = "l1"              # Lasso regularization
    VARIANCE_THRESHOLD = "variance"       # Remove low-variance features


@dataclass
class SelectionResult:
    """
    Result of feature selection.

    Attributes:
        selected_features: Names of selected features
        selected_indices: Indices of selected features
        scores: Feature importance scores (if available)
        metadata: Additional selection information
    """
    selected_features: List[str]
    selected_indices: List[int]
    scores: Optional[np.ndarray]
    metadata: Dict[str, Any]

    def get_feature_ranking(self) -> List[Tuple[str, float]]:
        """
        Get features ranked by importance.

        Returns:
            List of (feature_name, score) tuples, sorted by score descending
        """
        if self.scores is None:
            raise ValueError("No scores available for ranking")

        ranking = list(zip(self.selected_features, self.scores[self.selected_indices]))
        return sorted(ranking, key=lambda x: abs(x[1]), reverse=True)


class FeatureSelector:
    """
    Feature selection system with multiple selection strategies.

    Supports:
    - Univariate methods (correlation, F-test, mutual information)
    - Wrapper methods (RFE)
    - Embedded methods (model-based, L1 regularization)
    - Filter methods (variance threshold)

    All methods are deterministic given the same random seed.
    """

    def __init__(
        self,
        method: SelectionMethod,
        use_cv: bool = False,
        cv_folds: int = 5,
        random_seed: Optional[int] = None
    ):
        """
        Initialize feature selector.

        Args:
            method: Selection method to use
            use_cv: Whether to use cross-validation (for RFE, L1)
            cv_folds: Number of CV folds
            random_seed: Random seed for reproducibility
        """
        self.method = method
        self.use_cv = use_cv
        self.cv_folds = cv_folds
        self.random_seed = random_seed

        # Will be set during selection
        self._selector = None
        self._scaler = None

    def select_features(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        k: Optional[int] = None,
        threshold: Optional[float] = None,
        alpha: Optional[float] = None,
        remove_correlated: bool = False,
        correlation_threshold: float = 0.95
    ) -> SelectionResult:
        """
        Select features using the configured method.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target variable (n_samples,)
            feature_names: Names of features
            k: Number of features to select (for k-based methods)
            threshold: Threshold for variance/correlation methods
            alpha: Regularization strength for L1
            remove_correlated: Whether to remove highly correlated features first
            correlation_threshold: Correlation threshold for removal

        Returns:
            SelectionResult with selected features and metadata
        """
        # Validate inputs
        if X.shape[1] != len(feature_names):
            raise ValueError(
                f"Feature count mismatch: X has {X.shape[1]} columns "
                f"but {len(feature_names)} names provided"
            )

        # Remove highly correlated features first if requested
        if remove_correlated:
            X, feature_names = self._remove_correlated_features(
                X, feature_names, correlation_threshold
            )

        # Apply selection method
        if self.method == SelectionMethod.CORRELATION:
            return self._correlation_selection(X, y, feature_names, k)

        elif self.method == SelectionMethod.F_TEST:
            return self._f_test_selection(X, y, feature_names, k)

        elif self.method == SelectionMethod.MUTUAL_INFO:
            return self._mutual_info_selection(X, y, feature_names, k)

        elif self.method == SelectionMethod.RFE:
            return self._rfe_selection(X, y, feature_names, k)

        elif self.method == SelectionMethod.MODEL_BASED:
            return self._model_based_selection(X, y, feature_names, k)

        elif self.method == SelectionMethod.L1_REGULARIZATION:
            return self._l1_selection(X, y, feature_names, alpha)

        elif self.method == SelectionMethod.VARIANCE_THRESHOLD:
            return self._variance_threshold_selection(X, y, feature_names, threshold)

        else:
            raise ValueError(f"Unknown selection method: {self.method}")

    def _correlation_selection(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        k: Optional[int]
    ) -> SelectionResult:
        """Select features by correlation with target."""
        # Compute correlations
        correlations = np.array([
            np.corrcoef(X[:, i], y)[0, 1] for i in range(X.shape[1])
        ])

        # Handle NaN (constant features)
        correlations = np.nan_to_num(correlations, nan=0.0)

        # Rank by absolute correlation
        abs_corr = np.abs(correlations)
        ranking = np.argsort(abs_corr)[::-1]  # Descending

        # Select top k
        if k is None:
            k = X.shape[1]
        k = min(k, X.shape[1])

        selected_indices = ranking[:k].tolist()
        selected_features = [feature_names[i] for i in selected_indices]

        return SelectionResult(
            selected_features=selected_features,
            selected_indices=selected_indices,
            scores=correlations,
            metadata={
                'method': 'correlation',
                'ranking': ranking.tolist(),
                'top_correlation': float(abs_corr[ranking[0]])
            }
        )

    def _f_test_selection(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        k: Optional[int]
    ) -> SelectionResult:
        """Select features using F-test statistic."""
        if k is None:
            k = X.shape[1]

        selector = SelectKBest(score_func=f_regression, k=k)
        selector.fit(X, y)

        selected_indices = selector.get_support(indices=True).tolist()
        selected_features = [feature_names[i] for i in selected_indices]

        return SelectionResult(
            selected_features=selected_features,
            selected_indices=selected_indices,
            scores=selector.scores_,
            metadata={
                'method': 'f_test',
                'p_values': selector.pvalues_.tolist()
            }
        )

    def _mutual_info_selection(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        k: Optional[int]
    ) -> SelectionResult:
        """Select features using mutual information."""
        if k is None:
            k = X.shape[1]

        selector = SelectKBest(
            score_func=lambda X, y: mutual_info_regression(
                X, y, random_state=self.random_seed
            ),
            k=k
        )
        selector.fit(X, y)

        selected_indices = selector.get_support(indices=True).tolist()
        selected_features = [feature_names[i] for i in selected_indices]

        return SelectionResult(
            selected_features=selected_features,
            selected_indices=selected_indices,
            scores=selector.scores_,
            metadata={'method': 'mutual_info'}
        )

    def _rfe_selection(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        k: Optional[int]
    ) -> SelectionResult:
        """Recursive feature elimination with linear model."""
        from sklearn.linear_model import Ridge

        if k is None:
            k = max(1, X.shape[1] // 2)  # Select half by default

        # Use Ridge regression as base estimator
        estimator = Ridge(random_state=self.random_seed)

        if self.use_cv:
            from sklearn.feature_selection import RFECV
            selector = RFECV(
                estimator=estimator,
                step=1,
                cv=self.cv_folds,
                scoring='r2',
                n_jobs=-1
            )
        else:
            selector = RFE(estimator=estimator, n_features_to_select=k, step=1)

        selector.fit(X, y)

        selected_indices = selector.get_support(indices=True).tolist()
        selected_features = [feature_names[i] for i in selected_indices]

        metadata = {
            'method': 'rfe',
            'ranking': selector.ranking_.tolist()
        }

        if self.use_cv:
            metadata['cv_scores'] = selector.cv_results_['mean_test_score'].tolist()
            metadata['optimal_n_features'] = selector.n_features_

        return SelectionResult(
            selected_features=selected_features,
            selected_indices=selected_indices,
            scores=None,
            metadata=metadata
        )

    def _model_based_selection(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        k: Optional[int]
    ) -> SelectionResult:
        """Select features using random forest importance."""
        # Train random forest
        rf = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=self.random_seed,
            n_jobs=-1
        )
        rf.fit(X, y)

        # Get feature importances
        importances = rf.feature_importances_

        # Select top k
        if k is None:
            k = X.shape[1]
        k = min(k, X.shape[1])

        ranking = np.argsort(importances)[::-1]
        selected_indices = ranking[:k].tolist()
        selected_features = [feature_names[i] for i in selected_indices]

        return SelectionResult(
            selected_features=selected_features,
            selected_indices=selected_indices,
            scores=importances,
            metadata={
                'method': 'random_forest',
                'importances': importances.tolist(),
                'ranking': ranking.tolist()
            }
        )

    def _l1_selection(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        alpha: Optional[float]
    ) -> SelectionResult:
        """Select features using L1 (Lasso) regularization."""
        # Standardize features (important for L1)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Use cross-validation to select alpha if not provided
        if alpha is None or self.use_cv:
            lasso = LassoCV(
                cv=self.cv_folds,
                random_state=self.random_seed,
                n_jobs=-1
            )
        else:
            lasso = Lasso(alpha=alpha, random_state=self.random_seed)

        lasso.fit(X_scaled, y)

        # Features with non-zero coefficients are selected
        nonzero_mask = lasso.coef_ != 0
        selected_indices = np.where(nonzero_mask)[0].tolist()
        selected_features = [feature_names[i] for i in selected_indices]

        metadata = {
            'method': 'lasso',
            'coefficients': lasso.coef_.tolist(),
            'n_selected': len(selected_indices)
        }

        if hasattr(lasso, 'alpha_'):
            metadata['alpha'] = float(lasso.alpha_)

        return SelectionResult(
            selected_features=selected_features,
            selected_indices=selected_indices,
            scores=np.abs(lasso.coef_),
            metadata=metadata
        )

    def _variance_threshold_selection(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        threshold: Optional[float]
    ) -> SelectionResult:
        """Remove features with low variance."""
        if threshold is None:
            threshold = 0.01

        selector = VarianceThreshold(threshold=threshold)
        selector.fit(X)

        selected_indices = selector.get_support(indices=True).tolist()
        selected_features = [feature_names[i] for i in selected_indices]

        return SelectionResult(
            selected_features=selected_features,
            selected_indices=selected_indices,
            scores=selector.variances_,
            metadata={
                'method': 'variance_threshold',
                'threshold': threshold,
                'variances': selector.variances_.tolist()
            }
        )

    def _remove_correlated_features(
        self,
        X: np.ndarray,
        feature_names: List[str],
        threshold: float
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Remove highly correlated features.

        For each pair with correlation > threshold, keeps the first one.

        Args:
            X: Feature matrix
            feature_names: Feature names
            threshold: Correlation threshold

        Returns:
            Filtered X and feature_names
        """
        # Compute correlation matrix
        corr_matrix = np.corrcoef(X, rowvar=False)

        # Handle NaN (constant features)
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

        # Find features to remove
        to_remove = set()
        n_features = X.shape[1]

        for i in range(n_features):
            if i in to_remove:
                continue
            for j in range(i + 1, n_features):
                if j in to_remove:
                    continue
                if abs(corr_matrix[i, j]) > threshold:
                    to_remove.add(j)

        # Keep features not in removal set
        to_keep = [i for i in range(n_features) if i not in to_remove]

        X_filtered = X[:, to_keep]
        feature_names_filtered = [feature_names[i] for i in to_keep]

        return X_filtered, feature_names_filtered

    def __repr__(self) -> str:
        return f"FeatureSelector(method={self.method.value})"
