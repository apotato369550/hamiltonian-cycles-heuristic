"""
Feature Engineering for ML (Prompt 8).

Implements:
- Feature scaling strategies (standardization, min-max, robust)
- Non-linear transformations (log, sqrt, Box-Cox)
- Feature interactions
- Dimensionality reduction (PCA)
- Advanced feature selection
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
from scipy import stats


class ScalingStrategy(Enum):
    """Feature scaling strategy."""
    STANDARDIZATION = "standardization"  # Mean=0, std=1
    MIN_MAX = "min_max"  # Range [0, 1]
    ROBUST = "robust"  # Median=0, IQR-based
    NONE = "none"


class TransformationType(Enum):
    """Non-linear transformation type."""
    LOG = "log"  # log(x + 1)
    SQRT = "sqrt"  # sqrt(x) for x >= 0
    SQUARE = "square"  # x^2
    RECIPROCAL = "reciprocal"  # 1/x
    BOX_COX = "box_cox"  # Automated power transform


@dataclass
class FeatureScaler:
    """
    Feature scaling transformer.

    Handles different scaling strategies with fit/transform pattern.
    """
    strategy: ScalingStrategy = ScalingStrategy.STANDARDIZATION

    def __post_init__(self):
        """Initialize scaler."""
        self.scaler = None
        self.is_fitted = False
        self.feature_names = None

    def fit(
        self,
        X: pd.DataFrame
    ) -> 'FeatureScaler':
        """
        Fit scaler on training data.

        Args:
            X: Training features

        Returns:
            self
        """
        self.feature_names = X.columns.tolist()

        if self.strategy == ScalingStrategy.STANDARDIZATION:
            self.scaler = StandardScaler()
        elif self.strategy == ScalingStrategy.MIN_MAX:
            self.scaler = MinMaxScaler()
        elif self.strategy == ScalingStrategy.ROBUST:
            self.scaler = RobustScaler()
        elif self.strategy == ScalingStrategy.NONE:
            self.scaler = None
        else:
            raise ValueError(f"Unknown scaling strategy: {self.strategy}")

        if self.scaler is not None:
            self.scaler.fit(X.values)

        self.is_fitted = True
        return self

    def transform(
        self,
        X: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Transform features using fitted scaler.

        Args:
            X: Features to transform

        Returns:
            Scaled features (DataFrame with same column names)
        """
        if not self.is_fitted:
            raise RuntimeError("Scaler not fitted. Call fit() first.")

        if self.scaler is None:
            return X.copy()

        X_scaled = self.scaler.transform(X.values)
        return pd.DataFrame(X_scaled, columns=self.feature_names, index=X.index)

    def fit_transform(
        self,
        X: pd.DataFrame
    ) -> pd.DataFrame:
        """Fit and transform in one step."""
        self.fit(X)
        return self.transform(X)


class NonLinearTransformer:
    """
    Apply non-linear transformations to features.

    Useful for:
    - Log transform: right-skewed features
    - Square root: milder transformation
    - Box-Cox: automated power transform
    """

    def __init__(
        self,
        transformation: TransformationType = TransformationType.LOG,
        features: Optional[List[str]] = None  # If None, transform all
    ):
        """
        Initialize transformer.

        Args:
            transformation: Type of transformation
            features: List of features to transform (None = all)
        """
        self.transformation = transformation
        self.features = features
        self.lambda_params = {}  # For Box-Cox
        self.is_fitted = False

    def fit(
        self,
        X: pd.DataFrame
    ) -> 'NonLinearTransformer':
        """
        Fit transformer (needed for Box-Cox).

        Args:
            X: Training features

        Returns:
            self
        """
        features_to_transform = self.features if self.features else X.columns.tolist()

        if self.transformation == TransformationType.BOX_COX:
            # Fit Box-Cox parameters
            for col in features_to_transform:
                if col in X.columns:
                    # Box-Cox requires positive values
                    values = X[col].values
                    if np.all(values > 0):
                        _, fitted_lambda = stats.boxcox(values)
                        self.lambda_params[col] = fitted_lambda

        self.is_fitted = True
        return self

    def transform(
        self,
        X: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Transform features.

        Args:
            X: Features to transform

        Returns:
            Transformed features
        """
        X_transformed = X.copy()
        features_to_transform = self.features if self.features else X.columns.tolist()

        for col in features_to_transform:
            if col not in X.columns:
                continue

            values = X[col].values

            if self.transformation == TransformationType.LOG:
                # log(x + 1) to handle zeros
                X_transformed[col] = np.log1p(np.maximum(values, 0))

            elif self.transformation == TransformationType.SQRT:
                # sqrt(x) for x >= 0
                X_transformed[col] = np.sqrt(np.maximum(values, 0))

            elif self.transformation == TransformationType.SQUARE:
                X_transformed[col] = values ** 2

            elif self.transformation == TransformationType.RECIPROCAL:
                # 1/x, avoid division by zero
                X_transformed[col] = 1.0 / (values + 1e-10)

            elif self.transformation == TransformationType.BOX_COX:
                if col in self.lambda_params:
                    X_transformed[col] = stats.boxcox(values, lmbda=self.lambda_params[col])

        return X_transformed

    def fit_transform(
        self,
        X: pd.DataFrame
    ) -> pd.DataFrame:
        """Fit and transform in one step."""
        self.fit(X)
        return self.transform(X)


class FeatureInteractionGenerator:
    """
    Generate interaction features (products of feature pairs).

    For linear models, interactions can capture non-linear relationships.
    """

    def __init__(
        self,
        interaction_pairs: Optional[List[Tuple[str, str]]] = None,
        degree: int = 2,  # Maximum interaction degree
        max_interactions: Optional[int] = None  # Limit number of interactions
    ):
        """
        Initialize interaction generator.

        Args:
            interaction_pairs: Specific pairs to interact (None = all pairs)
            degree: Interaction degree (2 = pairwise, 3 = three-way, etc.)
            max_interactions: Maximum number of interactions to generate
        """
        self.interaction_pairs = interaction_pairs
        self.degree = degree
        self.max_interactions = max_interactions
        self.generated_features = []

    def fit(
        self,
        X: pd.DataFrame
    ) -> 'FeatureInteractionGenerator':
        """
        Determine which interactions to create.

        Args:
            X: Training features

        Returns:
            self
        """
        if self.interaction_pairs is not None:
            # Use specified pairs
            self.generated_features = self.interaction_pairs
        else:
            # Generate all pairwise interactions
            features = X.columns.tolist()
            pairs = []

            for i in range(len(features)):
                for j in range(i + 1, len(features)):
                    pairs.append((features[i], features[j]))

            # Limit if requested
            if self.max_interactions is not None:
                pairs = pairs[:self.max_interactions]

            self.generated_features = pairs

        return self

    def transform(
        self,
        X: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Generate interaction features.

        Args:
            X: Features

        Returns:
            DataFrame with original + interaction features
        """
        X_with_interactions = X.copy()

        for feat_a, feat_b in self.generated_features:
            if feat_a in X.columns and feat_b in X.columns:
                interaction_name = f"{feat_a}_x_{feat_b}"
                X_with_interactions[interaction_name] = X[feat_a] * X[feat_b]

        return X_with_interactions

    def fit_transform(
        self,
        X: pd.DataFrame
    ) -> pd.DataFrame:
        """Fit and transform in one step."""
        self.fit(X)
        return self.transform(X)


class PCAReducer:
    """
    Dimensionality reduction via PCA.

    Useful for:
    - Reducing correlated features
    - Visualization (2-3 components)
    - Regularization via dimension reduction
    """

    def __init__(
        self,
        n_components: Optional[int] = None,
        variance_threshold: Optional[float] = None  # Keep components explaining this much variance
    ):
        """
        Initialize PCA reducer.

        Args:
            n_components: Number of components (or None for variance_threshold)
            variance_threshold: Keep components explaining this fraction of variance
        """
        self.n_components = n_components
        self.variance_threshold = variance_threshold
        self.pca = None
        self.is_fitted = False
        self.original_feature_names = None

    def fit(
        self,
        X: pd.DataFrame
    ) -> 'PCAReducer':
        """
        Fit PCA on training data.

        Args:
            X: Training features

        Returns:
            self
        """
        self.original_feature_names = X.columns.tolist()

        if self.variance_threshold is not None:
            # Fit with all components first
            pca_full = PCA()
            pca_full.fit(X.values)

            # Find number of components for threshold
            cumsum = np.cumsum(pca_full.explained_variance_ratio_)
            n_components = int(np.searchsorted(cumsum, self.variance_threshold) + 1)
            self.n_components = n_components

        # Fit final PCA
        self.pca = PCA(n_components=self.n_components)
        self.pca.fit(X.values)
        self.is_fitted = True

        return self

    def transform(
        self,
        X: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Transform features to principal components.

        Args:
            X: Features to transform

        Returns:
            DataFrame with PC columns
        """
        if not self.is_fitted:
            raise RuntimeError("PCA not fitted. Call fit() first.")

        X_pca = self.pca.transform(X.values)

        # Create column names
        n_components = X_pca.shape[1]
        columns = [f"PC{i+1}" for i in range(n_components)]

        return pd.DataFrame(X_pca, columns=columns, index=X.index)

    def fit_transform(
        self,
        X: pd.DataFrame
    ) -> pd.DataFrame:
        """Fit and transform in one step."""
        self.fit(X)
        return self.transform(X)

    def get_explained_variance(self) -> np.ndarray:
        """Get explained variance ratio for each component."""
        if not self.is_fitted:
            raise RuntimeError("PCA not fitted")
        return self.pca.explained_variance_ratio_

    def get_component_loadings(self) -> pd.DataFrame:
        """
        Get feature loadings on each component.

        Returns:
            DataFrame with features as rows, components as columns
        """
        if not self.is_fitted:
            raise RuntimeError("PCA not fitted")

        n_components = self.pca.n_components_
        columns = [f"PC{i+1}" for i in range(n_components)]

        return pd.DataFrame(
            self.pca.components_.T,
            columns=columns,
            index=self.original_feature_names
        )


class AdvancedFeatureSelector:
    """
    Advanced feature selection beyond Phase 3.

    Includes:
    - Correlation-based removal
    - L1-based selection
    - Importance-based selection
    """

    @staticmethod
    def remove_correlated_features(
        X: pd.DataFrame,
        threshold: float = 0.95
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Remove highly correlated features.

        For each pair with correlation > threshold, keep the first feature.

        Args:
            X: Features
            threshold: Correlation threshold

        Returns:
            (X_reduced, removed_features)
        """
        corr_matrix = X.corr().abs()

        # Upper triangle (avoid double-counting)
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )

        # Find features to drop
        to_drop = [col for col in upper.columns if any(upper[col] > threshold)]

        X_reduced = X.drop(columns=to_drop)
        return X_reduced, to_drop

    @staticmethod
    def select_by_importance(
        X: pd.DataFrame,
        importance_scores: Dict[str, float],
        top_k: int
    ) -> pd.DataFrame:
        """
        Select top-k features by importance.

        Args:
            X: Features
            importance_scores: Dict mapping feature_name -> importance
            top_k: Number of features to keep

        Returns:
            X with only top-k features
        """
        # Sort by importance
        sorted_features = sorted(
            importance_scores.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )

        # Select top-k
        top_features = [feat for feat, _ in sorted_features[:top_k]]

        return X[top_features]
