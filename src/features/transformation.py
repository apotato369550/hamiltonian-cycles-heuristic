"""
Feature Transformation and Engineering.

Provides tools for transforming and combining features to create derived
features with improved predictive power. Supports non-linear transformations,
feature interactions, standardization, and binning.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Tuple, Any
import numpy as np
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    KBinsDiscretizer
)


class TransformationType(Enum):
    """Types of non-linear transformations."""
    LOG = "log"
    SQRT = "sqrt"
    POLYNOMIAL = "polynomial"
    INVERSE = "inverse"


@dataclass
class TransformationConfig:
    """
    Configuration for feature transformation.

    Attributes:
        transformations: Dict mapping feature names to transformation types
        interactions: List of (feat1, feat2, operation) tuples
        standardization: Type of standardization ('zscore', 'minmax', 'robust', None)
        binning: Dict mapping feature names to binning strategy
        polynomial_degree: Degree for polynomial transformations
        n_bins: Number of bins for discretization
        handle_zeros: How to handle zeros/negatives in log/inverse ('offset', 'replace', 'skip')
    """
    transformations: Dict[str, TransformationType] = field(default_factory=dict)
    interactions: List[Tuple[str, str, str]] = field(default_factory=list)
    standardization: Optional[str] = None
    binning: Dict[str, str] = field(default_factory=dict)
    polynomial_degree: int = 2
    n_bins: int = 5
    handle_zeros: str = 'offset'  # 'offset', 'replace', 'skip'


class FeatureTransformer:
    """
    Feature transformation system for creating derived features.

    Supports:
    - Non-linear transformations (log, sqrt, polynomial, inverse)
    - Feature interactions (product, ratio, difference)
    - Standardization (z-score, min-max, robust)
    - Binning/discretization (uniform, quantile)

    All transformations handle edge cases (zeros, negatives, infinities).
    """

    def __init__(self):
        """Initialize transformer."""
        self._fitted_scalers = {}
        self._fitted_discretizers = {}
        self._is_fitted = False

    def fit_transform(
        self,
        X: np.ndarray,
        feature_names: List[str],
        config: TransformationConfig
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Fit transformer on data and transform.

        Args:
            X: Feature matrix (n_samples, n_features)
            feature_names: Original feature names
            config: Transformation configuration

        Returns:
            Transformed features and new feature names
        """
        self._is_fitted = False
        return self.transform(X, feature_names, config, fit=True)

    def transform(
        self,
        X: np.ndarray,
        feature_names: List[str],
        config: TransformationConfig,
        fit: bool = False
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Transform features according to config.

        Args:
            X: Feature matrix
            feature_names: Original feature names
            config: Transformation configuration
            fit: Whether to fit scalers/discretizers

        Returns:
            Transformed features and new feature names
        """
        if X.shape[1] != len(feature_names):
            raise ValueError(
                f"Feature count mismatch: X has {X.shape[1]} columns "
                f"but {len(feature_names)} names provided"
            )

        # Start with original features
        X_transformed = X.copy()
        new_names = feature_names.copy()

        # Step 1: Apply non-linear transformations
        if config.transformations:
            X_transformed, new_names = self._apply_transformations(
                X_transformed, new_names, config
            )

        # Step 2: Create feature interactions
        if config.interactions:
            X_transformed, new_names = self._create_interactions(
                X_transformed, new_names, config
            )

        # Step 3: Apply binning
        if config.binning:
            X_transformed, new_names = self._apply_binning(
                X_transformed, new_names, config, fit
            )

        # Step 4: Apply standardization (always last)
        if config.standardization:
            X_transformed = self._apply_standardization(
                X_transformed, config.standardization, fit
            )

        if fit:
            self._is_fitted = True

        return X_transformed, new_names

    def _apply_transformations(
        self,
        X: np.ndarray,
        feature_names: List[str],
        config: TransformationConfig
    ) -> Tuple[np.ndarray, List[str]]:
        """Apply non-linear transformations to specified features."""
        X_new = X.copy()
        new_names = feature_names.copy()

        for feat_name, transform_type in config.transformations.items():
            if feat_name not in feature_names:
                continue

            feat_idx = feature_names.index(feat_name)
            feat_values = X[:, feat_idx]

            # Apply transformation
            if transform_type == TransformationType.LOG:
                transformed, new_name = self._log_transform(
                    feat_values, feat_name, config.handle_zeros
                )
            elif transform_type == TransformationType.SQRT:
                transformed, new_name = self._sqrt_transform(
                    feat_values, feat_name, config.handle_zeros
                )
            elif transform_type == TransformationType.POLYNOMIAL:
                transformed, new_name = self._polynomial_transform(
                    feat_values, feat_name, config.polynomial_degree
                )
            elif transform_type == TransformationType.INVERSE:
                transformed, new_name = self._inverse_transform(
                    feat_values, feat_name, config.handle_zeros
                )
            else:
                continue

            # Add to feature matrix
            X_new = np.column_stack([X_new, transformed])
            new_names.append(new_name)

        return X_new, new_names

    def _log_transform(
        self,
        values: np.ndarray,
        name: str,
        handle_zeros: str
    ) -> Tuple[np.ndarray, str]:
        """Apply log transformation."""
        values = values.copy()

        if handle_zeros == 'offset':
            # Shift values to be positive, then add small offset to avoid log(0)
            min_val = np.min(values)
            if min_val <= 0:
                # Shift all values to be positive
                values = values - min_val + 1.0
            # Add small offset to avoid log(0)
            min_positive = np.min(values[values > 0]) if np.any(values > 0) else 1.0
            offset = min_positive * 0.01
            values = np.log(values + offset)
        elif handle_zeros == 'replace':
            # Replace non-positive with small value
            values[values <= 0] = 1e-10
            values = np.log(values)
        else:  # skip
            # Only transform positive values, others become 0
            mask = values > 0
            result = np.zeros_like(values)
            result[mask] = np.log(values[mask])
            values = result

        return values, f"{name}_log"

    def _sqrt_transform(
        self,
        values: np.ndarray,
        name: str,
        handle_zeros: str
    ) -> Tuple[np.ndarray, str]:
        """Apply square root transformation."""
        values = values.copy()

        if handle_zeros == 'offset':
            # Shift to make all positive
            if np.min(values) < 0:
                values = values - np.min(values) + 1e-10
            values = np.sqrt(values)
        else:
            # Take absolute value
            values = np.sqrt(np.abs(values))

        return values, f"{name}_sqrt"

    def _polynomial_transform(
        self,
        values: np.ndarray,
        name: str,
        degree: int
    ) -> Tuple[np.ndarray, str]:
        """Apply polynomial transformation."""
        if degree == 2:
            return values ** 2, f"{name}_squared"
        elif degree == 3:
            return values ** 3, f"{name}_cubed"
        else:
            return values ** degree, f"{name}_pow{degree}"

    def _inverse_transform(
        self,
        values: np.ndarray,
        name: str,
        handle_zeros: str
    ) -> Tuple[np.ndarray, str]:
        """Apply inverse (1/x) transformation."""
        values = values.copy()

        if handle_zeros == 'offset':
            # Add small offset to avoid division by zero
            offset = 1e-10
            values = 1.0 / (values + offset)
        elif handle_zeros == 'replace':
            # Replace zeros with small value
            values[values == 0] = 1e-10
            values = 1.0 / values
        else:  # skip
            # Set zeros to zero in result
            result = np.zeros_like(values)
            mask = values != 0
            result[mask] = 1.0 / values[mask]
            values = result

        # Handle infinities
        values = np.nan_to_num(values, nan=0.0, posinf=1e10, neginf=-1e10)

        return values, f"{name}_inverse"

    def _create_interactions(
        self,
        X: np.ndarray,
        feature_names: List[str],
        config: TransformationConfig
    ) -> Tuple[np.ndarray, List[str]]:
        """Create feature interactions."""
        X_new = X.copy()
        new_names = feature_names.copy()

        for feat1_name, feat2_name, operation in config.interactions:
            if feat1_name not in feature_names or feat2_name not in feature_names:
                continue

            idx1 = feature_names.index(feat1_name)
            idx2 = feature_names.index(feat2_name)

            values1 = X[:, idx1]
            values2 = X[:, idx2]

            if operation == 'product':
                interaction = values1 * values2
                new_name = f"{feat1_name}_x_{feat2_name}"

            elif operation == 'ratio':
                # Handle division by zero
                denominator = values2.copy()
                denominator[denominator == 0] = 1e-10
                interaction = values1 / denominator
                interaction = np.nan_to_num(interaction, nan=0.0, posinf=1e10, neginf=-1e10)
                new_name = f"{feat1_name}_div_{feat2_name}"

            elif operation == 'difference':
                interaction = values1 - values2
                new_name = f"{feat1_name}_minus_{feat2_name}"

            elif operation == 'sum':
                interaction = values1 + values2
                new_name = f"{feat1_name}_plus_{feat2_name}"

            else:
                continue

            # Add to feature matrix
            X_new = np.column_stack([X_new, interaction])
            new_names.append(new_name)

        return X_new, new_names

    def _apply_binning(
        self,
        X: np.ndarray,
        feature_names: List[str],
        config: TransformationConfig,
        fit: bool
    ) -> Tuple[np.ndarray, List[str]]:
        """Apply binning/discretization to features."""
        X_new = X.copy()
        new_names = feature_names.copy()

        for feat_name, strategy in config.binning.items():
            if feat_name not in feature_names:
                continue

            feat_idx = feature_names.index(feat_name)
            feat_values = X[:, feat_idx].reshape(-1, 1)

            # Create or use fitted discretizer
            key = f"{feat_name}_{strategy}"

            if fit or key not in self._fitted_discretizers:
                # Fit new discretizer
                if strategy == 'uniform':
                    discretizer = KBinsDiscretizer(
                        n_bins=config.n_bins,
                        encode='onehot-dense',
                        strategy='uniform'
                    )
                elif strategy == 'quantile':
                    discretizer = KBinsDiscretizer(
                        n_bins=config.n_bins,
                        encode='onehot-dense',
                        strategy='quantile'
                    )
                else:
                    continue

                binned = discretizer.fit_transform(feat_values)
                self._fitted_discretizers[key] = discretizer
            else:
                # Use fitted discretizer
                discretizer = self._fitted_discretizers[key]
                binned = discretizer.transform(feat_values)

            # Create bin feature names
            bin_names = [f"{feat_name}_bin_{i}" for i in range(binned.shape[1])]

            # Add to feature matrix
            X_new = np.column_stack([X_new, binned])
            new_names.extend(bin_names)

        return X_new, new_names

    def _apply_standardization(
        self,
        X: np.ndarray,
        method: str,
        fit: bool
    ) -> np.ndarray:
        """Apply standardization to all features."""
        if method not in ['zscore', 'minmax', 'robust']:
            raise ValueError(f"Unknown standardization method: {method}")

        # Create or use fitted scaler
        if fit or method not in self._fitted_scalers:
            if method == 'zscore':
                scaler = StandardScaler()
            elif method == 'minmax':
                scaler = MinMaxScaler()
            else:  # robust
                scaler = RobustScaler()

            X_scaled = scaler.fit_transform(X)
            self._fitted_scalers[method] = scaler
        else:
            scaler = self._fitted_scalers[method]
            X_scaled = scaler.transform(X)

        return X_scaled

    def get_transformation_summary(self, config: TransformationConfig) -> Dict[str, Any]:
        """
        Get summary of transformations to be applied.

        Args:
            config: Transformation configuration

        Returns:
            Summary dictionary
        """
        summary = {
            'num_transformations': len(config.transformations),
            'num_interactions': len(config.interactions),
            'num_binnings': len(config.binning),
            'standardization': config.standardization,
            'transformations': {
                name: ttype.value
                for name, ttype in config.transformations.items()
            },
            'interactions': [
                f"{f1} {op} {f2}"
                for f1, f2, op in config.interactions
            ]
        }

        return summary

    def __repr__(self) -> str:
        fitted_str = "fitted" if self._is_fitted else "not fitted"
        return f"FeatureTransformer({fitted_str})"
