"""
ML Models for Anchor Prediction (Prompts 3 and 4).

Prompt 3: Linear Regression Baseline
- OLS, Ridge, Lasso, ElasticNet
- Feature importance extraction
- Model diagnostics

Prompt 4: Tree-Based Models
- Decision Tree
- Random Forest
- Gradient Boosting (XGBoost/LightGBM if available)
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any, Union
import numpy as np
import pandas as pd

from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet
)
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


class ModelType(Enum):
    """Type of ML model."""
    # Linear models
    LINEAR_OLS = "linear_ols"
    LINEAR_RIDGE = "linear_ridge"
    LINEAR_LASSO = "linear_lasso"
    LINEAR_ELASTICNET = "linear_elasticnet"

    # Tree models
    DECISION_TREE = "decision_tree"
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"


@dataclass
class ModelResult:
    """
    Result of model training and prediction.

    Attributes:
        model_type: Type of model
        predictions: Predicted values
        metrics: Performance metrics (R², MAE, RMSE)
        feature_importance: Feature importance scores
        coefficients: Model coefficients (for linear models)
        metadata: Additional model information
    """
    model_type: ModelType
    predictions: np.ndarray
    metrics: Dict[str, float]
    feature_importance: Optional[Dict[str, float]] = None
    coefficients: Optional[Dict[str, float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_summary(self) -> str:
        """Get human-readable summary."""
        summary = []
        summary.append(f"Model: {self.model_type.value}")
        summary.append("Metrics:")
        for metric, value in self.metrics.items():
            summary.append(f"  {metric}: {value:.4f}")

        if self.feature_importance:
            top_features = sorted(
                self.feature_importance.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )[:5]
            summary.append("Top 5 Features:")
            for feat, importance in top_features:
                summary.append(f"  {feat}: {importance:.4f}")

        return "\n".join(summary)


class LinearRegressionModel:
    """
    Linear regression models (Prompt 3).

    Supports:
    - OLS (Ordinary Least Squares)
    - Ridge (L2 regularization)
    - Lasso (L1 regularization)
    - ElasticNet (L1 + L2)

    Provides:
    - Feature importance via coefficients
    - Model diagnostics (residuals, R²)
    - Standardized coefficients for comparison
    """

    def __init__(
        self,
        model_type: ModelType = ModelType.LINEAR_RIDGE,
        alpha: float = 1.0,
        l1_ratio: float = 0.5,  # For ElasticNet
        fit_intercept: bool = True,
        standardize_features: bool = True,
        random_seed: Optional[int] = None
    ):
        """
        Initialize linear regression model.

        Args:
            model_type: Type of linear model
            alpha: Regularization strength (for Ridge/Lasso/ElasticNet)
            l1_ratio: L1 ratio for ElasticNet (1.0 = Lasso, 0.0 = Ridge)
            fit_intercept: Whether to fit intercept
            standardize_features: Whether to standardize features (recommended for regularized models)
            random_seed: Random seed
        """
        if model_type not in [
            ModelType.LINEAR_OLS,
            ModelType.LINEAR_RIDGE,
            ModelType.LINEAR_LASSO,
            ModelType.LINEAR_ELASTICNET
        ]:
            raise ValueError(f"Invalid linear model type: {model_type}")

        self.model_type = model_type
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.fit_intercept = fit_intercept
        self.standardize_features = standardize_features
        self.random_seed = random_seed

        self.model = None
        self.scaler = None
        self.feature_names = None
        self.is_fitted = False

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series]
    ) -> 'LinearRegressionModel':
        """
        Fit linear regression model.

        Args:
            X: Feature matrix (N x F)
            y: Target labels (N,)

        Returns:
            self
        """
        # Convert to numpy if DataFrame/Series
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
            X = X.values
        else:
            self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]

        if isinstance(y, pd.Series):
            y = y.values

        # Standardize features if requested
        if self.standardize_features:
            self.scaler = StandardScaler()
            X = self.scaler.fit_transform(X)

        # Create model
        if self.model_type == ModelType.LINEAR_OLS:
            self.model = LinearRegression(fit_intercept=self.fit_intercept)

        elif self.model_type == ModelType.LINEAR_RIDGE:
            self.model = Ridge(
                alpha=self.alpha,
                fit_intercept=self.fit_intercept,
                random_state=self.random_seed
            )

        elif self.model_type == ModelType.LINEAR_LASSO:
            self.model = Lasso(
                alpha=self.alpha,
                fit_intercept=self.fit_intercept,
                random_state=self.random_seed
            )

        elif self.model_type == ModelType.LINEAR_ELASTICNET:
            self.model = ElasticNet(
                alpha=self.alpha,
                l1_ratio=self.l1_ratio,
                fit_intercept=self.fit_intercept,
                random_state=self.random_seed
            )

        # Fit model
        self.model.fit(X, y)
        self.is_fitted = True

        return self

    def predict(
        self,
        X: Union[np.ndarray, pd.DataFrame]
    ) -> np.ndarray:
        """
        Predict using fitted model.

        Args:
            X: Feature matrix (N x F)

        Returns:
            Predictions (N,)
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        # Convert to numpy if DataFrame
        if isinstance(X, pd.DataFrame):
            X = X.values

        # Standardize if needed
        if self.standardize_features:
            X = self.scaler.transform(X)

        return self.model.predict(X)

    def evaluate(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series]
    ) -> ModelResult:
        """
        Evaluate model on test data.

        Args:
            X: Feature matrix
            y: True labels

        Returns:
            ModelResult with predictions and metrics
        """
        if isinstance(y, pd.Series):
            y = y.values

        # Predict
        predictions = self.predict(X)

        # Compute metrics
        metrics = {
            'r2': r2_score(y, predictions),
            'mae': mean_absolute_error(y, predictions),
            'rmse': np.sqrt(mean_squared_error(y, predictions)),
            'mse': mean_squared_error(y, predictions)
        }

        # Extract feature importance
        feature_importance = self.get_feature_importance()

        # Extract coefficients
        coefficients = self.get_coefficients()

        return ModelResult(
            model_type=self.model_type,
            predictions=predictions,
            metrics=metrics,
            feature_importance=feature_importance,
            coefficients=coefficients,
            metadata={
                'alpha': self.alpha,
                'l1_ratio': self.l1_ratio if self.model_type == ModelType.LINEAR_ELASTICNET else None,
                'standardized': self.standardize_features,
                'n_features': len(self.feature_names)
            }
        )

    def get_coefficients(self) -> Dict[str, float]:
        """Get model coefficients."""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted")

        coefs = {}
        for name, coef in zip(self.feature_names, self.model.coef_):
            coefs[name] = float(coef)

        # Add intercept
        if self.fit_intercept:
            coefs['intercept'] = float(self.model.intercept_)

        return coefs

    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance via standardized coefficients.

        For linear models, importance = |coefficient| × feature_std
        This makes coefficients comparable across features with different scales.
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted")

        # Use absolute coefficients as importance
        importance = {}
        for name, coef in zip(self.feature_names, self.model.coef_):
            importance[name] = abs(float(coef))

        return importance

    def get_diagnostics(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series]
    ) -> Dict[str, Any]:
        """
        Get model diagnostics (residuals, etc.).

        Args:
            X: Feature matrix
            y: True labels

        Returns:
            Dictionary with diagnostic information
        """
        if isinstance(y, pd.Series):
            y = y.values

        predictions = self.predict(X)
        residuals = y - predictions

        diagnostics = {
            'residuals': residuals,
            'residuals_mean': float(np.mean(residuals)),
            'residuals_std': float(np.std(residuals)),
            'residuals_min': float(np.min(residuals)),
            'residuals_max': float(np.max(residuals)),
            'residuals_skew': float(pd.Series(residuals).skew()),
            'residuals_kurtosis': float(pd.Series(residuals).kurtosis())
        }

        return diagnostics


class TreeBasedModel:
    """
    Tree-based models (Prompt 4).

    Supports:
    - Decision Tree
    - Random Forest
    - Gradient Boosting

    Provides:
    - Feature importance via impurity reduction
    - No feature scaling required
    - Handles non-linearities automatically
    """

    def __init__(
        self,
        model_type: ModelType = ModelType.RANDOM_FOREST,
        max_depth: Optional[int] = None,
        min_samples_leaf: int = 1,
        min_samples_split: int = 2,
        n_estimators: int = 100,  # For ensemble models
        max_features: Union[str, int, float] = 'sqrt',  # For RF
        learning_rate: float = 0.1,  # For GB
        random_seed: Optional[int] = None
    ):
        """
        Initialize tree-based model.

        Args:
            model_type: Type of tree model
            max_depth: Maximum depth of trees
            min_samples_leaf: Minimum samples per leaf
            min_samples_split: Minimum samples to split node
            n_estimators: Number of trees (for ensembles)
            max_features: Max features per split (for RF)
            learning_rate: Learning rate (for GB)
            random_seed: Random seed
        """
        if model_type not in [
            ModelType.DECISION_TREE,
            ModelType.RANDOM_FOREST,
            ModelType.GRADIENT_BOOSTING
        ]:
            raise ValueError(f"Invalid tree model type: {model_type}")

        self.model_type = model_type
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.learning_rate = learning_rate
        self.random_seed = random_seed

        self.model = None
        self.feature_names = None
        self.is_fitted = False

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series]
    ) -> 'TreeBasedModel':
        """
        Fit tree-based model.

        Args:
            X: Feature matrix (N x F)
            y: Target labels (N,)

        Returns:
            self
        """
        # Convert to numpy if DataFrame/Series
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
            X = X.values
        else:
            self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]

        if isinstance(y, pd.Series):
            y = y.values

        # Create model
        if self.model_type == ModelType.DECISION_TREE:
            self.model = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                min_samples_split=self.min_samples_split,
                random_state=self.random_seed
            )

        elif self.model_type == ModelType.RANDOM_FOREST:
            self.model = RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                min_samples_split=self.min_samples_split,
                max_features=self.max_features,
                random_state=self.random_seed,
                n_jobs=-1  # Use all CPUs
            )

        elif self.model_type == ModelType.GRADIENT_BOOSTING:
            self.model = GradientBoostingRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                min_samples_split=self.min_samples_split,
                learning_rate=self.learning_rate,
                random_state=self.random_seed
            )

        # Fit model
        self.model.fit(X, y)
        self.is_fitted = True

        return self

    def predict(
        self,
        X: Union[np.ndarray, pd.DataFrame]
    ) -> np.ndarray:
        """
        Predict using fitted model.

        Args:
            X: Feature matrix (N x F)

        Returns:
            Predictions (N,)
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        # Convert to numpy if DataFrame
        if isinstance(X, pd.DataFrame):
            X = X.values

        return self.model.predict(X)

    def evaluate(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series]
    ) -> ModelResult:
        """
        Evaluate model on test data.

        Args:
            X: Feature matrix
            y: True labels

        Returns:
            ModelResult with predictions and metrics
        """
        if isinstance(y, pd.Series):
            y = y.values

        # Predict
        predictions = self.predict(X)

        # Compute metrics
        metrics = {
            'r2': r2_score(y, predictions),
            'mae': mean_absolute_error(y, predictions),
            'rmse': np.sqrt(mean_squared_error(y, predictions)),
            'mse': mean_squared_error(y, predictions)
        }

        # Extract feature importance
        feature_importance = self.get_feature_importance()

        return ModelResult(
            model_type=self.model_type,
            predictions=predictions,
            metrics=metrics,
            feature_importance=feature_importance,
            coefficients=None,  # Tree models don't have coefficients
            metadata={
                'max_depth': self.max_depth,
                'n_estimators': self.n_estimators if self.model_type != ModelType.DECISION_TREE else None,
                'n_features': len(self.feature_names)
            }
        )

    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance from tree model.

        Based on impurity reduction (Gini importance).
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted")

        importance = {}
        for name, imp in zip(self.feature_names, self.model.feature_importances_):
            importance[name] = float(imp)

        return importance
