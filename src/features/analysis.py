"""
Feature validation and analysis tools.

Provides utilities for:
- Feature validation (range checks, constant detection, correlation analysis)
- Exploratory data analysis (correlation matrices, PCA)
- Feature-target correlation
- Feature distribution analysis
"""

from typing import Dict, List, Tuple, Optional, Set
import numpy as np
from scipy import stats
from scipy.stats import pearsonr


class FeatureAnalyzer:
    """
    Analyzes feature matrices for quality and relationships.

    Provides methods for:
    - Sanity checks (range validation, constant features, correlation)
    - Exploratory data analysis
    - Feature-target correlation analysis
    - Feature distribution analysis
    """

    def __init__(self, features: np.ndarray, feature_names: List[str]):
        """
        Initialize feature analyzer.

        Args:
            features: NxF feature matrix
            feature_names: List of F feature names
        """
        self.features = features
        self.feature_names = feature_names
        self.n_samples = features.shape[0]
        self.n_features = features.shape[1]

        if len(feature_names) != self.n_features:
            raise ValueError(
                f"Feature count mismatch: {self.n_features} features, "
                f"{len(feature_names)} names"
            )

    def validate_ranges(
        self,
        expected_ranges: Optional[Dict[str, Tuple[float, float]]] = None
    ) -> Dict[str, List[str]]:
        """
        Validate feature values are in expected ranges.

        Args:
            expected_ranges: Optional dict mapping feature name to (min, max).
                           If None, just checks for NaN/Inf.

        Returns:
            Dictionary with validation results:
            - 'nan_features': Features with NaN values
            - 'inf_features': Features with infinite values
            - 'out_of_range': Features outside expected ranges
        """
        results = {
            'nan_features': [],
            'inf_features': [],
            'out_of_range': []
        }

        for i, name in enumerate(self.feature_names):
            feature_values = self.features[:, i]

            # Check for NaN
            if np.any(np.isnan(feature_values)):
                results['nan_features'].append(name)

            # Check for Inf
            if np.any(np.isinf(feature_values)):
                results['inf_features'].append(name)

            # Check ranges if provided
            if expected_ranges and name in expected_ranges:
                min_val, max_val = expected_ranges[name]
                if np.any(feature_values < min_val) or np.any(feature_values > max_val):
                    results['out_of_range'].append(name)

        return results

    def find_constant_features(
        self,
        variance_threshold: float = 1e-10
    ) -> List[str]:
        """
        Find features with zero or near-zero variance.

        Args:
            variance_threshold: Variance below this is considered constant

        Returns:
            List of constant feature names
        """
        variances = np.var(self.features, axis=0)
        constant_features = [
            self.feature_names[i]
            for i in range(self.n_features)
            if variances[i] < variance_threshold
        ]
        return constant_features

    def compute_correlation_matrix(
        self,
        method: str = 'pearson'
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Compute correlation matrix for all features.

        Args:
            method: Correlation method ('pearson', 'spearman', 'kendall')

        Returns:
            correlation_matrix: FxF correlation matrix
            feature_names: List of feature names (for reference)
        """
        if method == 'pearson':
            corr_matrix = np.corrcoef(self.features, rowvar=False)
        elif method == 'spearman':
            # Compute Spearman rank correlation
            from scipy.stats import spearmanr
            corr_matrix, _ = spearmanr(self.features, axis=0)
        elif method == 'kendall':
            # Kendall tau (expensive for large feature sets)
            from scipy.stats import kendalltau
            corr_matrix = np.zeros((self.n_features, self.n_features))
            for i in range(self.n_features):
                for j in range(self.n_features):
                    if i == j:
                        corr_matrix[i, j] = 1.0
                    else:
                        tau, _ = kendalltau(self.features[:, i], self.features[:, j])
                        corr_matrix[i, j] = tau
        else:
            raise ValueError(f"Unknown correlation method: {method}")

        return corr_matrix, self.feature_names

    def find_highly_correlated_pairs(
        self,
        threshold: float = 0.95
    ) -> List[Tuple[str, str, float]]:
        """
        Find pairs of features with high correlation.

        Args:
            threshold: Correlation above this (in absolute value) is considered high

        Returns:
            List of (feature1, feature2, correlation) tuples
        """
        corr_matrix, _ = self.compute_correlation_matrix()

        high_corr_pairs = []
        for i in range(self.n_features):
            for j in range(i + 1, self.n_features):
                corr = corr_matrix[i, j]
                if abs(corr) >= threshold:
                    high_corr_pairs.append((
                        self.feature_names[i],
                        self.feature_names[j],
                        corr
                    ))

        # Sort by absolute correlation
        high_corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)

        return high_corr_pairs

    def perform_pca(
        self,
        n_components: Optional[int] = None
    ) -> Dict[str, np.ndarray]:
        """
        Perform PCA on features.

        Args:
            n_components: Number of components (None = all)

        Returns:
            Dictionary with:
            - 'components': Principal components
            - 'explained_variance': Variance explained by each component
            - 'explained_variance_ratio': Proportion of variance explained
            - 'transformed': Features in PC space
        """
        # Center features
        features_centered = self.features - np.mean(self.features, axis=0)

        # Compute covariance matrix
        cov_matrix = np.cov(features_centered, rowvar=False)

        # Eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # Sort by eigenvalue (descending)
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Select components
        if n_components is not None:
            eigenvalues = eigenvalues[:n_components]
            eigenvectors = eigenvectors[:, :n_components]

        # Transform features
        transformed = features_centered @ eigenvectors

        # Compute explained variance ratio
        total_variance = np.sum(eigenvalues)
        explained_variance_ratio = eigenvalues / total_variance if total_variance > 0 else eigenvalues

        return {
            'components': eigenvectors,
            'explained_variance': eigenvalues,
            'explained_variance_ratio': explained_variance_ratio,
            'transformed': transformed
        }

    def correlate_with_target(
        self,
        target: np.ndarray,
        method: str = 'pearson'
    ) -> List[Tuple[str, float, float]]:
        """
        Compute correlation between each feature and target variable.

        Args:
            target: Target variable (length N)
            method: Correlation method ('pearson' or 'spearman')

        Returns:
            List of (feature_name, correlation, p_value) sorted by abs(correlation)
        """
        if len(target) != self.n_samples:
            raise ValueError(
                f"Target length {len(target)} doesn't match samples {self.n_samples}"
            )

        correlations = []

        for i, name in enumerate(self.feature_names):
            feature_values = self.features[:, i]

            if method == 'pearson':
                corr, p_value = pearsonr(feature_values, target)
            elif method == 'spearman':
                from scipy.stats import spearmanr
                corr, p_value = spearmanr(feature_values, target)
            else:
                raise ValueError(f"Unknown method: {method}")

            correlations.append((name, corr, p_value))

        # Sort by absolute correlation
        correlations.sort(key=lambda x: abs(x[1]), reverse=True)

        return correlations

    def analyze_distributions(
        self
    ) -> Dict[str, Dict[str, float]]:
        """
        Analyze distribution of each feature.

        Returns:
            Dictionary mapping feature name to distribution statistics:
            - mean, std, min, max, skewness, kurtosis
        """
        distributions = {}

        for i, name in enumerate(self.feature_names):
            feature_values = self.features[:, i]

            distributions[name] = {
                'mean': np.mean(feature_values),
                'std': np.std(feature_values),
                'min': np.min(feature_values),
                'max': np.max(feature_values),
                'median': np.median(feature_values),
                'q25': np.percentile(feature_values, 25),
                'q75': np.percentile(feature_values, 75),
                'skewness': float(stats.skew(feature_values)),
                'kurtosis': float(stats.kurtosis(feature_values))
            }

        return distributions

    def detect_outliers(
        self,
        method: str = 'iqr',
        threshold: float = 3.0
    ) -> Dict[str, np.ndarray]:
        """
        Detect outliers in features.

        Args:
            method: 'iqr' (interquartile range) or 'zscore'
            threshold: For IQR: multiplier (default 3.0 = extreme outliers)
                      For z-score: absolute z-score threshold

        Returns:
            Dictionary mapping feature name to boolean array (True = outlier)
        """
        outliers = {}

        for i, name in enumerate(self.feature_names):
            feature_values = self.features[:, i]

            if method == 'iqr':
                q25 = np.percentile(feature_values, 25)
                q75 = np.percentile(feature_values, 75)
                iqr = q75 - q25

                lower_bound = q25 - threshold * iqr
                upper_bound = q75 + threshold * iqr

                is_outlier = (feature_values < lower_bound) | (feature_values > upper_bound)

            elif method == 'zscore':
                mean = np.mean(feature_values)
                std = np.std(feature_values)

                if std > 0:
                    z_scores = np.abs((feature_values - mean) / std)
                    is_outlier = z_scores > threshold
                else:
                    is_outlier = np.zeros(len(feature_values), dtype=bool)

            else:
                raise ValueError(f"Unknown outlier detection method: {method}")

            outliers[name] = is_outlier

        return outliers

    def get_feature_importance_by_variance(
        self,
        top_k: Optional[int] = None
    ) -> List[Tuple[str, float]]:
        """
        Rank features by variance (simple importance metric).

        Args:
            top_k: Return only top k features (None = all)

        Returns:
            List of (feature_name, variance) sorted by variance
        """
        variances = np.var(self.features, axis=0)

        feature_variance = [
            (self.feature_names[i], variances[i])
            for i in range(self.n_features)
        ]

        # Sort by variance
        feature_variance.sort(key=lambda x: x[1], reverse=True)

        if top_k is not None:
            feature_variance = feature_variance[:top_k]

        return feature_variance

    def summary_report(self) -> str:
        """
        Generate a comprehensive summary report.

        Returns:
            Multi-line string with summary statistics
        """
        lines = []
        lines.append("=" * 70)
        lines.append("Feature Analysis Summary Report")
        lines.append("=" * 70)
        lines.append(f"Number of samples: {self.n_samples}")
        lines.append(f"Number of features: {self.n_features}")
        lines.append("")

        # Validation
        validation = self.validate_ranges()
        lines.append("Validation Results:")
        lines.append(f"  Features with NaN: {len(validation['nan_features'])}")
        lines.append(f"  Features with Inf: {len(validation['inf_features'])}")
        if validation['nan_features']:
            lines.append(f"    {validation['nan_features'][:5]}")
        if validation['inf_features']:
            lines.append(f"    {validation['inf_features'][:5]}")
        lines.append("")

        # Constant features
        constant = self.find_constant_features()
        lines.append(f"Constant features (var < 1e-10): {len(constant)}")
        if constant:
            lines.append(f"  {constant[:5]}")
        lines.append("")

        # Highly correlated pairs
        high_corr = self.find_highly_correlated_pairs(threshold=0.95)
        lines.append(f"Highly correlated pairs (|r| > 0.95): {len(high_corr)}")
        for pair in high_corr[:5]:
            lines.append(f"  {pair[0]} <-> {pair[1]}: {pair[2]:.3f}")
        lines.append("")

        # PCA
        pca = self.perform_pca(n_components=5)
        lines.append("PCA Results (top 5 components):")
        cumsum = np.cumsum(pca['explained_variance_ratio'])
        for i in range(min(5, len(pca['explained_variance_ratio']))):
            lines.append(
                f"  PC{i+1}: {pca['explained_variance_ratio'][i]:.3f} "
                f"(cumsum: {cumsum[i]:.3f})"
            )
        lines.append("")

        # Feature importance by variance
        top_variance = self.get_feature_importance_by_variance(top_k=10)
        lines.append("Top 10 features by variance:")
        for name, var in top_variance:
            lines.append(f"  {name:50s} {var:12.4f}")

        lines.append("=" * 70)

        return "\n".join(lines)
