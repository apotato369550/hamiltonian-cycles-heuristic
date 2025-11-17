"""
ML Problem Formulation and Dataset Preparation (Prompt 1).

Handles:
- ML problem type selection (regression, binary, multiclass, ranking)
- Missing value handling
- Constant feature removal
- Outlier handling
- Train/validation/test splitting with stratification

Also includes Prompt 2: Train-Test Split Strategy
- Random split
- Graph-based split
- Stratified graph split
- Graph-type holdout
- Size-based holdout
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any, Callable
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class MLProblemType(Enum):
    """Type of ML problem formulation."""
    REGRESSION = "regression"  # Predict continuous anchor quality score
    BINARY = "binary"  # Predict if vertex is in top-k%
    MULTICLASS = "multiclass"  # Predict quality tier (excellent/good/mediocre/poor)
    RANKING = "ranking"  # Predict relative ordering (not implemented in Phase 4)


class SplitStrategy(Enum):
    """Strategy for splitting data into train/val/test sets."""
    RANDOM = "random"  # Random shuffle (baseline)
    GRAPH_BASED = "graph_based"  # Split by graph (no graph in multiple sets)
    STRATIFIED_GRAPH = "stratified_graph"  # Graph-based with stratification by type
    GRAPH_TYPE_HOLDOUT = "graph_type_holdout"  # Hold out entire graph type
    SIZE_HOLDOUT = "size_holdout"  # Train on small, test on large graphs


@dataclass
class DatasetPreparator:
    """
    Prepares features and labels for ML training.

    Handles:
    - Missing value imputation or removal
    - Constant feature removal
    - Outlier clipping or removal
    - Feature/label validation

    Attributes:
        problem_type: Type of ML problem
        remove_constant_features: Remove features with variance < threshold
        constant_threshold: Variance threshold for constant detection
        handle_outliers: Method for outlier handling ('clip', 'remove', 'none')
        outlier_percentiles: Percentiles for clipping (e.g., (1, 99))
        handle_missing: Method for missing values ('mean', 'median', 'remove', 'none')
        random_seed: Random seed for reproducibility
    """
    problem_type: MLProblemType = MLProblemType.REGRESSION
    remove_constant_features: bool = True
    constant_threshold: float = 1e-6
    handle_outliers: str = 'clip'  # 'clip', 'remove', 'none'
    outlier_percentiles: Tuple[float, float] = (1.0, 99.0)
    handle_missing: str = 'mean'  # 'mean', 'median', 'remove', 'none'
    random_seed: Optional[int] = None

    def __post_init__(self):
        """Validate configuration."""
        if self.handle_outliers not in ['clip', 'remove', 'none']:
            raise ValueError(f"Invalid outlier handling: {self.handle_outliers}")
        if self.handle_missing not in ['mean', 'median', 'remove', 'none']:
            raise ValueError(f"Invalid missing value handling: {self.handle_missing}")

    def prepare(
        self,
        features: pd.DataFrame,
        labels: pd.Series,
        feature_metadata: Optional[Dict[str, Any]] = None
    ) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
        """
        Prepare dataset for ML training.

        Args:
            features: Feature matrix (N x F)
            labels: Target labels (N,)
            feature_metadata: Optional metadata about features

        Returns:
            Tuple of (prepared_features, prepared_labels, metadata)
            metadata contains: removed_features, imputation_values, outlier_bounds, etc.
        """
        if len(features) != len(labels):
            raise ValueError(f"Feature/label length mismatch: {len(features)} vs {len(labels)}")

        metadata = {
            'original_shape': features.shape,
            'original_feature_count': len(features.columns),
            'removed_features': [],
            'imputation_values': {},
            'outlier_bounds': {},
            'constant_features': [],
        }

        # Copy to avoid modifying original
        X = features.copy()
        y = labels.copy()

        # 1. Handle missing values
        X, y, missing_info = self._handle_missing_values(X, y)
        metadata['missing_info'] = missing_info

        # 2. Remove constant features
        if self.remove_constant_features:
            X, const_features = self._remove_constant_features(X)
            metadata['constant_features'] = const_features
            metadata['removed_features'].extend(const_features)

        # 3. Handle outliers
        if self.handle_outliers != 'none':
            X, y, outlier_info = self._handle_outliers(X, y)
            metadata['outlier_info'] = outlier_info

        metadata['final_shape'] = X.shape
        metadata['final_feature_count'] = len(X.columns)

        return X, y, metadata

    def _handle_missing_values(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
        """Handle missing values in features and labels."""
        info = {
            'features_with_missing': [],
            'labels_with_missing': 0,
            'rows_removed': 0,
            'imputation_values': {}
        }

        # Check for missing in labels
        missing_labels = y.isna().sum()
        if missing_labels > 0:
            info['labels_with_missing'] = int(missing_labels)
            # Always remove rows with missing labels
            valid_idx = ~y.isna()
            X = X[valid_idx]
            y = y[valid_idx]
            info['rows_removed'] = int(missing_labels)

        # Check for missing in features
        missing_counts = X.isna().sum()
        features_with_missing = missing_counts[missing_counts > 0].index.tolist()
        info['features_with_missing'] = features_with_missing

        if len(features_with_missing) > 0:
            if self.handle_missing == 'remove':
                # Remove rows with any missing values
                valid_idx = ~X.isna().any(axis=1)
                removed = len(X) - valid_idx.sum()
                X = X[valid_idx]
                y = y[valid_idx]
                info['rows_removed'] += int(removed)

            elif self.handle_missing in ['mean', 'median']:
                # Impute with mean or median
                for col in features_with_missing:
                    if self.handle_missing == 'mean':
                        fill_value = X[col].mean()
                    else:  # median
                        fill_value = X[col].median()

                    X[col].fillna(fill_value, inplace=True)
                    info['imputation_values'][col] = float(fill_value)

            # else: 'none' - leave as-is

        return X, y, info

    def _remove_constant_features(
        self,
        X: pd.DataFrame
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Remove features with variance below threshold."""
        variances = X.var()
        constant_features = variances[variances < self.constant_threshold].index.tolist()

        if len(constant_features) > 0:
            X = X.drop(columns=constant_features)

        return X, constant_features

    def _handle_outliers(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
        """Handle outliers via clipping or removal."""
        info = {
            'method': self.handle_outliers,
            'bounds': {},
            'rows_removed': 0,
            'values_clipped': {}
        }

        if self.handle_outliers == 'clip':
            # Clip each feature to percentile bounds
            for col in X.columns:
                lower = np.percentile(X[col], self.outlier_percentiles[0])
                upper = np.percentile(X[col], self.outlier_percentiles[1])

                clipped_count = ((X[col] < lower) | (X[col] > upper)).sum()
                if clipped_count > 0:
                    X[col] = X[col].clip(lower, upper)
                    info['bounds'][col] = (float(lower), float(upper))
                    info['values_clipped'][col] = int(clipped_count)

        elif self.handle_outliers == 'remove':
            # Remove rows where any feature is outlier
            mask = pd.Series(True, index=X.index)

            for col in X.columns:
                lower = np.percentile(X[col], self.outlier_percentiles[0])
                upper = np.percentile(X[col], self.outlier_percentiles[1])

                mask &= (X[col] >= lower) & (X[col] <= upper)
                info['bounds'][col] = (float(lower), float(upper))

            removed = len(X) - mask.sum()
            X = X[mask]
            y = y[mask]
            info['rows_removed'] = int(removed)

        return X, y, info


@dataclass
class DatasetSplit:
    """
    Result of train/val/test split.

    Attributes:
        X_train, X_val, X_test: Feature matrices
        y_train, y_val, y_test: Label vectors
        train_graphs, val_graphs, test_graphs: Graph IDs in each split
        metadata: Split statistics and information
    """
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series
    train_graphs: Optional[List[Any]] = None
    val_graphs: Optional[List[Any]] = None
    test_graphs: Optional[List[Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_summary(self) -> str:
        """Get human-readable summary of split."""
        summary = []
        summary.append("Dataset Split Summary:")
        summary.append(f"  Train: {len(self.X_train)} samples")
        summary.append(f"  Val:   {len(self.X_val)} samples")
        summary.append(f"  Test:  {len(self.X_test)} samples")

        if self.train_graphs is not None:
            summary.append(f"  Train graphs: {len(self.train_graphs)}")
            summary.append(f"  Val graphs:   {len(self.val_graphs)}")
            summary.append(f"  Test graphs:  {len(self.test_graphs)}")

        if 'strategy' in self.metadata:
            summary.append(f"  Strategy: {self.metadata['strategy']}")

        return "\n".join(summary)


class TrainTestSplitter:
    """
    Implements various train/test splitting strategies (Prompt 2).

    Supports:
    - Random split (baseline)
    - Graph-based split (no graph in multiple sets)
    - Stratified graph split (balanced by graph type)
    - Graph-type holdout (train on some types, test on others)
    - Size-based holdout (train on small, test on large)
    """

    def __init__(
        self,
        strategy: SplitStrategy = SplitStrategy.STRATIFIED_GRAPH,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        random_seed: Optional[int] = None
    ):
        """
        Initialize splitter.

        Args:
            strategy: Splitting strategy
            train_ratio: Proportion for training
            val_ratio: Proportion for validation
            test_ratio: Proportion for testing
            random_seed: Random seed for reproducibility
        """
        if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
            raise ValueError("Train/val/test ratios must sum to 1.0")

        self.strategy = strategy
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_seed = random_seed

    def split(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        graph_ids: Optional[pd.Series] = None,
        graph_types: Optional[pd.Series] = None,
        graph_sizes: Optional[pd.Series] = None,
        holdout_graph_type: Optional[str] = None,
        size_threshold: Optional[int] = None
    ) -> DatasetSplit:
        """
        Split dataset according to configured strategy.

        Args:
            X: Feature matrix
            y: Labels
            graph_ids: Graph identifier for each sample
            graph_types: Graph type for each sample (e.g., 'euclidean', 'metric')
            graph_sizes: Graph size (number of vertices) for each sample
            holdout_graph_type: For GRAPH_TYPE_HOLDOUT strategy
            size_threshold: For SIZE_HOLDOUT strategy

        Returns:
            DatasetSplit object with train/val/test sets
        """
        if self.strategy == SplitStrategy.RANDOM:
            return self._random_split(X, y)

        elif self.strategy == SplitStrategy.GRAPH_BASED:
            if graph_ids is None:
                raise ValueError("graph_ids required for GRAPH_BASED strategy")
            return self._graph_based_split(X, y, graph_ids)

        elif self.strategy == SplitStrategy.STRATIFIED_GRAPH:
            if graph_ids is None or graph_types is None:
                raise ValueError("graph_ids and graph_types required for STRATIFIED_GRAPH")
            return self._stratified_graph_split(X, y, graph_ids, graph_types)

        elif self.strategy == SplitStrategy.GRAPH_TYPE_HOLDOUT:
            if graph_types is None or holdout_graph_type is None:
                raise ValueError("graph_types and holdout_graph_type required for GRAPH_TYPE_HOLDOUT")
            return self._graph_type_holdout(X, y, graph_ids, graph_types, holdout_graph_type)

        elif self.strategy == SplitStrategy.SIZE_HOLDOUT:
            if graph_sizes is None or size_threshold is None:
                raise ValueError("graph_sizes and size_threshold required for SIZE_HOLDOUT")
            return self._size_holdout(X, y, graph_ids, graph_sizes, size_threshold)

        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def _random_split(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> DatasetSplit:
        """Random split (baseline)."""
        # First split: train+val vs test
        X_trainval, X_test, y_trainval, y_test = train_test_split(
            X, y,
            test_size=self.test_ratio,
            random_state=self.random_seed
        )

        # Second split: train vs val
        val_ratio_adjusted = self.val_ratio / (self.train_ratio + self.val_ratio)
        X_train, X_val, y_train, y_val = train_test_split(
            X_trainval, y_trainval,
            test_size=val_ratio_adjusted,
            random_state=self.random_seed
        )

        return DatasetSplit(
            X_train=X_train,
            X_val=X_val,
            X_test=X_test,
            y_train=y_train,
            y_val=y_val,
            y_test=y_test,
            metadata={'strategy': 'random'}
        )

    def _graph_based_split(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        graph_ids: pd.Series
    ) -> DatasetSplit:
        """Split by graph: entire graphs in train/val/test."""
        unique_graphs = graph_ids.unique()
        n_graphs = len(unique_graphs)

        # Shuffle graphs
        np.random.seed(self.random_seed)
        shuffled_graphs = np.random.permutation(unique_graphs)

        # Split graph IDs
        n_train = int(n_graphs * self.train_ratio)
        n_val = int(n_graphs * self.val_ratio)

        train_graph_ids = shuffled_graphs[:n_train]
        val_graph_ids = shuffled_graphs[n_train:n_train + n_val]
        test_graph_ids = shuffled_graphs[n_train + n_val:]

        # Select samples by graph ID
        train_mask = graph_ids.isin(train_graph_ids)
        val_mask = graph_ids.isin(val_graph_ids)
        test_mask = graph_ids.isin(test_graph_ids)

        return DatasetSplit(
            X_train=X[train_mask],
            X_val=X[val_mask],
            X_test=X[test_mask],
            y_train=y[train_mask],
            y_val=y[val_mask],
            y_test=y[test_mask],
            train_graphs=train_graph_ids.tolist(),
            val_graphs=val_graph_ids.tolist(),
            test_graphs=test_graph_ids.tolist(),
            metadata={'strategy': 'graph_based', 'n_graphs': n_graphs}
        )

    def _stratified_graph_split(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        graph_ids: pd.Series,
        graph_types: pd.Series
    ) -> DatasetSplit:
        """Graph-based split with stratification by graph type."""
        # Group graphs by type
        graph_type_map = {}
        for gid in graph_ids.unique():
            gtype = graph_types[graph_ids == gid].iloc[0]
            if gtype not in graph_type_map:
                graph_type_map[gtype] = []
            graph_type_map[gtype].append(gid)

        # Split each graph type proportionally
        train_graphs = []
        val_graphs = []
        test_graphs = []

        np.random.seed(self.random_seed)

        for gtype, gids in graph_type_map.items():
            gids = np.array(gids)
            np.random.shuffle(gids)

            n = len(gids)
            n_train = max(1, int(n * self.train_ratio))
            n_val = max(1, int(n * self.val_ratio))

            train_graphs.extend(gids[:n_train])
            val_graphs.extend(gids[n_train:n_train + n_val])
            test_graphs.extend(gids[n_train + n_val:])

        # Select samples
        train_mask = graph_ids.isin(train_graphs)
        val_mask = graph_ids.isin(val_graphs)
        test_mask = graph_ids.isin(test_graphs)

        return DatasetSplit(
            X_train=X[train_mask],
            X_val=X[val_mask],
            X_test=X[test_mask],
            y_train=y[train_mask],
            y_val=y[val_mask],
            y_test=y[test_mask],
            train_graphs=train_graphs,
            val_graphs=val_graphs,
            test_graphs=test_graphs,
            metadata={
                'strategy': 'stratified_graph',
                'graph_types': list(graph_type_map.keys()),
                'graphs_per_type': {k: len(v) for k, v in graph_type_map.items()}
            }
        )

    def _graph_type_holdout(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        graph_ids: pd.Series,
        graph_types: pd.Series,
        holdout_type: str
    ) -> DatasetSplit:
        """Hold out entire graph type for testing."""
        # Test set = holdout type
        test_mask = graph_types == holdout_type

        # Train/val = remaining types
        trainval_mask = ~test_mask
        X_trainval = X[trainval_mask]
        y_trainval = y[trainval_mask]
        graph_ids_trainval = graph_ids[trainval_mask]

        # Split train/val using graph-based split
        unique_graphs = graph_ids_trainval.unique()
        n_graphs = len(unique_graphs)

        np.random.seed(self.random_seed)
        shuffled_graphs = np.random.permutation(unique_graphs)

        # Adjust ratios for train/val only
        val_ratio_adjusted = self.val_ratio / (self.train_ratio + self.val_ratio)
        n_val = max(1, int(n_graphs * val_ratio_adjusted))

        val_graph_ids = shuffled_graphs[:n_val]
        train_graph_ids = shuffled_graphs[n_val:]

        train_mask_trainval = graph_ids_trainval.isin(train_graph_ids)
        val_mask_trainval = graph_ids_trainval.isin(val_graph_ids)

        test_graph_ids = graph_ids[test_mask].unique()

        return DatasetSplit(
            X_train=X_trainval[train_mask_trainval],
            X_val=X_trainval[val_mask_trainval],
            X_test=X[test_mask],
            y_train=y_trainval[train_mask_trainval],
            y_val=y_trainval[val_mask_trainval],
            y_test=y[test_mask],
            train_graphs=train_graph_ids.tolist(),
            val_graphs=val_graph_ids.tolist(),
            test_graphs=test_graph_ids.tolist(),
            metadata={
                'strategy': 'graph_type_holdout',
                'holdout_type': holdout_type,
                'training_types': list(set(graph_types[trainval_mask]))
            }
        )

    def _size_holdout(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        graph_ids: pd.Series,
        graph_sizes: pd.Series,
        size_threshold: int
    ) -> DatasetSplit:
        """Train on small graphs, test on large graphs."""
        # Test set = large graphs
        test_mask = graph_sizes >= size_threshold

        # Train/val = small graphs
        trainval_mask = ~test_mask
        X_trainval = X[trainval_mask]
        y_trainval = y[trainval_mask]
        graph_ids_trainval = graph_ids[trainval_mask]

        # Split train/val using graph-based split
        unique_graphs = graph_ids_trainval.unique()
        n_graphs = len(unique_graphs)

        np.random.seed(self.random_seed)
        shuffled_graphs = np.random.permutation(unique_graphs)

        val_ratio_adjusted = self.val_ratio / (self.train_ratio + self.val_ratio)
        n_val = max(1, int(n_graphs * val_ratio_adjusted))

        val_graph_ids = shuffled_graphs[:n_val]
        train_graph_ids = shuffled_graphs[n_val:]

        train_mask_trainval = graph_ids_trainval.isin(train_graph_ids)
        val_mask_trainval = graph_ids_trainval.isin(val_graph_ids)

        test_graph_ids = graph_ids[test_mask].unique()

        return DatasetSplit(
            X_train=X_trainval[train_mask_trainval],
            X_val=X_trainval[val_mask_trainval],
            X_test=X[test_mask],
            y_train=y_trainval[train_mask_trainval],
            y_val=y_trainval[val_mask_trainval],
            y_test=y[test_mask],
            train_graphs=train_graph_ids.tolist(),
            val_graphs=val_graph_ids.tolist(),
            test_graphs=test_graph_ids.tolist(),
            metadata={
                'strategy': 'size_holdout',
                'size_threshold': size_threshold,
                'train_size_range': (
                    int(graph_sizes[trainval_mask].min()),
                    int(graph_sizes[trainval_mask].max())
                ),
                'test_size_range': (
                    int(graph_sizes[test_mask].min()),
                    int(graph_sizes[test_mask].max())
                )
            }
        )
