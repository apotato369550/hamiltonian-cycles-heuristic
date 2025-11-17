"""
Test Suite for Phase 4: Machine Learning Component (Prompts 1-4).

Tests:
- Dataset preparation (missing values, outliers, constant features)
- Train/test splitting strategies
- Linear regression models
- Tree-based models
- Model evaluation and feature importance
"""

import unittest
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import warnings

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.dataset import (
    MLProblemType,
    DatasetPreparator,
    SplitStrategy,
    TrainTestSplitter,
    DatasetSplit
)

from ml.models import (
    LinearRegressionModel,
    TreeBasedModel,
    ModelType,
    ModelResult
)


class TestDatasetPreparator(unittest.TestCase):
    """Test dataset preparation functionality (Prompt 1)."""

    def setUp(self):
        """Create test dataset."""
        np.random.seed(42)
        self.n = 100
        self.n_features = 10

        # Create features with some issues
        self.X = pd.DataFrame(
            np.random.randn(self.n, self.n_features),
            columns=[f'feat_{i}' for i in range(self.n_features)]
        )

        # Add missing values
        self.X.iloc[0, 0] = np.nan
        self.X.iloc[1, 1] = np.nan

        # Add constant feature
        self.X['const_feat'] = 1.0

        # Add outliers
        self.X.iloc[2, 2] = 100.0  # Extreme outlier

        self.y = pd.Series(np.random.randn(self.n))

    def test_preparator_initialization(self):
        """Test preparator initialization."""
        prep = DatasetPreparator(
            problem_type=MLProblemType.REGRESSION,
            remove_constant_features=True
        )
        self.assertEqual(prep.problem_type, MLProblemType.REGRESSION)
        self.assertTrue(prep.remove_constant_features)

    def test_handle_missing_mean(self):
        """Test missing value imputation with mean."""
        prep = DatasetPreparator(handle_missing='mean')
        X_clean, y_clean, metadata = prep.prepare(self.X, self.y)

        # No missing values in result
        self.assertEqual(X_clean.isna().sum().sum(), 0)
        self.assertGreater(len(metadata['missing_info']['features_with_missing']), 0)

    def test_handle_missing_remove(self):
        """Test missing value removal."""
        prep = DatasetPreparator(handle_missing='remove')
        X_clean, y_clean, metadata = prep.prepare(self.X, self.y)

        # Rows with missing values removed
        self.assertLess(len(X_clean), len(self.X))
        self.assertEqual(len(X_clean), len(y_clean))

    def test_remove_constant_features(self):
        """Test constant feature removal."""
        prep = DatasetPreparator(
            remove_constant_features=True,
            constant_threshold=1e-6,
            handle_missing='mean'
        )
        X_clean, y_clean, metadata = prep.prepare(self.X, self.y)

        # Constant feature removed
        self.assertNotIn('const_feat', X_clean.columns)
        self.assertIn('const_feat', metadata['constant_features'])

    def test_handle_outliers_clip(self):
        """Test outlier clipping."""
        prep = DatasetPreparator(
            handle_outliers='clip',
            outlier_percentiles=(1, 99),
            handle_missing='mean'
        )
        X_clean, y_clean, metadata = prep.prepare(self.X, self.y)

        # Outlier should be clipped
        self.assertLess(X_clean.iloc[:, 2].max(), 100.0)

    def test_handle_outliers_remove(self):
        """Test outlier removal."""
        prep = DatasetPreparator(
            handle_outliers='remove',
            outlier_percentiles=(1, 99),
            handle_missing='mean'
        )
        X_clean, y_clean, metadata = prep.prepare(self.X, self.y)

        # Rows with outliers removed
        self.assertLess(len(X_clean), len(self.X))

    def test_prepare_metadata(self):
        """Test metadata returned by prepare."""
        prep = DatasetPreparator(handle_missing='mean')
        X_clean, y_clean, metadata = prep.prepare(self.X, self.y)

        # Check metadata structure
        self.assertIn('original_shape', metadata)
        self.assertIn('final_shape', metadata)
        self.assertIn('removed_features', metadata)
        self.assertIn('missing_info', metadata)


class TestTrainTestSplitter(unittest.TestCase):
    """Test train/test splitting strategies (Prompt 2)."""

    def setUp(self):
        """Create test dataset with graph metadata."""
        np.random.seed(42)
        self.n_graphs = 10
        self.n_vertices_per_graph = 20
        self.n_samples = self.n_graphs * self.n_vertices_per_graph
        self.n_features = 5

        # Create features and labels
        self.X = pd.DataFrame(
            np.random.randn(self.n_samples, self.n_features),
            columns=[f'feat_{i}' for i in range(self.n_features)]
        )
        self.y = pd.Series(np.random.randn(self.n_samples))

        # Create graph metadata
        self.graph_ids = pd.Series([i for i in range(self.n_graphs) for _ in range(self.n_vertices_per_graph)])
        self.graph_types = pd.Series([
            'type_A' if i < 5 else 'type_B'
            for i in range(self.n_graphs)
            for _ in range(self.n_vertices_per_graph)
        ])
        self.graph_sizes = pd.Series([
            20 if i < 7 else 50
            for i in range(self.n_graphs)
            for _ in range(self.n_vertices_per_graph)
        ])

    def test_random_split(self):
        """Test random splitting."""
        splitter = TrainTestSplitter(
            strategy=SplitStrategy.RANDOM,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            random_seed=42
        )
        split = splitter.split(self.X, self.y)

        # Check sizes
        total = len(split.X_train) + len(split.X_val) + len(split.X_test)
        self.assertEqual(total, len(self.X))

        # Check approximate ratios (within 5%)
        self.assertAlmostEqual(len(split.X_train) / len(self.X), 0.7, delta=0.05)
        self.assertAlmostEqual(len(split.X_val) / len(self.X), 0.15, delta=0.05)

    def test_graph_based_split(self):
        """Test graph-based splitting."""
        splitter = TrainTestSplitter(
            strategy=SplitStrategy.GRAPH_BASED,
            train_ratio=0.6,
            val_ratio=0.2,
            test_ratio=0.2,
            random_seed=42
        )
        split = splitter.split(self.X, self.y, graph_ids=self.graph_ids)

        # No graph should appear in multiple splits
        train_set = set(split.train_graphs)
        val_set = set(split.val_graphs)
        test_set = set(split.test_graphs)

        self.assertEqual(len(train_set & val_set), 0)
        self.assertEqual(len(train_set & test_set), 0)
        self.assertEqual(len(val_set & test_set), 0)

        # All graphs accounted for
        self.assertEqual(len(train_set) + len(val_set) + len(test_set), self.n_graphs)

    def test_stratified_graph_split(self):
        """Test stratified graph splitting."""
        splitter = TrainTestSplitter(
            strategy=SplitStrategy.STRATIFIED_GRAPH,
            train_ratio=0.6,
            val_ratio=0.2,
            test_ratio=0.2,
            random_seed=42
        )
        split = splitter.split(
            self.X, self.y,
            graph_ids=self.graph_ids,
            graph_types=self.graph_types
        )

        # Check that both types are represented in each split
        train_types = set(self.graph_types[self.X.index.isin(split.X_train.index)])
        val_types = set(self.graph_types[self.X.index.isin(split.X_val.index)])
        test_types = set(self.graph_types[self.X.index.isin(split.X_test.index)])

        # At least one graph of each type in train (primary set)
        self.assertGreaterEqual(len(train_types), 1)

    def test_graph_type_holdout(self):
        """Test graph type holdout strategy."""
        splitter = TrainTestSplitter(
            strategy=SplitStrategy.GRAPH_TYPE_HOLDOUT,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            random_seed=42
        )
        split = splitter.split(
            self.X, self.y,
            graph_ids=self.graph_ids,
            graph_types=self.graph_types,
            holdout_graph_type='type_B'
        )

        # Test set should only contain holdout type
        test_types = set(self.graph_types[self.X.index.isin(split.X_test.index)])
        self.assertEqual(test_types, {'type_B'})

        # Train/val should not contain holdout type
        train_types = set(self.graph_types[self.X.index.isin(split.X_train.index)])
        val_types = set(self.graph_types[self.X.index.isin(split.X_val.index)])
        self.assertNotIn('type_B', train_types)
        self.assertNotIn('type_B', val_types)

    def test_size_holdout(self):
        """Test size-based holdout strategy."""
        splitter = TrainTestSplitter(
            strategy=SplitStrategy.SIZE_HOLDOUT,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            random_seed=42
        )
        split = splitter.split(
            self.X, self.y,
            graph_ids=self.graph_ids,
            graph_sizes=self.graph_sizes,
            size_threshold=30
        )

        # Test set should only contain large graphs
        test_sizes = self.graph_sizes[self.X.index.isin(split.X_test.index)]
        self.assertTrue(all(test_sizes >= 30))

        # Train/val should only contain small graphs
        train_sizes = self.graph_sizes[self.X.index.isin(split.X_train.index)]
        val_sizes = self.graph_sizes[self.X.index.isin(split.X_val.index)]
        self.assertTrue(all(train_sizes < 30))
        self.assertTrue(all(val_sizes < 30))

    def test_split_summary(self):
        """Test split summary generation."""
        splitter = TrainTestSplitter(strategy=SplitStrategy.RANDOM, random_seed=42)
        split = splitter.split(self.X, self.y)

        summary = split.get_summary()
        self.assertIn("Train:", summary)
        self.assertIn("Val:", summary)
        self.assertIn("Test:", summary)


class TestLinearRegressionModel(unittest.TestCase):
    """Test linear regression models (Prompt 3)."""

    def setUp(self):
        """Create toy dataset."""
        np.random.seed(42)
        self.n = 100
        self.n_features = 5

        # Create simple linear relationship
        self.X = pd.DataFrame(
            np.random.randn(self.n, self.n_features),
            columns=[f'feat_{i}' for i in range(self.n_features)]
        )

        # y = 2*feat_0 + 1*feat_1 + noise
        self.y = pd.Series(
            2 * self.X['feat_0'] + 1 * self.X['feat_1'] + 0.1 * np.random.randn(self.n)
        )

        # Split data
        split_idx = int(0.8 * self.n)
        self.X_train = self.X.iloc[:split_idx]
        self.y_train = self.y.iloc[:split_idx]
        self.X_test = self.X.iloc[split_idx:]
        self.y_test = self.y.iloc[split_idx:]

    def test_ols_fit_predict(self):
        """Test OLS fitting and prediction."""
        model = LinearRegressionModel(
            model_type=ModelType.LINEAR_OLS,
            standardize_features=False
        )
        model.fit(self.X_train, self.y_train)

        predictions = model.predict(self.X_test)
        self.assertEqual(len(predictions), len(self.X_test))
        self.assertTrue(model.is_fitted)

    def test_ridge_fit_predict(self):
        """Test Ridge regression."""
        model = LinearRegressionModel(
            model_type=ModelType.LINEAR_RIDGE,
            alpha=1.0,
            standardize_features=True
        )
        model.fit(self.X_train, self.y_train)

        predictions = model.predict(self.X_test)
        self.assertEqual(len(predictions), len(self.X_test))

    def test_lasso_fit_predict(self):
        """Test Lasso regression."""
        model = LinearRegressionModel(
            model_type=ModelType.LINEAR_LASSO,
            alpha=0.1,
            standardize_features=True
        )
        model.fit(self.X_train, self.y_train)

        predictions = model.predict(self.X_test)
        self.assertEqual(len(predictions), len(self.X_test))

    def test_elasticnet_fit_predict(self):
        """Test ElasticNet regression."""
        model = LinearRegressionModel(
            model_type=ModelType.LINEAR_ELASTICNET,
            alpha=0.1,
            l1_ratio=0.5,
            standardize_features=True
        )
        model.fit(self.X_train, self.y_train)

        predictions = model.predict(self.X_test)
        self.assertEqual(len(predictions), len(self.X_test))

    def test_get_coefficients(self):
        """Test coefficient extraction."""
        model = LinearRegressionModel(model_type=ModelType.LINEAR_OLS)
        model.fit(self.X_train, self.y_train)

        coefs = model.get_coefficients()

        # Should have coefficient for each feature + intercept
        self.assertEqual(len(coefs), self.n_features + 1)
        self.assertIn('feat_0', coefs)
        self.assertIn('intercept', coefs)

        # feat_0 should have largest coefficient (since y = 2*feat_0 + ...)
        self.assertGreater(abs(coefs['feat_0']), abs(coefs['feat_2']))

    def test_get_feature_importance(self):
        """Test feature importance extraction."""
        model = LinearRegressionModel(model_type=ModelType.LINEAR_OLS)
        model.fit(self.X_train, self.y_train)

        importance = model.get_feature_importance()

        # Should have importance for each feature
        self.assertEqual(len(importance), self.n_features)

        # feat_0 should be most important
        self.assertEqual(
            max(importance, key=importance.get),
            'feat_0'
        )

    def test_evaluate(self):
        """Test model evaluation."""
        model = LinearRegressionModel(model_type=ModelType.LINEAR_OLS)
        model.fit(self.X_train, self.y_train)

        result = model.evaluate(self.X_test, self.y_test)

        # Check result structure
        self.assertIsInstance(result, ModelResult)
        self.assertEqual(result.model_type, ModelType.LINEAR_OLS)
        self.assertIn('r2', result.metrics)
        self.assertIn('mae', result.metrics)
        self.assertIn('rmse', result.metrics)

        # R² should be high for this simple linear relationship
        self.assertGreater(result.metrics['r2'], 0.8)

    def test_get_diagnostics(self):
        """Test diagnostic information."""
        model = LinearRegressionModel(model_type=ModelType.LINEAR_OLS)
        model.fit(self.X_train, self.y_train)

        diagnostics = model.get_diagnostics(self.X_test, self.y_test)

        # Check diagnostic fields
        self.assertIn('residuals', diagnostics)
        self.assertIn('residuals_mean', diagnostics)
        self.assertIn('residuals_std', diagnostics)

        # Residuals should be approximately centered at zero
        self.assertAlmostEqual(diagnostics['residuals_mean'], 0.0, delta=0.5)

    def test_predict_before_fit_raises(self):
        """Test that predicting before fitting raises error."""
        model = LinearRegressionModel(model_type=ModelType.LINEAR_OLS)

        with self.assertRaises(RuntimeError):
            model.predict(self.X_test)


class TestTreeBasedModel(unittest.TestCase):
    """Test tree-based models (Prompt 4)."""

    def setUp(self):
        """Create toy dataset."""
        np.random.seed(42)
        self.n = 100
        self.n_features = 5

        # Create non-linear relationship
        self.X = pd.DataFrame(
            np.random.randn(self.n, self.n_features),
            columns=[f'feat_{i}' for i in range(self.n_features)]
        )

        # y = feat_0^2 + feat_1 + noise
        self.y = pd.Series(
            self.X['feat_0'] ** 2 + self.X['feat_1'] + 0.1 * np.random.randn(self.n)
        )

        # Split data
        split_idx = int(0.8 * self.n)
        self.X_train = self.X.iloc[:split_idx]
        self.y_train = self.y.iloc[:split_idx]
        self.X_test = self.X.iloc[split_idx:]
        self.y_test = self.y.iloc[split_idx:]

    def test_decision_tree_fit_predict(self):
        """Test decision tree fitting."""
        model = TreeBasedModel(
            model_type=ModelType.DECISION_TREE,
            max_depth=5,
            random_seed=42
        )
        model.fit(self.X_train, self.y_train)

        predictions = model.predict(self.X_test)
        self.assertEqual(len(predictions), len(self.X_test))
        self.assertTrue(model.is_fitted)

    def test_random_forest_fit_predict(self):
        """Test random forest fitting."""
        model = TreeBasedModel(
            model_type=ModelType.RANDOM_FOREST,
            n_estimators=50,
            max_depth=5,
            random_seed=42
        )
        model.fit(self.X_train, self.y_train)

        predictions = model.predict(self.X_test)
        self.assertEqual(len(predictions), len(self.X_test))

    def test_gradient_boosting_fit_predict(self):
        """Test gradient boosting fitting."""
        model = TreeBasedModel(
            model_type=ModelType.GRADIENT_BOOSTING,
            n_estimators=50,
            max_depth=3,
            learning_rate=0.1,
            random_seed=42
        )
        model.fit(self.X_train, self.y_train)

        predictions = model.predict(self.X_test)
        self.assertEqual(len(predictions), len(self.X_test))

    def test_get_feature_importance(self):
        """Test feature importance extraction."""
        model = TreeBasedModel(
            model_type=ModelType.RANDOM_FOREST,
            n_estimators=50,
            random_seed=42
        )
        model.fit(self.X_train, self.y_train)

        importance = model.get_feature_importance()

        # Should have importance for each feature
        self.assertEqual(len(importance), self.n_features)

        # All importances should be non-negative
        self.assertTrue(all(v >= 0 for v in importance.values()))

        # Importances should sum to approximately 1.0
        self.assertAlmostEqual(sum(importance.values()), 1.0, delta=0.01)

    def test_evaluate(self):
        """Test model evaluation."""
        model = TreeBasedModel(
            model_type=ModelType.RANDOM_FOREST,
            n_estimators=50,
            random_seed=42
        )
        model.fit(self.X_train, self.y_train)

        result = model.evaluate(self.X_test, self.y_test)

        # Check result structure
        self.assertIsInstance(result, ModelResult)
        self.assertEqual(result.model_type, ModelType.RANDOM_FOREST)
        self.assertIn('r2', result.metrics)
        self.assertIn('mae', result.metrics)
        self.assertIsNone(result.coefficients)  # Tree models don't have coefficients

    def test_tree_handles_nonlinearity(self):
        """Test that tree models handle non-linearities better than linear."""
        # Fit both models
        linear_model = LinearRegressionModel(model_type=ModelType.LINEAR_OLS)
        linear_model.fit(self.X_train, self.y_train)
        linear_result = linear_model.evaluate(self.X_test, self.y_test)

        tree_model = TreeBasedModel(
            model_type=ModelType.RANDOM_FOREST,
            n_estimators=100,
            random_seed=42
        )
        tree_model.fit(self.X_train, self.y_train)
        tree_result = tree_model.evaluate(self.X_test, self.y_test)

        # Tree should have better R² on non-linear data
        self.assertGreater(tree_result.metrics['r2'], linear_result.metrics['r2'])


class TestModelIntegration(unittest.TestCase):
    """Integration tests for complete ML pipeline."""

    def setUp(self):
        """Create realistic dataset."""
        np.random.seed(42)
        self.n_graphs = 5
        self.n_vertices = 20
        self.n_samples = self.n_graphs * self.n_vertices
        self.n_features = 10

        # Create features
        self.X = pd.DataFrame(
            np.random.randn(self.n_samples, self.n_features),
            columns=[f'feat_{i}' for i in range(self.n_features)]
        )

        # Create labels with realistic relationship
        self.y = pd.Series(
            2 * self.X['feat_0'] +
            1 * self.X['feat_1'] -
            0.5 * self.X['feat_2'] +
            np.random.randn(self.n_samples) * 0.5
        )

        # Graph metadata
        self.graph_ids = pd.Series([i for i in range(self.n_graphs) for _ in range(self.n_vertices)])
        self.graph_types = pd.Series(['euclidean' if i < 3 else 'metric' for i in range(self.n_graphs) for _ in range(self.n_vertices)])

    def test_complete_pipeline(self):
        """Test complete ML pipeline from data prep to evaluation."""
        # 1. Prepare dataset
        preparator = DatasetPreparator(
            problem_type=MLProblemType.REGRESSION,
            remove_constant_features=True,
            handle_outliers='clip',
            handle_missing='mean'
        )
        X_prep, y_prep, _ = preparator.prepare(self.X, self.y)

        # 2. Split data
        splitter = TrainTestSplitter(
            strategy=SplitStrategy.GRAPH_BASED,
            random_seed=42
        )
        split = splitter.split(X_prep, y_prep, graph_ids=self.graph_ids)

        # 3. Train linear model
        linear_model = LinearRegressionModel(
            model_type=ModelType.LINEAR_RIDGE,
            alpha=1.0
        )
        linear_model.fit(split.X_train, split.y_train)
        linear_result = linear_model.evaluate(split.X_test, split.y_test)

        # 4. Train tree model
        tree_model = TreeBasedModel(
            model_type=ModelType.RANDOM_FOREST,
            n_estimators=50,
            random_seed=42
        )
        tree_model.fit(split.X_train, split.y_train)
        tree_result = tree_model.evaluate(split.X_test, split.y_test)

        # 5. Compare results
        self.assertGreater(linear_result.metrics['r2'], 0.0)
        self.assertGreater(tree_result.metrics['r2'], 0.0)

        # Both models should extract feature importance
        self.assertIsNotNone(linear_result.feature_importance)
        self.assertIsNotNone(tree_result.feature_importance)


if __name__ == '__main__':
    # Suppress sklearn warnings
    warnings.filterwarnings('ignore')
    unittest.main(verbosity=2)
