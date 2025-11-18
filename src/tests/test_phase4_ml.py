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

from ml.evaluation import (
    ModelEvaluator,
    ModelComparator,
    PerformanceMatrix,
    PerformanceMetrics
)

from ml.cross_validation import (
    CrossValidator,
    NestedCrossValidator,
    CVStrategy,
    CVResult
)

from ml.tuning import (
    HyperparameterTuner,
    ModelSpecificTuner,
    TuningStrategy
)

from ml.feature_engineering import (
    FeatureScaler,
    NonLinearTransformer,
    FeatureInteractionGenerator,
    PCAReducer,
    AdvancedFeatureSelector,
    ScalingStrategy,
    TransformationType
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


class TestModelEvaluator(unittest.TestCase):
    """Test model evaluation framework (Prompt 5)."""

    def setUp(self):
        """Create test data."""
        np.random.seed(42)
        self.n = 100

        self.y_true = np.random.randn(self.n)
        self.y_pred = self.y_true + 0.1 * np.random.randn(self.n)
        self.groups = np.array(['type_A'] * 50 + ['type_B'] * 50)

    def test_compute_metrics(self):
        """Test metric computation."""
        metrics = ModelEvaluator.compute_metrics(self.y_true, self.y_pred)

        self.assertIsInstance(metrics, PerformanceMetrics)
        self.assertGreater(metrics.r2, 0.8)  # Should be high correlation
        self.assertLess(metrics.mae, 0.5)
        self.assertAlmostEqual(metrics.mean_residual, 0.0, delta=0.1)

    def test_per_group_metrics(self):
        """Test per-group metric computation."""
        group_metrics = ModelEvaluator.compute_per_group_metrics(
            self.y_true, self.y_pred, self.groups
        )

        self.assertIn('type_A', group_metrics)
        self.assertIn('type_B', group_metrics)
        self.assertIsInstance(group_metrics['type_A'], PerformanceMetrics)

    def test_metrics_to_dict(self):
        """Test metrics conversion to dictionary."""
        metrics = ModelEvaluator.compute_metrics(self.y_true, self.y_pred)
        metrics_dict = metrics.to_dict()

        self.assertIn('r2', metrics_dict)
        self.assertIn('mae', metrics_dict)
        self.assertIn('rmse', metrics_dict)


class TestModelComparator(unittest.TestCase):
    """Test model comparison framework (Prompt 5)."""

    def setUp(self):
        """Create test data."""
        np.random.seed(42)
        self.n = 100

        self.y_true = np.random.randn(self.n)
        self.y_pred_a = self.y_true + 0.1 * np.random.randn(self.n)
        self.y_pred_b = self.y_true + 0.2 * np.random.randn(self.n)

    def test_compare_predictions(self):
        """Test pairwise model comparison."""
        comparator = ModelComparator(significance_level=0.05)

        result = comparator.compare_predictions(
            self.y_true,
            self.y_pred_a,
            self.y_pred_b,
            model_a_name="Model A",
            model_b_name="Model B"
        )

        self.assertEqual(result.model_a_name, "Model A")
        self.assertEqual(result.model_b_name, "Model B")
        self.assertIsNotNone(result.p_value)
        self.assertIsNotNone(result.effect_size)
        self.assertIsInstance(result.significant, bool)

    def test_compare_multiple_models(self):
        """Test multiple model comparison."""
        comparator = ModelComparator()

        predictions = {
            'model_1': self.y_pred_a,
            'model_2': self.y_pred_b,
            'model_3': self.y_true + 0.15 * np.random.randn(self.n)
        }

        results = comparator.compare_multiple_models(
            self.y_true,
            predictions
        )

        # Should have 3 pairwise comparisons
        self.assertEqual(len(results), 3)

    def test_wilcoxon_test(self):
        """Test non-parametric comparison."""
        comparator = ModelComparator(use_parametric=False)

        result = comparator.compare_predictions(
            self.y_true,
            self.y_pred_a,
            self.y_pred_b
        )

        self.assertEqual(result.test_type, "wilcoxon")


class TestPerformanceMatrix(unittest.TestCase):
    """Test performance matrix creation (Prompt 5)."""

    def setUp(self):
        """Create test data."""
        np.random.seed(42)
        self.n = 100

        self.y_true = np.random.randn(self.n)
        self.groups = np.array(['euclidean'] * 50 + ['metric'] * 50)

        self.predictions = {
            'linear': self.y_true + 0.1 * np.random.randn(self.n),
            'tree': self.y_true + 0.15 * np.random.randn(self.n)
        }

    def test_create_matrix(self):
        """Test matrix creation."""
        matrix = PerformanceMatrix.create_matrix(
            self.predictions,
            self.y_true,
            self.groups,
            metric_name='r2'
        )

        self.assertIsInstance(matrix, pd.DataFrame)
        self.assertEqual(len(matrix), 2)  # 2 models
        self.assertEqual(len(matrix.columns), 2)  # 2 groups


class TestCrossValidator(unittest.TestCase):
    """Test cross-validation framework (Prompt 6)."""

    def setUp(self):
        """Create test data."""
        np.random.seed(42)
        self.n = 100
        self.n_features = 5

        self.X = np.random.randn(self.n, self.n_features)
        self.y = np.random.randn(self.n)
        self.groups = np.array([i // 20 for i in range(self.n)])  # 5 groups

    def test_k_fold_split(self):
        """Test k-fold splitting."""
        cv = CrossValidator(strategy=CVStrategy.K_FOLD, n_folds=5, random_seed=42)
        folds = cv.split(self.X, self.y)

        self.assertEqual(len(folds), 5)
        self.assertIsNotNone(folds[0].train_indices)
        self.assertIsNotNone(folds[0].val_indices)

    def test_group_k_fold_split(self):
        """Test group k-fold splitting."""
        cv = CrossValidator(strategy=CVStrategy.GROUP_K_FOLD, n_folds=3, random_seed=42)
        folds = cv.split(self.X, self.y, groups=self.groups)

        self.assertEqual(len(folds), 3)

        # Check no group overlap
        for fold in folds:
            train_groups = set(fold.train_groups)
            val_groups = set(fold.val_groups)
            self.assertEqual(len(train_groups & val_groups), 0)

    def test_leave_one_group_out(self):
        """Test leave-one-group-out."""
        cv = CrossValidator(strategy=CVStrategy.LEAVE_ONE_GROUP_OUT)
        folds = cv.split(self.X, self.y, groups=self.groups)

        # Should have one fold per group
        n_groups = len(np.unique(self.groups))
        self.assertEqual(len(folds), n_groups)

    def test_cross_validate(self):
        """Test cross-validation with a model."""
        cv = CrossValidator(strategy=CVStrategy.K_FOLD, n_folds=3, random_seed=42)

        result = cv.cross_validate(
            model_class=LinearRegressionModel,
            model_params={'model_type': ModelType.LINEAR_OLS},
            X=self.X,
            y=self.y
        )

        self.assertIsInstance(result, CVResult)
        self.assertEqual(result.n_folds, 3)
        self.assertIn('r2', result.mean_metrics)
        self.assertIn('mae', result.mean_metrics)


class TestNestedCrossValidator(unittest.TestCase):
    """Test nested cross-validation (Prompt 6)."""

    def setUp(self):
        """Create test data."""
        np.random.seed(42)
        self.n = 80  # Smaller for nested CV
        self.n_features = 5

        self.X = np.random.randn(self.n, self.n_features)
        self.y = 2 * self.X[:, 0] + 1 * self.X[:, 1] + 0.1 * np.random.randn(self.n)
        self.groups = np.array([i // 20 for i in range(self.n)])

    def test_nested_cv(self):
        """Test nested cross-validation."""
        outer_cv = CrossValidator(strategy=CVStrategy.K_FOLD, n_folds=3, random_seed=42)
        inner_cv = CrossValidator(strategy=CVStrategy.K_FOLD, n_folds=2, random_seed=42)

        nested = NestedCrossValidator(outer_cv=outer_cv, inner_cv=inner_cv)

        param_grid = [
            {'model_type': ModelType.LINEAR_RIDGE, 'alpha': 0.1},
            {'model_type': ModelType.LINEAR_RIDGE, 'alpha': 1.0}
        ]

        result = nested.nested_cross_validate(
            model_class=LinearRegressionModel,
            param_grid=param_grid,
            X=self.X,
            y=self.y
        )

        self.assertIn('mean_outer_metrics', result)
        self.assertIn('best_params_per_fold', result)
        self.assertEqual(len(result['best_params_per_fold']), 3)


class TestHyperparameterTuner(unittest.TestCase):
    """Test hyperparameter tuning (Prompt 7)."""

    def setUp(self):
        """Create test data."""
        np.random.seed(42)
        self.n_train = 80
        self.n_val = 20
        self.n_features = 5

        X = np.random.randn(self.n_train + self.n_val, self.n_features)
        y = 2 * X[:, 0] + 1 * X[:, 1] + 0.1 * np.random.randn(self.n_train + self.n_val)

        self.X_train = X[:self.n_train]
        self.y_train = y[:self.n_train]
        self.X_val = X[self.n_train:]
        self.y_val = y[self.n_train:]

    def test_grid_search(self):
        """Test grid search."""
        tuner = HyperparameterTuner(
            strategy=TuningStrategy.GRID_SEARCH,
            scoring_metric='r2'
        )

        param_grid = {
            'model_type': [ModelType.LINEAR_RIDGE],
            'alpha': [0.1, 1.0, 10.0]
        }

        result = tuner.tune(
            model_class=LinearRegressionModel,
            param_grid=param_grid,
            X_train=self.X_train,
            y_train=self.y_train,
            X_val=self.X_val,
            y_val=self.y_val
        )

        self.assertIsNotNone(result.best_params)
        self.assertGreater(result.best_score, 0.0)
        self.assertEqual(result.n_trials, 3)

    def test_random_search(self):
        """Test random search."""
        tuner = HyperparameterTuner(
            strategy=TuningStrategy.RANDOM_SEARCH,
            n_trials=5,
            random_seed=42
        )

        param_grid = {
            'model_type': [ModelType.LINEAR_RIDGE],
            'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]
        }

        result = tuner.tune(
            model_class=LinearRegressionModel,
            param_grid=param_grid,
            X_train=self.X_train,
            y_train=self.y_train,
            X_val=self.X_val,
            y_val=self.y_val
        )

        self.assertEqual(result.n_trials, 5)
        self.assertLessEqual(len(result.all_params), 5)


class TestModelSpecificTuner(unittest.TestCase):
    """Test model-specific parameter grids (Prompt 7)."""

    def test_linear_param_grid(self):
        """Test linear model parameter grids."""
        grid = ModelSpecificTuner.get_linear_param_grid('ridge')
        self.assertIn('alpha', grid)
        self.assertGreater(len(grid['alpha']), 1)

    def test_tree_param_grid(self):
        """Test tree model parameter grids."""
        grid = ModelSpecificTuner.get_tree_param_grid('random_forest')
        self.assertIn('n_estimators', grid)
        self.assertIn('max_depth', grid)

    def test_coarse_grid(self):
        """Test coarse grid generation."""
        grid = ModelSpecificTuner.get_coarse_grid('ridge')
        self.assertIn('alpha', grid)
        # Coarse grid should be smaller
        self.assertLessEqual(len(grid['alpha']), 5)


class TestFeatureScaler(unittest.TestCase):
    """Test feature scaling (Prompt 8)."""

    def setUp(self):
        """Create test data."""
        np.random.seed(42)
        self.X = pd.DataFrame({
            'feat_0': np.random.randn(100) * 10 + 50,
            'feat_1': np.random.randn(100) * 2 + 10,
            'feat_2': np.random.randn(100) * 0.1 + 1
        })

    def test_standardization(self):
        """Test standardization."""
        scaler = FeatureScaler(strategy=ScalingStrategy.STANDARDIZATION)
        X_scaled = scaler.fit_transform(self.X)

        # Check mean ~ 0, std ~ 1
        for col in X_scaled.columns:
            self.assertAlmostEqual(X_scaled[col].mean(), 0.0, delta=1e-10)
            self.assertAlmostEqual(X_scaled[col].std(), 1.0, delta=1e-10)

    def test_min_max_scaling(self):
        """Test min-max scaling."""
        scaler = FeatureScaler(strategy=ScalingStrategy.MIN_MAX)
        X_scaled = scaler.fit_transform(self.X)

        # Check range [0, 1]
        for col in X_scaled.columns:
            self.assertAlmostEqual(X_scaled[col].min(), 0.0, delta=1e-10)
            self.assertAlmostEqual(X_scaled[col].max(), 1.0, delta=1e-10)

    def test_robust_scaling(self):
        """Test robust scaling."""
        scaler = FeatureScaler(strategy=ScalingStrategy.ROBUST)
        X_scaled = scaler.fit_transform(self.X)

        # Just check it runs without error
        self.assertEqual(X_scaled.shape, self.X.shape)

    def test_fit_transform_consistency(self):
        """Test fit and transform separately."""
        scaler = FeatureScaler(strategy=ScalingStrategy.STANDARDIZATION)
        scaler.fit(self.X)
        X_scaled = scaler.transform(self.X)

        # Should match fit_transform
        X_scaled2 = FeatureScaler(strategy=ScalingStrategy.STANDARDIZATION).fit_transform(self.X)

        pd.testing.assert_frame_equal(X_scaled, X_scaled2)


class TestNonLinearTransformer(unittest.TestCase):
    """Test non-linear transformations (Prompt 8)."""

    def setUp(self):
        """Create test data."""
        np.random.seed(42)
        self.X = pd.DataFrame({
            'feat_0': np.abs(np.random.randn(100)) + 1,  # Positive
            'feat_1': np.abs(np.random.randn(100)) + 0.1,
            'feat_2': np.random.randn(100)
        })

    def test_log_transform(self):
        """Test log transformation."""
        transformer = NonLinearTransformer(
            transformation=TransformationType.LOG,
            features=['feat_0', 'feat_1']
        )
        X_transformed = transformer.fit_transform(self.X)

        # Log should reduce skewness
        self.assertLess(X_transformed['feat_0'].max(), self.X['feat_0'].max())

    def test_sqrt_transform(self):
        """Test square root transformation."""
        transformer = NonLinearTransformer(
            transformation=TransformationType.SQRT,
            features=['feat_0']
        )
        X_transformed = transformer.fit_transform(self.X)

        # Check sqrt applied
        self.assertTrue(np.all(X_transformed['feat_0'] >= 0))

    def test_square_transform(self):
        """Test square transformation."""
        transformer = NonLinearTransformer(
            transformation=TransformationType.SQUARE,
            features=['feat_2']
        )
        X_transformed = transformer.fit_transform(self.X)

        # Check squared
        self.assertTrue(np.all(X_transformed['feat_2'] >= 0))


class TestFeatureInteractionGenerator(unittest.TestCase):
    """Test feature interaction generation (Prompt 8)."""

    def setUp(self):
        """Create test data."""
        np.random.seed(42)
        self.X = pd.DataFrame({
            'feat_0': np.random.randn(100),
            'feat_1': np.random.randn(100),
            'feat_2': np.random.randn(100)
        })

    def test_pairwise_interactions(self):
        """Test pairwise interaction generation."""
        generator = FeatureInteractionGenerator()
        X_interactions = generator.fit_transform(self.X)

        # Should have original + interaction features
        self.assertGreater(len(X_interactions.columns), len(self.X.columns))

        # Check interaction columns exist
        self.assertIn('feat_0_x_feat_1', X_interactions.columns)

    def test_specific_pairs(self):
        """Test specific interaction pairs."""
        generator = FeatureInteractionGenerator(
            interaction_pairs=[('feat_0', 'feat_1')]
        )
        X_interactions = generator.fit_transform(self.X)

        # Should have original + 1 interaction
        self.assertEqual(len(X_interactions.columns), len(self.X.columns) + 1)

    def test_max_interactions_limit(self):
        """Test interaction limit."""
        generator = FeatureInteractionGenerator(max_interactions=1)
        X_interactions = generator.fit_transform(self.X)

        # Should have original + 1 interaction
        self.assertEqual(len(X_interactions.columns), len(self.X.columns) + 1)


class TestPCAReducer(unittest.TestCase):
    """Test PCA dimensionality reduction (Prompt 8)."""

    def setUp(self):
        """Create test data."""
        np.random.seed(42)
        self.X = pd.DataFrame({
            'feat_0': np.random.randn(100),
            'feat_1': np.random.randn(100),
            'feat_2': np.random.randn(100),
            'feat_3': np.random.randn(100),
            'feat_4': np.random.randn(100)
        })

    def test_pca_with_n_components(self):
        """Test PCA with fixed number of components."""
        reducer = PCAReducer(n_components=2)
        X_pca = reducer.fit_transform(self.X)

        self.assertEqual(X_pca.shape[1], 2)
        self.assertIn('PC1', X_pca.columns)
        self.assertIn('PC2', X_pca.columns)

    def test_pca_with_variance_threshold(self):
        """Test PCA with variance threshold."""
        reducer = PCAReducer(variance_threshold=0.8)
        X_pca = reducer.fit_transform(self.X)

        # Should capture 80% of variance
        variance = reducer.get_explained_variance()
        self.assertGreaterEqual(np.sum(variance), 0.8)

    def test_component_loadings(self):
        """Test component loadings extraction."""
        reducer = PCAReducer(n_components=2)
        reducer.fit(self.X)

        loadings = reducer.get_component_loadings()
        self.assertEqual(loadings.shape[0], 5)  # 5 original features
        self.assertEqual(loadings.shape[1], 2)  # 2 components


class TestAdvancedFeatureSelector(unittest.TestCase):
    """Test advanced feature selection (Prompt 8)."""

    def setUp(self):
        """Create test data with correlated features."""
        np.random.seed(42)
        n = 100

        feat_0 = np.random.randn(n)
        feat_1 = feat_0 + 0.01 * np.random.randn(n)  # Highly correlated

        self.X = pd.DataFrame({
            'feat_0': feat_0,
            'feat_1': feat_1,
            'feat_2': np.random.randn(n),
            'feat_3': np.random.randn(n)
        })

    def test_remove_correlated_features(self):
        """Test correlated feature removal."""
        X_reduced, removed = AdvancedFeatureSelector.remove_correlated_features(
            self.X,
            threshold=0.95
        )

        # feat_0 and feat_1 are highly correlated
        self.assertLess(len(X_reduced.columns), len(self.X.columns))
        self.assertGreater(len(removed), 0)

    def test_select_by_importance(self):
        """Test selection by importance."""
        importance_scores = {
            'feat_0': 0.5,
            'feat_1': 0.3,
            'feat_2': 0.8,
            'feat_3': 0.1
        }

        X_selected = AdvancedFeatureSelector.select_by_importance(
            self.X,
            importance_scores,
            top_k=2
        )

        self.assertEqual(len(X_selected.columns), 2)
        self.assertIn('feat_2', X_selected.columns)  # Highest importance
        self.assertIn('feat_0', X_selected.columns)  # Second highest


if __name__ == '__main__':
    # Suppress sklearn warnings
    warnings.filterwarnings('ignore')
    unittest.main(verbosity=2)
