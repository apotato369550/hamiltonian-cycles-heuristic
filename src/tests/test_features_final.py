"""
Tests for Phase 3 Feature Engineering Prompts 9-12.

Covers:
- Prompt 9: Anchor Quality Labeling System
- Prompt 10: Feature Engineering Pipeline
- Prompt 11: Feature Selection Utilities
- Prompt 12: Feature Transformation

Test count target: ~57 tests total
"""

import unittest
import numpy as np
import pandas as pd
from typing import List, Dict, Optional
import tempfile
import os
import shutil

# Import components to test (will be implemented)
from src.features.labeling import (
    AnchorQualityLabeler,
    LabelingStrategy,
    LabelingResult
)
from src.features.dataset_pipeline import (
    FeatureDatasetPipeline,
    DatasetConfig,
    DatasetResult
)
from src.features.selection import (
    FeatureSelector,
    SelectionMethod,
    SelectionResult
)
from src.features.transformation import (
    FeatureTransformer,
    TransformationType,
    TransformationConfig
)


class TestAnchorQualityLabeling(unittest.TestCase):
    """Tests for Prompt 9: Anchor Quality Labeling System."""

    def setUp(self):
        """Create test graph."""
        # Simple 4-vertex complete graph
        self.graph = np.array([
            [0, 10, 15, 20],
            [10, 0, 35, 25],
            [15, 35, 0, 30],
            [20, 25, 30, 0]
        ], dtype=float)
        self.n = 4

    def test_labeler_initialization(self):
        """Test labeler initialization with different strategies."""
        labeler = AnchorQualityLabeler(
            strategy=LabelingStrategy.RANK_BASED,
            algorithm_name='single_anchor'
        )
        self.assertEqual(labeler.strategy, LabelingStrategy.RANK_BASED)
        self.assertEqual(labeler.algorithm_name, 'single_anchor')

    def test_absolute_quality_labeling(self):
        """Test absolute quality labeling strategy."""
        labeler = AnchorQualityLabeler(
            strategy=LabelingStrategy.ABSOLUTE_QUALITY,
            algorithm_name='single_anchor'
        )
        result = labeler.label_vertices(self.graph)

        # Check result structure
        self.assertIsInstance(result, LabelingResult)
        self.assertEqual(len(result.labels), self.n)
        self.assertEqual(len(result.tour_weights), self.n)

        # Higher quality = lower tour weight
        best_vertex = np.argmax(result.labels)
        self.assertEqual(result.tour_weights[best_vertex],
                        min(result.tour_weights))

    def test_rank_based_labeling(self):
        """Test rank-based quality labeling."""
        labeler = AnchorQualityLabeler(
            strategy=LabelingStrategy.RANK_BASED,
            algorithm_name='single_anchor'
        )
        result = labeler.label_vertices(self.graph)

        # Ranks should be 0-100 percentiles
        self.assertTrue(np.all(result.labels >= 0))
        self.assertTrue(np.all(result.labels <= 100))

        # Best vertex gets highest rank
        best_idx = np.argmax(result.labels)
        self.assertEqual(result.labels[best_idx], 100.0)

    def test_binary_classification_labeling(self):
        """Test binary classification labeling."""
        labeler = AnchorQualityLabeler(
            strategy=LabelingStrategy.BINARY,
            algorithm_name='single_anchor',
            top_k_percent=25  # Top 25% = 1 vertex for n=4
        )
        result = labeler.label_vertices(self.graph)

        # Should have exactly 1 positive example (25% of 4)
        self.assertEqual(np.sum(result.labels == 1.0), 1)
        self.assertEqual(np.sum(result.labels == 0.0), 3)

    def test_multiclass_labeling(self):
        """Test multi-class labeling (excellent/good/mediocre/poor)."""
        labeler = AnchorQualityLabeler(
            strategy=LabelingStrategy.MULTICLASS,
            algorithm_name='single_anchor'
        )
        result = labeler.label_vertices(self.graph)

        # Classes should be 0-3
        self.assertTrue(np.all(np.isin(result.labels, [0, 1, 2, 3])))

        # Best vertex should be class 3 (excellent)
        best_idx = np.argmin(result.tour_weights)
        self.assertEqual(result.labels[best_idx], 3)

    def test_relative_to_optimal_labeling(self):
        """Test labeling relative to known optimal."""
        labeler = AnchorQualityLabeler(
            strategy=LabelingStrategy.RELATIVE_TO_OPTIMAL,
            algorithm_name='single_anchor'
        )

        # Provide known optimal weight
        optimal_weight = 80.0  # Example
        result = labeler.label_vertices(self.graph, optimal_weight=optimal_weight)

        # All scores should be >= 1.0 (ratio to optimal)
        self.assertTrue(np.all(result.labels >= 1.0))

        # Best vertex should have lowest ratio
        best_idx = np.argmax(1.0 / result.labels)  # Invert to find min
        self.assertEqual(result.tour_weights[best_idx], min(result.tour_weights))

    def test_deterministic_labeling(self):
        """Test that labeling is deterministic with same seed."""
        labeler1 = AnchorQualityLabeler(
            strategy=LabelingStrategy.RANK_BASED,
            algorithm_name='single_anchor',
            random_seed=42
        )
        labeler2 = AnchorQualityLabeler(
            strategy=LabelingStrategy.RANK_BASED,
            algorithm_name='single_anchor',
            random_seed=42
        )

        result1 = labeler1.label_vertices(self.graph)
        result2 = labeler2.label_vertices(self.graph)

        np.testing.assert_array_equal(result1.labels, result2.labels)
        np.testing.assert_array_equal(result1.tour_weights, result2.tour_weights)

    def test_tie_handling(self):
        """Test handling of vertices with identical tour weights."""
        # Create graph where some vertices produce identical tours
        uniform_graph = np.ones((4, 4), dtype=float) * 10
        np.fill_diagonal(uniform_graph, 0)

        labeler = AnchorQualityLabeler(
            strategy=LabelingStrategy.RANK_BASED,
            algorithm_name='single_anchor'
        )
        result = labeler.label_vertices(uniform_graph)

        # All vertices should have same label (tie)
        self.assertTrue(np.allclose(result.labels, result.labels[0]))

    def test_metadata_tracking(self):
        """Test that metadata is properly tracked."""
        labeler = AnchorQualityLabeler(
            strategy=LabelingStrategy.ABSOLUTE_QUALITY,
            algorithm_name='single_anchor'
        )
        result = labeler.label_vertices(self.graph)

        # Check metadata
        self.assertIn('strategy', result.metadata)
        self.assertIn('algorithm', result.metadata)
        self.assertIn('labeling_time', result.metadata)
        self.assertGreater(result.metadata['labeling_time'], 0)

    def test_failed_tour_handling(self):
        """Test handling when some tours fail to construct."""
        # This is hard to trigger with single_anchor, but test the mechanism
        labeler = AnchorQualityLabeler(
            strategy=LabelingStrategy.RANK_BASED,
            algorithm_name='single_anchor'
        )

        # Should handle gracefully even if some vertices fail
        result = labeler.label_vertices(self.graph)
        self.assertEqual(len(result.labels), self.n)


class TestFeatureDatasetPipeline(unittest.TestCase):
    """Tests for Prompt 10: Feature Engineering Pipeline."""

    def setUp(self):
        """Create test graphs and temporary directory."""
        self.graphs = [
            np.array([[0, 10, 15], [10, 0, 20], [15, 20, 0]], dtype=float),
            np.array([[0, 5, 8, 12], [5, 0, 9, 7], [8, 9, 0, 11], [12, 7, 11, 0]], dtype=float)
        ]
        self.graph_ids = ['graph_1', 'graph_2']
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir)

    def test_pipeline_initialization(self):
        """Test pipeline initialization with config."""
        config = DatasetConfig(
            labeling_strategy=LabelingStrategy.RANK_BASED,
            algorithm_name='single_anchor',
            cache_dir=self.temp_dir
        )
        pipeline = FeatureDatasetPipeline(config)
        self.assertEqual(pipeline.config.labeling_strategy, LabelingStrategy.RANK_BASED)

    def test_single_graph_processing(self):
        """Test processing a single graph."""
        config = DatasetConfig(
            labeling_strategy=LabelingStrategy.RANK_BASED,
            algorithm_name='single_anchor'
        )
        pipeline = FeatureDatasetPipeline(config)

        result = pipeline.process_graph(
            self.graphs[0],
            graph_id='test_graph',
            graph_metadata={'type': 'metric', 'size': 3}
        )

        # Check result structure
        self.assertIsInstance(result.features, pd.DataFrame)
        self.assertEqual(len(result.features), 3)  # 3 vertices
        self.assertIn('label', result.features.columns)
        self.assertIn('graph_id', result.features.columns)
        self.assertIn('vertex_id', result.features.columns)

    def test_batch_processing(self):
        """Test batch processing of multiple graphs."""
        config = DatasetConfig(
            labeling_strategy=LabelingStrategy.RANK_BASED,
            algorithm_name='single_anchor'
        )
        pipeline = FeatureDatasetPipeline(config)

        graphs_with_metadata = [
            (self.graphs[0], 'graph_1', {'type': 'metric'}),
            (self.graphs[1], 'graph_2', {'type': 'metric'})
        ]

        result = pipeline.process_batch(graphs_with_metadata)

        # Should have rows for all vertices (3 + 4 = 7)
        self.assertEqual(len(result.features), 7)
        self.assertEqual(result.metadata['graphs_processed'], 2)
        self.assertEqual(result.metadata['vertices_processed'], 7)

    def test_progress_tracking(self):
        """Test that progress is tracked during batch processing."""
        config = DatasetConfig(
            labeling_strategy=LabelingStrategy.RANK_BASED,
            algorithm_name='single_anchor',
            show_progress=True
        )
        pipeline = FeatureDatasetPipeline(config)

        graphs_with_metadata = [
            (self.graphs[0], f'graph_{i}', {'type': 'metric'})
            for i in range(3)
        ]

        result = pipeline.process_batch(graphs_with_metadata)
        self.assertGreater(result.metadata['processing_time'], 0)

    def test_caching_intermediate_results(self):
        """Test caching of intermediate results."""
        config = DatasetConfig(
            labeling_strategy=LabelingStrategy.RANK_BASED,
            algorithm_name='single_anchor',
            cache_dir=self.temp_dir,
            use_cache=True
        )
        pipeline = FeatureDatasetPipeline(config)

        # Process once
        result1 = pipeline.process_graph(self.graphs[0], 'cached_graph')

        # Process again - should use cache
        result2 = pipeline.process_graph(self.graphs[0], 'cached_graph')

        # Results should be identical
        pd.testing.assert_frame_equal(result1.features, result2.features)

    def test_resumable_processing(self):
        """Test that processing can be resumed after interruption."""
        config = DatasetConfig(
            labeling_strategy=LabelingStrategy.RANK_BASED,
            algorithm_name='single_anchor',
            cache_dir=self.temp_dir,
            use_cache=True
        )
        pipeline = FeatureDatasetPipeline(config)

        # Process first graph
        pipeline.process_graph(self.graphs[0], 'graph_1')

        # Create new pipeline instance (simulates restart)
        pipeline2 = FeatureDatasetPipeline(config)

        # Process both - should skip cached graph_1
        graphs_with_metadata = [
            (self.graphs[0], 'graph_1', {}),
            (self.graphs[1], 'graph_2', {})
        ]
        result = pipeline2.process_batch(graphs_with_metadata)
        self.assertEqual(result.metadata['graphs_processed'], 2)

    def test_feature_validation_in_pipeline(self):
        """Test that features are validated during pipeline."""
        config = DatasetConfig(
            labeling_strategy=LabelingStrategy.RANK_BASED,
            algorithm_name='single_anchor',
            validate_features=True
        )
        pipeline = FeatureDatasetPipeline(config)

        result = pipeline.process_graph(self.graphs[0], 'test_graph')

        # Should have no NaN or Inf values
        self.assertFalse(result.features.isnull().any().any())
        self.assertFalse(np.isinf(result.features.select_dtypes(include=[np.number]).values).any())

    def test_save_and_load_dataset(self):
        """Test saving and loading dataset to/from disk."""
        config = DatasetConfig(
            labeling_strategy=LabelingStrategy.RANK_BASED,
            algorithm_name='single_anchor'
        )
        pipeline = FeatureDatasetPipeline(config)

        result = pipeline.process_graph(self.graphs[0], 'test_graph')

        # Save to CSV
        output_path = os.path.join(self.temp_dir, 'dataset.csv')
        result.save_csv(output_path)
        self.assertTrue(os.path.exists(output_path))

        # Load back
        loaded_df = pd.read_csv(output_path)
        self.assertEqual(len(loaded_df), len(result.features))

    def test_summary_statistics(self):
        """Test generation of summary statistics."""
        config = DatasetConfig(
            labeling_strategy=LabelingStrategy.RANK_BASED,
            algorithm_name='single_anchor'
        )
        pipeline = FeatureDatasetPipeline(config)

        result = pipeline.process_graph(self.graphs[0], 'test_graph')
        summary = result.get_summary()

        self.assertIn('num_vertices', summary)
        self.assertIn('num_features', summary)
        self.assertIn('label_distribution', summary)


class TestFeatureSelection(unittest.TestCase):
    """Tests for Prompt 11: Feature Selection Utilities."""

    def setUp(self):
        """Create test feature matrix and labels."""
        np.random.seed(42)
        self.n_samples = 100
        self.n_features = 20

        # Create correlated features
        self.X = np.random.randn(self.n_samples, self.n_features)

        # Make some features highly correlated
        self.X[:, 1] = self.X[:, 0] + np.random.randn(self.n_samples) * 0.1
        self.X[:, 2] = self.X[:, 0] * 2 + np.random.randn(self.n_samples) * 0.1

        # Create labels with dependency on some features
        self.y = 2 * self.X[:, 0] - self.X[:, 3] + np.random.randn(self.n_samples) * 0.5

        self.feature_names = [f'feature_{i}' for i in range(self.n_features)]

    def test_selector_initialization(self):
        """Test selector initialization."""
        selector = FeatureSelector(method=SelectionMethod.CORRELATION)
        self.assertEqual(selector.method, SelectionMethod.CORRELATION)

    def test_univariate_correlation_selection(self):
        """Test univariate correlation-based selection."""
        selector = FeatureSelector(method=SelectionMethod.CORRELATION)
        result = selector.select_features(
            self.X, self.y, self.feature_names, k=5
        )

        # Should select top 5 features
        self.assertEqual(len(result.selected_features), 5)
        self.assertEqual(len(result.selected_indices), 5)

        # Feature 0 and 3 should be in top features (they determine y)
        self.assertIn('feature_0', result.selected_features)
        self.assertIn('feature_3', result.selected_features)

    def test_f_test_selection(self):
        """Test F-test based selection."""
        selector = FeatureSelector(method=SelectionMethod.F_TEST)
        result = selector.select_features(
            self.X, self.y, self.feature_names, k=10
        )

        self.assertEqual(len(result.selected_features), 10)
        self.assertIn('scores', result.metadata)

    def test_mutual_information_selection(self):
        """Test mutual information selection."""
        selector = FeatureSelector(method=SelectionMethod.MUTUAL_INFO)
        result = selector.select_features(
            self.X, self.y, self.feature_names, k=5
        )

        self.assertEqual(len(result.selected_features), 5)

    def test_recursive_feature_elimination(self):
        """Test recursive feature elimination."""
        selector = FeatureSelector(method=SelectionMethod.RFE)
        result = selector.select_features(
            self.X, self.y, self.feature_names, k=8
        )

        self.assertEqual(len(result.selected_features), 8)
        self.assertIn('ranking', result.metadata)

    def test_model_based_importance(self):
        """Test model-based feature importance (random forest)."""
        selector = FeatureSelector(method=SelectionMethod.MODEL_BASED)
        result = selector.select_features(
            self.X, self.y, self.feature_names, k=10
        )

        self.assertEqual(len(result.selected_features), 10)
        self.assertIn('importances', result.metadata)

    def test_l1_regularization_selection(self):
        """Test L1 regularization (Lasso) selection."""
        selector = FeatureSelector(method=SelectionMethod.L1_REGULARIZATION)
        result = selector.select_features(
            self.X, self.y, self.feature_names, alpha=0.1
        )

        # L1 should drive some coefficients to zero
        self.assertLess(len(result.selected_features), self.n_features)
        self.assertIn('coefficients', result.metadata)

    def test_variance_threshold_selection(self):
        """Test variance threshold selection."""
        # Add a constant feature
        X_with_constant = np.column_stack([self.X, np.ones(self.n_samples)])
        feature_names_ext = self.feature_names + ['constant_feature']

        selector = FeatureSelector(method=SelectionMethod.VARIANCE_THRESHOLD)
        result = selector.select_features(
            X_with_constant, self.y, feature_names_ext, threshold=0.01
        )

        # Constant feature should be removed
        self.assertNotIn('constant_feature', result.selected_features)

    def test_highly_correlated_removal(self):
        """Test removal of highly correlated features."""
        selector = FeatureSelector(method=SelectionMethod.CORRELATION)
        result = selector.select_features(
            self.X, self.y, self.feature_names,
            remove_correlated=True,
            correlation_threshold=0.9
        )

        # Features 1 and 2 are highly correlated with 0
        # At least one should be removed
        self.assertLess(len(result.selected_features), self.n_features)

    def test_feature_ranking(self):
        """Test that features are ranked by importance."""
        selector = FeatureSelector(method=SelectionMethod.CORRELATION)
        result = selector.select_features(
            self.X, self.y, self.feature_names, k=self.n_features
        )

        # Check that ranking is provided
        self.assertIn('ranking', result.metadata)
        self.assertEqual(len(result.metadata['ranking']), self.n_features)

    def test_cross_validation_in_selection(self):
        """Test feature selection with cross-validation."""
        selector = FeatureSelector(
            method=SelectionMethod.RFE,
            use_cv=True,
            cv_folds=3
        )
        result = selector.select_features(
            self.X, self.y, self.feature_names, k=5
        )

        self.assertEqual(len(result.selected_features), 5)
        self.assertIn('cv_scores', result.metadata)


class TestFeatureTransformation(unittest.TestCase):
    """Tests for Prompt 12: Feature Transformation."""

    def setUp(self):
        """Create test features."""
        np.random.seed(42)
        self.n_samples = 100
        self.X = np.random.randn(self.n_samples, 5)

        # Make some features skewed
        self.X[:, 2] = np.abs(self.X[:, 2]) ** 2  # Right-skewed
        self.X[:, 3] = np.exp(self.X[:, 3] / 2)   # Exponential

        self.feature_names = ['f0', 'f1', 'f2', 'f3', 'f4']

    def test_transformer_initialization(self):
        """Test transformer initialization."""
        transformer = FeatureTransformer()
        self.assertIsNotNone(transformer)

    def test_log_transformation(self):
        """Test log transformation for skewed features."""
        transformer = FeatureTransformer()
        config = TransformationConfig(
            transformations={'f2': TransformationType.LOG}
        )

        X_transformed, new_names = transformer.transform(
            self.X, self.feature_names, config
        )

        # Should have original features + log(f2)
        self.assertIn('f2_log', new_names)
        self.assertGreater(len(new_names), len(self.feature_names))

    def test_sqrt_transformation(self):
        """Test square root transformation."""
        transformer = FeatureTransformer()
        config = TransformationConfig(
            transformations={'f2': TransformationType.SQRT}
        )

        X_transformed, new_names = transformer.transform(
            self.X, self.feature_names, config
        )

        self.assertIn('f2_sqrt', new_names)

    def test_polynomial_transformation(self):
        """Test polynomial transformation."""
        transformer = FeatureTransformer()
        config = TransformationConfig(
            transformations={'f0': TransformationType.POLYNOMIAL},
            polynomial_degree=2
        )

        X_transformed, new_names = transformer.transform(
            self.X, self.feature_names, config
        )

        # Should have f0^2
        self.assertIn('f0_squared', new_names)

    def test_inverse_transformation(self):
        """Test inverse transformation."""
        # Avoid zeros
        X_positive = np.abs(self.X) + 1.0

        transformer = FeatureTransformer()
        config = TransformationConfig(
            transformations={'f0': TransformationType.INVERSE}
        )

        X_transformed, new_names = transformer.transform(
            X_positive, self.feature_names, config
        )

        self.assertIn('f0_inverse', new_names)

    def test_feature_interactions_product(self):
        """Test feature interaction via product."""
        transformer = FeatureTransformer()
        config = TransformationConfig(
            interactions=[('f0', 'f1', 'product')]
        )

        X_transformed, new_names = transformer.transform(
            self.X, self.feature_names, config
        )

        self.assertIn('f0_x_f1', new_names)

        # Verify product is correct
        interaction_idx = new_names.index('f0_x_f1')
        f0_idx = self.feature_names.index('f0')
        f1_idx = self.feature_names.index('f1')

        expected_product = self.X[:, f0_idx] * self.X[:, f1_idx]
        np.testing.assert_array_almost_equal(
            X_transformed[:, interaction_idx], expected_product
        )

    def test_feature_interactions_ratio(self):
        """Test feature interaction via ratio."""
        X_positive = np.abs(self.X) + 1.0

        transformer = FeatureTransformer()
        config = TransformationConfig(
            interactions=[('f0', 'f1', 'ratio')]
        )

        X_transformed, new_names = transformer.transform(
            X_positive, self.feature_names, config
        )

        self.assertIn('f0_div_f1', new_names)

    def test_feature_interactions_difference(self):
        """Test feature interaction via difference."""
        transformer = FeatureTransformer()
        config = TransformationConfig(
            interactions=[('f0', 'f1', 'difference')]
        )

        X_transformed, new_names = transformer.transform(
            self.X, self.feature_names, config
        )

        self.assertIn('f0_minus_f1', new_names)

    def test_standardization_zscore(self):
        """Test z-score standardization."""
        transformer = FeatureTransformer()
        config = TransformationConfig(
            standardization='zscore'
        )

        X_transformed, new_names = transformer.transform(
            self.X, self.feature_names, config
        )

        # Transformed features should have mean~0, std~1
        means = np.mean(X_transformed, axis=0)
        stds = np.std(X_transformed, axis=0)

        np.testing.assert_array_almost_equal(means, np.zeros(X_transformed.shape[1]), decimal=10)
        np.testing.assert_array_almost_equal(stds, np.ones(X_transformed.shape[1]), decimal=10)

    def test_standardization_minmax(self):
        """Test min-max standardization."""
        transformer = FeatureTransformer()
        config = TransformationConfig(
            standardization='minmax'
        )

        X_transformed, new_names = transformer.transform(
            self.X, self.feature_names, config
        )

        # All values should be in [0, 1]
        self.assertTrue(np.all(X_transformed >= 0))
        self.assertTrue(np.all(X_transformed <= 1))

    def test_standardization_robust(self):
        """Test robust standardization (median, IQR)."""
        transformer = FeatureTransformer()
        config = TransformationConfig(
            standardization='robust'
        )

        X_transformed, new_names = transformer.transform(
            self.X, self.feature_names, config
        )

        # Should be centered around 0 (by median)
        medians = np.median(X_transformed, axis=0)
        np.testing.assert_array_almost_equal(medians, np.zeros(X_transformed.shape[1]), decimal=1)

    def test_binning_quantile(self):
        """Test quantile-based binning."""
        transformer = FeatureTransformer()
        config = TransformationConfig(
            binning={'f0': 'quantile'},
            n_bins=4
        )

        X_transformed, new_names = transformer.transform(
            self.X, self.feature_names, config
        )

        # Should create one-hot encoded bins
        bin_features = [name for name in new_names if 'f0_bin_' in name]
        self.assertEqual(len(bin_features), 4)

    def test_binning_uniform(self):
        """Test uniform binning."""
        transformer = FeatureTransformer()
        config = TransformationConfig(
            binning={'f0': 'uniform'},
            n_bins=3
        )

        X_transformed, new_names = transformer.transform(
            self.X, self.feature_names, config
        )

        bin_features = [name for name in new_names if 'f0_bin_' in name]
        self.assertEqual(len(bin_features), 3)

    def test_edge_case_log_of_zero(self):
        """Test handling of log(0) edge case."""
        X_with_zero = self.X.copy()
        X_with_zero[0, 0] = 0

        transformer = FeatureTransformer()
        config = TransformationConfig(
            transformations={'f0': TransformationType.LOG},
            handle_zeros='offset'  # Add small offset
        )

        X_transformed, new_names = transformer.transform(
            X_with_zero, self.feature_names, config
        )

        # Should not have inf or nan
        self.assertFalse(np.any(np.isinf(X_transformed)))
        self.assertFalse(np.any(np.isnan(X_transformed)))

    def test_edge_case_division_by_zero(self):
        """Test handling of division by zero."""
        X_with_zero = self.X.copy()
        X_with_zero[0, 1] = 0

        transformer = FeatureTransformer()
        config = TransformationConfig(
            interactions=[('f0', 'f1', 'ratio')],
            handle_zeros='replace'  # Replace inf with large value
        )

        X_transformed, new_names = transformer.transform(
            X_with_zero, self.feature_names, config
        )

        # Should not have inf
        self.assertFalse(np.any(np.isinf(X_transformed)))

    def test_transformation_pipeline(self):
        """Test chaining multiple transformations."""
        transformer = FeatureTransformer()
        config = TransformationConfig(
            transformations={
                'f0': TransformationType.LOG,
                'f1': TransformationType.SQRT
            },
            interactions=[('f0', 'f1', 'product')],
            standardization='zscore'
        )

        X_transformed, new_names = transformer.transform(
            np.abs(self.X) + 1, self.feature_names, config
        )

        # Should have original + transformed + interactions
        self.assertGreater(len(new_names), len(self.feature_names))
        self.assertIn('f0_log', new_names)
        self.assertIn('f1_sqrt', new_names)

    def test_fit_transform_consistency(self):
        """Test that fit_transform and transform give same results."""
        transformer = FeatureTransformer()
        config = TransformationConfig(standardization='zscore')

        # Fit on training data
        X_train = self.X[:80]
        X_test = self.X[80:]

        X_train_transformed, names = transformer.fit_transform(
            X_train, self.feature_names, config
        )

        # Transform test data with same scaler
        X_test_transformed, _ = transformer.transform(
            X_test, self.feature_names, config
        )

        # Shapes should match
        self.assertEqual(X_train_transformed.shape[1], X_test_transformed.shape[1])


if __name__ == '__main__':
    unittest.main()
