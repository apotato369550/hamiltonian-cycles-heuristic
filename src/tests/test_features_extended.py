"""
Extended test suite for Phase 3 feature extraction (Prompts 5-8).

Tests cover:
- Neighborhood features
- Heuristic-specific features
- Graph context features
- Feature analysis tools
"""

import unittest
import numpy as np
from src.features import FeatureExtractorPipeline, FeatureAnalyzer
from src.features.extractors import (
    NeighborhoodFeatureExtractor,
    HeuristicFeatureExtractor,
    GraphContextFeatureExtractor
)


class TestNeighborhoodFeatureExtractor(unittest.TestCase):
    """Test neighborhood-based feature extraction."""

    def setUp(self):
        """Create test graph."""
        self.graph = np.array([
            [0, 1, 4, 7, 10],
            [1, 0, 2, 5, 8],
            [4, 2, 0, 3, 6],
            [7, 5, 3, 0, 4],
            [10, 8, 6, 4, 0]
        ], dtype=float)

    def test_extract_neighborhood_features(self):
        """Test basic neighborhood feature extraction."""
        extractor = NeighborhoodFeatureExtractor()
        features, names = extractor.extract(self.graph)

        self.assertEqual(features.shape[0], 5)
        self.assertEqual(features.shape[1], len(names))

    def test_knn_features(self):
        """Test k-NN feature extraction."""
        extractor = NeighborhoodFeatureExtractor(k_values=[1, 2])
        features, names = extractor.extract(self.graph)

        # Should have features for k=1 and k=2
        knn_names = [n for n in names if 'knn_' in n]
        self.assertEqual(len(knn_names), 6)  # 3 features per k value

    def test_density_features(self):
        """Test neighborhood density features."""
        extractor = NeighborhoodFeatureExtractor(density_percentiles=[50])
        features, names = extractor.extract(self.graph)

        # Should have density features
        density_names = [n for n in names if 'density_' in n]
        self.assertGreater(len(density_names), 0)

    def test_radial_features(self):
        """Test radial shell features."""
        extractor = NeighborhoodFeatureExtractor(n_shells=3)
        features, names = extractor.extract(self.graph)

        # Should have shell features
        shell_names = [n for n in names if 'shell_' in n]
        self.assertEqual(len(shell_names), 6)  # 2 features per shell

    def test_voronoi_region_size(self):
        """Test Voronoi region size computation."""
        extractor = NeighborhoodFeatureExtractor()
        features, names = extractor.extract(self.graph)

        voronoi_idx = names.index('voronoi_region_size')

        # Voronoi sizes should be proportions (0-1)
        self.assertTrue(np.all(features[:, voronoi_idx] >= 0))
        self.assertTrue(np.all(features[:, voronoi_idx] <= 1))

    def test_no_nan_values(self):
        """Test no NaN values produced."""
        extractor = NeighborhoodFeatureExtractor()
        features, _ = extractor.extract(self.graph)

        self.assertFalse(np.any(np.isnan(features)))


class TestHeuristicFeatureExtractor(unittest.TestCase):
    """Test heuristic-specific feature extraction."""

    def setUp(self):
        """Create test graph."""
        self.graph = np.array([
            [0, 1, 3, 5, 7],
            [1, 0, 2, 4, 6],
            [3, 2, 0, 3, 5],
            [5, 4, 3, 0, 2],
            [7, 6, 5, 2, 0]
        ], dtype=float)

    def test_extract_heuristic_features(self):
        """Test basic heuristic feature extraction."""
        extractor = HeuristicFeatureExtractor()
        features, names = extractor.extract(self.graph)

        self.assertEqual(features.shape[0], 5)
        self.assertEqual(features.shape[1], len(names))

    def test_anchor_edge_features(self):
        """Test anchor edge feature extraction."""
        extractor = HeuristicFeatureExtractor(
            include_tour_estimates=False,
            include_baseline_comparison=False
        )
        features, names = extractor.extract(self.graph)

        # Should have anchor edge features
        self.assertIn('anchor_edge_1', names)
        self.assertIn('anchor_edge_2', names)
        self.assertIn('anchor_sum', names)

        # Anchor edge 1 should be <= anchor edge 2
        edge1_idx = names.index('anchor_edge_1')
        edge2_idx = names.index('anchor_edge_2')

        for i in range(5):
            self.assertLessEqual(features[i, edge1_idx], features[i, edge2_idx])

    def test_tour_estimates(self):
        """Test tour quality estimates."""
        extractor = HeuristicFeatureExtractor(include_tour_estimates=True)
        features, names = extractor.extract(self.graph)

        # Should have tour estimate features
        self.assertIn('tour_estimate_nn', names)
        self.assertIn('tour_estimate_lower_bound', names)

        # Tour estimates should be positive
        nn_idx = names.index('tour_estimate_nn')
        self.assertTrue(np.all(features[:, nn_idx] >= 0))

    def test_baseline_comparison(self):
        """Test baseline comparison features."""
        extractor = HeuristicFeatureExtractor(include_baseline_comparison=True)
        features, names = extractor.extract(self.graph)

        # Should have baseline features
        self.assertIn('baseline_nn_cost', names)
        self.assertIn('baseline_anchor_cost', names)

    def test_no_nan_values(self):
        """Test no NaN values."""
        extractor = HeuristicFeatureExtractor()
        features, _ = extractor.extract(self.graph)

        self.assertFalse(np.any(np.isnan(features)))


class TestGraphContextFeatureExtractor(unittest.TestCase):
    """Test graph-level context feature extraction."""

    def setUp(self):
        """Create test graph."""
        self.graph = np.array([
            [0, 2, 3, 5],
            [2, 0, 4, 6],
            [3, 4, 0, 7],
            [5, 6, 7, 0]
        ], dtype=float)

    def test_extract_context_features(self):
        """Test basic context feature extraction."""
        extractor = GraphContextFeatureExtractor()
        features, names = extractor.extract(self.graph)

        self.assertEqual(features.shape[0], 4)
        self.assertEqual(features.shape[1], len(names))

    def test_graph_properties(self):
        """Test graph property features."""
        extractor = GraphContextFeatureExtractor(include_graph_properties=True)
        features, names = extractor.extract(self.graph)

        # Should have graph properties
        self.assertIn('graph_size', names)
        self.assertIn('graph_density', names)
        self.assertIn('graph_metricity_score', names)

        # Graph size should be constant and correct
        size_idx = names.index('graph_size')
        self.assertTrue(np.all(features[:, size_idx] == 4))

    def test_normalized_importance(self):
        """Test normalized importance features."""
        extractor = GraphContextFeatureExtractor(include_normalized_importance=True)
        features, names = extractor.extract(self.graph)

        # Should have normalized features
        self.assertIn('closeness_normalized', names)
        self.assertIn('degree_normalized', names)

        # Normalized values should be in [0, 1]
        closeness_idx = names.index('closeness_normalized')
        self.assertTrue(np.all(features[:, closeness_idx] >= 0))
        self.assertTrue(np.all(features[:, closeness_idx] <= 1))

    def test_metricity_score_range(self):
        """Test metricity score is in valid range."""
        extractor = GraphContextFeatureExtractor()
        features, names = extractor.extract(self.graph)

        metricity_idx = names.index('graph_metricity_score')

        # Metricity score should be in [0, 1]
        self.assertTrue(np.all(features[:, metricity_idx] >= 0))
        self.assertTrue(np.all(features[:, metricity_idx] <= 1))

    def test_no_nan_values(self):
        """Test no NaN values."""
        extractor = GraphContextFeatureExtractor()
        features, _ = extractor.extract(self.graph)

        self.assertFalse(np.any(np.isnan(features)))


class TestFeatureAnalyzer(unittest.TestCase):
    """Test feature analysis tools."""

    def setUp(self):
        """Create test feature matrix."""
        np.random.seed(42)
        self.features = np.random.rand(20, 10)
        self.feature_names = [f'feature_{i}' for i in range(10)]
        self.analyzer = FeatureAnalyzer(self.features, self.feature_names)

    def test_initialization(self):
        """Test analyzer initialization."""
        self.assertEqual(self.analyzer.n_samples, 20)
        self.assertEqual(self.analyzer.n_features, 10)

    def test_validate_ranges(self):
        """Test range validation."""
        results = self.analyzer.validate_ranges()

        # Should have no NaN or Inf in random data
        self.assertEqual(len(results['nan_features']), 0)
        self.assertEqual(len(results['inf_features']), 0)

    def test_find_constant_features(self):
        """Test constant feature detection."""
        # Add a constant feature
        features_with_constant = np.copy(self.features)
        features_with_constant[:, 0] = 5.0

        analyzer = FeatureAnalyzer(features_with_constant, self.feature_names)
        constant_features = analyzer.find_constant_features()

        self.assertIn('feature_0', constant_features)

    def test_correlation_matrix(self):
        """Test correlation matrix computation."""
        corr_matrix, names = self.analyzer.compute_correlation_matrix()

        # Should be square
        self.assertEqual(corr_matrix.shape[0], 10)
        self.assertEqual(corr_matrix.shape[1], 10)

        # Diagonal should be 1
        np.testing.assert_array_almost_equal(np.diag(corr_matrix), np.ones(10))

    def test_find_highly_correlated_pairs(self):
        """Test finding highly correlated features."""
        # Create perfectly correlated features
        features_corr = np.copy(self.features)
        features_corr[:, 1] = features_corr[:, 0]

        analyzer = FeatureAnalyzer(features_corr, self.feature_names)
        high_corr = analyzer.find_highly_correlated_pairs(threshold=0.95)

        # Should find the perfect correlation
        self.assertGreater(len(high_corr), 0)

    def test_pca(self):
        """Test PCA computation."""
        pca_results = self.analyzer.perform_pca(n_components=3)

        # Should have 3 components
        self.assertEqual(pca_results['components'].shape[1], 3)
        self.assertEqual(len(pca_results['explained_variance']), 3)

        # Explained variance should sum to <= 1
        self.assertLessEqual(np.sum(pca_results['explained_variance_ratio']), 1.0)

    def test_correlate_with_target(self):
        """Test feature-target correlation."""
        target = np.random.rand(20)
        correlations = self.analyzer.correlate_with_target(target)

        # Should have correlation for each feature
        self.assertEqual(len(correlations), 10)

        # Each entry should have (name, corr, p_value)
        for name, corr, p_value in correlations:
            self.assertIn(name, self.feature_names)
            self.assertTrue(-1 <= corr <= 1)
            self.assertTrue(0 <= p_value <= 1)

    def test_analyze_distributions(self):
        """Test distribution analysis."""
        distributions = self.analyzer.analyze_distributions()

        # Should have stats for each feature
        self.assertEqual(len(distributions), 10)

        # Each feature should have expected stats
        for name in self.feature_names:
            self.assertIn(name, distributions)
            stats = distributions[name]
            self.assertIn('mean', stats)
            self.assertIn('std', stats)
            self.assertIn('skewness', stats)

    def test_detect_outliers_iqr(self):
        """Test IQR outlier detection."""
        outliers = self.analyzer.detect_outliers(method='iqr')

        # Should have outlier mask for each feature
        self.assertEqual(len(outliers), 10)

        for name in self.feature_names:
            self.assertEqual(len(outliers[name]), 20)

    def test_detect_outliers_zscore(self):
        """Test z-score outlier detection."""
        outliers = self.analyzer.detect_outliers(method='zscore', threshold=3.0)

        # Should have outlier mask for each feature
        self.assertEqual(len(outliers), 10)

    def test_feature_importance_by_variance(self):
        """Test variance-based importance ranking."""
        importance = self.analyzer.get_feature_importance_by_variance(top_k=5)

        # Should return top 5
        self.assertEqual(len(importance), 5)

        # Should be sorted by variance
        variances = [var for name, var in importance]
        self.assertEqual(variances, sorted(variances, reverse=True))

    def test_summary_report(self):
        """Test summary report generation."""
        report = self.analyzer.summary_report()

        # Should be a string with content
        self.assertIsInstance(report, str)
        self.assertGreater(len(report), 100)


class TestExtendedPipelineIntegration(unittest.TestCase):
    """Test full pipeline with all extractors."""

    def setUp(self):
        """Create test graph and pipeline."""
        self.graph = np.array([
            [0, 1, 2, 3, 4],
            [1, 0, 3, 4, 5],
            [2, 3, 0, 5, 6],
            [3, 4, 5, 0, 7],
            [4, 5, 6, 7, 0]
        ], dtype=float)

    def test_all_extractors_pipeline(self):
        """Test pipeline with all extractor types."""
        pipeline = FeatureExtractorPipeline()
        pipeline.add_extractor(NeighborhoodFeatureExtractor())
        pipeline.add_extractor(HeuristicFeatureExtractor())
        pipeline.add_extractor(GraphContextFeatureExtractor())

        features, names = pipeline.extract_features(self.graph)

        # Should have features from all extractors
        self.assertTrue(any('neighborhood.' in n for n in names))
        self.assertTrue(any('heuristic.' in n for n in names))
        self.assertTrue(any('graph_context.' in n for n in names))

    def test_feature_analysis_workflow(self):
        """Test complete feature extraction and analysis workflow."""
        pipeline = FeatureExtractorPipeline()
        pipeline.add_extractor(NeighborhoodFeatureExtractor())
        pipeline.add_extractor(HeuristicFeatureExtractor())

        features, names = pipeline.extract_features(self.graph)

        # Analyze features
        analyzer = FeatureAnalyzer(features, names)
        report = analyzer.summary_report()

        # Should produce valid report
        self.assertIn('Feature Analysis', report)
        self.assertIn('Number of samples', report)


if __name__ == '__main__':
    unittest.main()
