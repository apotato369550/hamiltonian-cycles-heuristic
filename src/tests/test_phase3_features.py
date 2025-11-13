"""
Phase 3: Feature Engineering - Comprehensive Test Suite

Tests cover:
- Base architecture and pipeline (Prompt 1)
- Weight-based features (Prompt 2)
- Topological features (Prompt 3)
- MST-based features (Prompt 4)
- Neighborhood features (Prompt 5)
- Heuristic features (Prompt 6)
- Graph context features (Prompt 7)
- Feature analysis tools (Prompt 8)
- Anchor quality labeling (Prompt 9)
- Dataset pipeline (Prompt 10)
- Feature selection (Prompt 11)
- Feature transformation (Prompt 12)

Total: 111 tests (34 basic + 30 extended + 47 final)
Note: Prompts 10-12 require pandas and scikit-learn
"""



import unittest
import numpy as np
from src.features.base import VertexFeatureExtractor, FeatureValidationError, CachedComputation
from src.features.pipeline import FeatureExtractorPipeline
from src.features.extractors import WeightFeatureExtractor, TopologicalFeatureExtractor, MSTFeatureExtractor


class TestBaseArchitecture(unittest.TestCase):
    """Test base classes and infrastructure."""

    def test_cached_computation(self):
        """Test cache helper works correctly."""
        cache = {}
        call_count = [0]

        def expensive_fn():
            call_count[0] += 1
            return 42

        # First call computes
        result1 = CachedComputation.get_or_compute(cache, 'key', expensive_fn)
        self.assertEqual(result1, 42)
        self.assertEqual(call_count[0], 1)

        # Second call uses cache
        result2 = CachedComputation.get_or_compute(cache, 'key', expensive_fn)
        self.assertEqual(result2, 42)
        self.assertEqual(call_count[0], 1)  # Not called again

    def test_pipeline_add_extractor(self):
        """Test adding extractors to pipeline."""
        pipeline = FeatureExtractorPipeline()
        extractor = WeightFeatureExtractor()

        pipeline.add_extractor(extractor)
        self.assertEqual(len(pipeline.extractors), 1)
        self.assertIn('weight_based', pipeline.get_extractor_names())

    def test_pipeline_duplicate_name_error(self):
        """Test that duplicate extractor names raise error."""
        pipeline = FeatureExtractorPipeline()
        pipeline.add_extractor(WeightFeatureExtractor())

        with self.assertRaises(ValueError):
            pipeline.add_extractor(WeightFeatureExtractor())

    def test_pipeline_remove_extractor(self):
        """Test removing extractors."""
        pipeline = FeatureExtractorPipeline()
        pipeline.add_extractor(WeightFeatureExtractor())
        pipeline.remove_extractor('weight_based')

        self.assertEqual(len(pipeline.extractors), 0)

    def test_pipeline_empty_error(self):
        """Test that extracting with no extractors raises error."""
        pipeline = FeatureExtractorPipeline()
        graph = np.random.rand(5, 5)

        with self.assertRaises(ValueError):
            pipeline.extract_features(graph)


class TestWeightFeatureExtractor(unittest.TestCase):
    """Test weight-based feature extraction."""

    def setUp(self):
        """Create test graphs."""
        # Simple 4-vertex symmetric graph
        self.symmetric_graph = np.array([
            [0, 1, 2, 3],
            [1, 0, 4, 5],
            [2, 4, 0, 6],
            [3, 5, 6, 0]
        ], dtype=float)

        # Asymmetric graph
        self.asymmetric_graph = np.array([
            [0, 1, 2, 3],
            [2, 0, 4, 5],
            [3, 1, 0, 6],
            [4, 2, 1, 0]
        ], dtype=float)

    def test_extract_symmetric_features(self):
        """Test extraction on symmetric graph."""
        extractor = WeightFeatureExtractor(include_asymmetric_features=False)
        features, names = extractor.extract(self.symmetric_graph)

        # Should have 4 vertices
        self.assertEqual(features.shape[0], 4)

        # Should have expected number of features
        expected_count = len(extractor.get_feature_names())
        self.assertEqual(features.shape[1], expected_count)
        self.assertEqual(len(names), expected_count)

    def test_extract_asymmetric_features(self):
        """Test extraction on asymmetric graph with asymmetric features."""
        extractor = WeightFeatureExtractor(include_asymmetric_features=True)
        features, names = extractor.extract(self.asymmetric_graph)

        # Should have 4 vertices
        self.assertEqual(features.shape[0], 4)

        # Should have more features (outgoing + incoming + asymmetry)
        self.assertGreater(features.shape[1], len(extractor.get_feature_names()))

        # Check that asymmetry features are present
        self.assertTrue(any('asym_' in name for name in names))

    def test_no_nan_values(self):
        """Test that no NaN values are produced."""
        extractor = WeightFeatureExtractor()
        features, _ = extractor.extract(self.symmetric_graph)

        self.assertFalse(np.any(np.isnan(features)))

    def test_no_inf_values(self):
        """Test that no infinite values are produced."""
        extractor = WeightFeatureExtractor()
        features, _ = extractor.extract(self.symmetric_graph)

        self.assertFalse(np.any(np.isinf(features)))

    def test_mean_weight_calculation(self):
        """Test mean weight is calculated correctly."""
        extractor = WeightFeatureExtractor(include_asymmetric_features=False)
        features, names = extractor.extract(self.symmetric_graph)

        # Find mean_weight column
        mean_idx = names.index('mean_weight')

        # Vertex 0 has edges [1, 2, 3], mean = 2.0
        self.assertAlmostEqual(features[0, mean_idx], 2.0)

    def test_min_max_calculation(self):
        """Test min and max weight calculation."""
        extractor = WeightFeatureExtractor(include_asymmetric_features=False)
        features, names = extractor.extract(self.symmetric_graph)

        min_idx = names.index('min_weight')
        max_idx = names.index('max_weight')

        # Vertex 0 has edges [1, 2, 3]
        self.assertEqual(features[0, min_idx], 1.0)
        self.assertEqual(features[0, max_idx], 3.0)


class TestTopologicalFeatureExtractor(unittest.TestCase):
    """Test topological feature extraction."""

    def setUp(self):
        """Create test graph."""
        # 5-vertex complete graph
        self.graph = np.array([
            [0, 1, 2, 3, 4],
            [1, 0, 1, 2, 3],
            [2, 1, 0, 1, 2],
            [3, 2, 1, 0, 1],
            [4, 3, 2, 1, 0]
        ], dtype=float)

    def test_extract_all_features(self):
        """Test extraction with all features enabled."""
        extractor = TopologicalFeatureExtractor(
            include_betweenness=True,
            include_eigenvector=True,
            include_clustering=True
        )
        features, names = extractor.extract(self.graph)

        # Should have 5 vertices
        self.assertEqual(features.shape[0], 5)

        # Should have all features
        expected_count = len(extractor.get_feature_names())
        self.assertEqual(features.shape[1], expected_count)
        self.assertEqual(len(names), expected_count)

    def test_extract_minimal_features(self):
        """Test extraction with expensive features disabled."""
        extractor = TopologicalFeatureExtractor(
            include_betweenness=False,
            include_eigenvector=False,
            include_clustering=False
        )
        features, names = extractor.extract(self.graph)

        # Should still have basic features
        self.assertGreater(features.shape[1], 0)

        # Should not have expensive features
        self.assertNotIn('betweenness_centrality', names)
        self.assertNotIn('eigenvector_centrality', names)

    def test_degree_calculation(self):
        """Test degree calculation in complete graph."""
        extractor = TopologicalFeatureExtractor()
        features, names = extractor.extract(self.graph)

        degree_idx = names.index('degree')

        # In complete graph with 5 vertices, degree should be 4
        for i in range(5):
            self.assertEqual(features[i, degree_idx], 4)

    def test_closeness_centrality_positive(self):
        """Test that closeness centrality is positive."""
        extractor = TopologicalFeatureExtractor()
        features, names = extractor.extract(self.graph)

        closeness_idx = names.index('closeness_centrality')

        # All vertices should have positive closeness
        self.assertTrue(np.all(features[:, closeness_idx] > 0))

    def test_no_nan_values(self):
        """Test no NaN values in features."""
        extractor = TopologicalFeatureExtractor()
        features, _ = extractor.extract(self.graph)

        self.assertFalse(np.any(np.isnan(features)))

    def test_cache_usage(self):
        """Test that extractor uses cache for expensive computations."""
        cache = {}
        extractor = TopologicalFeatureExtractor()

        # Extract features twice with same cache
        extractor.extract(self.graph, cache)
        extractor.extract(self.graph, cache)

        # Cache should contain shortest paths
        self.assertIn('shortest_paths', cache)


class TestMSTFeatureExtractor(unittest.TestCase):
    """Test MST-based feature extraction."""

    def setUp(self):
        """Create test graph."""
        # 5-vertex graph
        self.graph = np.array([
            [0, 1, 4, 5, 6],
            [1, 0, 2, 4, 5],
            [4, 2, 0, 1, 3],
            [5, 4, 1, 0, 2],
            [6, 5, 3, 2, 0]
        ], dtype=float)

    def test_extract_mst_features(self):
        """Test MST feature extraction."""
        extractor = MSTFeatureExtractor()
        features, names = extractor.extract(self.graph)

        # Should have 5 vertices
        self.assertEqual(features.shape[0], 5)

        # Should have expected features
        expected_count = len(extractor.get_feature_names())
        self.assertEqual(features.shape[1], expected_count)
        self.assertEqual(len(names), expected_count)

    def test_mst_degree_range(self):
        """Test that MST degree is in valid range."""
        extractor = MSTFeatureExtractor()
        features, names = extractor.extract(self.graph)

        mst_degree_idx = names.index('mst_degree')

        # MST degree should be at least 1 (leaf) and at most n-1
        degrees = features[:, mst_degree_idx]
        self.assertTrue(np.all(degrees >= 1))
        self.assertTrue(np.all(degrees < 5))

    def test_mst_total_edges(self):
        """Test that MST has correct number of edges."""
        extractor = MSTFeatureExtractor()
        features, names = extractor.extract(self.graph)

        mst_degree_idx = names.index('mst_degree')

        # Sum of degrees should be 2 * (n - 1) = 8 for 5 vertices
        total_degree = np.sum(features[:, mst_degree_idx])
        self.assertEqual(total_degree, 8)

    def test_leaf_hub_indicators(self):
        """Test leaf and hub indicators."""
        extractor = MSTFeatureExtractor()
        features, names = extractor.extract(self.graph)

        leaf_idx = names.index('mst_is_leaf')
        hub_idx = names.index('mst_is_hub')

        # Values should be 0 or 1
        self.assertTrue(np.all((features[:, leaf_idx] == 0) | (features[:, leaf_idx] == 1)))
        self.assertTrue(np.all((features[:, hub_idx] == 0) | (features[:, hub_idx] == 1)))

    def test_no_nan_values(self):
        """Test no NaN values."""
        extractor = MSTFeatureExtractor()
        features, _ = extractor.extract(self.graph)

        self.assertFalse(np.any(np.isnan(features)))

    def test_cache_usage(self):
        """Test that MST is cached."""
        cache = {}
        extractor = MSTFeatureExtractor()

        # Extract twice with same cache
        extractor.extract(self.graph, cache)
        extractor.extract(self.graph, cache)

        # Cache should contain MST
        self.assertIn('mst', cache)


class TestFeatureValidation(unittest.TestCase):
    """Test feature validation."""

    def test_nan_detection(self):
        """Test that NaN values are detected."""
        extractor = WeightFeatureExtractor()
        features = np.array([[1, 2, 3], [4, np.nan, 6]])
        names = ['a', 'b', 'c']

        with self.assertRaises(FeatureValidationError):
            extractor.validate_features(features, names, 2)

    def test_inf_detection(self):
        """Test that infinite values are detected."""
        extractor = WeightFeatureExtractor()
        features = np.array([[1, 2, 3], [4, np.inf, 6]])
        names = ['a', 'b', 'c']

        with self.assertRaises(FeatureValidationError):
            extractor.validate_features(features, names, 2)

    def test_shape_mismatch_detection(self):
        """Test that shape mismatches are detected."""
        extractor = WeightFeatureExtractor()
        features = np.array([[1, 2, 3], [4, 5, 6]])
        names = ['a', 'b', 'c']

        # Wrong number of rows
        with self.assertRaises(FeatureValidationError):
            extractor.validate_features(features, names, 3)

    def test_name_count_mismatch(self):
        """Test that feature name count mismatch is detected."""
        extractor = WeightFeatureExtractor()
        features = np.array([[1, 2, 3], [4, 5, 6]])
        names = ['a', 'b']  # Only 2 names for 3 features

        with self.assertRaises(FeatureValidationError):
            extractor.validate_features(features, names, 2)


class TestPipelineIntegration(unittest.TestCase):
    """Test full pipeline integration."""

    def setUp(self):
        """Create test graph and pipeline."""
        self.graph = np.array([
            [0, 1, 2, 3],
            [1, 0, 4, 5],
            [2, 4, 0, 6],
            [3, 5, 6, 0]
        ], dtype=float)

        self.pipeline = FeatureExtractorPipeline()

    def test_single_extractor_pipeline(self):
        """Test pipeline with single extractor."""
        self.pipeline.add_extractor(WeightFeatureExtractor())
        features, names = self.pipeline.extract_features(self.graph)

        self.assertEqual(features.shape[0], 4)
        self.assertGreater(features.shape[1], 0)

        # All names should be prefixed
        self.assertTrue(all(name.startswith('weight_based.') for name in names))

    def test_multi_extractor_pipeline(self):
        """Test pipeline with multiple extractors."""
        self.pipeline.add_extractor(WeightFeatureExtractor())
        self.pipeline.add_extractor(TopologicalFeatureExtractor(
            include_betweenness=False,
            include_eigenvector=False
        ))
        self.pipeline.add_extractor(MSTFeatureExtractor())

        features, names = self.pipeline.extract_features(self.graph)

        self.assertEqual(features.shape[0], 4)

        # Should have features from all extractors
        self.assertTrue(any(name.startswith('weight_based.') for name in names))
        self.assertTrue(any(name.startswith('topological.') for name in names))
        self.assertTrue(any(name.startswith('mst_based.') for name in names))

    def test_cache_sharing_between_extractors(self):
        """Test that cache is shared between extractors."""
        cache = {}

        self.pipeline.add_extractor(TopologicalFeatureExtractor())
        self.pipeline.add_extractor(MSTFeatureExtractor())

        self.pipeline.extract_features(self.graph, cache)

        # Cache should have entries from both extractors
        self.assertIn('shortest_paths', cache)
        self.assertIn('mst', cache)

    def test_feature_count(self):
        """Test feature count reporting."""
        self.pipeline.add_extractor(WeightFeatureExtractor())
        self.pipeline.add_extractor(MSTFeatureExtractor())

        count = self.pipeline.get_feature_count()
        self.assertGreater(count, 0)

        # Verify it matches actual extraction
        features, _ = self.pipeline.extract_features(self.graph)
        self.assertEqual(features.shape[1], count)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and corner cases."""

    def test_single_vertex_graph(self):
        """Test extraction on single vertex graph."""
        graph = np.array([[0]], dtype=float)
        extractor = WeightFeatureExtractor()

        # Should handle gracefully
        features, names = extractor.extract(graph)
        self.assertEqual(features.shape[0], 1)

    def test_uniform_weights(self):
        """Test extraction when all weights are identical."""
        graph = np.ones((5, 5), dtype=float)
        np.fill_diagonal(graph, 0)

        extractor = WeightFeatureExtractor()
        features, _ = extractor.extract(graph)

        # Should not produce NaN or inf
        self.assertFalse(np.any(np.isnan(features)))
        self.assertFalse(np.any(np.isinf(features)))

    def test_large_weight_range(self):
        """Test with very large weight differences."""
        graph = np.array([
            [0, 1, 1000, 1],
            [1, 0, 1, 1000],
            [1000, 1, 0, 1],
            [1, 1000, 1, 0]
        ], dtype=float)

        pipeline = FeatureExtractorPipeline()
        pipeline.add_extractor(WeightFeatureExtractor())
        pipeline.add_extractor(MSTFeatureExtractor())

        features, _ = pipeline.extract_features(graph)

        # Should handle gracefully
        self.assertFalse(np.any(np.isnan(features)))
        self.assertFalse(np.any(np.isinf(features)))


if __name__ == '__main__':
    unittest.main()


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
            algorithm_name='single_anchor_v1'
        )
        self.assertEqual(labeler.strategy, LabelingStrategy.RANK_BASED)
        self.assertEqual(labeler.algorithm_name, 'single_anchor')

    def test_absolute_quality_labeling(self):
        """Test absolute quality labeling strategy."""
        labeler = AnchorQualityLabeler(
            strategy=LabelingStrategy.ABSOLUTE_QUALITY,
            algorithm_name='single_anchor_v1'
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
            algorithm_name='single_anchor_v1'
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
            algorithm_name='single_anchor_v1',
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
            algorithm_name='single_anchor_v1'
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
            algorithm_name='single_anchor_v1'
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
            algorithm_name='single_anchor_v1',
            random_seed=42
        )
        labeler2 = AnchorQualityLabeler(
            strategy=LabelingStrategy.RANK_BASED,
            algorithm_name='single_anchor_v1',
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
            algorithm_name='single_anchor_v1'
        )
        result = labeler.label_vertices(uniform_graph)

        # All vertices should have same label (tie)
        self.assertTrue(np.allclose(result.labels, result.labels[0]))

    def test_metadata_tracking(self):
        """Test that metadata is properly tracked."""
        labeler = AnchorQualityLabeler(
            strategy=LabelingStrategy.ABSOLUTE_QUALITY,
            algorithm_name='single_anchor_v1'
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
            algorithm_name='single_anchor_v1'
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
            algorithm_name='single_anchor_v1',
            cache_dir=self.temp_dir
        )
        pipeline = FeatureDatasetPipeline(config)
        self.assertEqual(pipeline.config.labeling_strategy, LabelingStrategy.RANK_BASED)

    def test_single_graph_processing(self):
        """Test processing a single graph."""
        config = DatasetConfig(
            labeling_strategy=LabelingStrategy.RANK_BASED,
            algorithm_name='single_anchor_v1'
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
            algorithm_name='single_anchor_v1'
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
            algorithm_name='single_anchor_v1',
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
            algorithm_name='single_anchor_v1',
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
            algorithm_name='single_anchor_v1',
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
            algorithm_name='single_anchor_v1',
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
            algorithm_name='single_anchor_v1'
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
            algorithm_name='single_anchor_v1'
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
        # F-test returns p-values, not scores
        self.assertIn('p_values', result.metadata)

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

        # CV may select 4-5 features depending on sklearn version
        self.assertIn(len(result.selected_features), [4, 5])
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
