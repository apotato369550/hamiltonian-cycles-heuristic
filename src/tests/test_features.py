"""
Test suite for feature extraction system.

Tests cover:
- Base architecture (VertexFeatureExtractor, FeatureExtractorPipeline)
- Weight-based features
- Topological features
- MST-based features
- Feature validation
- Edge cases
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
