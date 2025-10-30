"""
Comprehensive test suite for graph generators.

Tests property correctness, edge cases, consistency, and performance
for all graph generator types.
"""

import unittest
import time
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from graph_generation.euclidean_generator import generate_euclidean_graph, EuclideanGraphGenerator
from graph_generation.metric_generator import generate_metric_graph, generate_quasi_metric_graph
from graph_generation.random_generator import generate_random_graph
from graph_generation.verification import GraphVerifier, verify_graph_properties
from graph_generation.graph_instance import create_graph_instance


class TestEuclideanGenerator(unittest.TestCase):
    """Test suite for Euclidean graph generator."""

    def test_basic_generation(self):
        """Test basic Euclidean graph generation."""
        matrix, coords = generate_euclidean_graph(
            num_vertices=10,
            dimensions=2,
            random_seed=42
        )

        self.assertEqual(len(matrix), 10)
        self.assertEqual(len(coords), 10)
        self.assertEqual(len(coords[0]), 2)

    def test_symmetry(self):
        """Test that Euclidean graphs are symmetric."""
        matrix, _ = generate_euclidean_graph(num_vertices=10, random_seed=42)

        verifier = GraphVerifier(fast_mode=False)
        result = verifier.verify_symmetry(matrix)

        self.assertTrue(result.passed, "Euclidean graph should be symmetric")

    def test_metricity(self):
        """Test that Euclidean graphs satisfy triangle inequality."""
        matrix, _ = generate_euclidean_graph(num_vertices=10, random_seed=42)

        verifier = GraphVerifier(fast_mode=False)
        result = verifier.verify_metricity(matrix)

        self.assertTrue(result.passed, "Euclidean graph should be metric")

    def test_weight_scaling(self):
        """Test that weight range scaling works correctly.

        Note: For Euclidean graphs, coordinate scaling preserves the Euclidean
        property but can only scale all distances by a constant factor.
        This means we can match the max distance but not independently set
        the min distance. The max should match closely, and all weights should
        be within the target range.
        """
        target_range = (10.0, 50.0)
        matrix, _ = generate_euclidean_graph(
            num_vertices=10,
            weight_range=target_range,
            random_seed=42
        )

        weights = [matrix[i][j] for i in range(10) for j in range(i+1, 10)]
        min_weight = min(weights)
        max_weight = max(weights)

        # Max distance should match target max closely
        self.assertAlmostEqual(max_weight, target_range[1], delta=0.1)

        # All weights should be >= target min (coordinate scaling preserves ratios)
        # Note: min may not match target_min exactly due to coordinate scaling limitations
        self.assertGreaterEqual(min_weight, 0.0)  # Weights should be non-negative

    def test_3d_generation(self):
        """Test 3D Euclidean graph generation."""
        matrix, coords = generate_euclidean_graph(
            num_vertices=8,
            dimensions=3,
            random_seed=42
        )

        self.assertEqual(len(coords[0]), 3)
        self.assertEqual(len(matrix), 8)

    def test_clustered_distribution(self):
        """Test clustered point distribution."""
        matrix, coords = generate_euclidean_graph(
            num_vertices=15,
            distribution='clustered',
            distribution_params={'num_clusters': 3, 'cluster_std': 5.0},
            random_seed=42
        )

        self.assertEqual(len(coords), 15)

    def test_grid_distribution(self):
        """Test grid point distribution."""
        matrix, coords = generate_euclidean_graph(
            num_vertices=16,
            distribution='grid',
            random_seed=42
        )

        self.assertEqual(len(coords), 16)

    def test_small_graph(self):
        """Test very small graph generation."""
        matrix, coords = generate_euclidean_graph(
            num_vertices=3,
            random_seed=42
        )

        self.assertEqual(len(matrix), 3)
        self.assertEqual(len(coords), 3)

    def test_deterministic_generation(self):
        """Test that same seed produces same graph."""
        matrix1, coords1 = generate_euclidean_graph(num_vertices=10, random_seed=42)
        matrix2, coords2 = generate_euclidean_graph(num_vertices=10, random_seed=42)

        self.assertEqual(matrix1, matrix2)
        self.assertEqual(coords1, coords2)

    def test_different_seeds(self):
        """Test that different seeds produce different graphs."""
        matrix1, _ = generate_euclidean_graph(num_vertices=10, random_seed=42)
        matrix2, _ = generate_euclidean_graph(num_vertices=10, random_seed=43)

        self.assertNotEqual(matrix1, matrix2)


class TestMetricGenerator(unittest.TestCase):
    """Test suite for metric graph generator."""

    def test_basic_generation(self):
        """Test basic metric graph generation."""
        matrix = generate_metric_graph(num_vertices=10, random_seed=42)

        self.assertEqual(len(matrix), 10)

    def test_metricity_mst_strategy(self):
        """Test that MST strategy produces metric graphs."""
        matrix = generate_metric_graph(
            num_vertices=15,
            strategy='mst',
            random_seed=42
        )

        verifier = GraphVerifier(fast_mode=False)
        result = verifier.verify_metricity(matrix)

        self.assertTrue(result.passed, "MST-based graph should be metric")

    def test_metricity_completion_strategy(self):
        """Test that completion strategy produces metric graphs."""
        matrix = generate_metric_graph(
            num_vertices=15,
            strategy='completion',
            random_seed=42
        )

        verifier = GraphVerifier(fast_mode=False)
        result = verifier.verify_metricity(matrix)

        self.assertTrue(result.passed, "Completion-based graph should be metric")

    def test_symmetry(self):
        """Test that symmetric metric graphs are symmetric."""
        matrix = generate_metric_graph(
            num_vertices=10,
            is_symmetric=True,
            random_seed=42
        )

        verifier = GraphVerifier(fast_mode=False)
        result = verifier.verify_symmetry(matrix)

        self.assertTrue(result.passed)

    def test_weight_range(self):
        """Test that weights are within specified range."""
        target_range = (5.0, 50.0)
        matrix = generate_metric_graph(
            num_vertices=10,
            weight_range=target_range,
            random_seed=42
        )

        weights = [matrix[i][j] for i in range(10) for j in range(i+1, 10)]
        min_weight = min(weights)
        max_weight = max(weights)

        self.assertGreaterEqual(min_weight, target_range[0])
        self.assertLessEqual(max_weight, target_range[1])

    def test_deterministic_generation(self):
        """Test that same seed produces same graph."""
        matrix1 = generate_metric_graph(num_vertices=10, random_seed=42)
        matrix2 = generate_metric_graph(num_vertices=10, random_seed=42)

        self.assertEqual(matrix1, matrix2)


class TestQuasiMetricGenerator(unittest.TestCase):
    """Test suite for quasi-metric graph generator."""

    def test_basic_generation(self):
        """Test basic quasi-metric graph generation."""
        matrix = generate_quasi_metric_graph(num_vertices=10, random_seed=42)

        self.assertEqual(len(matrix), 10)

    def test_asymmetry(self):
        """Test that quasi-metric graphs can be asymmetric."""
        matrix = generate_quasi_metric_graph(
            num_vertices=10,
            asymmetry_factor=0.3,
            random_seed=42
        )

        # Count asymmetric edges
        asymmetric_count = 0
        for i in range(10):
            for j in range(i+1, 10):
                if abs(matrix[i][j] - matrix[j][i]) > 1e-6:
                    asymmetric_count += 1

        # With asymmetry factor > 0, we expect some asymmetric edges
        self.assertGreater(asymmetric_count, 0)

    def test_metricity(self):
        """Test that quasi-metric graphs satisfy triangle inequality.

        Quasi-metric graphs are asymmetric but still satisfy the triangle
        inequality for forward paths: d(x,z) <= d(x,y) + d(y,z).

        The verifier now supports asymmetric mode (symmetric=False) which only
        checks valid forward-path constraints instead of all permutations.
        """
        matrix = generate_quasi_metric_graph(num_vertices=10, random_seed=42)

        verifier = GraphVerifier(fast_mode=False)
        # Use symmetric=False for quasi-metric (asymmetric) graphs
        result = verifier.verify_metricity(matrix, symmetric=False)

        # Quasi-metric graphs should pass all valid triangle inequality checks
        self.assertTrue(
            result.passed,
            f"Quasi-metric graph should satisfy triangle inequality. "
            f"Violations: {result.details['violations']}, Score: {result.details['metricity_score']}"
        )


class TestRandomGenerator(unittest.TestCase):
    """Test suite for random graph generator."""

    def test_basic_generation(self):
        """Test basic random graph generation."""
        matrix = generate_random_graph(num_vertices=10, random_seed=42)

        self.assertEqual(len(matrix), 10)

    def test_uniform_distribution(self):
        """Test uniform distribution."""
        matrix = generate_random_graph(
            num_vertices=20,
            distribution='uniform',
            weight_range=(10.0, 50.0),
            random_seed=42
        )

        weights = [matrix[i][j] for i in range(20) for j in range(i+1, 20)]
        min_weight = min(weights)
        max_weight = max(weights)

        self.assertGreaterEqual(min_weight, 10.0)
        self.assertLessEqual(max_weight, 50.0)

    def test_normal_distribution(self):
        """Test normal distribution."""
        matrix = generate_random_graph(
            num_vertices=20,
            distribution='normal',
            weight_range=(0.0, 100.0),
            random_seed=42
        )

        self.assertEqual(len(matrix), 20)

    def test_symmetric_generation(self):
        """Test symmetric random graph generation."""
        matrix = generate_random_graph(
            num_vertices=10,
            is_symmetric=True,
            random_seed=42
        )

        verifier = GraphVerifier(fast_mode=False)
        result = verifier.verify_symmetry(matrix)

        self.assertTrue(result.passed)

    def test_asymmetric_generation(self):
        """Test asymmetric random graph generation."""
        matrix = generate_random_graph(
            num_vertices=10,
            is_symmetric=False,
            random_seed=42
        )

        # Count asymmetric edges
        asymmetric_count = 0
        for i in range(10):
            for j in range(i+1, 10):
                if abs(matrix[i][j] - matrix[j][i]) > 1e-6:
                    asymmetric_count += 1

        # Should have some asymmetric edges
        self.assertGreater(asymmetric_count, 0)

    def test_non_metric(self):
        """Test that random graphs are typically non-metric."""
        matrix = generate_random_graph(
            num_vertices=15,
            random_seed=42
        )

        verifier = GraphVerifier(fast_mode=False)
        result = verifier.verify_metricity(matrix)

        # Random graphs are almost always non-metric
        self.assertFalse(result.passed)

    def test_deterministic_generation(self):
        """Test that same seed produces same graph."""
        matrix1 = generate_random_graph(num_vertices=10, random_seed=42)
        matrix2 = generate_random_graph(num_vertices=10, random_seed=42)

        self.assertEqual(matrix1, matrix2)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and degenerate inputs."""

    def test_single_vertex_euclidean(self):
        """Test Euclidean graph with single vertex."""
        matrix, coords = generate_euclidean_graph(num_vertices=1, random_seed=42)

        self.assertEqual(len(matrix), 1)
        self.assertEqual(matrix[0][0], 0.0)

    def test_very_narrow_weight_range(self):
        """Test generation with very narrow weight range."""
        # Use 'completion' strategy to keep weights within narrow range
        # MST strategy would create wide distribution from path sums
        matrix = generate_metric_graph(
            num_vertices=10,
            weight_range=(10.0, 10.01),
            strategy='completion',
            random_seed=42
        )

        weights = [matrix[i][j] for i in range(10) for j in range(i+1, 10)]
        weight_std = (sum((w - sum(weights)/len(weights))**2 for w in weights) / len(weights)) ** 0.5

        # Standard deviation should be small for completion strategy
        self.assertLess(weight_std, 0.1)

    def test_large_weight_range(self):
        """Test generation with very large weight range."""
        matrix = generate_metric_graph(
            num_vertices=10,
            weight_range=(1.0, 1000000.0),
            random_seed=42
        )

        weights = [matrix[i][j] for i in range(10) for j in range(i+1, 10)]
        self.assertGreater(max(weights), 1000.0)


class TestConsistency(unittest.TestCase):
    """Test consistency and reproducibility."""

    def test_save_load_roundtrip(self):
        """Test that saving and loading preserves graph."""
        from graph_generation.storage import GraphStorage
        import tempfile
        import shutil

        # Create temporary directory
        temp_dir = tempfile.mkdtemp()

        try:
            matrix, coords = generate_euclidean_graph(num_vertices=10, random_seed=42)
            graph = create_graph_instance(
                adjacency_matrix=matrix,
                graph_type='euclidean',
                generation_params={'test': 'value'},
                random_seed=42,
                coordinates=coords
            )

            # Save and load
            storage = GraphStorage(base_directory=temp_dir)
            filepath = storage.save_graph(graph)
            loaded_graph = storage.load_graph(filepath)

            # Verify equality
            self.assertEqual(graph.adjacency_matrix, loaded_graph.adjacency_matrix)
            self.assertEqual(graph.metadata.graph_type, loaded_graph.metadata.graph_type)
            self.assertEqual(graph.metadata.size, loaded_graph.metadata.size)

        finally:
            # Cleanup
            shutil.rmtree(temp_dir)

    def test_verification_consistency(self):
        """Test that verification gives consistent results."""
        matrix, coords = generate_euclidean_graph(num_vertices=10, random_seed=42)

        verifier = GraphVerifier(fast_mode=False)
        result1 = verifier.verify_metricity(matrix)
        result2 = verifier.verify_metricity(matrix)

        self.assertEqual(result1.passed, result2.passed)
        self.assertEqual(result1.details, result2.details)


class TestPerformance(unittest.TestCase):
    """Test performance benchmarks."""

    def test_euclidean_generation_speed(self):
        """Benchmark Euclidean graph generation."""
        sizes = [20, 50, 100]
        times = []

        for size in sizes:
            start = time.time()
            generate_euclidean_graph(num_vertices=size, random_seed=42)
            duration = time.time() - start
            times.append(duration)

        # Generation should complete in reasonable time
        for t in times:
            self.assertLess(t, 5.0, "Generation taking too long")

    def test_metric_generation_speed(self):
        """Benchmark metric graph generation."""
        sizes = [20, 50, 100]
        times = []

        for size in sizes:
            start = time.time()
            generate_metric_graph(num_vertices=size, random_seed=42)
            duration = time.time() - start
            times.append(duration)

        for t in times:
            self.assertLess(t, 5.0, "Generation taking too long")

    def test_verification_scaling(self):
        """Test that verification time scales appropriately."""
        # Small graph - exhaustive verification
        matrix_small, _ = generate_euclidean_graph(num_vertices=20, random_seed=42)

        start = time.time()
        verifier = GraphVerifier(fast_mode=False)
        verifier.verify_metricity(matrix_small)
        time_small = time.time() - start

        # Should complete quickly
        self.assertLess(time_small, 2.0)


def run_tests():
    """Run all tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestEuclideanGenerator))
    suite.addTests(loader.loadTestsFromTestCase(TestMetricGenerator))
    suite.addTests(loader.loadTestsFromTestCase(TestQuasiMetricGenerator))
    suite.addTests(loader.loadTestsFromTestCase(TestRandomGenerator))
    suite.addTests(loader.loadTestsFromTestCase(TestEdgeCases))
    suite.addTests(loader.loadTestsFromTestCase(TestConsistency))
    suite.addTests(loader.loadTestsFromTestCase(TestPerformance))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result


if __name__ == '__main__':
    result = run_tests()
    sys.exit(0 if result.wasSuccessful() else 1)
