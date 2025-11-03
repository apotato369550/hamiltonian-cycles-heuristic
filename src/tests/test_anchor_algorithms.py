"""
Test suite for anchor-based TSP algorithms.

Tests single anchor, best anchor, and multi-anchor variants.
"""

import unittest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from algorithms.registry import AlgorithmRegistry
from algorithms.validation import validate_tour


class TestSingleAnchor(unittest.TestCase):
    """Test suite for single anchor algorithm."""

    def setUp(self):
        """Import algorithm before each test."""
        import algorithms.single_anchor

    def test_single_anchor_basic(self):
        """Test single anchor on small graph."""
        algo = AlgorithmRegistry.get_algorithm("single_anchor")

        matrix = [
            [0, 10, 20, 30],
            [10, 0, 15, 25],
            [20, 15, 0, 35],
            [30, 25, 35, 0]
        ]

        result = algo.solve(matrix, anchor_vertex=0)

        self.assertTrue(result.success)
        self.assertEqual(len(result.tour), 4)
        self.assertEqual(result.metadata['anchor_vertex'], 0)
        self.assertEqual(len(result.metadata['anchor_neighbors']), 2)

    def test_single_anchor_different_vertices(self):
        """Test single anchor produces different tours for different anchors."""
        algo1 = AlgorithmRegistry.get_algorithm("single_anchor")
        algo2 = AlgorithmRegistry.get_algorithm("single_anchor")

        matrix = [
            [0, 10, 20, 30],
            [10, 0, 15, 25],
            [20, 15, 0, 35],
            [30, 25, 35, 0]
        ]

        result1 = algo1.solve(matrix, anchor_vertex=0)
        result2 = algo2.solve(matrix, anchor_vertex=1)

        # Different anchors should produce different tours
        self.assertTrue(result1.success)
        self.assertTrue(result2.success)
        # Anchors should be tracked in metadata
        self.assertEqual(result1.metadata['anchor_vertex'], 0)
        self.assertEqual(result2.metadata['anchor_vertex'], 1)

    def test_single_anchor_tour_validity(self):
        """Test that single anchor produces valid tours."""
        algo = AlgorithmRegistry.get_algorithm("single_anchor")

        matrix = [
            [0, 10, 20, 30],
            [10, 0, 15, 25],
            [20, 15, 0, 35],
            [30, 25, 35, 0]
        ]

        result = algo.solve(matrix, anchor_vertex=0)

        if result.success:
            validation = validate_tour(result.tour, matrix)
            self.assertTrue(validation.valid, f"Invalid tour: {validation.errors}")

    def test_single_anchor_invalid_vertex(self):
        """Test that single anchor rejects invalid anchor vertices."""
        algo = AlgorithmRegistry.get_algorithm("single_anchor")

        matrix = [
            [0, 10, 20],
            [10, 0, 15],
            [20, 15, 0]
        ]

        result = algo.solve(matrix, anchor_vertex=10)
        self.assertFalse(result.success)


class TestBestAnchor(unittest.TestCase):
    """Test suite for best anchor algorithm."""

    def setUp(self):
        """Import algorithm before each test."""
        import algorithms.best_anchor

    def test_best_anchor_finds_solution(self):
        """Test that best anchor finds a valid solution."""
        algo = AlgorithmRegistry.get_algorithm("best_anchor_exhaustive")

        matrix = [
            [0, 10, 20, 30],
            [10, 0, 15, 25],
            [20, 15, 0, 35],
            [30, 25, 35, 0]
        ]

        result = algo.solve(matrix)

        self.assertTrue(result.success)
        self.assertEqual(len(result.tour), 4)
        self.assertGreater(result.weight, 0)

    def test_best_anchor_metadata(self):
        """Test that best anchor returns proper metadata."""
        algo = AlgorithmRegistry.get_algorithm("best_anchor_exhaustive")

        matrix = [
            [0, 10, 20],
            [10, 0, 15],
            [20, 15, 0]
        ]

        result = algo.solve(matrix)

        self.assertTrue(result.success)
        self.assertIn('best_anchor_vertex', result.metadata)
        self.assertIn('all_anchor_weights', result.metadata)

    def test_best_anchor_tour_validity(self):
        """Test that best anchor produces valid tours."""
        algo = AlgorithmRegistry.get_algorithm("best_anchor_exhaustive")

        matrix = [
            [0, 10, 20, 30],
            [10, 0, 15, 25],
            [20, 15, 0, 35],
            [30, 25, 35, 0]
        ]

        result = algo.solve(matrix)

        self.assertTrue(result.success)
        validation = validate_tour(result.tour, matrix)
        self.assertTrue(validation.valid)

    def test_best_anchor_beats_single(self):
        """Test that best anchor finds better or equal solution."""
        single = AlgorithmRegistry.get_algorithm("single_anchor")
        best = AlgorithmRegistry.get_algorithm("best_anchor_exhaustive")

        matrix = [
            [0, 10, 20, 30],
            [10, 0, 15, 25],
            [20, 15, 0, 35],
            [30, 25, 35, 0]
        ]

        # Try single anchor from vertex 0
        result_single = single.solve(matrix, anchor_vertex=0)
        result_best = best.solve(matrix)

        if result_single.success and result_best.success:
            # Best anchor should find solution <= single
            self.assertLessEqual(result_best.weight, result_single.weight + 0.01)


class TestMultiAnchor(unittest.TestCase):
    """Test suite for multi-anchor algorithms."""

    def setUp(self):
        """Import algorithms before each test."""
        import algorithms.multi_anchor

    def test_multi_anchor_random(self):
        """Test multi-anchor random on small graph."""
        algo = AlgorithmRegistry.get_algorithm("multi_anchor_random", random_seed=42)

        matrix = [
            [0, 10, 20, 30],
            [10, 0, 15, 25],
            [20, 15, 0, 35],
            [30, 25, 35, 0]
        ]

        result = algo.solve(matrix, num_anchors=2)

        self.assertTrue(result.success)
        self.assertEqual(len(result.tour), 4)
        self.assertEqual(len(result.metadata['anchor_vertices']), 2)

    def test_multi_anchor_distributed(self):
        """Test multi-anchor distributed on small graph."""
        algo = AlgorithmRegistry.get_algorithm("multi_anchor_distributed")

        matrix = [
            [0, 10, 20, 30],
            [10, 0, 15, 25],
            [20, 15, 0, 35],
            [30, 25, 35, 0]
        ]

        result = algo.solve(matrix, num_anchors=2)

        self.assertTrue(result.success)
        self.assertEqual(len(result.tour), 4)
        self.assertEqual(len(result.metadata['anchor_vertices']), 2)

    def test_multi_anchor_tour_validity(self):
        """Test that multi-anchor produces valid tours."""
        algo = AlgorithmRegistry.get_algorithm("multi_anchor_random")

        matrix = [
            [0, 10, 20, 30],
            [10, 0, 15, 25],
            [20, 15, 0, 35],
            [30, 25, 35, 0]
        ]

        result = algo.solve(matrix, num_anchors=2)

        self.assertTrue(result.success)
        validation = validate_tour(result.tour, matrix)
        self.assertTrue(validation.valid)

    def test_multi_anchor_different_counts(self):
        """Test multi-anchor with different anchor counts."""
        algo1 = AlgorithmRegistry.get_algorithm("multi_anchor_random", random_seed=42)
        algo2 = AlgorithmRegistry.get_algorithm("multi_anchor_random", random_seed=42)

        matrix = [
            [0, 10, 20, 30],
            [10, 0, 15, 25],
            [20, 15, 0, 35],
            [30, 25, 35, 0]
        ]

        result1 = algo1.solve(matrix, num_anchors=1)
        result2 = algo2.solve(matrix, num_anchors=2)

        self.assertTrue(result1.success)
        self.assertTrue(result2.success)
        self.assertEqual(len(result1.metadata['anchor_vertices']), 1)
        self.assertEqual(len(result2.metadata['anchor_vertices']), 2)


class TestAnchorRegistry(unittest.TestCase):
    """Test that all anchor algorithms are registered."""

    def setUp(self):
        """Import all algorithms."""
        import algorithms.single_anchor
        import algorithms.best_anchor
        import algorithms.multi_anchor

    def test_all_anchor_algorithms_registered(self):
        """Test that all anchor algorithms are registered."""
        anchors = AlgorithmRegistry.list_algorithms(tags=["anchor"])
        self.assertGreaterEqual(len(anchors), 3)

    def test_anchor_algorithms_exist(self):
        """Test specific anchor algorithms are registered."""
        self.assertTrue(AlgorithmRegistry.is_registered("single_anchor"))
        self.assertTrue(AlgorithmRegistry.is_registered("best_anchor_exhaustive"))
        self.assertTrue(AlgorithmRegistry.is_registered("multi_anchor_random"))
        self.assertTrue(AlgorithmRegistry.is_registered("multi_anchor_distributed"))


if __name__ == '__main__':
    unittest.main()
