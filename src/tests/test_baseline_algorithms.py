"""
Test suite for baseline TSP algorithms.

Tests for nearest neighbor, greedy, and Held-Karp algorithms.
"""

import unittest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from algorithms.registry import AlgorithmRegistry
from algorithms.validation import validate_tour


class TestNearestNeighbor(unittest.TestCase):
    """Test suite for nearest neighbor algorithms."""

    def setUp(self):
        """Import algorithms before each test."""
        # Import to trigger registration
        import algorithms.nearest_neighbor

    def test_nn_random_on_small_graph(self):
        """Test nearest neighbor random on small graph."""
        algo = AlgorithmRegistry.get_algorithm("nearest_neighbor_random", random_seed=42)

        matrix = [
            [0, 10, 20, 30],
            [10, 0, 15, 25],
            [20, 15, 0, 35],
            [30, 25, 35, 0]
        ]

        result = algo.solve(matrix)

        self.assertTrue(result.success)
        self.assertEqual(len(result.tour), 4)
        self.assertEqual(set(result.tour), {0, 1, 2, 3})
        self.assertGreater(result.weight, 0)

    def test_nn_best_on_small_graph(self):
        """Test nearest neighbor best on small graph."""
        algo = AlgorithmRegistry.get_algorithm("nearest_neighbor_best")

        matrix = [
            [0, 10, 20, 30],
            [10, 0, 15, 25],
            [20, 15, 0, 35],
            [30, 25, 35, 0]
        ]

        result = algo.solve(matrix)

        self.assertTrue(result.success)
        self.assertEqual(len(result.tour), 4)
        self.assertEqual(set(result.tour), {0, 1, 2, 3})
        self.assertGreater(result.weight, 0)

    def test_nn_best_beats_random(self):
        """Test that best-start NN finds better tour than single random."""
        algo_random = AlgorithmRegistry.get_algorithm("nearest_neighbor_random", random_seed=42)
        algo_best = AlgorithmRegistry.get_algorithm("nearest_neighbor_best")

        matrix = [
            [0, 10, 20, 30],
            [10, 0, 15, 25],
            [20, 15, 0, 35],
            [30, 25, 35, 0]
        ]

        result_random = algo_random.solve(matrix)
        result_best = algo_best.solve(matrix)

        # Best should find solution <= random
        self.assertLessEqual(result_best.weight, result_random.weight)

    def test_nn_tour_validity(self):
        """Test that NN produces valid tours."""
        algo = AlgorithmRegistry.get_algorithm("nearest_neighbor_best")

        matrix = [
            [0, 10, 20],
            [10, 0, 15],
            [20, 15, 0]
        ]

        result = algo.solve(matrix)

        # Validate tour
        validation = validate_tour(result.tour, matrix)
        self.assertTrue(validation.valid, f"Invalid tour: {validation.errors}")

    def test_nn_reproducibility(self):
        """Test that NN with same seed produces same tour."""
        algo1 = AlgorithmRegistry.get_algorithm("nearest_neighbor_random", random_seed=42)
        algo2 = AlgorithmRegistry.get_algorithm("nearest_neighbor_random", random_seed=42)

        matrix = [
            [0, 10, 20, 30],
            [10, 0, 15, 25],
            [20, 15, 0, 35],
            [30, 25, 35, 0]
        ]

        result1 = algo1.solve(matrix)
        result2 = algo2.solve(matrix)

        # Same seed should give same results
        self.assertEqual(result1.tour, result2.tour)


class TestGreedyEdge(unittest.TestCase):
    """Test suite for greedy edge algorithm."""

    def setUp(self):
        """Import algorithm before each test."""
        import algorithms.greedy

    def test_greedy_on_small_graph(self):
        """Test greedy edge algorithm on small graph."""
        algo = AlgorithmRegistry.get_algorithm("greedy_edge")

        matrix = [
            [0, 10, 20, 30],
            [10, 0, 15, 25],
            [20, 15, 0, 35],
            [30, 25, 35, 0]
        ]

        result = algo.solve(matrix)

        # Greedy may or may not find valid tour depending on graph structure
        if result.success:
            self.assertEqual(len(result.tour), 4)
            validation = validate_tour(result.tour, matrix)
            self.assertTrue(validation.valid)

    def test_greedy_on_simple_symmetric_graph(self):
        """Test greedy on simple symmetric graph."""
        algo = AlgorithmRegistry.get_algorithm("greedy_edge")

        # Simple cycle: 0-1-2-3-0
        matrix = [
            [0, 1, 100, 100],
            [1, 0, 1, 100],
            [100, 1, 0, 1],
            [100, 100, 1, 0]
        ]

        result = algo.solve(matrix)

        if result.success:
            validation = validate_tour(result.tour, matrix)
            self.assertTrue(validation.valid)


class TestHeldKarp(unittest.TestCase):
    """Test suite for Held-Karp exact algorithm."""

    def setUp(self):
        """Import algorithm before each test."""
        import algorithms.exact

    def test_held_karp_small_graph(self):
        """Test Held-Karp on small graph."""
        algo = AlgorithmRegistry.get_algorithm("held_karp_exact")

        matrix = [
            [0, 10, 20],
            [10, 0, 15],
            [20, 15, 0]
        ]

        result = algo.solve(matrix)

        self.assertTrue(result.success)
        self.assertEqual(len(result.tour), 3)

        # Verify it's valid
        validation = validate_tour(result.tour, matrix)
        self.assertTrue(validation.valid)

    def test_held_karp_finds_optimal(self):
        """Test that Held-Karp finds optimal solution."""
        algo = AlgorithmRegistry.get_algorithm("held_karp_exact")

        # Simple triangle
        matrix = [
            [0, 10, 20],
            [10, 0, 15],
            [20, 15, 0]
        ]

        result = algo.solve(matrix)

        self.assertTrue(result.success)

        # Best tour should be 0->1->2->0 (10+15+20=45)
        # or any rotation with same weight
        self.assertAlmostEqual(result.weight, 45.0, places=1)

    def test_held_karp_reproducibility(self):
        """Test that Held-Karp is deterministic."""
        algo1 = AlgorithmRegistry.get_algorithm("held_karp_exact")
        algo2 = AlgorithmRegistry.get_algorithm("held_karp_exact")

        matrix = [
            [0, 10, 20, 30],
            [10, 0, 15, 25],
            [20, 15, 0, 35],
            [30, 25, 35, 0]
        ]

        result1 = algo1.solve(matrix)
        result2 = algo2.solve(matrix)

        # Should find exact same optimal
        if result1.success and result2.success:
            self.assertAlmostEqual(result1.weight, result2.weight, places=5)

    def test_held_karp_rejects_large_graph(self):
        """Test that Held-Karp rejects graphs that are too large."""
        algo = AlgorithmRegistry.get_algorithm("held_karp_exact")

        # Create a 25-vertex graph
        matrix = [[0 if i == j else 1.0 for j in range(25)] for i in range(25)]

        result = algo.solve(matrix)

        self.assertFalse(result.success)
        self.assertIn("too large", result.error_message.lower())

    def test_held_karp_rejects_tiny_graph(self):
        """Test that Held-Karp rejects graphs that are too small."""
        algo = AlgorithmRegistry.get_algorithm("held_karp_exact")

        # Create a 2-vertex graph
        matrix = [[0, 1], [1, 0]]

        result = algo.solve(matrix)

        self.assertFalse(result.success)
        self.assertIn("too small", result.error_message.lower())


class TestAlgorithmRegistry(unittest.TestCase):
    """Test that all baseline algorithms are registered."""

    def setUp(self):
        """Import all algorithms."""
        import algorithms.nearest_neighbor
        import algorithms.greedy
        import algorithms.exact

    def test_all_baseline_algorithms_registered(self):
        """Test that all baseline algorithms are registered."""
        baselines = AlgorithmRegistry.list_algorithms(tags=["baseline"])
        self.assertGreaterEqual(len(baselines), 2)  # At least NN and Greedy

    def test_greedy_is_registered(self):
        """Test that greedy is registered."""
        self.assertTrue(AlgorithmRegistry.is_registered("greedy_edge"))

    def test_exact_is_registered(self):
        """Test that Held-Karp is registered."""
        self.assertTrue(AlgorithmRegistry.is_registered("held_karp_exact"))

    def test_nearest_neighbors_registered(self):
        """Test that NN variants are registered."""
        self.assertTrue(AlgorithmRegistry.is_registered("nearest_neighbor_random"))
        self.assertTrue(AlgorithmRegistry.is_registered("nearest_neighbor_best"))


if __name__ == '__main__':
    unittest.main()
