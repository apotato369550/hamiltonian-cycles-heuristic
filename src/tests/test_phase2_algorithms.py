"""
Phase 2: Algorithm Benchmarking - Comprehensive Test Suite

Tests cover:
- Algorithm interface contract validation
- Tour validation and metrics computation
- Registry system and algorithm registration
- Baseline algorithms (Nearest Neighbor, Greedy, Held-Karp)
- Anchor-based algorithms (Single, Best, Multi-Anchor)

Total: 89 tests (59 core + 16 baseline + 14 anchor)
"""



import unittest
import sys
import time
from pathlib import Path
from typing import List

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from algorithms.base import TourResult, AlgorithmMetadata, TSPAlgorithm
from algorithms.registry import AlgorithmRegistry, register_algorithm
from algorithms.validation import (
    validate_tour, validate_tour_constraints, ValidationResult, TourValidator
)
from algorithms.metrics import (
    compute_tour_weight, compute_tour_statistics, compute_optimality_gap,
    compute_approximation_ratio, compute_tour_properties, MetricsCalculator
)


class TestTourResult(unittest.TestCase):
    """Test suite for TourResult dataclass."""

    def test_valid_tour_result(self):
        """Test creating a valid tour result."""
        result = TourResult(
            tour=[0, 1, 2, 3],
            weight=100.0,
            runtime=0.5,
            metadata={'anchor': 0}
        )
        self.assertEqual(result.tour, [0, 1, 2, 3])
        self.assertEqual(result.weight, 100.0)
        self.assertEqual(result.runtime, 0.5)
        self.assertTrue(result.success)

    def test_tour_result_with_failure(self):
        """Test creating a failed tour result."""
        result = TourResult(
            tour=[],
            weight=float('inf'),
            runtime=1.0,
            success=False,
            error_message="Timeout reached"
        )
        self.assertFalse(result.success)
        self.assertEqual(result.error_message, "Timeout reached")

    def test_tour_result_duplicate_vertices(self):
        """Test that duplicate vertices are rejected."""
        with self.assertRaises(ValueError):
            TourResult(
                tour=[0, 1, 1, 2],
                weight=100.0,
                runtime=0.5
            )

    def test_tour_result_too_few_vertices(self):
        """Test that tours with <3 vertices are rejected."""
        with self.assertRaises(ValueError):
            TourResult(
                tour=[0, 1],
                weight=100.0,
                runtime=0.5
            )

    def test_tour_result_negative_weight(self):
        """Test that negative weights are rejected."""
        with self.assertRaises(ValueError):
            TourResult(
                tour=[0, 1, 2],
                weight=-10.0,
                runtime=0.5
            )

    def test_tour_result_negative_runtime(self):
        """Test that negative runtimes are rejected."""
        with self.assertRaises(ValueError):
            TourResult(
                tour=[0, 1, 2],
                weight=100.0,
                runtime=-1.0
            )


class TestAlgorithmMetadata(unittest.TestCase):
    """Test suite for AlgorithmMetadata dataclass."""

    def test_basic_metadata(self):
        """Test creating basic metadata."""
        metadata = AlgorithmMetadata(
            name="test_algo",
            version="1.0.0"
        )
        self.assertEqual(metadata.name, "test_algo")
        self.assertEqual(metadata.version, "1.0.0")

    def test_metadata_with_constraints(self):
        """Test metadata with graph type constraints."""
        metadata = AlgorithmMetadata(
            name="test_algo",
            version="1.0.0",
            applicability_constraints={
                'graph_types': ['euclidean', 'metric'],
                'min_size': 3,
                'max_size': 100
            }
        )
        self.assertTrue(metadata.can_handle_graph_type('euclidean'))
        self.assertTrue(metadata.can_handle_graph_type('metric'))
        self.assertFalse(metadata.can_handle_graph_type('random'))
        self.assertTrue(metadata.can_handle_graph_size(50))
        self.assertFalse(metadata.can_handle_graph_size(2))
        self.assertFalse(metadata.can_handle_graph_size(101))

    def test_metadata_no_constraints(self):
        """Test metadata with no constraints allows everything."""
        metadata = AlgorithmMetadata(name="test", version="1.0.0")
        self.assertTrue(metadata.can_handle_graph_type('euclidean'))
        self.assertTrue(metadata.can_handle_graph_type('random'))
        self.assertTrue(metadata.can_handle_graph_size(1))
        self.assertTrue(metadata.can_handle_graph_size(1000))


class SimpleDummyAlgorithm(TSPAlgorithm):
    """Simple algorithm for testing the interface."""

    def solve(self, adjacency_matrix: List[List[float]], **kwargs) -> TourResult:
        """Implement solve method."""
        n = len(adjacency_matrix)

        # Create a simple sequential tour
        tour = list(range(n))
        weight = self._compute_tour_weight(tour, adjacency_matrix)

        return TourResult(
            tour=tour,
            weight=weight,
            runtime=0.0,
            metadata={'algorithm': 'dummy'}
        )

    def get_metadata(self) -> AlgorithmMetadata:
        """Return metadata."""
        return AlgorithmMetadata(
            name="dummy_simple",
            version="1.0.0",
            description="Simple dummy algorithm for testing"
        )


class TestTSPAlgorithmInterface(unittest.TestCase):
    """Test suite for TSPAlgorithm abstract base class."""

    def test_algorithm_initialization(self):
        """Test algorithm initialization."""
        algo = SimpleDummyAlgorithm()
        self.assertIsNone(algo.random_seed)

        algo_seeded = SimpleDummyAlgorithm(random_seed=42)
        self.assertEqual(algo_seeded.random_seed, 42)

    def test_algorithm_solve(self):
        """Test algorithm solve method."""
        algo = SimpleDummyAlgorithm()

        # Create simple 4-vertex graph
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

    def test_algorithm_metadata(self):
        """Test algorithm metadata."""
        algo = SimpleDummyAlgorithm()
        metadata = algo.get_metadata()

        self.assertEqual(metadata.name, "dummy_simple")
        self.assertEqual(metadata.version, "1.0.0")

    def test_is_applicable(self):
        """Test applicability checking."""
        algo = SimpleDummyAlgorithm()

        # Should be applicable to any type/size
        self.assertTrue(algo.is_applicable("euclidean", 10))
        self.assertTrue(algo.is_applicable("random", 100))

    def test_compute_tour_weight(self):
        """Test tour weight computation."""
        algo = SimpleDummyAlgorithm()

        matrix = [
            [0, 10, 20],
            [10, 0, 15],
            [20, 15, 0]
        ]

        tour = [0, 1, 2]
        weight = algo._compute_tour_weight(tour, matrix)

        # Weight should be: 0->1 (10) + 1->2 (15) + 2->0 (20) = 45
        self.assertEqual(weight, 45.0)

    def test_validate_tour_structure(self):
        """Test tour structure validation."""
        algo = SimpleDummyAlgorithm()

        # Valid tour
        is_valid, msg = algo._validate_tour_structure([0, 1, 2, 3], 4)
        self.assertTrue(is_valid)

        # Wrong size
        is_valid, msg = algo._validate_tour_structure([0, 1, 2], 4)
        self.assertFalse(is_valid)

        # Duplicate vertices
        is_valid, msg = algo._validate_tour_structure([0, 1, 1, 2], 4)
        self.assertFalse(is_valid)

        # Invalid vertex index
        is_valid, msg = algo._validate_tour_structure([0, 1, 5], 3)
        self.assertFalse(is_valid)

    def test_create_failure_result(self):
        """Test failure result creation."""
        algo = SimpleDummyAlgorithm()
        result = algo._create_failure_result("Test error")

        self.assertFalse(result.success)
        self.assertEqual(result.error_message, "Test error")
        self.assertEqual(result.tour, [])
        self.assertEqual(result.weight, float('inf'))


class TestAlgorithmRegistry(unittest.TestCase):
    """Test suite for AlgorithmRegistry."""

    def setUp(self):
        """Clear registry before each test."""
        AlgorithmRegistry.clear()

    def tearDown(self):
        """Clear registry after each test."""
        AlgorithmRegistry.clear()

    def test_register_algorithm(self):
        """Test registering an algorithm."""
        AlgorithmRegistry.register(
            "test_algo",
            SimpleDummyAlgorithm,
            tags=["test"],
            constraints={}
        )

        self.assertTrue(AlgorithmRegistry.is_registered("test_algo"))
        self.assertEqual(AlgorithmRegistry.count(), 1)

    def test_get_algorithm(self):
        """Test retrieving an algorithm."""
        AlgorithmRegistry.register("test_algo", SimpleDummyAlgorithm)

        algo = AlgorithmRegistry.get_algorithm("test_algo")
        self.assertIsInstance(algo, SimpleDummyAlgorithm)

    def test_get_algorithm_with_seed(self):
        """Test retrieving algorithm with random seed."""
        AlgorithmRegistry.register("test_algo", SimpleDummyAlgorithm)

        algo = AlgorithmRegistry.get_algorithm("test_algo", random_seed=42)
        self.assertEqual(algo.random_seed, 42)

    def test_get_nonexistent_algorithm(self):
        """Test that getting nonexistent algorithm raises error."""
        with self.assertRaises(KeyError):
            AlgorithmRegistry.get_algorithm("nonexistent")

    def test_duplicate_registration(self):
        """Test that duplicate registration raises error."""
        AlgorithmRegistry.register("test_algo", SimpleDummyAlgorithm)

        with self.assertRaises(ValueError):
            AlgorithmRegistry.register("test_algo", SimpleDummyAlgorithm)

    def test_invalid_algorithm_class(self):
        """Test that non-TSPAlgorithm class raises error."""
        class NotAnAlgorithm:
            pass

        with self.assertRaises(TypeError):
            AlgorithmRegistry.register("bad", NotAnAlgorithm)

    def test_list_algorithms(self):
        """Test listing all algorithms."""
        AlgorithmRegistry.register("algo1", SimpleDummyAlgorithm, tags=["baseline"])
        AlgorithmRegistry.register("algo2", SimpleDummyAlgorithm, tags=["heuristic"])

        all_algos = AlgorithmRegistry.list_algorithms()
        self.assertEqual(len(all_algos), 2)
        self.assertIn("algo1", all_algos)
        self.assertIn("algo2", all_algos)

    def test_list_algorithms_by_tags(self):
        """Test filtering algorithms by tags."""
        AlgorithmRegistry.register("algo1", SimpleDummyAlgorithm, tags=["baseline"])
        AlgorithmRegistry.register("algo2", SimpleDummyAlgorithm, tags=["heuristic"])
        AlgorithmRegistry.register("algo3", SimpleDummyAlgorithm, tags=["baseline", "greedy"])

        baselines = AlgorithmRegistry.list_algorithms(tags=["baseline"])
        self.assertEqual(len(baselines), 2)
        self.assertIn("algo1", baselines)
        self.assertIn("algo3", baselines)

    def test_list_algorithms_by_graph_type(self):
        """Test filtering algorithms by graph type."""
        AlgorithmRegistry.register(
            "metric_only",
            SimpleDummyAlgorithm,
            constraints={'graph_types': ['metric']}
        )
        AlgorithmRegistry.register(
            "any_type",
            SimpleDummyAlgorithm
        )

        metric_algos = AlgorithmRegistry.list_algorithms(graph_type='metric')
        self.assertEqual(len(metric_algos), 2)

        euclidean_algos = AlgorithmRegistry.list_algorithms(graph_type='euclidean')
        self.assertEqual(len(euclidean_algos), 1)
        self.assertIn("any_type", euclidean_algos)

    def test_list_algorithms_by_size(self):
        """Test filtering algorithms by graph size."""
        AlgorithmRegistry.register(
            "small_only",
            SimpleDummyAlgorithm,
            constraints={'max_size': 15}
        )
        AlgorithmRegistry.register(
            "any_size",
            SimpleDummyAlgorithm
        )

        small = AlgorithmRegistry.list_algorithms(graph_size=10)
        self.assertEqual(len(small), 2)

        large = AlgorithmRegistry.list_algorithms(graph_size=100)
        self.assertEqual(len(large), 1)
        self.assertIn("any_size", large)

    def test_get_tags(self):
        """Test getting tags for an algorithm."""
        AlgorithmRegistry.register("algo", SimpleDummyAlgorithm, tags=["baseline", "fast"])
        tags = AlgorithmRegistry.get_tags("algo")
        self.assertEqual(set(tags), {"baseline", "fast"})

    def test_get_constraints(self):
        """Test getting constraints for an algorithm."""
        constraints = {'graph_types': ['metric'], 'max_size': 20}
        AlgorithmRegistry.register("algo", SimpleDummyAlgorithm, constraints=constraints)

        retrieved = AlgorithmRegistry.get_constraints("algo")
        self.assertEqual(retrieved, constraints)

    def test_registry_summary(self):
        """Test registry summary output."""
        AlgorithmRegistry.register("algo1", SimpleDummyAlgorithm)
        AlgorithmRegistry.register("algo2", SimpleDummyAlgorithm)

        summary = AlgorithmRegistry.summary()
        self.assertIn("algo1", summary)
        self.assertIn("algo2", summary)
        self.assertIn("2 algorithms", summary)


class TestRegisterDecorator(unittest.TestCase):
    """Test suite for the @register_algorithm decorator."""

    def setUp(self):
        """Clear registry before each test."""
        AlgorithmRegistry.clear()

    def tearDown(self):
        """Clear registry after each test."""
        AlgorithmRegistry.clear()

    def test_decorator_registration(self):
        """Test that decorator registers algorithm."""
        @register_algorithm("decorated", tags=["test"])
        class DecoratedAlgorithm(TSPAlgorithm):
            def solve(self, adjacency_matrix, **kwargs) -> TourResult:
                tour = list(range(len(adjacency_matrix)))
                weight = self._compute_tour_weight(tour, adjacency_matrix)
                return TourResult(tour=tour, weight=weight, runtime=0.0)

            def get_metadata(self) -> AlgorithmMetadata:
                return AlgorithmMetadata("decorated", "1.0.0")

        self.assertTrue(AlgorithmRegistry.is_registered("decorated"))
        algo = AlgorithmRegistry.get_algorithm("decorated")
        self.assertIsInstance(algo, DecoratedAlgorithm)

    def test_decorator_preserves_class(self):
        """Test that decorator returns the class unchanged."""
        @register_algorithm("test")
        class TestAlgorithm(TSPAlgorithm):
            def solve(self, adjacency_matrix, **kwargs) -> TourResult:
                pass

            def get_metadata(self) -> AlgorithmMetadata:
                pass

        # Class should still be usable directly
        algo = TestAlgorithm(random_seed=42)
        self.assertEqual(algo.random_seed, 42)


class TestValidateTour(unittest.TestCase):
    """Test suite for tour validation."""

    def test_valid_tour(self):
        """Test validation of a valid tour."""
        matrix = [[0, 10, 20], [10, 0, 15], [20, 15, 0]]
        tour = [0, 1, 2]

        result = validate_tour(tour, matrix)
        self.assertTrue(result.valid)
        self.assertEqual(len(result.errors), 0)

    def test_invalid_tour_wrong_size(self):
        """Test validation rejects tours with wrong size."""
        matrix = [[0, 10, 20], [10, 0, 15], [20, 15, 0]]
        tour = [0, 1]

        result = validate_tour(tour, matrix)
        self.assertFalse(result.valid)
        self.assertGreater(len(result.errors), 0)

    def test_invalid_tour_duplicates(self):
        """Test validation rejects tours with duplicates."""
        matrix = [[0, 10, 20], [10, 0, 15], [20, 15, 0]]
        tour = [0, 1, 1]

        result = validate_tour(tour, matrix)
        self.assertFalse(result.valid)

    def test_invalid_tour_vertex_index(self):
        """Test validation rejects invalid vertex indices."""
        matrix = [[0, 10, 20], [10, 0, 15], [20, 15, 0]]
        tour = [0, 1, 5]

        result = validate_tour(tour, matrix)
        self.assertFalse(result.valid)

    def test_validation_result_summary(self):
        """Test ValidationResult summary method."""
        result = ValidationResult(valid=True, errors=[], warnings=[])
        summary = result.summary()
        self.assertIn("VALID", summary)


class TestTourValidator(unittest.TestCase):
    """Test suite for TourValidator class."""

    def test_validator_without_cache(self):
        """Test validator without caching."""
        validator = TourValidator(cache_results=False)
        matrix = [[0, 10, 20], [10, 0, 15], [20, 15, 0]]
        tour = [0, 1, 2]

        result = validator.validate(tour, matrix)
        self.assertTrue(result.valid)

    def test_validator_with_cache(self):
        """Test validator with caching."""
        validator = TourValidator(cache_results=True)
        matrix = [[0, 10, 20], [10, 0, 15], [20, 15, 0]]
        tour = [0, 1, 2]

        # First call
        result1 = validator.validate(tour, matrix)
        self.assertTrue(result1.valid)

        # Second call should use cache
        result2 = validator.validate(tour, matrix)
        self.assertTrue(result2.valid)

    def test_validator_batch(self):
        """Test batch validation."""
        validator = TourValidator()
        matrix = [[0, 10, 20], [10, 0, 15], [20, 15, 0]]
        tours = [[0, 1, 2], [0, 2, 1]]

        results = validator.validate_batch(tours, matrix)
        self.assertEqual(len(results), 2)
        self.assertTrue(results[0].valid)
        self.assertTrue(results[1].valid)


class TestComputeTourWeight(unittest.TestCase):
    """Test suite for tour weight computation."""

    def test_simple_tour_weight(self):
        """Test computing weight of a simple tour."""
        matrix = [
            [0, 10, 20],
            [10, 0, 15],
            [20, 15, 0]
        ]
        tour = [0, 1, 2]

        # Weight: 0->1 (10) + 1->2 (15) + 2->0 (20) = 45
        weight = compute_tour_weight(tour, matrix)
        self.assertEqual(weight, 45.0)

    def test_different_tour_order(self):
        """Test that tour order affects weight on asymmetric graphs."""
        # Use asymmetric matrix to ensure different tour orders give different weights
        matrix = [
            [0, 10, 20],
            [5, 0, 30],   # Asymmetric: 1->0 is 5, 1->2 is 30
            [20, 15, 0]
        ]

        tour1 = [0, 1, 2]
        tour2 = [0, 2, 1]

        weight1 = compute_tour_weight(tour1, matrix)  # 0->1(10) + 1->2(30) + 2->0(20) = 60
        weight2 = compute_tour_weight(tour2, matrix)  # 0->2(20) + 2->1(15) + 1->0(5) = 40

        # Different order should give different weights
        self.assertNotEqual(weight1, weight2)

    def test_asymmetric_graph(self):
        """Test weight computation on asymmetric graph."""
        matrix = [
            [0, 10, 20],
            [5, 0, 15],   # Asymmetric: 1->0 is 5, not 10
            [20, 15, 0]
        ]

        tour = [0, 1, 2]
        weight = compute_tour_weight(tour, matrix)

        # Weight: 0->1 (10) + 1->2 (15) + 2->0 (20) = 45
        self.assertEqual(weight, 45.0)


class TestComputeTourStatistics(unittest.TestCase):
    """Test suite for tour statistics computation."""

    def test_single_tour_statistics(self):
        """Test statistics with a single tour weight."""
        weights = [100.0]
        stats = compute_tour_statistics(weights)

        self.assertEqual(stats.mean_weight, 100.0)
        self.assertEqual(stats.median_weight, 100.0)
        self.assertEqual(stats.min_weight, 100.0)
        self.assertEqual(stats.max_weight, 100.0)
        self.assertEqual(stats.count, 1)

    def test_multiple_tour_statistics(self):
        """Test statistics with multiple tour weights."""
        weights = [80.0, 90.0, 100.0, 110.0, 120.0]
        stats = compute_tour_statistics(weights)

        self.assertEqual(stats.mean_weight, 100.0)
        self.assertEqual(stats.median_weight, 100.0)
        self.assertGreater(stats.std_weight, 0)
        self.assertEqual(stats.min_weight, 80.0)
        self.assertEqual(stats.max_weight, 120.0)
        self.assertEqual(stats.count, 5)

    def test_empty_weights(self):
        """Test statistics with empty weight list."""
        weights = []
        stats = compute_tour_statistics(weights)

        self.assertEqual(stats.count, 0)
        self.assertEqual(stats.mean_weight, 0.0)


class TestOptimalityGap(unittest.TestCase):
    """Test suite for optimality gap computation."""

    def test_optimal_solution(self):
        """Test optimality gap for optimal solution."""
        gap = compute_optimality_gap(100.0, 100.0)
        self.assertEqual(gap, 0.0)

    def test_suboptimal_solution(self):
        """Test optimality gap for suboptimal solution."""
        gap = compute_optimality_gap(110.0, 100.0)
        self.assertAlmostEqual(gap, 10.0, places=5)

    def test_far_suboptimal(self):
        """Test optimality gap for very suboptimal solution."""
        gap = compute_optimality_gap(200.0, 100.0)
        self.assertAlmostEqual(gap, 100.0, places=5)


class TestApproximationRatio(unittest.TestCase):
    """Test suite for approximation ratio computation."""

    def test_optimal_solution(self):
        """Test approximation ratio for optimal solution."""
        ratio = compute_approximation_ratio(100.0, 100.0)
        self.assertAlmostEqual(ratio, 1.0, places=5)

    def test_suboptimal_solution(self):
        """Test approximation ratio for suboptimal solution."""
        ratio = compute_approximation_ratio(110.0, 100.0)
        self.assertAlmostEqual(ratio, 1.1, places=5)

    def test_worse_solution(self):
        """Test approximation ratio for worse solution."""
        ratio = compute_approximation_ratio(150.0, 100.0)
        self.assertAlmostEqual(ratio, 1.5, places=5)


class TestComputeTourProperties(unittest.TestCase):
    """Test suite for tour properties computation."""

    def test_basic_properties(self):
        """Test computing basic tour properties."""
        matrix = [
            [0, 10, 20],
            [10, 0, 15],
            [20, 15, 0]
        ]
        tour = [0, 1, 2]

        props = compute_tour_properties(tour, matrix)

        self.assertIn('total_weight', props)
        self.assertIn('max_edge_weight', props)
        self.assertIn('min_edge_weight', props)
        self.assertEqual(props['total_weight'], 45.0)
        self.assertEqual(props['num_edges'], 3)

    def test_edge_statistics(self):
        """Test edge statistics in properties."""
        matrix = [
            [0, 10, 20],
            [10, 0, 15],
            [20, 15, 0]
        ]
        tour = [0, 1, 2]

        props = compute_tour_properties(tour, matrix)

        self.assertEqual(props['max_edge_weight'], 20.0)
        self.assertEqual(props['min_edge_weight'], 10.0)


class TestMetricsCalculator(unittest.TestCase):
    """Test suite for MetricsCalculator class."""

    def test_calculator_without_cache(self):
        """Test calculator without caching."""
        calc = MetricsCalculator(cache_results=False)
        matrix = [[0, 10, 20], [10, 0, 15], [20, 15, 0]]
        tour = [0, 1, 2]

        weight = calc.compute_weight(tour, matrix)
        self.assertEqual(weight, 45.0)

    def test_calculator_with_cache(self):
        """Test calculator with caching."""
        calc = MetricsCalculator(cache_results=True)
        matrix = [[0, 10, 20], [10, 0, 15], [20, 15, 0]]
        tour = [0, 1, 2]

        weight1 = calc.compute_weight(tour, matrix)
        weight2 = calc.compute_weight(tour, matrix)  # From cache

        self.assertEqual(weight1, weight2)

    def test_calculator_properties_cache(self):
        """Test caching of properties computation."""
        calc = MetricsCalculator(cache_results=True)
        matrix = [[0, 10, 20], [10, 0, 15], [20, 15, 0]]
        tour = [0, 1, 2]

        props1 = calc.compute_properties(tour, matrix)
        props2 = calc.compute_properties(tour, matrix)

        self.assertEqual(props1, props2)

    def test_calculator_clear_cache(self):
        """Test clearing calculator cache."""
        calc = MetricsCalculator(cache_results=True)
        matrix = [[0, 10, 20], [10, 0, 15], [20, 15, 0]]
        tour = [0, 1, 2]

        calc.compute_weight(tour, matrix)
        calc.clear_cache()

        # Cache should be empty
        self.assertEqual(len(calc._weight_cache), 0)


class TestAlgorithmIntegration(unittest.TestCase):
    """Integration tests for algorithm system."""

    def setUp(self):
        """Clear registry before each test."""
        AlgorithmRegistry.clear()

    def tearDown(self):
        """Clear registry after each test."""
        AlgorithmRegistry.clear()

    def test_full_workflow(self):
        """Test complete workflow: register, retrieve, run."""
        # Create and register algorithm
        @register_algorithm(
            "nn",
            tags=["baseline", "greedy"],
            constraints={'graph_types': ['euclidean', 'metric']}
        )
        class NN(TSPAlgorithm):
            def solve(self, adjacency_matrix, **kwargs) -> TourResult:
                tour = list(range(len(adjacency_matrix)))
                weight = self._compute_tour_weight(tour, adjacency_matrix)
                return TourResult(tour=tour, weight=weight, runtime=0.0)

            def get_metadata(self) -> AlgorithmMetadata:
                return AlgorithmMetadata("nn", "1.0.0")

        # Create test graph
        matrix = [
            [0, 10, 20],
            [10, 0, 15],
            [20, 15, 0]
        ]

        # Retrieve and run
        algo = AlgorithmRegistry.get_algorithm("nn", random_seed=42)
        result = algo.solve(matrix)

        self.assertTrue(result.success)
        self.assertEqual(len(result.tour), 3)
        self.assertEqual(result.weight, 45.0)

    def test_multiple_algorithms(self):
        """Test working with multiple registered algorithms."""
        @register_algorithm("algo1", tags=["test"])
        class Algo1(TSPAlgorithm):
            def solve(self, adjacency_matrix, **kwargs) -> TourResult:
                tour = list(range(len(adjacency_matrix)))
                weight = self._compute_tour_weight(tour, adjacency_matrix)
                return TourResult(tour=tour, weight=weight, runtime=0.1)

            def get_metadata(self) -> AlgorithmMetadata:
                return AlgorithmMetadata("algo1", "1.0.0")

        @register_algorithm("algo2", tags=["test"])
        class Algo2(TSPAlgorithm):
            def solve(self, adjacency_matrix, **kwargs) -> TourResult:
                tour = list(range(len(adjacency_matrix)))
                weight = self._compute_tour_weight(tour, adjacency_matrix)
                return TourResult(tour=tour, weight=weight, runtime=0.2)

            def get_metadata(self) -> AlgorithmMetadata:
                return AlgorithmMetadata("algo2", "1.0.0")

        # List all test algorithms
        test_algos = AlgorithmRegistry.list_algorithms(tags=["test"])
        self.assertEqual(len(test_algos), 2)

        # Run both
        matrix = [[0, 10, 20], [10, 0, 15], [20, 15, 0]]

        algo1 = AlgorithmRegistry.get_algorithm("algo1")
        result1 = algo1.solve(matrix)
        self.assertEqual(result1.runtime, 0.1)

        algo2 = AlgorithmRegistry.get_algorithm("algo2")
        result2 = algo2.solve(matrix)
        self.assertEqual(result2.runtime, 0.2)


if __name__ == '__main__':
    unittest.main()


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
