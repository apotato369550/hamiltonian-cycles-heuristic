"""
Core algorithm interface and data structures for TSP benchmarking.

This module defines the fundamental abstractions for all TSP algorithms:
- TourResult: dataclass for algorithm output
- AlgorithmMetadata: dataclass for algorithm properties
- TSPAlgorithm: abstract base class for all algorithms

All algorithms must implement the TSPAlgorithm interface to be registered
and used in the benchmarking pipeline.
"""

from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
import time
import numpy as np


@dataclass
class TourResult:
    """
    Complete result of running a TSP algorithm on a graph.

    Attributes:
        tour: List of vertex indices forming a Hamiltonian cycle
        weight: Total cost (sum of edge weights in the tour)
        runtime: Wall-clock runtime in seconds
        metadata: Algorithm-specific metadata (anchor vertices, parameters used, etc.)
        success: Whether algorithm completed successfully
        error_message: Error description if success=False
    """
    tour: List[int]
    weight: float
    runtime: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    success: bool = True
    error_message: str = ""

    def __post_init__(self):
        """Validate tour result after construction."""
        if self.success and self.tour:
            if len(self.tour) < 3:
                raise ValueError(f"Tour must have at least 3 vertices, got {len(self.tour)}")
            if len(set(self.tour)) != len(self.tour):
                raise ValueError("Tour contains duplicate vertices")

        if self.weight < 0:
            raise ValueError(f"Tour weight must be non-negative, got {self.weight}")

        if self.runtime < 0:
            raise ValueError(f"Runtime must be non-negative, got {self.runtime}")


@dataclass
class AlgorithmMetadata:
    """
    Metadata describing an algorithm's properties and capabilities.

    Attributes:
        name: Algorithm name/identifier (e.g., 'nearest_neighbor')
        version: Version string for reproducibility
        parameters: Current parameter values
        applicability_constraints: Dict describing which graph types work
        description: Human-readable description
    """
    name: str
    version: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    applicability_constraints: Dict[str, Any] = field(default_factory=dict)
    description: str = ""

    def can_handle_graph_type(self, graph_type: str) -> bool:
        """Check if algorithm can handle given graph type."""
        allowed = self.applicability_constraints.get('graph_types', None)
        if allowed is None:
            return True  # No constraints
        return graph_type in allowed

    def can_handle_graph_size(self, size: int) -> bool:
        """Check if algorithm can handle given graph size."""
        min_size = self.applicability_constraints.get('min_size', None)
        max_size = self.applicability_constraints.get('max_size', None)

        if min_size is not None and size < min_size:
            return False
        if max_size is not None and size > max_size:
            return False

        return True


class TSPAlgorithm(ABC):
    """
    Abstract base class for all TSP algorithms.

    All algorithms must implement:
    - solve(): Run the algorithm and return a TourResult
    - get_metadata(): Return algorithm metadata

    Subclasses can optionally override is_applicable() for custom constraints.
    """

    def __init__(self, random_seed: Optional[int] = None):
        """
        Initialize algorithm with optional random seed.

        Args:
            random_seed: Random seed for reproducibility (None for non-deterministic)
        """
        self.random_seed = random_seed
        if random_seed is not None:
            np.random.seed(random_seed)

    @abstractmethod
    def solve(self, adjacency_matrix: List[List[float]], **kwargs) -> TourResult:
        """
        Solve the TSP on the given graph.

        Args:
            adjacency_matrix: 2D list/array of edge weights
            **kwargs: Algorithm-specific parameters

        Returns:
            TourResult with tour, weight, runtime, and metadata
        """
        pass

    @abstractmethod
    def get_metadata(self) -> AlgorithmMetadata:
        """
        Return metadata describing this algorithm.

        Returns:
            AlgorithmMetadata object
        """
        pass

    def is_applicable(self, graph_type: str, graph_size: int) -> bool:
        """
        Check if algorithm is applicable to given graph.

        Args:
            graph_type: Type of graph ('euclidean', 'metric', 'random', 'quasi_metric')
            graph_size: Number of vertices

        Returns:
            True if algorithm can handle this graph type and size
        """
        metadata = self.get_metadata()
        return (
            metadata.can_handle_graph_type(graph_type) and
            metadata.can_handle_graph_size(graph_size)
        )

    @staticmethod
    def _track_runtime(func):
        """
        Decorator to track wall-clock runtime of algorithm execution.

        Usage in solve() method:
            @TSPAlgorithm._track_runtime
            def _solve_impl(self, adjacency_matrix, **kwargs) -> TourResult:
                # implementation
        """
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()

            if isinstance(result, TourResult):
                result.runtime = end_time - start_time

            return result
        return wrapper

    def _compute_tour_weight(self, tour: List[int], adjacency_matrix: List[List[float]]) -> float:
        """
        Compute total weight of a tour.

        Args:
            tour: List of vertex indices
            adjacency_matrix: 2D adjacency matrix

        Returns:
            Sum of edge weights in the tour
        """
        if len(tour) < 2:
            return 0.0

        total_weight = 0.0
        for i in range(len(tour)):
            current = tour[i]
            next_vertex = tour[(i + 1) % len(tour)]  # Wrap around to start

            if current >= len(adjacency_matrix) or next_vertex >= len(adjacency_matrix):
                raise ValueError(f"Invalid vertex index in tour: {current} or {next_vertex}")

            total_weight += adjacency_matrix[current][next_vertex]

        return total_weight

    def _validate_tour_structure(self, tour: List[int], num_vertices: int) -> Tuple[bool, str]:
        """
        Validate basic tour structure.

        Args:
            tour: List of vertex indices
            num_vertices: Expected number of vertices

        Returns:
            Tuple of (is_valid, error_message)
        """
        if len(tour) != num_vertices:
            return False, f"Tour has {len(tour)} vertices, expected {num_vertices}"

        if len(set(tour)) != num_vertices:
            return False, "Tour contains duplicate vertices"

        for v in tour:
            if v < 0 or v >= num_vertices:
                return False, f"Invalid vertex index: {v}"

        return True, ""

    def _create_failure_result(self, error_message: str) -> TourResult:
        """
        Create a failure TourResult with appropriate error message.

        Args:
            error_message: Description of failure

        Returns:
            TourResult with success=False
        """
        return TourResult(
            tour=[],
            weight=float('inf'),
            runtime=0.0,
            metadata={},
            success=False,
            error_message=error_message
        )
