"""
Tour validation functions for TSP benchmarking.

Provides comprehensive validation of tours and quality metrics computation.
"""

from dataclasses import dataclass
from typing import List, Dict, Tuple, Any, Optional
import numpy as np


@dataclass
class ValidationResult:
    """Result of validating a tour."""
    valid: bool
    errors: List[str]
    warnings: List[str]

    def summary(self) -> str:
        """Get human-readable summary."""
        lines = []
        if self.valid:
            lines.append("Tour is VALID")
        else:
            lines.append("Tour is INVALID")

        if self.errors:
            lines.append("Errors:")
            for error in self.errors:
                lines.append(f"  - {error}")

        if self.warnings:
            lines.append("Warnings:")
            for warning in self.warnings:
                lines.append(f"  - {warning}")

        return "\n".join(lines)


def validate_tour(
    tour: List[int],
    adjacency_matrix: List[List[float]]
) -> ValidationResult:
    """
    Validate that a tour is a proper Hamiltonian cycle.

    Args:
        tour: List of vertex indices
        adjacency_matrix: 2D adjacency matrix

    Returns:
        ValidationResult with validity status and any errors/warnings
    """
    errors = []
    warnings = []

    num_vertices = len(adjacency_matrix)

    # Check tour length
    if len(tour) != num_vertices:
        errors.append(
            f"Tour has {len(tour)} vertices, expected {num_vertices}"
        )
        return ValidationResult(valid=False, errors=errors, warnings=warnings)

    # Check for duplicates
    if len(set(tour)) != num_vertices:
        errors.append("Tour contains duplicate vertices")

    # Check vertex validity
    for v in tour:
        if not isinstance(v, (int, np.integer)):
            errors.append(f"Invalid vertex type: {type(v)}")
            break

        if v < 0 or v >= num_vertices:
            errors.append(f"Vertex index {v} out of range [0, {num_vertices-1}]")

    # If structural errors exist, stop here
    if errors:
        return ValidationResult(valid=False, errors=errors, warnings=warnings)

    # Check edges exist
    for i in range(len(tour)):
        current = int(tour[i])
        next_vertex = int(tour[(i + 1) % len(tour)])

        try:
            weight = adjacency_matrix[current][next_vertex]
            if weight < 0:
                errors.append(
                    f"Negative weight edge {current}->{next_vertex}: {weight}"
                )
        except (IndexError, TypeError):
            errors.append(
                f"Cannot access edge {current}->{next_vertex} in adjacency matrix"
            )

    # Return final result
    is_valid = len(errors) == 0
    return ValidationResult(valid=is_valid, errors=errors, warnings=warnings)


def validate_tour_constraints(
    tour: List[int],
    constraints: Dict[str, Any]
) -> ValidationResult:
    """
    Validate tour against specific constraints.

    Args:
        tour: List of vertex indices
        constraints: Dict of constraints to check
                     - 'must_include_edges': List[(u, v)] edges that must be in tour
                     - 'must_start_with': vertex that tour must start with
                     - 'required_subpath': List of vertices that must appear consecutively

    Returns:
        ValidationResult with validation status
    """
    errors = []
    warnings = []

    # Check must_include_edges
    if 'must_include_edges' in constraints:
        required_edges = constraints['must_include_edges']
        for u, v in required_edges:
            # Check if edge u->v exists in tour
            found = False
            for i in range(len(tour)):
                if tour[i] == u and tour[(i + 1) % len(tour)] == v:
                    found = True
                    break
            if not found:
                errors.append(f"Required edge {u}->{v} not in tour")

    # Check must_start_with
    if 'must_start_with' in constraints:
        start_vertex = constraints['must_start_with']
        if len(tour) > 0 and tour[0] != start_vertex:
            errors.append(f"Tour must start with {start_vertex}, starts with {tour[0]}")

    # Check required_subpath
    if 'required_subpath' in constraints:
        subpath = constraints['required_subpath']
        # Check if subpath appears consecutively in tour
        found = False
        for i in range(len(tour)):
            match = True
            for j, vertex in enumerate(subpath):
                if tour[(i + j) % len(tour)] != vertex:
                    match = False
                    break
            if match:
                found = True
                break

        if not found:
            errors.append(f"Required subpath {subpath} not found in tour")

    is_valid = len(errors) == 0
    return ValidationResult(valid=is_valid, errors=errors, warnings=warnings)


class TourValidator:
    """Class for batch validation with optional caching."""

    def __init__(self, cache_results: bool = False):
        """
        Initialize validator.

        Args:
            cache_results: Whether to cache validation results
        """
        self.cache_results = cache_results
        self._cache: Dict[Tuple, ValidationResult] = {}

    def validate(
        self,
        tour: List[int],
        adjacency_matrix: List[List[float]],
        use_cache: bool = True
    ) -> ValidationResult:
        """
        Validate a tour, using cache if enabled.

        Args:
            tour: Tour to validate
            adjacency_matrix: Adjacency matrix
            use_cache: Whether to use cache (if enabled)

        Returns:
            ValidationResult
        """
        if self.cache_results and use_cache:
            tour_tuple = tuple(tour)
            if tour_tuple in self._cache:
                return self._cache[tour_tuple]

        result = validate_tour(tour, adjacency_matrix)

        if self.cache_results:
            self._cache[tuple(tour)] = result

        return result

    def validate_batch(
        self,
        tours: List[List[int]],
        adjacency_matrix: List[List[float]]
    ) -> List[ValidationResult]:
        """
        Validate multiple tours.

        Args:
            tours: List of tours
            adjacency_matrix: Adjacency matrix

        Returns:
            List of ValidationResult objects
        """
        return [self.validate(tour, adjacency_matrix) for tour in tours]

    def clear_cache(self) -> None:
        """Clear the validation cache."""
        self._cache = {}


def validate_adjacency_matrix(
    matrix: List[List[float]]
) -> ValidationResult:
    """
    Validate that a matrix is a valid adjacency matrix for TSP.

    Args:
        matrix: Adjacency matrix

    Returns:
        ValidationResult
    """
    errors = []
    warnings = []

    # Check it's square
    if not all(len(row) == len(matrix) for row in matrix):
        errors.append("Adjacency matrix is not square")
        return ValidationResult(valid=False, errors=errors, warnings=warnings)

    # Check size is at least 3
    if len(matrix) < 3:
        errors.append(f"Adjacency matrix too small: {len(matrix)} vertices")

    # Check diagonal is zero
    for i in range(len(matrix)):
        if matrix[i][i] != 0:
            warnings.append(f"Non-zero diagonal element: matrix[{i}][{i}] = {matrix[i][i]}")

    # Check weights are non-negative
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            if matrix[i][j] < 0:
                errors.append(f"Negative weight: matrix[{i}][{j}] = {matrix[i][j]}")

    # Check for NaN or Inf
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            val = matrix[i][j]
            if np.isnan(val) or np.isinf(val):
                errors.append(f"Invalid value at [{i}][{j}]: {val}")

    is_valid = len(errors) == 0
    return ValidationResult(valid=is_valid, errors=errors, warnings=warnings)
