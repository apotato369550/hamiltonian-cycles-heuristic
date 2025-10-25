"""
Graph property verification system.

Provides standalone verification of graph properties independent
of generation, with support for both fast sampling and exhaustive checking.
"""

import random
import math
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import numpy as np


@dataclass
class VerificationResult:
    """Result of a graph property verification."""
    property_name: str
    passed: bool
    details: Dict[str, Any]
    errors: List[str]

    def __repr__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return f"VerificationResult({self.property_name}: {status})"


class GraphVerifier:
    """
    Comprehensive graph property verifier.

    Supports both fast sampling-based checks and exhaustive verification.
    """

    def __init__(self, fast_mode: bool = True, sample_size: int = 1000):
        """
        Initialize the verifier.

        Args:
            fast_mode: If True, use sampling for large graphs
            sample_size: Number of samples for fast mode checks
        """
        self.fast_mode = fast_mode
        self.sample_size = sample_size

    def verify_all(
        self,
        adjacency_matrix: List[List[float]],
        coordinates: Optional[List[Tuple[float, ...]]] = None,
        claimed_properties: Optional[Dict[str, Any]] = None
    ) -> List[VerificationResult]:
        """
        Verify all graph properties.

        Args:
            adjacency_matrix: Graph adjacency matrix
            coordinates: Optional vertex coordinates
            claimed_properties: Properties claimed by generator

        Returns:
            List of verification results
        """
        results = []

        # Verify symmetry
        results.append(self.verify_symmetry(adjacency_matrix))

        # Verify metricity
        results.append(self.verify_metricity(adjacency_matrix))

        # Verify weight statistics
        results.append(self.verify_weight_statistics(adjacency_matrix))

        # Verify Euclidean distances if coordinates provided
        if coordinates is not None:
            results.append(self.verify_euclidean_distances(adjacency_matrix, coordinates))

        # Compare with claimed properties if provided
        if claimed_properties is not None:
            results.append(self.verify_claimed_properties(adjacency_matrix, claimed_properties))

        return results

    def verify_symmetry(self, adjacency_matrix: List[List[float]]) -> VerificationResult:
        """
        Verify that the graph is symmetric.

        Checks that weight(i,j) == weight(j,i) for all i,j.
        """
        n = len(adjacency_matrix)
        errors = []
        asymmetric_pairs = []

        TOLERANCE = 1e-9  # Floating-point comparison tolerance

        for i in range(n):
            for j in range(i + 1, n):
                if abs(adjacency_matrix[i][j] - adjacency_matrix[j][i]) > TOLERANCE:
                    asymmetric_pairs.append((i, j, adjacency_matrix[i][j], adjacency_matrix[j][i]))
                    if len(errors) < 10:  # Limit error messages
                        errors.append(
                            f"Asymmetric edge ({i},{j}): "
                            f"forward={adjacency_matrix[i][j]:.6f}, "
                            f"backward={adjacency_matrix[j][i]:.6f}"
                        )

        passed = len(asymmetric_pairs) == 0

        return VerificationResult(
            property_name="symmetry",
            passed=passed,
            details={
                'is_symmetric': passed,
                'asymmetric_count': len(asymmetric_pairs),
                'total_edges': n * (n - 1) // 2
            },
            errors=errors
        )

    def verify_metricity(self, adjacency_matrix: List[List[float]]) -> VerificationResult:
        """
        Verify that the graph satisfies the triangle inequality.

        For large graphs in fast mode, samples triplets randomly.
        """
        n = len(adjacency_matrix)
        errors = []
        violations = []

        if self.fast_mode and n > 50:
            # Sample random triplets
            total_checked = min(self.sample_size, n * (n - 1) * (n - 2) // 6)
            checked = 0

            while checked < total_checked:
                i, j, k = sorted(random.sample(range(n), 3))
                checked += 1

                # Check all three triangle inequalities
                triplet_violations = self._check_triplet(adjacency_matrix, i, j, k)
                violations.extend(triplet_violations)

                if triplet_violations and len(errors) < 10:
                    for v in triplet_violations:
                        errors.append(
                            f"Triangle inequality violated at ({v['i']},{v['j']},{v['k']}): "
                            f"{v['direct']:.2f} > {v['indirect']:.2f}"
                        )

            details = {
                'is_metric': len(violations) == 0,
                'triplets_checked': total_checked,
                'violations': len(violations),
                'sampling_mode': True
            }
        else:
            # Exhaustive check
            total_triplets = 0
            for i in range(n):
                for j in range(i + 1, n):
                    for k in range(j + 1, n):
                        total_triplets += 1
                        triplet_violations = self._check_triplet(adjacency_matrix, i, j, k)
                        violations.extend(triplet_violations)

                        if triplet_violations and len(errors) < 10:
                            for v in triplet_violations:
                                errors.append(
                                    f"Triangle inequality violated at ({v['i']},{v['j']},{v['k']}): "
                                    f"{v['direct']:.2f} > {v['indirect']:.2f}"
                                )

            details = {
                'is_metric': len(violations) == 0,
                'triplets_checked': total_triplets,
                'violations': len(violations),
                'sampling_mode': False
            }

        # Calculate metricity score
        if details['triplets_checked'] > 0:
            details['metricity_score'] = 1.0 - (len(violations) / details['triplets_checked'])
        else:
            details['metricity_score'] = 1.0

        passed = len(violations) == 0

        return VerificationResult(
            property_name="metricity",
            passed=passed,
            details=details,
            errors=errors
        )

    def _check_triplet(
        self,
        matrix: List[List[float]],
        i: int,
        j: int,
        k: int
    ) -> List[Dict[str, Any]]:
        """
        Check triangle inequality for a single triplet.

        Returns list of violations (empty if none).
        """
        violations = []
        TOLERANCE = 1e-9  # Floating-point tolerance

        # Check i-j <= i-k + k-j
        if matrix[i][j] > matrix[i][k] + matrix[k][j] + TOLERANCE:
            violations.append({
                'i': i, 'j': j, 'k': k,
                'direct': matrix[i][j],
                'indirect': matrix[i][k] + matrix[k][j],
                'violation': matrix[i][j] - (matrix[i][k] + matrix[k][j])
            })

        # Check i-k <= i-j + j-k
        if matrix[i][k] > matrix[i][j] + matrix[j][k] + TOLERANCE:
            violations.append({
                'i': i, 'j': k, 'k': j,
                'direct': matrix[i][k],
                'indirect': matrix[i][j] + matrix[j][k],
                'violation': matrix[i][k] - (matrix[i][j] + matrix[j][k])
            })

        # Check j-k <= i-j + i-k
        if matrix[j][k] > matrix[i][j] + matrix[i][k] + TOLERANCE:
            violations.append({
                'i': j, 'j': k, 'k': i,
                'direct': matrix[j][k],
                'indirect': matrix[i][j] + matrix[i][k],
                'violation': matrix[j][k] - (matrix[i][j] + matrix[i][k])
            })

        return violations

    def verify_weight_statistics(
        self,
        adjacency_matrix: List[List[float]]
    ) -> VerificationResult:
        """Verify weight distribution statistics."""
        n = len(adjacency_matrix)
        errors = []

        # Extract all edge weights (upper triangle only)
        weights = []
        for i in range(n):
            for j in range(i + 1, n):
                weights.append(adjacency_matrix[i][j])

        if not weights:
            return VerificationResult(
                property_name="weight_statistics",
                passed=True,
                details={'vertices': n, 'edges': 0},
                errors=["Graph has no edges"]
            )

        # Calculate statistics
        weight_min = min(weights)
        weight_max = max(weights)
        weight_mean = sum(weights) / len(weights)
        weight_std = self._calculate_std(weights)

        # Check for invalid weights
        for i, w in enumerate(weights):
            if math.isnan(w) or math.isinf(w):
                errors.append(f"Invalid weight at index {i}: {w}")
            if w < 0:
                errors.append(f"Negative weight at index {i}: {w}")

        details = {
            'weight_range': (weight_min, weight_max),
            'weight_mean': weight_mean,
            'weight_std': weight_std,
            'total_edges': len(weights),
            'unique_weights': len(set(weights))
        }

        passed = len(errors) == 0

        return VerificationResult(
            property_name="weight_statistics",
            passed=passed,
            details=details,
            errors=errors
        )

    def verify_euclidean_distances(
        self,
        adjacency_matrix: List[List[float]],
        coordinates: List[Tuple[float, ...]]
    ) -> VerificationResult:
        """
        Verify that edge weights match Euclidean distances from coordinates.

        Args:
            adjacency_matrix: Graph adjacency matrix
            coordinates: Vertex coordinates
        """
        n = len(adjacency_matrix)
        errors = []
        mismatches = []

        TOLERANCE = 1e-6  # Tolerance for floating-point comparison

        for i in range(n):
            for j in range(i + 1, n):
                # Calculate Euclidean distance
                expected_distance = self._euclidean_distance(coordinates[i], coordinates[j])
                actual_weight = adjacency_matrix[i][j]

                if abs(expected_distance - actual_weight) > TOLERANCE:
                    mismatches.append((i, j, expected_distance, actual_weight))
                    if len(errors) < 10:
                        errors.append(
                            f"Distance mismatch at ({i},{j}): "
                            f"expected={expected_distance:.6f}, "
                            f"actual={actual_weight:.6f}"
                        )

        passed = len(mismatches) == 0

        return VerificationResult(
            property_name="euclidean_distances",
            passed=passed,
            details={
                'matches_coordinates': passed,
                'mismatch_count': len(mismatches),
                'total_edges': n * (n - 1) // 2
            },
            errors=errors
        )

    def verify_claimed_properties(
        self,
        adjacency_matrix: List[List[float]],
        claimed: Dict[str, Any]
    ) -> VerificationResult:
        """
        Verify that actual properties match claimed properties.

        Args:
            adjacency_matrix: Graph adjacency matrix
            claimed: Dictionary of claimed properties
        """
        errors = []
        discrepancies = {}

        # Verify size
        if 'size' in claimed:
            actual_size = len(adjacency_matrix)
            if actual_size != claimed['size']:
                errors.append(f"Size mismatch: claimed={claimed['size']}, actual={actual_size}")
                discrepancies['size'] = (claimed['size'], actual_size)

        # Verify weight range
        if 'weight_range' in claimed:
            weights = [adjacency_matrix[i][j] for i in range(len(adjacency_matrix))
                      for j in range(i + 1, len(adjacency_matrix))]
            if weights:
                actual_range = (min(weights), max(weights))
                claimed_range = tuple(claimed['weight_range'])

                TOLERANCE = 1e-6
                if (abs(actual_range[0] - claimed_range[0]) > TOLERANCE or
                    abs(actual_range[1] - claimed_range[1]) > TOLERANCE):
                    errors.append(
                        f"Weight range mismatch: claimed={claimed_range}, actual={actual_range}"
                    )
                    discrepancies['weight_range'] = (claimed_range, actual_range)

        passed = len(errors) == 0

        return VerificationResult(
            property_name="claimed_properties",
            passed=passed,
            details={'discrepancies': discrepancies},
            errors=errors
        )

    def _euclidean_distance(
        self,
        point1: Tuple[float, ...],
        point2: Tuple[float, ...]
    ) -> float:
        """Calculate Euclidean distance between two points."""
        return math.sqrt(sum((p1 - p2) ** 2 for p1, p2 in zip(point1, point2)))

    def _calculate_std(self, values: List[float]) -> float:
        """Calculate standard deviation."""
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return math.sqrt(variance)


def verify_graph_properties(
    adjacency_matrix: List[List[float]],
    coordinates: Optional[List[Tuple[float, ...]]] = None,
    fast_mode: bool = True
):
    """
    Convenience function to verify graph properties and return GraphProperties object.

    Args:
        adjacency_matrix: Graph adjacency matrix
        coordinates: Optional vertex coordinates
        fast_mode: Whether to use fast sampling mode

    Returns:
        GraphProperties object from graph_instance module
    """
    from .graph_instance import GraphProperties

    verifier = GraphVerifier(fast_mode=fast_mode)

    n = len(adjacency_matrix)
    weights = [adjacency_matrix[i][j] for i in range(n) for j in range(i + 1, n)]

    if not weights:
        return GraphProperties(
            is_metric=True,
            is_symmetric=True,
            weight_range=(0.0, 0.0),
            weight_mean=0.0,
            weight_std=0.0,
            density=0.0
        )

    # Verify symmetry
    symmetry_result = verifier.verify_symmetry(adjacency_matrix)

    # Verify metricity
    metricity_result = verifier.verify_metricity(adjacency_matrix)

    # Calculate statistics
    weight_min = min(weights)
    weight_max = max(weights)
    weight_mean = sum(weights) / len(weights)
    weight_std = verifier._calculate_std(weights)

    return GraphProperties(
        is_metric=metricity_result.passed,
        is_symmetric=symmetry_result.passed,
        weight_range=(weight_min, weight_max),
        weight_mean=weight_mean,
        weight_std=weight_std,
        density=1.0,  # Complete graphs have density 1.0
        metricity_score=metricity_result.details.get('metricity_score'),
        triangle_violations=metricity_result.details.get('violations', 0)
    )


def print_verification_report(results: List[VerificationResult]) -> None:
    """
    Print a formatted verification report.

    Args:
        results: List of verification results
    """
    print("=" * 70)
    print("GRAPH VERIFICATION REPORT")
    print("=" * 70)

    all_passed = all(r.passed for r in results)

    for result in results:
        status = "✓ PASS" if result.passed else "✗ FAIL"
        print(f"\n{result.property_name.upper()}: {status}")

        # Print details
        for key, value in result.details.items():
            print(f"  {key}: {value}")

        # Print errors if any
        if result.errors:
            print(f"\n  Errors ({len(result.errors)}):")
            for error in result.errors[:5]:  # Limit to first 5 errors
                print(f"    - {error}")
            if len(result.errors) > 5:
                print(f"    ... and {len(result.errors) - 5} more")

    print("\n" + "=" * 70)
    overall_status = "ALL CHECKS PASSED" if all_passed else "SOME CHECKS FAILED"
    print(f"Overall: {overall_status}")
    print("=" * 70)
