"""
Held-Karp dynamic programming algorithm for exact TSP solution.

Implements the O(n^2 * 2^n) Held-Karp algorithm for small TSP instances.
Only applicable to graphs with n <= 20 vertices due to exponential complexity.
"""

from typing import List, Optional, Tuple, Dict
import time
import numpy as np
from .base import TSPAlgorithm, TourResult, AlgorithmMetadata
from .registry import register_algorithm


@register_algorithm(
    "held_karp_exact",
    tags=["exact", "optimal"],
    constraints={"max_size": 20}
)
class HeldKarpAlgorithm(TSPAlgorithm):
    """
    Held-Karp algorithm for computing optimal TSP tour.

    Time complexity: O(n^2 * 2^n)
    Space complexity: O(n * 2^n)

    Only practical for n <= 18 (2^18 = 262144 is manageable).
    """

    def solve(self, adjacency_matrix: List[List[float]], **kwargs) -> TourResult:
        """
        Solve TSP optimally using Held-Karp dynamic programming.

        Args:
            adjacency_matrix: 2D adjacency matrix
            **kwargs: timeout_seconds (default 60)

        Returns:
            TourResult with optimal tour or failure if too large
        """
        timeout = kwargs.get('timeout_seconds', 60)
        start_time = time.time()

        num_vertices = len(adjacency_matrix)

        # Check applicability
        if num_vertices < 3:
            return self._create_failure_result(
                f"Graph too small: {num_vertices} vertices (minimum 3)"
            )

        if num_vertices > 20:
            return self._create_failure_result(
                f"Graph too large: {num_vertices} vertices (maximum 20)"
            )

        # Convert to numpy for efficiency
        matrix = np.array(adjacency_matrix, dtype=float)

        try:
            tour, weight = self._held_karp(matrix, timeout, start_time)

            runtime = time.time() - start_time

            return TourResult(
                tour=tour,
                weight=float(weight),
                runtime=runtime,
                metadata={'algorithm': 'held_karp'},
                success=True
            )

        except TimeoutError:
            runtime = time.time() - start_time
            return TourResult(
                tour=[],
                weight=float('inf'),
                runtime=runtime,
                metadata={},
                success=False,
                error_message=f"Timeout after {runtime:.1f}s (limit: {timeout}s)"
            )

        except Exception as e:
            runtime = time.time() - start_time
            return self._create_failure_result(f"Error: {str(e)}")

    def get_metadata(self) -> AlgorithmMetadata:
        """Return algorithm metadata."""
        return AlgorithmMetadata(
            name="held_karp_exact",
            version="1.0.0",
            description="Held-Karp optimal algorithm",
            applicability_constraints={"max_size": 20}
        )

    def _held_karp(
        self,
        matrix: np.ndarray,
        timeout: float,
        start_time: float
    ) -> Tuple[List[int], float]:
        """
        Core Held-Karp algorithm using dynamic programming.

        Returns:
            (tour, weight) tuple with optimal solution
        """
        n = len(matrix)

        # dp[mask][i] = minimum cost to visit vertices in mask, ending at i, starting from 0
        dp = [[float('inf')] * n for _ in range(1 << n)]
        parent = [[-1] * n for _ in range(1 << n)]

        # Base case: starting at vertex 0
        dp[1][0] = 0

        # Build up solutions for larger subsets
        for mask in range(1, 1 << n):
            # Check timeout
            if mask % 1000 == 0 and time.time() - start_time > timeout:
                raise TimeoutError()

            for u in range(n):
                if not (mask & (1 << u)):
                    continue

                if dp[mask][u] == float('inf'):
                    continue

                # Try extending to next vertex
                for v in range(n):
                    if mask & (1 << v):
                        continue  # Already visited

                    new_mask = mask | (1 << v)
                    new_cost = dp[mask][u] + matrix[u][v]

                    if new_cost < dp[new_mask][v]:
                        dp[new_mask][v] = new_cost
                        parent[new_mask][v] = u

        # Find minimum cost to visit all vertices and return to start
        full_mask = (1 << n) - 1
        best_cost = float('inf')
        best_last = -1

        for i in range(1, n):
            if dp[full_mask][i] == float('inf'):
                continue

            cost = dp[full_mask][i] + matrix[i][0]

            if cost < best_cost:
                best_cost = cost
                best_last = i

        if best_last < 0:
            raise ValueError("No valid tour found")

        # Reconstruct tour
        tour = []
        mask = full_mask
        current = best_last

        while current >= 0:
            tour.append(current)

            prev = parent[mask][current]
            if prev < 0:
                break

            mask ^= (1 << current)  # Remove current from mask
            current = prev

        tour.reverse()

        return tour, best_cost

    def _generate_subsets(self, n: int, size: int) -> List[int]:
        """Generate all bitmask subsets containing exactly 'size' vertices."""
        subsets = []

        def backtrack(mask: int, count: int, start: int):
            if count == size:
                subsets.append(mask)
                return

            for i in range(start, n):
                backtrack(mask | (1 << i), count + 1, i + 1)

        backtrack(0, 0, 0)
        return subsets
