"""
Euclidean graph generator for TSP instances.

Generates graphs where vertices are points in 2D or 3D space
and edge weights are geometric distances.
"""

import random
import math
from typing import List, Tuple, Optional, Literal
import numpy as np


DistributionType = Literal['uniform', 'clustered', 'grid', 'radial']


class EuclideanGraphGenerator:
    """
    Generator for Euclidean TSP graphs.

    Creates graphs where vertices are points in Euclidean space
    and edge weights are geometric distances between points.
    """

    def __init__(self, random_seed: Optional[int] = None):
        """
        Initialize the generator.

        Args:
            random_seed: Random seed for reproducibility
        """
        self.random_seed = random_seed
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)

    def generate(
        self,
        num_vertices: int,
        dimensions: int = 2,
        coord_bounds: Tuple[float, float] = (0.0, 100.0),
        weight_range: Optional[Tuple[float, float]] = None,
        distribution: DistributionType = 'uniform',
        distribution_params: Optional[dict] = None
    ) -> Tuple[List[List[float]], List[Tuple[float, ...]]]:
        """
        Generate a Euclidean graph.

        Args:
            num_vertices: Number of vertices/points
            dimensions: Dimensionality (2 or 3)
            coord_bounds: (min, max) bounds for coordinate generation
            weight_range: Optional (min, max) for scaling edge weights
            distribution: Point distribution type
            distribution_params: Parameters specific to the distribution

        Returns:
            Tuple of (adjacency_matrix, coordinates)
        """
        if num_vertices < 1:
            raise ValueError("Number of vertices must be at least 1")

        if dimensions not in [2, 3]:
            raise ValueError("Dimensions must be 2 or 3")

        # Generate point coordinates
        coordinates = self._generate_coordinates(
            num_vertices, dimensions, coord_bounds,
            distribution, distribution_params or {}
        )

        # Scale coordinates to achieve desired weight range if specified
        # This preserves the Euclidean property (weights match geometric distances)
        if weight_range is not None:
            coordinates = self._scale_coordinates(coordinates, weight_range)

        # Compute pairwise distances from (possibly scaled) coordinates
        adjacency_matrix = self._compute_distance_matrix(coordinates)

        return adjacency_matrix, coordinates

    def _generate_coordinates(
        self,
        num_vertices: int,
        dimensions: int,
        coord_bounds: Tuple[float, float],
        distribution: DistributionType,
        distribution_params: dict
    ) -> List[Tuple[float, ...]]:
        """Generate vertex coordinates based on distribution type."""
        if distribution == 'uniform':
            return self._uniform_distribution(num_vertices, dimensions, coord_bounds)
        elif distribution == 'clustered':
            return self._clustered_distribution(num_vertices, dimensions, coord_bounds, distribution_params)
        elif distribution == 'grid':
            return self._grid_distribution(num_vertices, dimensions, coord_bounds)
        elif distribution == 'radial':
            return self._radial_distribution(num_vertices, dimensions, coord_bounds, distribution_params)
        else:
            raise ValueError(f"Unknown distribution type: {distribution}")

    def _uniform_distribution(
        self,
        num_vertices: int,
        dimensions: int,
        coord_bounds: Tuple[float, float]
    ) -> List[Tuple[float, ...]]:
        """Generate uniformly distributed random points."""
        min_coord, max_coord = coord_bounds
        coordinates = []

        for _ in range(num_vertices):
            point = tuple(
                random.uniform(min_coord, max_coord)
                for _ in range(dimensions)
            )
            coordinates.append(point)

        return coordinates

    def _clustered_distribution(
        self,
        num_vertices: int,
        dimensions: int,
        coord_bounds: Tuple[float, float],
        params: dict
    ) -> List[Tuple[float, ...]]:
        """
        Generate clustered points.

        Params:
            num_clusters: Number of clusters (default: 3)
            cluster_std: Standard deviation within clusters (default: 5.0)
        """
        num_clusters = params.get('num_clusters', 3)
        cluster_std = params.get('cluster_std', 5.0)

        min_coord, max_coord = coord_bounds
        coord_range = max_coord - min_coord

        # Generate cluster centers
        cluster_centers = []
        for _ in range(num_clusters):
            center = tuple(
                random.uniform(min_coord + 0.1 * coord_range, max_coord - 0.1 * coord_range)
                for _ in range(dimensions)
            )
            cluster_centers.append(center)

        # Assign vertices to clusters and generate points
        coordinates = []
        for i in range(num_vertices):
            cluster_idx = i % num_clusters
            center = cluster_centers[cluster_idx]

            # Generate point near cluster center
            point = tuple(
                np.clip(
                    np.random.normal(center[d], cluster_std),
                    min_coord,
                    max_coord
                )
                for d in range(dimensions)
            )
            coordinates.append(point)

        return coordinates

    def _grid_distribution(
        self,
        num_vertices: int,
        dimensions: int,
        coord_bounds: Tuple[float, float]
    ) -> List[Tuple[float, ...]]:
        """Generate points on a regular grid."""
        min_coord, max_coord = coord_bounds

        # Calculate grid dimensions
        points_per_dim = int(math.ceil(num_vertices ** (1.0 / dimensions)))

        coordinates = []
        spacing = (max_coord - min_coord) / (points_per_dim - 1) if points_per_dim > 1 else 0

        if dimensions == 2:
            for i in range(points_per_dim):
                for j in range(points_per_dim):
                    if len(coordinates) >= num_vertices:
                        break
                    x = min_coord + i * spacing
                    y = min_coord + j * spacing
                    coordinates.append((x, y))
                if len(coordinates) >= num_vertices:
                    break

        elif dimensions == 3:
            for i in range(points_per_dim):
                for j in range(points_per_dim):
                    for k in range(points_per_dim):
                        if len(coordinates) >= num_vertices:
                            break
                        x = min_coord + i * spacing
                        y = min_coord + j * spacing
                        z = min_coord + k * spacing
                        coordinates.append((x, y, z))
                    if len(coordinates) >= num_vertices:
                        break
                if len(coordinates) >= num_vertices:
                    break

        return coordinates[:num_vertices]

    def _radial_distribution(
        self,
        num_vertices: int,
        dimensions: int,
        coord_bounds: Tuple[float, float],
        params: dict
    ) -> List[Tuple[float, ...]]:
        """
        Generate points in a radial pattern.

        Params:
            num_rings: Number of concentric rings (default: 3)
        """
        num_rings = params.get('num_rings', 3)

        min_coord, max_coord = coord_bounds
        center = (max_coord + min_coord) / 2
        max_radius = (max_coord - min_coord) / 2

        coordinates = []
        points_per_ring = num_vertices // num_rings
        remainder = num_vertices % num_rings

        for ring_idx in range(num_rings):
            radius = max_radius * (ring_idx + 1) / num_rings
            points_in_ring = points_per_ring + (1 if ring_idx < remainder else 0)

            for i in range(points_in_ring):
                angle = 2 * math.pi * i / points_in_ring

                if dimensions == 2:
                    x = center + radius * math.cos(angle)
                    y = center + radius * math.sin(angle)
                    coordinates.append((x, y))
                elif dimensions == 3:
                    # Use spherical coordinates
                    phi = math.acos(1 - 2 * random.random())  # Polar angle
                    x = center + radius * math.sin(phi) * math.cos(angle)
                    y = center + radius * math.sin(phi) * math.sin(angle)
                    z = center + radius * math.cos(phi)
                    coordinates.append((x, y, z))

        return coordinates

    def _compute_distance_matrix(
        self,
        coordinates: List[Tuple[float, ...]]
    ) -> List[List[float]]:
        """Compute pairwise Euclidean distances."""
        n = len(coordinates)
        matrix = [[0.0] * n for _ in range(n)]

        for i in range(n):
            for j in range(i + 1, n):
                distance = self._euclidean_distance(coordinates[i], coordinates[j])
                matrix[i][j] = distance
                matrix[j][i] = distance  # Symmetric

        return matrix

    def _euclidean_distance(
        self,
        point1: Tuple[float, ...],
        point2: Tuple[float, ...]
    ) -> float:
        """Calculate Euclidean distance between two points."""
        if len(point1) != len(point2):
            raise ValueError("Points must have same dimensionality")

        squared_sum = sum((p1 - p2) ** 2 for p1, p2 in zip(point1, point2))
        return math.sqrt(squared_sum)

    def _scale_coordinates(
        self,
        coordinates: List[Tuple[float, ...]],
        weight_range: Tuple[float, float]
    ) -> List[Tuple[float, ...]]:
        """
        Scale coordinates so that resulting edge weights fit the desired range.

        This preserves the Euclidean property by scaling the coordinate space
        rather than the weights directly.

        Args:
            coordinates: Original coordinate list
            weight_range: Desired (min, max) for edge weights

        Returns:
            Scaled coordinates
        """
        if len(coordinates) < 2:
            # Can't scale with fewer than 2 points
            return coordinates

        # Compute current distance range
        n = len(coordinates)
        distances = []
        for i in range(n):
            for j in range(i + 1, n):
                dist = self._euclidean_distance(coordinates[i], coordinates[j])
                distances.append(dist)

        if not distances:
            return coordinates

        current_min = min(distances)
        current_max = max(distances)

        # Handle edge cases
        EPSILON = 1e-10
        if current_max < EPSILON or current_min == current_max:
            # All points coincident or all distances equal - can't meaningfully scale
            return coordinates

        # Calculate scaling factor to map distance range to target range
        # If current range is [current_min, current_max]
        # and we want [target_min, target_max]
        # We need: scaled_distance = scale_factor * distance
        # So: target_max = scale_factor * current_max
        # And: target_min = scale_factor * current_min
        # This means: scale_factor = (target_max - target_min) / (current_max - current_min)
        # BUT we also need to offset, so we use a linear transformation

        target_min, target_max = weight_range

        # Use linear transformation: new_dist = a * old_dist + b
        # where new_min = a * old_min + b = target_min
        # and   new_max = a * old_max + b = target_max
        # Solving: a = (target_max - target_min) / (current_max - current_min)
        #          b = target_min - a * current_min

        # But for coordinate scaling, we can only scale (multiply), not translate distances
        # So we use: scale_factor = target_max / current_max
        # This ensures max distance matches target_max
        # Then the min will be approximately target_min * (current_min / current_max)

        # Better approach: use the midpoint
        # Scale to match the range span
        scale_factor = (target_max - target_min) / (current_max - current_min)

        # Find centroid to scale around
        dimensions = len(coordinates[0])
        centroid = tuple(
            sum(coord[d] for coord in coordinates) / len(coordinates)
            for d in range(dimensions)
        )

        # Scale coordinates relative to centroid
        # This multiplies all distances by scale_factor
        scaled_coordinates = []
        for coord in coordinates:
            scaled_coord = tuple(
                centroid[d] + (coord[d] - centroid[d]) * scale_factor
                for d in range(dimensions)
            )
            scaled_coordinates.append(scaled_coord)

        # Now scaled distances are in range [current_min * scale_factor, current_max * scale_factor]
        # Which is [current_min * (target_max - target_min) / (current_max - current_min),
        #           current_max * (target_max - target_min) / (current_max - current_min)]
        # = [target_min - something, target_max - something]

        # We need to apply additional scaling to shift this to [target_min, target_max]
        # Recompute distances
        new_distances = []
        for i in range(n):
            for j in range(i + 1, n):
                dist = self._euclidean_distance(scaled_coordinates[i], scaled_coordinates[j])
                new_distances.append(dist)

        new_min = min(new_distances)
        new_max = max(new_distances)

        # Apply a final scaling factor to map [new_min, new_max] to [target_min, target_max]
        # We want new_max * final_scale = target_max
        final_scale = target_max / new_max if new_max > EPSILON else 1.0

        final_coordinates = []
        for coord in scaled_coordinates:
            final_coord = tuple(
                centroid[d] + (coord[d] - centroid[d]) * final_scale
                for d in range(dimensions)
            )
            final_coordinates.append(final_coord)

        return final_coordinates

    def _scale_weights(
        self,
        matrix: List[List[float]],
        weight_range: Tuple[float, float]
    ) -> List[List[float]]:
        """
        Scale edge weights to fit within specified range.

        Handles the coordinate scaling problem by mapping the actual
        distance range to the desired weight range.
        """
        target_min, target_max = weight_range

        # Find current min/max (excluding diagonal zeros)
        n = len(matrix)
        distances = [matrix[i][j] for i in range(n) for j in range(i + 1, n)]

        if not distances:
            return matrix

        current_min = min(distances)
        current_max = max(distances)

        # Handle edge case where all distances are the same
        if current_min == current_max:
            # Set all edges to middle of target range
            mid_weight = (target_min + target_max) / 2
            scaled_matrix = [[0.0] * n for _ in range(n)]
            for i in range(n):
                for j in range(i + 1, n):
                    scaled_matrix[i][j] = mid_weight
                    scaled_matrix[j][i] = mid_weight
            return scaled_matrix

        # Handle edge case where points are identical
        EPSILON = 1e-10
        if current_max < EPSILON:
            # All points are essentially at the same location
            # Use minimum weight for all edges
            scaled_matrix = [[0.0] * n for _ in range(n)]
            for i in range(n):
                for j in range(i + 1, n):
                    scaled_matrix[i][j] = target_min
                    scaled_matrix[j][i] = target_min
            return scaled_matrix

        # Linear scaling
        scaled_matrix = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                if i != j:
                    normalized = (matrix[i][j] - current_min) / (current_max - current_min)
                    scaled_weight = target_min + normalized * (target_max - target_min)
                    scaled_matrix[i][j] = scaled_weight
                    scaled_matrix[j][i] = scaled_weight

        return scaled_matrix


def generate_euclidean_graph(
    num_vertices: int,
    dimensions: int = 2,
    coord_bounds: Tuple[float, float] = (0.0, 100.0),
    weight_range: Optional[Tuple[float, float]] = None,
    distribution: DistributionType = 'uniform',
    distribution_params: Optional[dict] = None,
    random_seed: Optional[int] = None
) -> Tuple[List[List[float]], List[Tuple[float, ...]]]:
    """
    Convenience function to generate a Euclidean graph.

    Args:
        num_vertices: Number of vertices
        dimensions: 2D or 3D (2 or 3)
        coord_bounds: (min, max) coordinate bounds
        weight_range: Optional (min, max) for edge weights
        distribution: Type of point distribution
        distribution_params: Distribution-specific parameters
        random_seed: Random seed for reproducibility

    Returns:
        Tuple of (adjacency_matrix, coordinates)
    """
    generator = EuclideanGraphGenerator(random_seed=random_seed)
    return generator.generate(
        num_vertices=num_vertices,
        dimensions=dimensions,
        coord_bounds=coord_bounds,
        weight_range=weight_range,
        distribution=distribution,
        distribution_params=distribution_params
    )
