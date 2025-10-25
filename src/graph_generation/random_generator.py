"""
Random graph generator for TSP instances.

Generates graphs with random edge weights without structural constraints.
These serve as baseline "chaotic" graphs for algorithm testing.
"""

import random
from typing import List, Tuple, Optional, Literal, Dict
import numpy as np
from scipy import stats


DistributionType = Literal['uniform', 'normal', 'exponential', 'bimodal', 'power_law']


class RandomGraphGenerator:
    """
    Generator for random TSP graphs without structural constraints.

    Creates graphs where edge weights are drawn independently from
    specified distributions. These graphs typically do not satisfy
    the triangle inequality.
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
        weight_range: Tuple[float, float] = (1.0, 100.0),
        distribution: DistributionType = 'uniform',
        is_symmetric: bool = True,
        distribution_params: Optional[Dict] = None
    ) -> List[List[float]]:
        """
        Generate a random graph.

        Args:
            num_vertices: Number of vertices
            weight_range: (min, max) for edge weights
            distribution: Type of weight distribution
            is_symmetric: Whether weight(i,j) == weight(j,i)
            distribution_params: Additional parameters for the distribution

        Returns:
            Adjacency matrix
        """
        if num_vertices < 1:
            raise ValueError("Number of vertices must be at least 1")

        params = distribution_params or {}
        matrix = [[0.0] * num_vertices for _ in range(num_vertices)]

        if is_symmetric:
            # Generate upper triangle, mirror to lower triangle
            for i in range(num_vertices):
                for j in range(i + 1, num_vertices):
                    weight = self._sample_weight(weight_range, distribution, params)
                    matrix[i][j] = weight
                    matrix[j][i] = weight
        else:
            # Generate all edges independently
            for i in range(num_vertices):
                for j in range(num_vertices):
                    if i != j:
                        weight = self._sample_weight(weight_range, distribution, params)
                        matrix[i][j] = weight

        return matrix

    def _sample_weight(
        self,
        weight_range: Tuple[float, float],
        distribution: DistributionType,
        params: Dict
    ) -> float:
        """Sample a single edge weight from the specified distribution."""
        min_weight, max_weight = weight_range

        if distribution == 'uniform':
            return self._uniform_sample(min_weight, max_weight)

        elif distribution == 'normal':
            return self._normal_sample(min_weight, max_weight, params)

        elif distribution == 'exponential':
            return self._exponential_sample(min_weight, max_weight, params)

        elif distribution == 'bimodal':
            return self._bimodal_sample(min_weight, max_weight, params)

        elif distribution == 'power_law':
            return self._power_law_sample(min_weight, max_weight, params)

        else:
            raise ValueError(f"Unknown distribution: {distribution}")

    def _uniform_sample(self, min_weight: float, max_weight: float) -> float:
        """Sample from uniform distribution."""
        return random.uniform(min_weight, max_weight)

    def _normal_sample(
        self,
        min_weight: float,
        max_weight: float,
        params: Dict
    ) -> float:
        """
        Sample from normal distribution.

        Params:
            mean_ratio: Where to center the mean (0.0 to 1.0), default 0.5
            std_ratio: Standard deviation as ratio of range, default 0.2
        """
        mean_ratio = params.get('mean_ratio', 0.5)
        std_ratio = params.get('std_ratio', 0.2)

        range_size = max_weight - min_weight
        mean = min_weight + mean_ratio * range_size
        std = std_ratio * range_size

        # Sample and clip to range
        value = np.random.normal(mean, std)
        return np.clip(value, min_weight, max_weight)

    def _exponential_sample(
        self,
        min_weight: float,
        max_weight: float,
        params: Dict
    ) -> float:
        """
        Sample from exponential distribution.

        Params:
            scale_ratio: Scale parameter as ratio of range, default 0.3
        """
        scale_ratio = params.get('scale_ratio', 0.3)
        range_size = max_weight - min_weight
        scale = scale_ratio * range_size

        # Sample and shift/clip to range
        value = min_weight + np.random.exponential(scale)
        return min(value, max_weight)

    def _bimodal_sample(
        self,
        min_weight: float,
        max_weight: float,
        params: Dict
    ) -> float:
        """
        Sample from bimodal distribution (mixture of two normals).

        Params:
            mode1_ratio: Position of first mode (0.0 to 1.0), default 0.25
            mode2_ratio: Position of second mode (0.0 to 1.0), default 0.75
            std_ratio: Standard deviation ratio, default 0.1
            mix_ratio: Probability of first mode, default 0.5
        """
        mode1_ratio = params.get('mode1_ratio', 0.25)
        mode2_ratio = params.get('mode2_ratio', 0.75)
        std_ratio = params.get('std_ratio', 0.1)
        mix_ratio = params.get('mix_ratio', 0.5)

        range_size = max_weight - min_weight
        std = std_ratio * range_size

        # Choose which mode to sample from
        if random.random() < mix_ratio:
            mean = min_weight + mode1_ratio * range_size
        else:
            mean = min_weight + mode2_ratio * range_size

        value = np.random.normal(mean, std)
        return np.clip(value, min_weight, max_weight)

    def _power_law_sample(
        self,
        min_weight: float,
        max_weight: float,
        params: Dict
    ) -> float:
        """
        Sample from power law distribution.

        Params:
            alpha: Power law exponent, default 2.0 (higher = more skewed)
        """
        alpha = params.get('alpha', 2.0)

        # Use inverse transform sampling for power law
        u = random.random()

        # Power law CDF: F(x) = 1 - (x/xmin)^(-alpha+1)
        # Inverse: x = xmin * (1-u)^(-1/(alpha-1))
        if alpha == 1.0:
            # Special case: uniform in log space
            value = min_weight * np.exp(u * np.log(max_weight / min_weight))
        else:
            # General power law
            x_min_normalized = 1.0
            x_max_normalized = max_weight / min_weight

            # Sample in normalized space
            if alpha > 1:
                term = (1 - u) * x_min_normalized**(1-alpha) + u * x_max_normalized**(1-alpha)
                x_normalized = term**(1/(1-alpha))
            else:
                x_normalized = x_min_normalized * (1 - u)**(1/(alpha-1))

            value = min_weight * x_normalized

        return np.clip(value, min_weight, max_weight)


class StructuredRandomGraphGenerator:
    """
    Generator for random graphs with local structure.

    Generates graphs where edge weights depend on vertex proximity
    or other structural features, while still being primarily random.
    """

    def __init__(self, random_seed: Optional[int] = None):
        """Initialize the generator."""
        self.random_seed = random_seed
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)

    def generate_distance_based(
        self,
        num_vertices: int,
        weight_range: Tuple[float, float] = (1.0, 100.0),
        locality_factor: float = 0.5
    ) -> List[List[float]]:
        """
        Generate random graph where nearby vertices (by index) have correlated weights.

        Args:
            num_vertices: Number of vertices
            weight_range: (min, max) edge weights
            locality_factor: 0.0 = fully random, 1.0 = strong locality

        Returns:
            Adjacency matrix
        """
        min_weight, max_weight = weight_range
        range_size = max_weight - min_weight
        matrix = [[0.0] * num_vertices for _ in range(num_vertices)]

        for i in range(num_vertices):
            for j in range(i + 1, num_vertices):
                # Distance in index space
                index_distance = abs(i - j)
                max_distance = num_vertices - 1

                # Base random weight
                base_weight = random.uniform(min_weight, max_weight)

                # Apply locality bias
                if locality_factor > 0:
                    # Closer vertices (smaller index distance) get lower weights
                    distance_ratio = index_distance / max_distance
                    locality_bias = locality_factor * distance_ratio * range_size
                    weight = base_weight + locality_bias
                    weight = np.clip(weight, min_weight, max_weight)
                else:
                    weight = base_weight

                matrix[i][j] = weight
                matrix[j][i] = weight

        return matrix

    def generate_cluster_based(
        self,
        num_vertices: int,
        weight_range: Tuple[float, float] = (1.0, 100.0),
        num_clusters: int = 3,
        intra_cluster_weight_ratio: float = 0.3
    ) -> List[List[float]]:
        """
        Generate random graph with cluster structure.

        Edges within clusters have weights from lower part of range,
        edges between clusters from upper part.

        Args:
            num_vertices: Number of vertices
            weight_range: (min, max) edge weights
            num_clusters: Number of clusters
            intra_cluster_weight_ratio: Ratio of range for intra-cluster edges

        Returns:
            Adjacency matrix
        """
        min_weight, max_weight = weight_range
        range_size = max_weight - min_weight

        # Assign vertices to clusters
        cluster_size = num_vertices // num_clusters
        vertex_clusters = []
        for i in range(num_vertices):
            cluster_id = min(i // cluster_size, num_clusters - 1)
            vertex_clusters.append(cluster_id)

        matrix = [[0.0] * num_vertices for _ in range(num_vertices)]

        for i in range(num_vertices):
            for j in range(i + 1, num_vertices):
                if vertex_clusters[i] == vertex_clusters[j]:
                    # Intra-cluster edge: lower weights
                    local_max = min_weight + intra_cluster_weight_ratio * range_size
                    weight = random.uniform(min_weight, local_max)
                else:
                    # Inter-cluster edge: higher weights
                    local_min = min_weight + intra_cluster_weight_ratio * range_size
                    weight = random.uniform(local_min, max_weight)

                matrix[i][j] = weight
                matrix[j][i] = weight

        return matrix


def generate_random_graph(
    num_vertices: int,
    weight_range: Tuple[float, float] = (1.0, 100.0),
    distribution: DistributionType = 'uniform',
    is_symmetric: bool = True,
    distribution_params: Optional[Dict] = None,
    random_seed: Optional[int] = None
) -> List[List[float]]:
    """
    Convenience function to generate a random graph.

    Args:
        num_vertices: Number of vertices
        weight_range: (min, max) edge weights
        distribution: Weight distribution type
        is_symmetric: Whether graph should be symmetric
        distribution_params: Distribution-specific parameters
        random_seed: Random seed for reproducibility

    Returns:
        Adjacency matrix
    """
    generator = RandomGraphGenerator(random_seed=random_seed)
    return generator.generate(
        num_vertices=num_vertices,
        weight_range=weight_range,
        distribution=distribution,
        is_symmetric=is_symmetric,
        distribution_params=distribution_params
    )


def analyze_metricity(adjacency_matrix: List[List[float]]) -> Dict[str, float]:
    """
    Analyze how metric a random graph is.

    Returns statistics about triangle inequality violations.

    Args:
        adjacency_matrix: Graph to analyze

    Returns:
        Dictionary with metricity statistics
    """
    n = len(adjacency_matrix)
    total_triplets = 0
    satisfied_triplets = 0
    violations = []

    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                # Check all three triangle inequalities
                triplet_satisfied = True

                # Check i-j <= i-k + k-j
                if adjacency_matrix[i][j] > adjacency_matrix[i][k] + adjacency_matrix[k][j]:
                    triplet_satisfied = False
                    violations.append(
                        adjacency_matrix[i][j] - (adjacency_matrix[i][k] + adjacency_matrix[k][j])
                    )

                # Check i-k <= i-j + j-k
                if adjacency_matrix[i][k] > adjacency_matrix[i][j] + adjacency_matrix[j][k]:
                    triplet_satisfied = False
                    violations.append(
                        adjacency_matrix[i][k] - (adjacency_matrix[i][j] + adjacency_matrix[j][k])
                    )

                # Check j-k <= i-j + i-k
                if adjacency_matrix[j][k] > adjacency_matrix[i][j] + adjacency_matrix[i][k]:
                    triplet_satisfied = False
                    violations.append(
                        adjacency_matrix[j][k] - (adjacency_matrix[i][j] + adjacency_matrix[i][k])
                    )

                total_triplets += 1
                if triplet_satisfied:
                    satisfied_triplets += 1

    metricity_score = satisfied_triplets / total_triplets if total_triplets > 0 else 1.0

    return {
        'metricity_score': metricity_score,
        'total_triplets': total_triplets,
        'satisfied_triplets': satisfied_triplets,
        'violation_count': len(violations),
        'mean_violation': sum(violations) / len(violations) if violations else 0.0,
        'max_violation': max(violations) if violations else 0.0
    }
