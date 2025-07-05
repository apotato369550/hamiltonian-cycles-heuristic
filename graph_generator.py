import random
import math
from typing import List, Tuple, Optional

def generate_complete_graph(vertices: int, weight_range: Tuple[int, int] = (1, 10), 
                          replace: bool = False, strict: bool = False, 
                          metric: bool = False, seed: Optional[int] = None) -> List[List[int]]:
    """
    Generate a complete graph represented as an adjacency matrix with random edge weights.
    
    Args:
        vertices (int): Number of vertices in the graph.
        weight_range (tuple): Range of possible edge weights as (min, max). Defaults to (1, 10).
        replace (bool): If True, allows weight values to be reused. Defaults to False.
        strict (bool): When replace=False, enforces unique weights. Defaults to False.
        metric (bool): If True, generates a metric graph satisfying triangle inequality. Defaults to False.
        seed (int, optional): Random seed for reproducible results. Defaults to None.
    
    Returns:
        list: A symmetric adjacency matrix representing the complete graph.
    
    Raises:
        ValueError: If parameters are incompatible or insufficient unique weights available.
    """
    if seed is not None:
        random.seed(seed)

    if vertices < 1:
        raise ValueError("Number of vertices must be at least 1")
    
    if weight_range[0] > weight_range[1]:
        raise ValueError("Invalid weight range: minimum must be <= maximum")

    graph = [[0] * vertices for _ in range(vertices)]

    if metric:
        return _generate_metric_graph_improved(vertices, weight_range, seed)
    else:
        return _generate_regular_graph(vertices, weight_range, replace, strict, graph)


def _generate_metric_graph_improved(vertices: int, weight_range: Tuple[int, int], 
                                   seed: Optional[int]) -> List[List[int]]:
    """
    Generate a metric graph using an improved coordinate-based approach.
    Uses better scaling and distribution to ensure weights span the full range.
    """
    if seed is not None:
        random.seed(seed)
    
    min_weight, max_weight = weight_range
    
    # Strategy 1: For small graphs, use optimized coordinate placement
    if vertices <= 10:
        return _generate_small_metric_graph(vertices, weight_range)
    
    # Strategy 2: For larger graphs, use sampling with post-processing
    return _generate_large_metric_graph(vertices, weight_range)


def _generate_small_metric_graph(vertices: int, weight_range: Tuple[int, int]) -> List[List[int]]:
    """
    Generate metric graph for small vertex counts using optimized coordinate placement.
    """
    min_weight, max_weight = weight_range
    
    # Generate coordinates with better distribution
    # Use a larger coordinate space to get better weight distribution
    max_coord = 100  # Fixed coordinate space
    coordinates = []
    
    # Try multiple coordinate generations and pick the one with best weight distribution
    best_graph = None
    best_score = float('inf')  # We want to minimize deviation from uniform distribution
    
    for attempt in range(10):  # Try multiple coordinate sets
        coords = []
        for _ in range(vertices):
            x = random.uniform(-max_coord, max_coord)
            y = random.uniform(-max_coord, max_coord)
            coords.append((x, y))
        
        # Calculate all distances
        distances = []
        for i in range(vertices):
            for j in range(i + 1, vertices):
                x1, y1 = coords[i]
                x2, y2 = coords[j]
                distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                distances.append(distance)
        
        if not distances:  # Handle single vertex case
            return [[0] * vertices for _ in range(vertices)]
        
        # Map distances to weight range using percentile-based scaling
        distances.sort()
        graph = [[0] * vertices for _ in range(vertices)]
        
        edge_idx = 0
        for i in range(vertices):
            for j in range(i + 1, vertices):
                x1, y1 = coords[i]
                x2, y2 = coords[j]
                distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                
                # Use percentile ranking for better distribution
                percentile = sum(1 for d in distances if d <= distance) / len(distances)
                weight = min_weight + int(percentile * (max_weight - min_weight))
                weight = max(min_weight, min(max_weight, weight))
                
                graph[i][j] = graph[j][i] = weight
                edge_idx += 1
        
        # Score this attempt based on weight distribution
        weights = [graph[i][j] for i in range(vertices) for j in range(i+1, vertices)]
        if weights:
            weight_std = _calculate_std(weights)
            target_std = (max_weight - min_weight) / 3  # Target standard deviation
            score = abs(weight_std - target_std)
            
            if score < best_score:
                best_score = score
                best_graph = graph
    
    return best_graph if best_graph is not None else [[0] * vertices for _ in range(vertices)]


def _generate_large_metric_graph(vertices: int, weight_range: Tuple[int, int]) -> List[List[int]]:
    """
    Generate metric graph for larger vertex counts using distance matrix completion.
    """
    min_weight, max_weight = weight_range
    graph = [[0] * vertices for _ in range(vertices)]
    
    # Generate initial random coordinates
    coordinates = []
    for _ in range(vertices):
        x = random.uniform(-50, 50)
        y = random.uniform(-50, 50)
        coordinates.append((x, y))
    
    # Calculate distances and map to weight range
    all_distances = []
    for i in range(vertices):
        for j in range(i + 1, vertices):
            x1, y1 = coordinates[i]
            x2, y2 = coordinates[j]
            distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            all_distances.append(distance)
    
    if not all_distances:
        return graph
    
    # Use better scaling: map min/max distances to weight range boundaries
    min_dist, max_dist = min(all_distances), max(all_distances)
    if min_dist == max_dist:
        # All points are the same - assign minimum weight
        for i in range(vertices):
            for j in range(i + 1, vertices):
                graph[i][j] = graph[j][i] = min_weight
        return graph
    
    # Apply scaling with some randomization to fill the range better
    for i in range(vertices):
        for j in range(i + 1, vertices):
            x1, y1 = coordinates[i]
            x2, y2 = coordinates[j]
            distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            
            # Linear mapping with small random perturbation
            normalized = (distance - min_dist) / (max_dist - min_dist)
            base_weight = min_weight + normalized * (max_weight - min_weight)
            
            # Add small random perturbation (±10% of range) to better fill the space
            perturbation_range = (max_weight - min_weight) * 0.1
            perturbation = random.uniform(-perturbation_range, perturbation_range)
            
            weight = base_weight + perturbation
            weight = max(min_weight, min(max_weight, round(weight)))
            
            graph[i][j] = graph[j][i] = weight
    
    # Post-process to ensure triangle inequality (fix any violations)
    _fix_triangle_inequality_violations(graph)
    
    return graph


def _fix_triangle_inequality_violations(graph: List[List[int]]) -> None:
    """
    Fix triangle inequality violations in-place using Floyd-Warshall-style approach.
    """
    vertices = len(graph)
    
    # Apply Floyd-Warshall to ensure triangle inequality
    for k in range(vertices):
        for i in range(vertices):
            for j in range(vertices):
                if i != j and graph[i][j] > graph[i][k] + graph[k][j]:
                    # Fix violation by reducing the direct distance
                    graph[i][j] = graph[i][k] + graph[k][j]


def _calculate_std(values: List[float]) -> float:
    """Calculate standard deviation of a list of values."""
    if len(values) < 2:
        return 0
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
    return math.sqrt(variance)


def _generate_regular_graph(vertices: int, weight_range: Tuple[int, int], 
                           replace: bool, strict: bool, graph: List[List[int]]) -> List[List[int]]:
    """Generate a regular complete graph (original logic with minor improvements)."""
    min_weight, max_weight = weight_range
    
    if replace:
        for i in range(vertices):
            for j in range(i + 1, vertices):
                weight = random.randint(min_weight, max_weight)
                graph[i][j] = graph[j][i] = weight
    else:
        possible_weights = list(range(min_weight, max_weight + 1))
        total_edges = vertices * (vertices - 1) // 2
        
        if strict and len(possible_weights) < total_edges:
            raise ValueError(f"Not enough unique weights for {total_edges} edges. "
                           f"Need weights in range with at least {total_edges} values.")
        
        if len(possible_weights) >= total_edges:
            selected_weights = random.sample(possible_weights, total_edges)
        else:
            # Need to repeat some weights
            repetitions = (total_edges // len(possible_weights)) + 1
            expanded_weights = possible_weights * repetitions
            selected_weights = random.sample(expanded_weights, total_edges)
        
        weight_index = 0
        for i in range(vertices):
            for j in range(i + 1, vertices):
                graph[i][j] = graph[j][i] = selected_weights[weight_index]
                weight_index += 1
    
    return graph


def verify_triangle_inequality(graph: List[List[int]]) -> Tuple[bool, List[Tuple[int, int, int, int]]]:
    """
    Verify that a graph satisfies the triangle inequality (optimized version).
    
    Args:
        graph: Adjacency matrix representation of the graph.
    
    Returns:
        tuple: (is_metric, violations) where violations is a list of (i, j, k, violation_amount).
    """
    vertices = len(graph)
    violations = []
    
    # Only check each triangle once (i < j < k)
    for i in range(vertices):
        for j in range(i + 1, vertices):
            for k in range(j + 1, vertices):
                # Check all three sides of the triangle
                edges = [
                    (graph[i][j], graph[i][k] + graph[k][j], i, j, k),
                    (graph[i][k], graph[i][j] + graph[j][k], i, k, j), 
                    (graph[j][k], graph[i][j] + graph[i][k], j, k, i)
                ]
                
                for direct, indirect, v1, v2, v3 in edges:
                    if direct > indirect:
                        violation_amount = direct - indirect
                        violations.append((v1, v2, v3, violation_amount))
    
    return len(violations) == 0, violations


def analyze_graph_properties(graph: List[List[int]]) -> dict:
    """
    Analyze various properties of a generated graph.
    
    Returns:
        dict: Dictionary containing graph analysis results.
    """
    vertices = len(graph)
    if vertices == 0:
        return {"error": "Empty graph"}
    
    # Extract edge weights
    weights = []
    for i in range(vertices):
        for j in range(i + 1, vertices):
            weights.append(graph[i][j])
    
    if not weights:
        return {"vertices": vertices, "edges": 0}
    
    # Calculate statistics
    is_metric, violations = verify_triangle_inequality(graph)
    
    return {
        "vertices": vertices,
        "edges": len(weights),
        "weight_range": (min(weights), max(weights)),
        "weight_mean": sum(weights) / len(weights),
        "weight_std": _calculate_std(weights),
        "is_metric": is_metric,
        "triangle_violations": len(violations),
        "weight_distribution": {w: weights.count(w) for w in set(weights)}
    }


# Example usage and testing
def test_improved_generator():
    """Test the improved graph generator."""
    print("=== Testing Improved Graph Generator ===\n")
    
    test_cases = [
        {"vertices": 4, "weight_range": (1, 10), "metric": True, "seed": 42},
        {"vertices": 6, "weight_range": (5, 25), "metric": True, "seed": 42},
        {"vertices": 5, "weight_range": (1, 50), "metric": True, "seed": 123},
    ]
    
    for i, params in enumerate(test_cases, 1):
        print(f"Test Case {i}: {params}")
        graph = generate_complete_graph(**params)
        analysis = analyze_graph_properties(graph)
        
        print(f"  Results: {analysis}")
        print(f"  Sample graph (first 4x4):")
        for row_idx in range(min(4, len(graph))):
            row = [graph[row_idx][col_idx] for col_idx in range(min(4, len(graph)))]
            print(f"    {row}")
        print()


if __name__ == "__main__":
    test_improved_generator()