import random
# Function to create a complete graph with weighted edges.
# Parameters:
# vertices - Number of vertices in the graph (int).
# weight_range - Tuple specifying the range of weights for edges (default: (1, 10)).
# replace - Boolean indicating whether to sample weights with or without replacement (default: False).
# strict - Boolean indicating whether to enforce strict uniqueness of weights (default: False).
# seed - An integer seed to base randomly generated numbers on. 
def generate_complete_graph(vertices, weight_range=(1, 10), replace=False, strict=False, seed=None):
    """
    Generate a complete graph represented as an adjacency matrix with random edge weights.
    This function creates a complete undirected graph where every vertex is connected to every other vertex.
    Edge weights are randomly assigned based on the specified parameters.
    Args:
        vertices (int): Number of vertices in the graph.
        weight_range (tuple, optional): Range of possible edge weights as (min, max). Defaults to (1, 10).
        replace (bool, optional): If True, allows weight values to be reused. If False, attempts to use unique weights.
            Defaults to False.
        strict (bool, optional): When replace=False, if True enforces unique weights by raising an error if there aren't
            enough unique values available. If False, allows duplicates if necessary. Defaults to False.
        seed (int, optional): Random seed for reproducible results. Defaults to None.
    Returns:
        list: A symmetric adjacency matrix representing the complete graph, where each element [i][j]
            contains the weight of the edge between vertices i and j.
    Raises:
        ValueError: If strict=True and there aren't enough unique weights available in the weight_range
            to assign to all edges.
    Examples:
        >>> graph = generate_complete_graph(4, weight_range=(1,5))
        >>> graph = generate_complete_graph(3, replace=True, seed=42)
    """
    # If a seed is provided, set the random seed for reproducibility.
    if seed is not None:
        random.seed(seed)

    # Initialize an empty adjacency matrix for the graph.
    graph = [[0] * vertices for _ in range(vertices)]

    # If weights are to be assigned with replacement.
    if replace:
        for i in range(vertices):
            for j in range(i + 1, vertices):
                # Randomly assign a weight to the edge (i, j) and mirror it for (j, i).
                weight = random.randint(*weight_range)
                graph[i][j] = graph[j][i] = weight
    else:
        # Create a list of possible weights within the specified range.
        possible_weights = list(range(weight_range[0], weight_range[1] + 1))
        total_edges = vertices * (vertices - 1) // 2  # Total number of edges in a complete graph.

        if strict:
            # Strict mode: Ensure there are enough unique weights for all edges.
            if len(possible_weights) < total_edges:
                raise ValueError(f"Not enough unique weights for the number of edges. "
                                 f"Must need at least {total_edges} total edges for a graph with {vertices} vertices.")
            # Randomly sample unique weights for all edges.
            selected_weights = random.sample(possible_weights, total_edges)
        else:
            # Relaxed mode: Allow duplicate weights by expanding the pool of possible weights.
            selected_weights = random.sample(possible_weights * ((total_edges // len(possible_weights)) + 1), total_edges)

        # Assign weights to edges in the adjacency matrix.
        weight_index = 0
        for i in range(vertices):
            for j in range(i + 1, vertices):
                graph[i][j] = graph[j][i] = selected_weights[weight_index]
                weight_index += 1

    return graph