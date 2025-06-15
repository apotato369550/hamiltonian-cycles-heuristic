import random
import itertools
import json
import csv
import functools
from hamiltonian import hamiltonian_cycle_heuristic

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

# Utility function to print the adjacency matrix of a graph.
def print_graph(graph):
    """
    Prints the adjacency matrix representation of a graph in a formatted way.

    This function takes a 2D list (matrix) representing a graph and prints it with a header
    and footer for better visualization.

    Args:
        graph (List[List[int]]): A 2D list representing the adjacency matrix of the graph,
                                where graph[i][j] indicates if there is an edge between vertices i and j.

    Returns:
        None

    Example:
        >>> graph = [[0, 1, 0], [1, 0, 1], [0, 1, 0]]
        >>> print_graph(graph)
        ----- PRINTING GRAPH -----
        [0, 1, 0]
        [1, 0, 1]
        [0, 1, 0]
        ----- GRAPH PRINTED -----
    """
    print("----- PRINTING GRAPH -----")
    for row in graph:
        print(row)
    print("----- GRAPH PRINTED -----")

# graph - list of lists, each list containing an int
# vertex - starting vertex. int representing the index of the sublist
# status: works as normal.
def greedy_algorithm(graph, vertex):
    vertices = len(graph)
    visited = [False] * vertices
    cycle = [vertex]
    total_weight = 0
    visited[vertex] = True

    current_vertex = vertex

    for _ in range(vertices - 1):
        next_vertex = -1
        lowest_weight = float('inf')

        for j in range(vertices):
            if not visited[j] and graph[current_vertex][j] < lowest_weight:
                lowest_weight = graph[current_vertex][j]
                next_vertex = j

        if next_vertex == -1:
            raise ValueError("No unvisited vertices found. Graph may be disconnected.")

        cycle.append(next_vertex)
        total_weight += graph[current_vertex][next_vertex]
        visited[next_vertex] = True
        current_vertex = next_vertex

    total_weight += graph[current_vertex][vertex]  # Return to start
    cycle.append(vertex)

    return cycle, total_weight

def calculate_cycle_weight(graph, cycle):
    return sum(graph[cycle[i]][cycle[(i + 1) % len(cycle)]] for i in range(len(cycle)))

# status: lacks printing the last vertex.
def find_optimal_cycle(graph, vertex):
    n = len(graph)
    vertices = list(range(n))
    vertices.remove(vertex)
    lowest_weight = float('inf')
    best_cycle = None

    for permutation in itertools.permutations(vertices):
        cycle = [vertex] + list(permutation) + [vertex]
        weight = calculate_cycle_weight(graph, cycle + [vertex])
        if weight < lowest_weight:
            lowest_weight = weight
            best_cycle = cycle

    return best_cycle, lowest_weight

# Fixing the anchor-based heuristic: ensure all vertices are visited before returning to start

def find_optimal_cycle_held_karp(graph, start_vertex):
    """
    Find the optimal Hamiltonian cycle (TSP solution) using the Held-Karp dynamic programming algorithm.
    
    Parameters:
    - graph: Adjacency matrix represented as a list of lists with edge weights
    - start_vertex: The vertex to start and end the cycle at
    
    Returns:
    - A tuple containing (optimal_cycle, optimal_weight)
    """
    n = len(graph)
    
    # Handle trivial cases
    if n <= 1:
        return [start_vertex, start_vertex], 0
    if n == 2:
        return [start_vertex, 1-start_vertex, start_vertex], graph[start_vertex][1-start_vertex] * 2
    
    # Convert vertex indices to 0..n-1 for ease of bit manipulation
    # We'll convert back at the end
    original_start = start_vertex
    
    # Create memo table for storing results of subproblems
    # memo[(S, v)] = (cost, predecessor)
    # where S is a set of vertices (represented as a bitmask)
    # and v is the last vertex visited
    memo = {}
    
    # Initialize base cases: distance from start to every other vertex
    for v in range(n):
        if v != start_vertex:
            # Set S contains only vertex v
            S = 1 << v
            memo[(S, v)] = (graph[start_vertex][v], start_vertex)
    
    # Solve for subsets of increasing size
    for subset_size in range(2, n):
        for subset in itertools.combinations(range(n), subset_size):
            # Skip if the subset contains the start vertex
            if start_vertex in subset:
                continue
                
            # Create bitmask for this subset
            S = 0
            for v in subset:
                S |= 1 << v
                
            # For each vertex in the subset
            for v in subset:
                # Create bitmask without vertex v
                S_v = S & ~(1 << v)
                
                min_cost = float('inf')
                min_prev = None
                
                # Try all possible previous vertices
                for u in subset:
                    if u == v:
                        continue
                        
                    # Calculate cost of reaching v through u
                    cost = memo[(S_v, u)][0] + graph[u][v]
                    
                    if cost < min_cost:
                        min_cost = cost
                        min_prev = u
                
                # Store result
                memo[(S, v)] = (min_cost, min_prev)
    
    # Find optimal cost to return to start_vertex
    all_vertices_except_start = 0
    for v in range(n):
        if v != start_vertex:
            all_vertices_except_start |= 1 << v
    
    min_cost_back = float('inf')
    min_last = None
    for v in range(n):
        if v != start_vertex:
            cost = memo[(all_vertices_except_start, v)][0] + graph[v][start_vertex]
            if cost < min_cost_back:
                min_cost_back = cost
                min_last = v
    
    # Reconstruct the optimal path
    optimal_path = [start_vertex]
    
    # Start from the last vertex before returning to start
    S = all_vertices_except_start
    v = min_last
    
    # Trace backward until we reach the start
    while v != start_vertex:
        optimal_path.append(v)
        
        # Get previous vertex in the path
        prev_v = memo[(S, v)][1]
        
        # Update S by removing v
        S &= ~(1 << v)
        
        # Move to previous vertex
        v = prev_v
    
    # Complete the cycle by returning to the start
    optimal_path.append(start_vertex)
    
    # The path is in reverse order, so reverse it
    optimal_path.reverse()

    print("Held-Karp Results: ")
    print(f"Cycle: {optimal_path} Weight: {min_cost_back}")
    print(f"Length of cycle: {len(optimal_path)}")
    
    return optimal_path, min_cost_back


def construct_greedy_cycle(graph, start, anchor1, anchor2):
    """Constructs a greedy cycle given a start vertex and 2 anchor points. Ensures Hamiltonian cycle."""
    vertices_count = len(graph)
    
    # Initialize with only the start point in the visited set
    visited = set([start])
    path = [start]
    current_vertex = start
    total_weight = 0
    
    # First, we need to visit anchor1
    if anchor1 not in visited:
        path.append(anchor1)
        visited.add(anchor1)
        total_weight += graph[current_vertex][anchor1]
        current_vertex = anchor1
        #print(f"Added anchor1: {anchor1}")
        #print(f"Visited: {visited}")
    
    # Visit all remaining vertices except anchor2
    while len(visited) < vertices_count - 1:  # -1 because we'll add anchor2 last
        next_vertex = None
        lowest_weight = float("inf")
        
        for i in range(vertices_count):
            if i not in visited and i != anchor2 and graph[current_vertex][i] < lowest_weight:
                next_vertex = i
                lowest_weight = graph[current_vertex][i]
        
        # If we can't find a next vertex, break the loop
        if next_vertex is None:
            break
            
        #print(f"Adding vertex: {next_vertex}")
        visited.add(next_vertex)
        path.append(next_vertex)
        total_weight += lowest_weight
        current_vertex = next_vertex
        #print(f"Visited: {visited}")
    
    # Now add anchor2 if it's not already visited
    if anchor2 not in visited:
        #print("Adding second anchor!!!")
        path.append(anchor2)
        visited.add(anchor2)
        total_weight += graph[current_vertex][anchor2]
        current_vertex = anchor2
        #print(f"Added anchor2: {anchor2}")
        #print(f"Visited: {visited}")
    
    # Complete the cycle by returning to start
    total_weight += graph[current_vertex][start]
    path.append(start)
    
    return path, total_weight
    

def low_anchor_heuristic(graph, vertex):
    def find_two_lowest_indices(values, vertex):
        if len(values) < 2:
            raise ValueError("List must contain at least two elements.")
        sorted_indices = sorted((i for i in range(len(values)) if i != vertex), key=lambda i: values[i])
        return sorted_indices[:2]
    anchors = find_two_lowest_indices(graph[vertex], vertex)
    # print(f"Anchors for {vertex}: {anchors}")

    cycle_1, lowest_weight_1 = construct_greedy_cycle(graph, vertex, anchors[0], anchors[1])
    cycle_2, lowest_weight_2 = construct_greedy_cycle(graph, vertex, anchors[1], anchors[0])

    if lowest_weight_1 < lowest_weight_2:
        return cycle_1, lowest_weight_1 
    return cycle_2, lowest_weight_2

def high_anchor_heuristic(graph, vertex):
    def find_two_highest_indices(values, vertex):
        if len(values) < 2:
            raise ValueError("List must contain at least two elements.")
        sorted_indices = sorted(
            (i for i in range(len(values)) if i != vertex),
            key=lambda i: values[i],
            reverse=True  # Sort from highest to lowest
        )
        return sorted_indices[:2]
    

    anchors = find_two_highest_indices(graph[vertex], vertex)
    # print(f"Anchors for {vertex}: {anchors}")

    cycle_1, lowest_weight_1 = construct_greedy_cycle(graph, vertex, anchors[0], anchors[1])
    cycle_2, lowest_weight_2 = construct_greedy_cycle(graph, vertex, anchors[1], anchors[0])

    if lowest_weight_1 < lowest_weight_2:
        return cycle_1, lowest_weight_1 
    return cycle_2, lowest_weight_2

def random_anchor_heuristic(graph, vertex):
    def find_two_random_indices(values, vertex):
        if len(values) < 3:
            raise ValueError("List must contain at least three elements to exclude one and pick two.")

        candidates = [i for i in range(len(values)) if i != vertex]
        return random.sample(candidates, 2)

    anchors = find_two_random_indices(graph[vertex], vertex)
    # rint(f"Anchors for {vertex}: {anchors}")

    cycle_1, lowest_weight_1 = construct_greedy_cycle(graph, vertex, anchors[0], anchors[1])
    cycle_2, lowest_weight_2 = construct_greedy_cycle(graph, vertex, anchors[1], anchors[0])

    if lowest_weight_1 < lowest_weight_2:
        return cycle_1, lowest_weight_1 
    return cycle_2, lowest_weight_2


def run_base_benchmark(graph, vertex):
    results = {}
    results["optimal"] = find_optimal_cycle_held_karp(graph, vertex)
    results["greedy"] = greedy_algorithm(graph, vertex)
    results["low_anchor"] = low_anchor_heuristic(graph, vertex)
    results["high_anchor"] = high_anchor_heuristic(graph, vertex)
    results["random_anchor"] = random_anchor_heuristic(graph, vertex)
    return results

def save_adjacency_matrix(matrix, filename, format="csv"):
    """
    Save an adjacency matrix to a file.
    
    Parameters:
    - matrix: 2D list representing the adjacency matrix
    - filename: Name of the file to save to
    - format: File format, either 'csv' or 'json' (default: 'csv')
    """
    if format.lower() == "csv":
        with open(filename, 'w', newline='') as file:
            writer = csv.writer(file)
            for row in matrix:
                writer.writerow(row)
        print(f"Adjacency matrix saved to {filename} in CSV format")
    
    elif format.lower() == "json":
        with open(filename, 'w') as file:
            json.dump(matrix, file)
        print(f"Adjacency matrix saved to {filename} in JSON format")
    
    else:
        raise ValueError("Unsupported format. Use 'csv' or 'json'")


def load_adjacency_matrix(filename, format="csv"):
    """
    Load an adjacency matrix from a file.
    
    Parameters:
    - filename: Name of the file to load from
    - format: File format, either 'csv' or 'json' (default: 'csv')
    
    Returns:
    - 2D list representing the adjacency matrix
    """
    if format.lower() == "csv":
        matrix = []
        with open(filename, 'r', newline='') as file:
            reader = csv.reader(file)
            for row in reader:
                # Convert strings to appropriate numeric values
                matrix.append([float(val) if '.' in val else int(val) for val in row])
        print(f"Adjacency matrix loaded from {filename}")
        return matrix
    
    elif format.lower() == "json":
        with open(filename, 'r') as file:
            matrix = json.load(file)
        print(f"Adjacency matrix loaded from {filename}")
        return matrix
    
    else:
        raise ValueError("Unsupported format. Use 'csv' or 'json'")
