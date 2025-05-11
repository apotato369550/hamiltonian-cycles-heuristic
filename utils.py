import random
import itertools

# Function to create a complete graph with weighted edges.
# Parameters:
# vertices - Number of vertices in the graph (int).
# weight_range - Tuple specifying the range of weights for edges (default: (1, 10)).
# replace - Boolean indicating whether to sample weights with or without replacement (default: False).
# strict - Boolean indicating whether to enforce strict uniqueness of weights (default: False).
# seed - An integer seed to base randomly generated numbers on. 
def generate_complete_graph(vertices, weight_range=(1, 10), replace=False, strict=False, seed=None):
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

def construct_greedy_cycle(graph, start, anchors):
    """Constructs a greedy cycle given a start vertex and 2 anchor points. Ensures Hamiltonian cycle."""
    return
# try to debug this.
# still not working.

def low_anchor_heuristic(graph, vertex):
    return

def high_anchor_heuristic(graph, vertex):
    return

def random_anchor_heuristic(graph, vertex):
    return


def run_benchmark(graph, vertex):
    results = {}
    results["optimal"] = find_optimal_cycle(graph, vertex)
    results["greedy"] = greedy_algorithm(graph, vertex)
    # results["low_anchor"] = low_anchor_heuristic(graph, vertex)
    # results["high_anchor"] = high_anchor_heuristic(graph, vertex)
    # results["random_anchor"] = random_anchor_heuristic(graph, vertex)
    return results
