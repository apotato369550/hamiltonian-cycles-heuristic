
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

def best_anchor_heuristic(graph):
    """
    Applies low_anchor_heuristic to all vertices and returns the cycle with the lowest total weight.
    
    Args:
        graph: 2D list/array representing adjacency matrix of a complete graph
        
    Returns:
        tuple: (best_cycle, best_weight) where best_cycle is the path and best_weight is the total weight
    """
    vertices_count = len(graph)
    best_cycle = None
    best_weight = float('inf')
    best_start_vertex = None
    
    # Try low_anchor_heuristic for each vertex as starting point
    for vertex in range(vertices_count):
        try:
            cycle, weight = low_anchor_heuristic(graph, vertex)
            
            # Keep track of the best cycle found so far
            if weight < best_weight:
                best_weight = weight
                best_cycle = cycle
                best_start_vertex = vertex
                
        except (ValueError, IndexError) as e:
            # Skip vertices that cause errors (e.g., if graph has < 3 vertices)
            continue
    
    if best_cycle is None:
        raise ValueError("No valid cycle could be constructed from any starting vertex")
    
    return best_cycle, best_weight