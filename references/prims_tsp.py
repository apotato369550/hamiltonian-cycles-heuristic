"""
Novel TSP heuristics inspired by Prim's algorithm and anchoring concept
"""

def calculate_cycle_cost(cycle, graph):
    """Calculate total cost of a cycle"""
    if len(cycle) < 2:
        return 0
    total = 0
    for i in range(len(cycle) - 1):
        total += graph[cycle[i]][cycle[i + 1]]
    return total

def prims_tsp(graph, start_vertex):
    """
    Prim's TSP: Adaptive bidirectional greedy using Prim's approach
    Search space includes all vertices in current path, degrees limited to 2
    """
    n = len(graph)
    if n < 2:
        return [start_vertex], 0
    
    # Track vertices and their connections
    path = [start_vertex]
    visited = {start_vertex}
    degree = [0] * n
    
    while len(path) < n:
        best_cost = float('inf')
        best_vertex = None
        best_from = None
        
        # Search from all vertices in current path
        for v in path:
            if degree[v] < 2:  # Can still connect
                for u in range(n):
                    if u not in visited and graph[v][u] < best_cost:
                        best_cost = graph[v][u]
                        best_vertex = u
                        best_from = v
        
        if best_vertex is None:
            break
        
        # Add vertex to path
        path.append(best_vertex)
        visited.add(best_vertex)
        degree[best_from] += 1
        degree[best_vertex] += 1
    
    # Complete cycle
    cycle = path + [start_vertex]
    return cycle, calculate_cycle_cost(cycle, graph)

def prims_anchoring_tsp(graph, start_vertex):
    """
    Prim's Anchoring TSP: Use two lowest-cost edges as anchors,
    then apply Prim's TSP with anchor constraints
    """
    n = len(graph)
    if n < 3:
        return [start_vertex] * (n + 1), 0
    
    # Find two lowest-cost edges from start vertex
    edges = [(graph[start_vertex][i], i) for i in range(n) if i != start_vertex]
    edges.sort()
    anchor1, anchor2 = edges[0][1], edges[1][1]
    
    # Initialize with anchors
    path = [start_vertex, anchor1]
    visited = {start_vertex, anchor1}
    degree = [0] * n
    degree[start_vertex] = 1
    degree[anchor1] = 1
    
    # Reserve anchor2 for final connection
    while len(path) < n - 1:
        best_cost = float('inf')
        best_vertex = None
        best_from = None
        
        # Search from all vertices in current path
        for v in path:
            if degree[v] < 2:
                for u in range(n):
                    if u not in visited and u != anchor2 and graph[v][u] < best_cost:
                        best_cost = graph[v][u]
                        best_vertex = u
                        best_from = v
        
        if best_vertex is None:
            break
        
        path.append(best_vertex)
        visited.add(best_vertex)
        degree[best_from] += 1
        degree[best_vertex] += 1
    
    # Add anchor2 and complete cycle
    path.append(anchor2)
    cycle = path + [start_vertex]
    return cycle, calculate_cycle_cost(cycle, graph)

def best_prims_tsp(graph):
    """
    Best Prim's TSP: Apply Prim's TSP from all vertices, return best result
    """
    n = len(graph)
    best_cycle = None
    best_cost = float('inf')
    
    for start in range(n):
        cycle, cost = prims_tsp(graph, start)
        if cost < best_cost:
            best_cost = cost
            best_cycle = cycle
    
    return best_cycle, best_cost

def best_prims_anchoring_tsp(graph):
    """
    Best Prim's Anchoring TSP: Apply Prim's Anchoring TSP from all vertices
    """
    n = len(graph)
    best_cycle = None
    best_cost = float('inf')
    
    for start in range(n):
        cycle, cost = prims_anchoring_tsp(graph, start)
        if cost < best_cost:
            best_cost = cost
            best_cycle = cycle
    
    return best_cycle, best_cost