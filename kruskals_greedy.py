def greedy_edge_tsp(graph, start_vertex=0):
    """
    Solve TSP using a greedy edge-based approach similar to Kruskal's algorithm.
    
    This algorithm picks the shortest edges that don't create cycles until we have
    a path visiting all vertices, then completes the Hamiltonian cycle.
    
    Parameters:
    - graph: Adjacency matrix represented as a list of lists with edge weights
    - start_vertex: The vertex to start and end the cycle at (default: 0)
    
    Returns:
    - A tuple containing (cycle_path, total_weight)
    """
    n = len(graph)
    if n <= 1:
        return [start_vertex], 0
    
    # Create list of all edges with their weights
    edges = []
    for i in range(n):
        for j in range(i + 1, n):  # Only consider each edge once (undirected graph)
            edges.append((graph[i][j], i, j))
    
    # Sort edges by weight (ascending)
    edges.sort()
    
    # Track vertex degrees and connected components
    vertex_degree = [0] * n
    parent = list(range(n))  # For union-find to detect cycles
    
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py
            return True
        return False
    
    selected_edges = []
    
    # Greedily select edges
    for weight, u, v in edges:
        # Check if adding this edge would create a cycle (except when completing the path)
        # or if it would give any vertex degree > 2
        if vertex_degree[u] < 2 and vertex_degree[v] < 2:
            # Check if adding this edge creates a cycle before we have n-1 edges
            if len(selected_edges) < n - 1:
                if find(u) != find(v):  # No cycle created
                    selected_edges.append((u, v, weight))
                    vertex_degree[u] += 1
                    vertex_degree[v] += 1
                    union(u, v)
                    
                    if len(selected_edges) == n - 1:
                        break
    
    # If we don't have n-1 edges, the graph might be disconnected or we need to adjust
    if len(selected_edges) < n - 1:
        # Fallback: complete the path by adding necessary edges
        remaining_vertices = set(range(n))
        connected_vertices = set()
        
        for u, v, _ in selected_edges:
            connected_vertices.add(u)
            connected_vertices.add(v)
        
        # Add edges to connect remaining vertices
        for vertex in remaining_vertices - connected_vertices:
            # Find the closest connected vertex
            min_weight = float('inf')
            best_connection = None
            
            for connected_vertex in connected_vertices:
                if graph[vertex][connected_vertex] < min_weight:
                    min_weight = graph[vertex][connected_vertex]
                    best_connection = connected_vertex
            
            if best_connection is not None:
                selected_edges.append((vertex, best_connection, min_weight))
                vertex_degree[vertex] += 1
                vertex_degree[best_connection] += 1
                connected_vertices.add(vertex)
    
    # Build adjacency list from selected edges
    adj_list = [[] for _ in range(n)]
    total_weight = 0
    
    for u, v, weight in selected_edges:
        adj_list[u].append(v)
        adj_list[v].append(u)
        total_weight += weight
    
    # Find path starting from start_vertex
    def find_path(current, visited, path):
        path.append(current)
        visited.add(current)
        
        if len(path) == n:
            return path
        
        for neighbor in adj_list[current]:
            if neighbor not in visited:
                result = find_path(neighbor, visited, path)
                if result:
                    return result
        
        # Backtrack
        path.pop()
        visited.remove(current)
        return None
    
    # Find the path through all vertices
    path = find_path(start_vertex, set(), [])
    
    if not path or len(path) != n:
        # Fallback: create a simple path and complete it
        unvisited = set(range(n))
        path = [start_vertex]
        unvisited.remove(start_vertex)
        current = start_vertex
        
        while unvisited:
            # Find closest unvisited vertex
            min_dist = float('inf')
            next_vertex = None
            
            for vertex in unvisited:
                if graph[current][vertex] < min_dist:
                    min_dist = graph[current][vertex]
                    next_vertex = vertex
            
            if next_vertex is not None:
                path.append(next_vertex)
                unvisited.remove(next_vertex)
                current = next_vertex
        
        # Calculate total weight for this path
        total_weight = 0
        for i in range(len(path) - 1):
            total_weight += graph[path[i]][path[i + 1]]
    
    # Complete the cycle by returning to start
    if len(path) == n:
        total_weight += graph[path[-1]][start_vertex]
        path.append(start_vertex)
    
    return path, total_weight


def greedy_edge_tsp_v2(graph, start_vertex=0):
    """
    Alternative implementation of greedy edge-based TSP solver.
    
    This version more strictly follows the Kruskal-like approach by maintaining
    the constraint that each vertex can have at most 2 connections until the final step.
    
    Parameters:
    - graph: Adjacency matrix represented as a list of lists with edge weights
    - start_vertex: The vertex to start and end the cycle at (default: 0)
    
    Returns:
    - A tuple containing (cycle_path, total_weight)
    """
    n = len(graph)
    if n <= 1:
        return [start_vertex], 0
    
    # Create and sort all edges
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((graph[i][j], i, j))
    edges.sort()
    
    # Track connections for each vertex
    connections = [[] for _ in range(n)]
    selected_edges = []
    
    # Union-Find for cycle detection
    parent = list(range(n))
    
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py
            return True
        return False
    
    # Select edges greedily
    for weight, u, v in edges:
        # Check constraints:
        # 1. Each vertex can have at most 2 connections
        # 2. Don't create a cycle unless we have n-1 edges already
        if len(connections[u]) < 2 and len(connections[v]) < 2:
            # If we have n-1 edges, we can create the final cycle
            if len(selected_edges) == n - 1:
                # Check if this edge completes our Hamiltonian cycle
                if find(u) != find(v):
                    continue  # This wouldn't complete the cycle
                selected_edges.append((u, v, weight))
                connections[u].append(v)
                connections[v].append(u)
                break
            else:
                # Don't create cycles yet
                if find(u) != find(v):
                    selected_edges.append((u, v, weight))
                    connections[u].append(v)
                    connections[v].append(u)
                    union(u, v)
                    
                    if len(selected_edges) == n - 1:
                        # Now we need to find the edge that completes the cycle
                        # Find the two vertices with degree 1
                        endpoints = [i for i in range(n) if len(connections[i]) == 1]
                        if len(endpoints) == 2:
                            u_end, v_end = endpoints
                            closing_weight = graph[u_end][v_end]
                            selected_edges.append((u_end, v_end, closing_weight))
                            connections[u_end].append(v_end)
                            connections[v_end].append(u_end)
                            break
    
    # Build the cycle path starting from start_vertex
    if len(selected_edges) == n:
        # We have a complete cycle, now traverse it
        visited = set()
        path = []
        current = start_vertex
        
        while len(path) < n:
            path.append(current)
            visited.add(current)
            
            # Find next unvisited neighbor
            next_vertex = None
            for neighbor in connections[current]:
                if neighbor not in visited:
                    next_vertex = neighbor
                    break
            
            if next_vertex is None and len(path) < n:
                # This shouldn't happen in a proper Hamiltonian cycle
                break
            
            current = next_vertex
        
        # Complete the cycle
        path.append(start_vertex)
        
        # Calculate total weight
        total_weight = sum(selected_edges[i][2] for i in range(len(selected_edges)))
        
        return path, total_weight
    
    else:
        # Fallback to simpler approach if we couldn't build a proper cycle
        return greedy_edge_tsp(graph, start_vertex)


# Test function to compare with existing algorithms
def test_greedy_edge_comparison(graph, start_vertex=0):
    """
    Test function to compare the greedy edge approach with existing algorithms.
    
    Parameters:
    - graph: Adjacency matrix
    - start_vertex: Starting vertex for the cycle
    
    Returns:
    - Dictionary with results from different algorithms
    """
    results = {}
    
    # Test both versions of greedy edge
    results["greedy_edge_v1"] = greedy_edge_tsp(graph, start_vertex)
    results["greedy_edge_v2"] = greedy_edge_tsp_v2(graph, start_vertex)
    
    # If you have the existing functions available, you can uncomment these:
    # results["nearest_neighbor"] = greedy_algorithm(graph, start_vertex)  # Your existing "greedy"
    # results["low_anchor"] = low_anchor_heuristic(graph, start_vertex)
    # results["high_anchor"] = high_anchor_heuristic(graph, start_vertex)
    
    return results


def greedy_edge_tsp_v3(graph):
    """
    Pure greedy edge-based TSP solver that ignores starting vertex.
    
    This version truly follows the Kruskal-like approach by selecting the cheapest
    edges without regard to any starting vertex. The cycle naturally emerges from
    the edge selection process.
    
    Parameters:
    - graph: Adjacency matrix represented as a list of lists with edge weights
    
    Returns:
    - A tuple containing (cycle_path, total_weight)
    """
    n = len(graph)
    if n <= 1:
        return [0], 0
    if n == 2:
        return [0, 1, 0], graph[0][1] * 2
    
    # Create and sort all edges by weight
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((graph[i][j], i, j))
    edges.sort()
    
    # Track vertex degrees
    degree = [0] * n
    selected_edges = []
    
    # Union-Find for cycle detection
    parent = list(range(n))
    
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py
            return True
        return False
    
    # Select edges following these rules:
    # 1. No vertex can have degree > 2
    # 2. Don't create cycles until we have exactly n edges (complete Hamiltonian cycle)
    # 3. Must form a single connected component
    
    for weight, u, v in edges:
        # Check degree constraint
        if degree[u] >= 2 or degree[v] >= 2:
            continue
            
        # If we have n-1 edges, we need to complete the cycle
        if len(selected_edges) == n - 1:
            # This edge should connect the two endpoints (vertices with degree 1)
            endpoints = [i for i in range(n) if degree[i] == 1]
            if len(endpoints) == 2 and {u, v} == set(endpoints):
                selected_edges.append((u, v, weight))
                degree[u] += 1
                degree[v] += 1
                break
        else:
            # Don't create a cycle yet
            if find(u) != find(v):
                selected_edges.append((u, v, weight))
                degree[u] += 1
                degree[v] += 1
                union(u, v)
    
    # Build adjacency list from selected edges
    adj_list = [[] for _ in range(n)]
    total_weight = 0
    
    for u, v, weight in selected_edges:
        adj_list[u].append(v)
        adj_list[v].append(u)
        total_weight += weight
    
    # Find any vertex to start the cycle traversal (doesn't matter which one)
    start = 0
    
    # Traverse the cycle
    path = []
    current = start
    prev = -1
    
    for _ in range(n):
        path.append(current)
        
        # Find the next vertex (not the previous one)
        next_vertex = None
        for neighbor in adj_list[current]:
            if neighbor != prev:
                next_vertex = neighbor
                break
        
        prev = current
        current = next_vertex
    
    # Complete the cycle by returning to start
    path.append(start)
    
    return path, total_weight


# Example usage:
if __name__ == "__main__":
    # Example 4x4 complete graph
    example_graph = [
        [0, 10, 15, 20],
        [10, 0, 35, 25],
        [15, 35, 0, 30],
        [20, 25, 30, 0]
    ]
    
    print("Testing Greedy Edge-Based TSP Solver")
    print("=" * 40)
    
    cycle_v3, weight_v3 = greedy_edge_tsp_v3(example_graph)
    print(f"Greedy Edge V3: {cycle_v3}, Weight: {weight_v3}")
    
    cycle_v1, weight_v1 = greedy_edge_tsp(example_graph, 0)
    print(f"Greedy Edge V1: {cycle_v1}, Weight: {weight_v1}")
    
    cycle_v2, weight_v2 = greedy_edge_tsp_v2(example_graph, 0)
    print(f"Greedy Edge V2: {cycle_v2}, Weight: {weight_v2}")
    
    # Test with a larger graph
    print("\nTesting with 5x5 graph:")
    graph_5x5 = [
        [0, 12, 10, 19, 8],
        [12, 0, 3, 7, 6],
        [10, 3, 0, 2, 20],
        [19, 7, 2, 0, 4],
        [8, 6, 20, 4, 0]
    ]
    
    cycle_v3_5, weight_v3_5 = greedy_edge_tsp_v3(graph_5x5)
    print(f"Greedy Edge V3 (5x5): {cycle_v3_5}, Weight: {weight_v3_5}")
    
    cycle_v1_5, weight_v1_5 = greedy_edge_tsp(graph_5x5, 0)
    print(f"Greedy Edge V1 (5x5): {cycle_v1_5}, Weight: {weight_v1_5}")
    
    cycle_v2_5, weight_v2_5 = greedy_edge_tsp_v2(graph_5x5, 0)
    print(f"Greedy Edge V2 (5x5): {cycle_v2_5}, Weight: {weight_v2_5}")