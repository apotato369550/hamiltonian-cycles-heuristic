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