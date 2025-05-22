def hamiltonian_cycle_heuristic(graph, start, anchors=None, max_depth=-1, early_exit=False):
    """
    Implements a Hamiltonian cycle heuristic based on an anchor-based greedy approach with constraints and adaptive bridging depth.
    
    Args:
        graph: A complete, weighted graph represented as an adjacency matrix (list of lists of integers).
        start: The starting vertex, which is one of the anchor vertices.
        anchors: A list of integers representing the anchor vertices (including start).
                If None, anchors will be generated randomly based on graph size.
        max_depth: Maximum number of intermediate vertices allowed when connecting one anchor to the next.
                  If -1, an adaptive depth strategy is used.
        early_exit: If True, greedily connect to the target anchor if it's the lowest-cost edge available.
    
    Returns:
        A tuple containing:
        - cycle: A list of vertex indices representing the Hamiltonian cycle
        - total_weight: The sum of weights in the cycle
    """
    import itertools
    import random
    
    # Get number of vertices in the graph
    n = len(graph)
    
    # If anchors is not provided, generate them randomly
    if anchors is None:
        # Calculate number of anchors based on n = 4k formula
        # k = n/4, where k is number of anchors
        k = max(2, n // 4)  # Ensure at least 2 anchors
        
        # Ensure start is included in anchors
        potential_anchors = [v for v in range(n) if v != start]
        # Randomly select k-1 additional anchors (start is already one anchor)
        random_anchors = random.sample(potential_anchors, k-1)
        anchors = [start] + random_anchors
    
    # 1. Sanity checks
    k = len(anchors)  # Number of anchors
    
    if n < 4 * k:
        raise ValueError(f"Graph must have at least {4 * k} vertices for {k} anchors. Current: {n}")
    
    # Check if start is in anchors
    if start not in anchors:
        raise ValueError(f"Start vertex {start} must be in the anchor list")
    
    # If max_depth is specified, check if it allows at least one intermediate vertex per bridge
    if max_depth != -1:
        # Each anchor pair needs a bridge
        num_bridges = k
        # Each bridge can have at most max_depth intermediate vertices
        max_intermediate_vertices = num_bridges * max_depth
        # We need n - k vertices for intermediate vertices
        if max_intermediate_vertices < n - k:
            raise ValueError(f"max_depth={max_depth} is too small to accommodate all {n-k} non-anchor vertices")
    
    # 2. Anchor permutation logic with start and end fixed
    # Remove start from anchors to avoid duplicates in permutations
    other_anchors = [a for a in anchors if a != start]
    
    # Generate all permutations of other anchors and add start at beginning and end
    best_cycle = None
    best_weight = float('inf')
    
    for perm in itertools.permutations(other_anchors):
        anchor_path = [start] + list(perm) + [start]
        print(f"Anchor path: {anchor_path}")
        
        # 3. Perform anchor-bridging traversal for this permutation
        cycle, weight = _build_cycle_from_anchors(graph, anchor_path, n, max_depth, early_exit)
        print(f"Current weight: {weight}")
        
        # Track best solution
        if weight < best_weight:
            best_weight = weight
            best_cycle = cycle
    
    return best_cycle, best_weight

def _build_cycle_from_anchors(graph, anchor_path, n, max_depth, early_exit):
    """
    Builds a Hamiltonian cycle by connecting anchors using greedy bridging with constraints.
    
    Args:
        graph: Adjacency matrix
        anchor_path: Ordered list of anchors to visit
        n: Number of vertices in the graph
        max_depth: Maximum bridge depth (-1 for adaptive)
        early_exit: Flag for early connection to target anchor
    
    Returns:
        Tuple of (cycle, total_weight)
    """
    visited = set(anchor_path[:-1])  # Mark all anchors except the last one as visited
    cycle = [anchor_path[0]]  # Start with the first anchor
    total_weight = 0
    
    # Set of all anchors for blacklisting direct anchor-to-anchor connections
    all_anchors = set(anchor_path)
    
    # Calculate total non-anchor vertices to distribute
    total_non_anchors = n - len(set(anchor_path))
    num_bridges = len(anchor_path) - 1
    
    # Calculate vertices per bridge (trying to distribute evenly)
    base_vertices_per_bridge = total_non_anchors // num_bridges
    extra = total_non_anchors % num_bridges
    
    # For each pair of consecutive anchors
    for i in range(len(anchor_path) - 1):
        from_anchor = anchor_path[i]
        to_anchor = anchor_path[i+1]
        
        # Determine adaptive depth for this bridge if requested
        if max_depth == -1:
            # More vertices for bridges between distant anchors
            anchor_distance = graph[from_anchor][to_anchor]
            # Simple adaptive strategy - allocate more vertices to longer distances
            # Allocate base vertices plus extra if available
            current_bridge_vertices = base_vertices_per_bridge + (1 if i < extra else 0)
        else:
            current_bridge_vertices = max_depth
        
        # Connect the anchors with a greedy path
        current = from_anchor
        depth = 0
        
        while current != to_anchor and depth < current_bridge_vertices:
            next_vertex = _find_next_vertex(graph, current, visited, to_anchor, early_exit, all_anchors)
            
            # If we found a valid next vertex
            if next_vertex is not None:
                # Add the edge
                total_weight += graph[current][next_vertex]
                cycle.append(next_vertex)
                
                # If we've reached the target anchor, we're done with this bridge
                if next_vertex == to_anchor:
                    break
                
                # Mark as visited and continue
                visited.add(next_vertex)
                current = next_vertex
                depth += 1
            else:
                # Force connection to target anchor if no other valid vertices
                total_weight += graph[current][to_anchor]
                cycle.append(to_anchor)
                break
        
        # Force connection to target anchor if we've reached depth limit
        if current != to_anchor:
            total_weight += graph[current][to_anchor]
            cycle.append(to_anchor)
    
    # 4. Visit remaining unvisited vertices
    # At this point, we've visited all anchors in the specified order
    # Now we need to visit any remaining unvisited vertices before returning to start
    current = cycle[-1]
    all_vertices = set(range(n))
    remaining = all_vertices - visited - {anchor_path[-1]}
    
    while remaining:
        # Find the closest unvisited vertex
        next_vertex = min(remaining, key=lambda v: graph[current][v])
        total_weight += graph[current][next_vertex]
        cycle.append(next_vertex)
        current = next_vertex
        remaining.remove(next_vertex)
    
    # Complete the cycle by returning to the starting vertex
    if current != anchor_path[0]:
        total_weight += graph[current][anchor_path[0]]
        cycle.append(anchor_path[0])
    
    return cycle, total_weight

def _find_next_vertex(graph, current, visited, target, early_exit, all_anchors=None):
    """
    Finds the next vertex to visit using a greedy approach.
    
    Args:
        graph: Adjacency matrix
        current: Current vertex
        visited: Set of already visited vertices
        target: Target anchor vertex
        early_exit: If True, connect to target if it's the lowest-cost option
        all_anchors: Set of all anchor vertices to avoid direct anchor-to-anchor connections
    
    Returns:
        Next vertex to visit, or None if no valid vertex exists
    """
    n = len(graph)
    min_weight = float('inf')
    next_vertex = None
    
    for v in range(n):
        # Skip visited vertices
        if v in visited:
            continue
            
        # Skip other anchors (except target) to prevent direct anchor-to-anchor connections
        if all_anchors and v in all_anchors and v != target:
            continue
        
        weight = graph[current][v]
        
        # If this is the target and early_exit is True, connect immediately if it's the best option
        if v == target and early_exit and weight < min_weight:
            return target
            
        # Otherwise, find the minimum weight edge to an unvisited vertex
        if weight < min_weight:
            min_weight = weight
            next_vertex = v
    
    return next_vertex