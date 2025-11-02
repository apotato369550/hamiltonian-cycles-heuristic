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
    
    # 2. Find the two lowest-weight edges for each anchor
    anchor_edges = _find_anchor_edges(graph, anchors)

    # print(f"Anchor edges: {anchor_edges}")
    
    # 3. Generate all anchor permutations and edge configurations
    other_anchors = [a for a in anchors if a != start]
    
    best_cycle = None
    best_weight = float('inf')
    
    # Try all permutations of anchor order (start fixed at beginning and end)
    for perm in itertools.permutations(other_anchors):
        anchor_path = [start] + list(perm) + [start]
        
        # print(f"Anchor path: f{anchor_path}")

        # Try all 2^k edge direction configurations for this permutation
        num_anchors = len(anchors)
        for config in range(2 ** num_anchors):
            # Convert configuration number to binary representation
            edge_config = _get_edge_configuration(config, num_anchors)
            # print(f"Current edge config: {edge_config}")
            
            # Build cycle with this anchor order and edge configuration
            cycle, weight = _build_cycle_with_edge_config(
                graph, anchor_path, anchor_edges, edge_config, n, max_depth, early_exit
            )
            
            #print(f"Current Cycle: {cycle}")
            # print(f"Current Weight: {weight}")

            # Track best solution
            if weight < best_weight:
                best_weight = weight
                best_cycle = cycle
    
    return best_cycle, best_weight

def _find_anchor_edges(graph, anchors):
    """
    Finds the two lowest-weight edges for each anchor vertex.
    
    Args:
        graph: Adjacency matrix
        anchors: List of anchor vertices
    
    Returns:
        Dictionary mapping each anchor to its two lowest-weight edges
    """
    anchor_edges = {}
    all_anchors_set = set(anchors)
    
    for anchor in anchors:
        # Find all edges from this anchor to non-anchor vertices
        edges = []
        for v in range(len(graph)):
            if v != anchor and v not in all_anchors_set:
                edges.append((graph[anchor][v], v))
        
        # Sort by weight and take the two lowest-weight edges
        edges.sort()
        if len(edges) >= 2:
            anchor_edges[anchor] = [edges[0][1], edges[1][1]]  # Store vertex indices only
        elif len(edges) == 1:
            anchor_edges[anchor] = [edges[0][1], edges[0][1]]  # Duplicate if only one available
        else:
            # If no non-anchor vertices available, this is an edge case
            anchor_edges[anchor] = [None, None]
    
    return anchor_edges

def _get_edge_configuration(config_num, num_anchors):
    """
    Converts a configuration number to a binary representation for edge directions.
    
    Args:
        config_num: Integer representing the configuration (0 to 2^num_anchors - 1)
        num_anchors: Number of anchors
    
    Returns:
        List of booleans indicating edge direction for each anchor
        (False = use first edge as entry, True = use second edge as entry)
    """
    config = []
    for i in range(num_anchors):
        config.append((config_num >> i) & 1 == 1)
    return config

def _build_cycle_with_edge_config(graph, anchor_path, anchor_edges, edge_config, n, max_depth, early_exit):
    """
    Builds a Hamiltonian cycle using specific anchor order and edge configuration.
    
    Args:
        graph: Adjacency matrix
        anchor_path: Ordered list of anchors to visit
        anchor_edges: Dictionary of anchor edges
        edge_config: List of booleans for edge direction configuration
        n: Number of vertices in the graph
        max_depth: Maximum bridge depth (-1 for adaptive)
        early_exit: Flag for early connection to target anchor
    
    Returns:
        Tuple of (cycle, total_weight)
    """
    # Create mapping from anchor to its index in the original anchors list
    unique_anchors = list(set(anchor_path))
    anchor_to_index = {anchor: i for i, anchor in enumerate(unique_anchors)}
    
    visited = set(unique_anchors)  # Mark all anchors as visited initially
    cycle = [anchor_path[0]]  # Start with the first anchor
    total_weight = 0
    
    # Set of all anchors for blacklisting direct anchor-to-anchor connections
    all_anchors = set(anchor_path)
    
    # Calculate total non-anchor vertices to distribute
    total_non_anchors = n - len(set(anchor_path))
    num_bridges = len(anchor_path) - 1
    
    # Calculate vertices per bridge (trying to distribute evenly)
    base_vertices_per_bridge = total_non_anchors // num_bridges if num_bridges > 0 else 0
    extra = total_non_anchors % num_bridges if num_bridges > 0 else 0
    
    # For each pair of consecutive anchors
    for i in range(len(anchor_path) - 1):
        from_anchor = anchor_path[i]
        to_anchor = anchor_path[i + 1]
        
        # Get the entry and exit vertices for the anchors based on configuration
        from_anchor_idx = anchor_to_index.get(from_anchor, 0)
        to_anchor_idx = anchor_to_index.get(to_anchor, 0)
        
        # Determine exit vertex for from_anchor
        if from_anchor in anchor_edges and anchor_edges[from_anchor][0] is not None:
            from_exit_idx = 1 if edge_config[from_anchor_idx] else 0
            from_exit = anchor_edges[from_anchor][from_exit_idx]
        else:
            from_exit = None
        
        # Determine entry vertex for to_anchor
        if to_anchor in anchor_edges and anchor_edges[to_anchor][0] is not None:
            to_entry_idx = 0 if edge_config[to_anchor_idx] else 1
            to_entry = anchor_edges[to_anchor][to_entry_idx]
        else:
            to_entry = None
        
        # Determine adaptive depth for this bridge if requested
        if max_depth == -1:
            current_bridge_vertices = base_vertices_per_bridge + (1 if i < extra else 0)
        else:
            current_bridge_vertices = max_depth
        
        # Build the bridge from from_anchor to to_anchor using specified entry/exit points
        current = from_anchor
        depth = 0
        
        # First, try to connect to the exit vertex of from_anchor if specified
        if from_exit is not None and from_exit not in visited:
            total_weight += graph[current][from_exit]
            cycle.append(from_exit)
            visited.add(from_exit)
            current = from_exit
            depth += 1
        
        # Build bridge to to_anchor, preferring to connect via to_entry if possible
        while current != to_anchor and depth < current_bridge_vertices:
            # If we can connect to the entry vertex of to_anchor, prioritize that
            if to_entry is not None and to_entry not in visited and depth == current_bridge_vertices - 1:
                total_weight += graph[current][to_entry]
                cycle.append(to_entry)
                visited.add(to_entry)
                current = to_entry
                depth += 1
                break
            
            # Otherwise, use greedy selection
            next_vertex = _find_next_vertex(graph, current, visited, to_anchor, early_exit, all_anchors)
            
            if next_vertex is not None:
                total_weight += graph[current][next_vertex]
                cycle.append(next_vertex)
                
                if next_vertex == to_anchor:
                    break
                
                visited.add(next_vertex)
                current = next_vertex
                depth += 1
            else:
                # Force connection to target anchor
                total_weight += graph[current][to_anchor]
                cycle.append(to_anchor)
                break
        
        # Connect to to_entry if we haven't reached to_anchor yet
        if current != to_anchor:
            if to_entry is not None and to_entry not in visited:
                total_weight += graph[current][to_entry]
                cycle.append(to_entry)
                visited.add(to_entry)
                current = to_entry
            
            # Finally connect to to_anchor
            if current != to_anchor:
                total_weight += graph[current][to_anchor]
                cycle.append(to_anchor)
    
    # Visit remaining unvisited vertices
    current = cycle[-1]
    all_vertices = set(range(n))
    remaining = all_vertices - visited
    
    while remaining:
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
def best_multi_anchor_heuristic(graph, num_anchors, max_depth=-1, early_exit=False):
    """
    Tries all possible anchor combinations of a given size and returns the best Hamiltonian cycle.
    
    Args:
        graph: A complete, weighted graph represented as an adjacency matrix
        num_anchors: Number of anchors to use (must be >= 2)
        max_depth: Maximum bridge depth (-1 for adaptive)
        early_exit: Flag for early connection to target anchor
        
    Returns:
        Tuple of (best_cycle, best_weight) where:
        - best_cycle: List of vertices representing the best Hamiltonian cycle found
        - best_weight: Total weight of the best cycle
    """
    import itertools
    
    n = len(graph)
    
    # Validation
    if num_anchors < 2:
        raise ValueError("Number of anchors must be at least 2")
    
    if num_anchors > n:
        raise ValueError(f"Number of anchors ({num_anchors}) cannot exceed number of vertices ({n})")
    
    # Check if we have enough vertices for the anchor structure
    if n < 4 * num_anchors:
        print(f"Warning: Graph has {n} vertices but ideally needs {4 * num_anchors} for {num_anchors} anchors")
    
    best_cycle = None
    best_weight = float('inf')
    best_anchors = None
    total_combinations = 0
    
    # Generate all possible anchor combinations
    for anchor_combination in itertools.combinations(range(n), num_anchors):
        total_combinations += 1
        anchors = list(anchor_combination)
        
        # Try each anchor in this combination as the starting vertex
        for start_vertex in anchors:
            try:
                cycle, weight = hamiltonian_cycle_heuristic(
                    graph, 
                    start_vertex, 
                    anchors=anchors, 
                    max_depth=max_depth, 
                    early_exit=early_exit
                )
                
                if weight < best_weight:
                    best_weight = weight
                    best_cycle = cycle
                    best_anchors = anchors
                    
            except (ValueError, IndexError) as e:
                # Skip combinations that cause errors
                continue
    
    if best_cycle is None:
        raise ValueError("No valid Hamiltonian cycle could be constructed from any anchor combination")
    
    # print(f"Evaluated {total_combinations} anchor combinations")
    # print(f"Best anchors found: {best_anchors}")
    return best_cycle, best_weight


def smart_multi_anchor_heuristic(graph, num_anchors, max_depth=-1, early_exit=False, selection_strategy="diverse"):
    """
    A smarter version that uses heuristics to select promising anchor combinations
    instead of trying all possible combinations.
    
    Args:
        graph: A complete, weighted graph represented as an adjacency matrix
        num_anchors: Number of anchors to use
        max_depth: Maximum bridge depth (-1 for adaptive)
        early_exit: Flag for early connection to target anchor
        selection_strategy: Strategy for selecting anchor combinations
                          - "diverse": Select anchors that are far apart
                          - "central": Select vertices with low average distances
                          - "mixed": Combine diverse and central strategies
        
    Returns:
        Tuple of (best_cycle, best_weight, best_anchors)
    """
    n = len(graph)
    
    if num_anchors < 2:
        raise ValueError("Number of anchors must be at least 2")
    
    if num_anchors > n:
        raise ValueError(f"Number of anchors ({num_anchors}) cannot exceed number of vertices ({n})")
    
    # Generate promising anchor combinations based on strategy
    if selection_strategy == "diverse":
        anchor_combinations = _generate_diverse_anchors(graph, num_anchors)
    elif selection_strategy == "central":
        anchor_combinations = _generate_central_anchors(graph, num_anchors)
    elif selection_strategy == "mixed":
        diverse_combos = _generate_diverse_anchors(graph, num_anchors, limit=10)
        central_combos = _generate_central_anchors(graph, num_anchors, limit=10)
        anchor_combinations = diverse_combos + central_combos
    else:
        raise ValueError(f"Unknown selection strategy: {selection_strategy}")
    
    best_cycle = None
    best_weight = float('inf')
    best_anchors = None
    
    # print(f"Trying {len(anchor_combinations)} promising anchor combinations")
    
    for anchors in anchor_combinations:
        # Try each anchor as starting vertex
        for start_vertex in anchors:
            try:
                cycle, weight = hamiltonian_cycle_heuristic(
                    graph, 
                    start_vertex, 
                    anchors=anchors, 
                    max_depth=max_depth, 
                    early_exit=early_exit
                )
                
                if weight < best_weight:
                    best_weight = weight
                    best_cycle = cycle
                    best_anchors = anchors
                    
            except (ValueError, IndexError) as e:
                continue
    
    if best_cycle is None:
        raise ValueError("No valid Hamiltonian cycle could be constructed")
    
    # print(f"Best anchors found: {best_anchors}")
    return best_cycle, best_weight


def _generate_diverse_anchors(graph, num_anchors, limit=20):
    """
    Generate anchor combinations where anchors are maximally distant from each other.
    """
    import itertools
    n = len(graph)
    
    # Calculate all pairwise distances and rank combinations by total distance
    combinations_with_scores = []
    
    for combo in itertools.combinations(range(n), num_anchors):
        total_distance = 0
        for i in range(len(combo)):
            for j in range(i + 1, len(combo)):
                total_distance += graph[combo[i]][combo[j]]
        combinations_with_scores.append((combo, total_distance))
    
    # Sort by total distance (descending - we want diverse anchors)
    combinations_with_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Return top combinations (convert back to lists)
    return [list(combo[0]) for combo in combinations_with_scores[:limit]]


def _generate_central_anchors(graph, num_anchors, limit=20):
    """
    Generate anchor combinations using vertices with low average distances to all other vertices.
    """
    import itertools
    n = len(graph)
    
    # Calculate centrality score for each vertex (lower is better)
    centrality_scores = []
    for v in range(n):
        avg_distance = sum(graph[v]) / (n - 1)
        centrality_scores.append((v, avg_distance))
    
    # Sort by centrality score
    centrality_scores.sort(key=lambda x: x[1])
    
    # Take the most central vertices as candidates
    central_candidates = [v[0] for v in centrality_scores[:min(num_anchors * 3, n)]]
    
    # Generate combinations from these candidates
    combinations_with_scores = []
    for combo in itertools.combinations(central_candidates, num_anchors):
        total_centrality = sum(centrality_scores[v][1] for v in combo)
        combinations_with_scores.append((combo, total_centrality))
    
    # Sort by total centrality (ascending - we want central vertices)
    combinations_with_scores.sort(key=lambda x: x[1])
    
    return [list(combo[0]) for combo in combinations_with_scores[:limit]]