import heapq
from itertools import combinations
import random

def find_k_lowest_indices(values, exclude_vertex, k=2):
    """Find k vertices with lowest edge weights, excluding specified vertex."""
    if len(values) - 1 < k:  # -1 because we exclude one vertex
        raise ValueError(f"Graph must have at least {k+1} vertices.")
    
    candidates = [(values[i], i) for i in range(len(values)) if i != exclude_vertex]
    candidates.sort()
    return [idx for _, idx in candidates[:k]]

def calculate_insertion_cost(graph, path, new_vertex, position):
    """Calculate cost of inserting new_vertex at given position in path."""
    if position == 0 or position >= len(path):
        return float('inf')
    
    prev_vertex = path[position - 1]
    next_vertex = path[position]
    
    # Cost = remove old edge + add two new edges
    old_cost = graph[prev_vertex][next_vertex]
    new_cost = graph[prev_vertex][new_vertex] + graph[new_vertex][next_vertex]
    
    return new_cost - old_cost

def construct_adaptive_anchor_cycle(graph, start, initial_anchors, adaptation_rate=0.3):
    """Constructs cycle with adaptive anchor selection during construction."""
    vertices_count = len(graph)
    visited = set([start])
    path = [start]
    current_vertex = start
    total_weight = 0
    
    # Copy anchors list so we can modify it
    anchors = initial_anchors.copy()
    steps_since_adaptation = 0
    
    while len(visited) < vertices_count:
        next_vertex = None
        
        # If we have anchors and should use them
        if anchors and random.random() < (1 - adaptation_rate):
            # Try to use next anchor
            for anchor in anchors:
                if anchor not in visited:
                    next_vertex = anchor
                    anchors.remove(anchor)
                    break
        
        # If no anchor selected, use greedy choice
        if next_vertex is None:
            lowest_weight = float("inf")
            for i in range(vertices_count):
                if i not in visited and graph[current_vertex][i] < lowest_weight:
                    next_vertex = i
                    lowest_weight = graph[current_vertex][i]
        
        if next_vertex is None:
            break
            
        visited.add(next_vertex)
        path.append(next_vertex)
        total_weight += graph[current_vertex][next_vertex]
        current_vertex = next_vertex
        steps_since_adaptation += 1
        
        # Adaptive anchor reselection
        if steps_since_adaptation >= 3 and len(visited) < vertices_count - 2:
            unvisited = [i for i in range(vertices_count) if i not in visited]
            if len(unvisited) >= 2:
                # Find new anchors based on current position
                anchor_candidates = [(graph[current_vertex][v], v) for v in unvisited]
                anchor_candidates.sort()
                anchors = [v for _, v in anchor_candidates[:2]]
                steps_since_adaptation = 0
    
    # Complete the cycle
    total_weight += graph[current_vertex][start]
    path.append(start)
    
    return path, total_weight

def construct_multi_anchor_cycle(graph, start, anchors):
    """Constructs cycle using multiple anchors (3 or more)."""
    vertices_count = len(graph)
    visited = set([start])
    path = [start]
    current_vertex = start
    total_weight = 0
    
    # Visit anchors in order of distance from start
    anchor_distances = [(graph[start][anchor], anchor) for anchor in anchors if anchor != start]
    anchor_distances.sort()
    ordered_anchors = [anchor for _, anchor in anchor_distances]
    
    # Visit anchors first
    for anchor in ordered_anchors:
        if anchor not in visited:
            path.append(anchor)
            visited.add(anchor)
            total_weight += graph[current_vertex][anchor]
            current_vertex = anchor
    
    # Fill in remaining vertices greedily
    while len(visited) < vertices_count:
        next_vertex = None
        lowest_weight = float("inf")
        
        for i in range(vertices_count):
            if i not in visited and graph[current_vertex][i] < lowest_weight:
                next_vertex = i
                lowest_weight = graph[current_vertex][i]
        
        if next_vertex is None:
            break
            
        visited.add(next_vertex)
        path.append(next_vertex)
        total_weight += lowest_weight
        current_vertex = next_vertex
    
    # Complete the cycle
    total_weight += graph[current_vertex][start]
    path.append(start)
    
    return path, total_weight

def construct_insertion_guided_cycle(graph, start, anchors):
    """Constructs cycle using cheapest insertion with anchor guidance."""
    vertices_count = len(graph)
    
    # Start with a partial tour of start and first anchor
    if not anchors or anchors[0] == start:
        # Fallback to basic greedy if no valid anchors
        return construct_greedy_cycle_basic(graph, start)
    
    path = [start, anchors[0], start]
    visited = set([start, anchors[0]])
    total_weight = graph[start][anchors[0]] + graph[anchors[0]][start]
    
    # Insert remaining vertices using cheapest insertion
    unvisited_anchors = [a for a in anchors[1:] if a not in visited]
    
    while len(visited) < vertices_count:
        best_vertex = None
        best_position = None
        best_cost = float('inf')
        
        # Prioritize remaining anchors
        candidates = unvisited_anchors if unvisited_anchors else [i for i in range(vertices_count) if i not in visited]
        
        for vertex in candidates:
            if vertex in visited:
                continue
                
            # Find best insertion position for this vertex
            for pos in range(1, len(path)):
                cost = calculate_insertion_cost(graph, path, vertex, pos)
                if cost < best_cost:
                    best_cost = cost
                    best_vertex = vertex
                    best_position = pos
        
        if best_vertex is None:
            break
            
        # Insert the vertex
        path.insert(best_position, best_vertex)
        visited.add(best_vertex)
        total_weight += best_cost
        
        # Remove from unvisited anchors if applicable
        if best_vertex in unvisited_anchors:
            unvisited_anchors.remove(best_vertex)
    
    return path, total_weight

def construct_probabilistic_cycle(graph, start, anchors, temperature=1.0):
    """Constructs cycle using probabilistic selection influenced by anchors."""
    vertices_count = len(graph)
    visited = set([start])
    path = [start]
    current_vertex = start
    total_weight = 0
    
    anchor_set = set(anchors)
    
    while len(visited) < vertices_count:
        candidates = [i for i in range(vertices_count) if i not in visited]
        if not candidates:
            break
        
        # Calculate probabilities
        weights = []
        for candidate in candidates:
            base_cost = graph[current_vertex][candidate]
            # Boost probability for anchors
            if candidate in anchor_set:
                adjusted_cost = base_cost * 0.5  # Make anchors more attractive
            else:
                adjusted_cost = base_cost
            
            # Convert to probability (lower cost = higher probability)
            prob = 1.0 / (adjusted_cost ** (1.0 / temperature))
            weights.append(prob)
        
        # Weighted random selection
        total_prob = sum(weights)
        r = random.random() * total_prob
        cumulative = 0
        next_vertex = candidates[-1]  # fallback
        
        for i, weight in enumerate(weights):
            cumulative += weight
            if r <= cumulative:
                next_vertex = candidates[i]
                break
        
        visited.add(next_vertex)
        path.append(next_vertex)
        total_weight += graph[current_vertex][next_vertex]
        current_vertex = next_vertex
        
        # Remove from anchor set if visited
        anchor_set.discard(next_vertex)
    
    # Complete the cycle
    total_weight += graph[current_vertex][start]
    path.append(start)
    
    return path, total_weight

def construct_greedy_cycle_basic(graph, start):
    """Basic greedy cycle construction for fallback."""
    vertices_count = len(graph)
    visited = set([start])
    path = [start]
    current_vertex = start
    total_weight = 0
    
    while len(visited) < vertices_count:
        next_vertex = None
        lowest_weight = float("inf")
        
        for i in range(vertices_count):
            if i not in visited and graph[current_vertex][i] < lowest_weight:
                next_vertex = i
                lowest_weight = graph[current_vertex][i]
        
        if next_vertex is None:
            break
            
        visited.add(next_vertex)
        path.append(next_vertex)
        total_weight += lowest_weight
        current_vertex = next_vertex
    
    total_weight += graph[current_vertex][start]
    path.append(start)
    
    return path, total_weight

# Enhanced heuristic implementations

def adaptive_anchor_heuristic(graph, vertex, adaptation_rate=0.3):
    """Enhanced version with adaptive anchor selection during construction."""
    anchors = find_k_lowest_indices(graph[vertex], vertex, k=2)
    
    cycle_1, weight_1 = construct_adaptive_anchor_cycle(graph, vertex, anchors, adaptation_rate)
    cycle_2, weight_2 = construct_adaptive_anchor_cycle(graph, vertex, anchors[::-1], adaptation_rate)
    
    return (cycle_1, weight_1) if weight_1 < weight_2 else (cycle_2, weight_2)

def multi_anchor_heuristic(graph, vertex, num_anchors=3):
    """Uses more than 2 anchors for better guidance."""
    anchors = find_k_lowest_indices(graph[vertex], vertex, k=min(num_anchors, len(graph) - 1))
    
    best_cycle = None
    best_weight = float('inf')
    
    # Try different permutations of anchor ordering
    for perm in combinations(anchors, min(len(anchors), num_anchors)):
        cycle, weight = construct_multi_anchor_cycle(graph, vertex, list(perm))
        if weight < best_weight:
            best_weight = weight
            best_cycle = cycle
    
    return best_cycle, best_weight

def insertion_anchor_heuristic(graph, vertex):
    """Combines anchor guidance with cheapest insertion strategy."""
    anchors = find_k_lowest_indices(graph[vertex], vertex, k=3)
    
    cycle_1, weight_1 = construct_insertion_guided_cycle(graph, vertex, anchors)
    cycle_2, weight_2 = construct_insertion_guided_cycle(graph, vertex, anchors[::-1])
    
    return (cycle_1, weight_1) if weight_1 < weight_2 else (cycle_2, weight_2)

def probabilistic_anchor_heuristic(graph, vertex, temperature=1.5):
    """Uses probabilistic selection with anchor bias."""
    anchors = find_k_lowest_indices(graph[vertex], vertex, k=3)
    
    best_cycle = None
    best_weight = float('inf')
    
    # Run multiple times due to randomness
    for _ in range(5):
        cycle, weight = construct_probabilistic_cycle(graph, vertex, anchors, temperature)
        if weight < best_weight:
            best_weight = weight
            best_cycle = cycle
    
    return best_cycle, best_weight

def hybrid_anchor_heuristic(graph, vertex):
    """Combines multiple strategies and returns the best result."""
    strategies = [
        lambda: adaptive_anchor_heuristic(graph, vertex),
        lambda: multi_anchor_heuristic(graph, vertex, 3),
        lambda: insertion_anchor_heuristic(graph, vertex),
        lambda: probabilistic_anchor_heuristic(graph, vertex)
    ]
    
    best_cycle = None
    best_weight = float('inf')
    
    for strategy in strategies:
        try:
            cycle, weight = strategy()
            if weight < best_weight:
                best_weight = weight
                best_cycle = cycle
        except Exception:
            continue  # Skip failed strategies
    
    return best_cycle, best_weight

def smart_anchor_heuristic(graph, vertex):
    """Intelligently selects anchors based on graph structure analysis."""
    vertices_count = len(graph)
    
    # Calculate vertex importance scores
    importance_scores = []
    for i in range(vertices_count):
        if i == vertex:
            importance_scores.append(-1)  # Exclude start vertex
            continue
            
        # Score based on multiple factors
        avg_distance = sum(graph[i]) / vertices_count
        min_distance = min(graph[i])
        max_distance = max(graph[i])
        distance_variance = sum((graph[i][j] - avg_distance) ** 2 for j in range(vertices_count)) / vertices_count
        
        # Prefer vertices that are:
        # 1. Close to start vertex
        # 2. Have low average distances to other vertices
        # 3. Have low variance (well-connected)
        score = (graph[vertex][i] * 0.4 + avg_distance * 0.3 + distance_variance * 0.3)
        importance_scores.append(score)
    
    # Select anchors based on importance scores
    anchor_candidates = [(score, i) for i, score in enumerate(importance_scores) if score >= 0]
    anchor_candidates.sort()
    anchors = [i for _, i in anchor_candidates[:3]]
    
    return construct_multi_anchor_cycle(graph, vertex, anchors)