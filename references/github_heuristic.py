"""
Refactored TSP heuristics from traveling_salesman.py
Adapted to work with adjacency matrices and return cycles with costs
"""

import heapq
import functools
from collections import defaultdict

def calculate_cycle_cost(cycle, graph):
    """Calculate total cost of a cycle"""
    if len(cycle) < 2:
        return 0
    total = 0
    for i in range(len(cycle) - 1):
        total += graph[cycle[i]][cycle[i + 1]]
    return total

def union_find_create():
    """Create union-find data structure"""
    return {'group_id': 0, 'groups': {}, 'id': {}}

def union_find_union(uf, a, b):
    """Union operation for union-find"""
    A, B = a in uf['id'], b in uf['id']
    if A and B and uf['id'][a] != uf['id'][b]:
        _merge(uf, a, b)
    elif A and B:
        return False
    elif A or B:
        _add(uf, a, b)
    else:
        _create(uf, a, b)
    return True

def _merge(uf, a, b):
    """Merge two groups"""
    obs, targ = sorted((uf['id'][a], uf['id'][b]), key=lambda i: len(uf['groups'][i]))
    for node in uf['groups'][obs]:
        uf['id'][node] = targ
    uf['groups'][targ] |= uf['groups'][obs]
    del uf['groups'][obs]

def _add(uf, a, b):
    """Add new city to existing group"""
    a, b = (a, b) if a in uf['id'] else (b, a)
    targ = uf['id'][a]
    uf['id'][b] = targ
    uf['groups'][targ] |= {b}

def _create(uf, a, b):
    """Create new group"""
    uf['groups'][uf['group_id']] = {a, b}
    uf['id'][a] = uf['id'][b] = uf['group_id']
    uf['group_id'] += 1

def build_mst(graph):
    """Build minimum spanning tree using Kruskal's algorithm"""
    n = len(graph)
    edges = []
    
    # Create edge heap
    h = []
    for i in range(n):
        for j in range(i + 1, n):
            heapq.heappush(h, (graph[i][j], i, j))
    
    # Build MST
    uf = union_find_create()
    mst_edges = []
    while len(mst_edges) < n - 1 and h:
        dist, a, b = heapq.heappop(h)
        if union_find_union(uf, a, b):
            mst_edges.append((a, b))
    
    return mst_edges

def mst_to_graph(mst_edges):
    """Convert MST edge list to adjacency list"""
    graph = defaultdict(list)
    for a, b in mst_edges:
        graph[a].append(b)
        graph[b].append(a)
    return graph

def dfs_preorder(graph, start, visited=None):
    """DFS preorder traversal"""
    if visited is None:
        visited = set()
    
    path = [start]
    visited.add(start)
    
    for neighbor in graph[start]:
        if neighbor not in visited:
            sub_path = dfs_preorder(graph, neighbor, visited)
            path.extend(sub_path)
    
    return path

def heuristic_path(graph, start_vertex):
    """MST + preorder traversal heuristic"""
    n = len(graph)
    mst_edges = build_mst(graph)
    mst_graph = mst_to_graph(mst_edges)
    
    best_cycle = None
    best_cost = float('inf')
    
    # Try each vertex as starting point
    for start in range(n):
        visited = set()
        path = dfs_preorder(mst_graph, start, visited)
        cycle = path + [start]  # Complete the cycle
        cost = calculate_cycle_cost(cycle, graph)
        
        if cost < best_cost:
            best_cost = cost
            best_cycle = cycle
    
    return best_cycle, best_cost

def path_relaxation(cycle, graph):
    """Relax path by relocating vertices"""
    def reduced_cost(i):
        """Cost reduction from removing vertex i"""
        if i == 0 or i == len(cycle) - 1:
            return 0
        a, b, c = cycle[i-1], cycle[i], cycle[i+1]
        return graph[a][c] - graph[a][b] - graph[b][c]
    
    def insertion_cost(a, b, c):
        """Cost of inserting b between a and c"""
        return graph[a][b] + graph[b][c] - graph[a][c]
    
    points = cycle[:-1]  # Remove duplicate start vertex
    adjusted = set()
    
    while len(adjusted) < len(points) - 1:
        for i in range(1, len(points)):
            if points[i] not in adjusted:
                adjusted.add(points[i])
                rc = reduced_cost(i)
                best = 0
                best_index = i
                
                for j in range(len(points)):
                    if j != i and j != i - 1:
                        next_j = (j + 1) % len(points)
                        ic = insertion_cost(points[j], points[i], points[next_j])
                        total_cost = ic + rc
                        if total_cost < best:
                            best = total_cost
                            best_index = j
                
                # Relocate vertex if beneficial
                if best < 0 and best_index != i:
                    vertex = points.pop(i)
                    insert_pos = best_index if best_index < i else best_index
                    points.insert(insert_pos + 1, vertex)
                    break
    
    return points + [points[0]], calculate_cycle_cost(points + [points[0]], graph)

def relaxed_heur_path(graph, start_vertex):
    """Heuristic path with relaxation"""
    cycle, cost = heuristic_path(graph, start_vertex)
    
    prev_cost = float('inf')
    while cost < prev_cost:
        prev_cost = cost
        cycle, cost = path_relaxation(cycle, graph)
    
    return cycle, cost

def partial_tour_optimization(cycle, graph, k):
    """Optimize k-length subpaths using dynamic programming"""
    @functools.lru_cache(None)
    def dp_helper(prev_idx, used_mask, segment):
        """DP helper for optimizing segment"""
        if used_mask == target_mask:
            return graph[segment[prev_idx]][end_vertex], [end_vertex]
        
        best_cost = float('inf')
        best_path = []
        
        for i in range(len(segment)):
            if not (used_mask & (1 << i)):
                cost, path = dp_helper(i, used_mask | (1 << i), segment)
                cost += graph[segment[prev_idx]][segment[i]]
                if cost < best_cost:
                    best_cost = cost
                    best_path = [segment[i]] + path
        
        return best_cost, best_path
    
    points = cycle[:-1]  # Remove duplicate start
    n = len(points)
    
    for i in range(n):
        if i + k + 1 < n:
            # Optimize segment from i to i+k+1
            start_vertex = points[i]
            end_vertex = points[i + k + 1]
            segment = points[i + 1:i + k + 1]
            
            if len(segment) == 0:
                continue
            
            target_mask = (1 << len(segment)) - 1
            cost, optimized_path = dp_helper(0, 1, [start_vertex] + segment)
            dp_helper.cache_clear()
            
            # Replace segment with optimized path
            points = points[:i + 1] + optimized_path[1:-1] + points[i + k + 1:]
    
    return points + [points[0]], calculate_cycle_cost(points + [points[0]], graph)

def k_optimized_path(graph, start_vertex, k=None):
    """k-optimized heuristic path"""
    if k is None:
        k = len(graph) // 4
    
    cycle, cost = relaxed_heur_path(graph, start_vertex)
    return partial_tour_optimization(cycle, graph, k)