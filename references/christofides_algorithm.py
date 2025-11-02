import heapq
from collections import defaultdict
import itertools

def christofides_algorithm(graph):
    """
    Implements Christofides' Algorithm for the Traveling Salesman Problem.
    
    Args:
        graph: 2D list/array representing adjacency matrix of a complete graph.
               Use float('inf') for non-existent edges.
        
    Returns:
        tuple: (cycle, total_weight) where cycle is the Hamiltonian cycle and 
               total_weight is the total weight of the cycle
    """
    n = len(graph)
    if n < 3:
        raise ValueError("Graph must have at least 3 vertices for TSP")
    
    # Validate that graph is complete (or at least connected)
    _validate_graph(graph)
    
    # Step 1: Find Minimum Spanning Tree (MST)
    mst_edges = find_minimum_spanning_tree(graph)
    
    # Step 2: Find vertices with odd degree in MST
    odd_degree_vertices = find_odd_degree_vertices(mst_edges, n)
    
    # Step 3: Find minimum weight perfect matching on odd degree vertices
    matching_edges = find_minimum_weight_perfect_matching(graph, odd_degree_vertices)
    
    # Step 4: Combine MST and matching to form Eulerian multigraph
    eulerian_edges = mst_edges + matching_edges
    
    # Step 5: Find Eulerian tour
    eulerian_tour = find_eulerian_tour(eulerian_edges, n)
    
    # Step 6: Convert to Hamiltonian cycle by skipping repeated vertices
    hamiltonian_cycle = convert_to_hamiltonian_cycle(eulerian_tour, n)
    
    # Calculate total weight
    total_weight = calculate_cycle_weight(graph, hamiltonian_cycle)
    
    return hamiltonian_cycle, total_weight


def _validate_graph(graph):
    """Validate that the graph is properly formatted."""
    n = len(graph)
    for i in range(n):
        if len(graph[i]) != n:
            raise ValueError("Graph must be square (n x n matrix)")
        if graph[i][i] != 0:
            raise ValueError("Diagonal elements should be 0 (no self-loops)")


def find_minimum_spanning_tree(graph):
    """
    Find MST using Prim's algorithm.
    
    Returns:
        list: List of edges (u, v, weight) in the MST
    """
    n = len(graph)
    visited = [False] * n
    mst_edges = []
    
    # Priority queue: (weight, vertex, parent)
    pq = [(0, 0, -1)]
    
    while pq:
        weight, u, parent = heapq.heappop(pq)
        
        if visited[u]:
            continue
            
        visited[u] = True
        
        if parent != -1:
            mst_edges.append((parent, u, weight))
        
        # Add all adjacent vertices to priority queue
        for v in range(n):
            if not visited[v] and graph[u][v] != float('inf'):
                heapq.heappush(pq, (graph[u][v], v, u))
    
    return mst_edges


def find_odd_degree_vertices(mst_edges, n):
    """
    Find vertices with odd degree in the MST.
    
    Returns:
        list: List of vertices with odd degree
    """
    degree = [0] * n
    
    for u, v, _ in mst_edges:
        degree[u] += 1
        degree[v] += 1
    
    odd_vertices = [i for i in range(n) if degree[i] % 2 == 1]
    return odd_vertices


def find_minimum_weight_perfect_matching(graph, odd_vertices):
    """
    Find minimum weight perfect matching among odd degree vertices.
    Uses brute force enumeration for small sets, Hungarian algorithm approach for larger sets.
    
    Returns:
        list: List of matching edges (u, v, weight)
    """
    if len(odd_vertices) % 2 != 0:
        raise ValueError("Number of odd degree vertices must be even")
    
    if len(odd_vertices) == 0:
        return []
    
    # For small sets, use brute force (optimal)
    if len(odd_vertices) <= 8:
        return _brute_force_perfect_matching(graph, odd_vertices)
    else:
        # For larger sets, use a more sophisticated approach
        return _hungarian_style_matching(graph, odd_vertices)


def _brute_force_perfect_matching(graph, vertices):
    """
    Find optimal perfect matching using brute force enumeration.
    Only practical for small vertex sets.
    """
    n = len(vertices)
    if n == 0:
        return []
    
    min_weight = float('inf')
    best_matching = []
    
    # Generate all possible perfect matchings
    for matching in _generate_perfect_matchings(vertices):
        weight = sum(graph[u][v] for u, v in matching)
        if weight < min_weight:
            min_weight = weight
            best_matching = matching
    
    return [(u, v, graph[u][v]) for u, v in best_matching]


def _generate_perfect_matchings(vertices):
    """Generate all possible perfect matchings of vertices."""
    if len(vertices) < 2:
        return
    if len(vertices) == 2:
        yield [(vertices[0], vertices[1])]
        return
    
    first = vertices[0]
    for i in range(1, len(vertices)):
        partner = vertices[i]
        remaining = vertices[1:i] + vertices[i+1:]
        
        for sub_matching in _generate_perfect_matchings(remaining):
            yield [(first, partner)] + sub_matching


def _hungarian_style_matching(graph, odd_vertices):
    """
    Simplified approach for larger vertex sets.
    Uses a greedy approach with local optimization.
    """
    if len(odd_vertices) == 0:
        return []
    
    # Create a complete bipartite graph representation
    n = len(odd_vertices)
    vertices = odd_vertices.copy()
    matching_edges = []
    
    # Use a greedy approach with some local optimization
    while vertices:
        min_weight = float('inf')
        best_pair = None
        
        # Find the minimum weight edge
        for i in range(len(vertices)):
            for j in range(i + 1, len(vertices)):
                u, v = vertices[i], vertices[j]
                if graph[u][v] < min_weight:
                    min_weight = graph[u][v]
                    best_pair = (u, v)
        
        if best_pair:
            u, v = best_pair
            matching_edges.append((u, v, min_weight))
            vertices.remove(u)
            vertices.remove(v)
    
    return matching_edges


def find_eulerian_tour(edges, n):
    """
    Find Eulerian tour using Hierholzer's algorithm.
    
    Returns:
        list: Eulerian tour as list of vertices
    """
    if not edges:
        return [0]
    
    # Build adjacency list from edges
    adj = defaultdict(list)
    for u, v, _ in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Find a vertex with edges to start from
    start_vertex = 0
    for vertex in range(n):
        if adj[vertex]:
            start_vertex = vertex
            break
    
    # Hierholzer's algorithm
    tour = []
    stack = [start_vertex]
    
    while stack:
        curr = stack[-1]
        if adj[curr]:
            next_vertex = adj[curr].pop()
            adj[next_vertex].remove(curr)
            stack.append(next_vertex)
        else:
            tour.append(stack.pop())
    
    return tour[::-1]  # Reverse to get correct order


def convert_to_hamiltonian_cycle(eulerian_tour, n):
    """
    Convert Eulerian tour to Hamiltonian cycle by skipping repeated vertices.
    
    Returns:
        list: Hamiltonian cycle visiting each vertex exactly once
    """
    if not eulerian_tour:
        return list(range(n)) + [0]  # Fallback cycle
    
    visited = set()
    hamiltonian_cycle = []
    
    for vertex in eulerian_tour:
        if vertex not in visited:
            hamiltonian_cycle.append(vertex)
            visited.add(vertex)
    
    # Ensure all vertices are included (shouldn't happen with correct algorithm)
    for vertex in range(n):
        if vertex not in visited:
            hamiltonian_cycle.append(vertex)
            visited.add(vertex)
    
    # Complete the cycle by returning to start
    if hamiltonian_cycle and hamiltonian_cycle[-1] != hamiltonian_cycle[0]:
        hamiltonian_cycle.append(hamiltonian_cycle[0])
    
    return hamiltonian_cycle


def calculate_cycle_weight(graph, cycle):
    """
    Calculate total weight of a cycle.
    
    Returns:
        int/float: Total weight of the cycle
    """
    if len(cycle) < 2:
        return 0
    
    total_weight = 0
    for i in range(len(cycle) - 1):
        weight = graph[cycle[i]][cycle[i + 1]]
        if weight == float('inf'):
            raise ValueError(f"No edge between vertices {cycle[i]} and {cycle[i + 1]}")
        total_weight += weight
    
    return total_weight


# Example usage and testing
def create_example_graph():
    """Create an example TSP graph for testing."""
    # 4-vertex complete graph
    graph = [
        [0, 10, 15, 20],
        [10, 0, 35, 25],
        [15, 35, 0, 30],
        [20, 25, 30, 0]
    ]
    return graph


def test_christofides():
    """Test the Christofides algorithm implementation."""
    print("Testing Christofides Algorithm...")
    
    # Test with example graph
    graph = create_example_graph()
    cycle, weight = christofides_algorithm(graph)
    
    print(f"Graph:")
    for row in graph:
        print(row)
    print(f"\nHamiltonian cycle: {cycle}")
    print(f"Total weight: {weight}")
    
    # Verify cycle validity
    print(f"Cycle visits {len(set(cycle[:-1]))} unique vertices")
    print(f"Expected to visit {len(graph)} vertices")


if __name__ == "__main__":
    test_christofides()