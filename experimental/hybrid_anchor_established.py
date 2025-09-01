import heapq
from collections import defaultdict
from graph_generator import calculate_cycle_cost
from utils.base_heuristics import ExperimentalHeuristic
from anchoring.low_anchor_heuristic import LowAnchorHeuristic

class HybridAnchorEstablished(ExperimentalHeuristic):
    """
    Hybrid heuristic combining low anchor strategy with Christofides-like construction.
    Uses low anchor to select optimal starting points and anchors, then applies
    Christofides algorithm with anchor-biased construction.
    """

    def __init__(self):
        self.graph = None

    def solve(self, graph):
        """
        Solve TSP using hybrid anchor-established approach.

        Args:
            graph: 2D list/array representing adjacency matrix of a complete graph

        Returns:
            list: Hamiltonian cycle as a list of vertices, including return to start
        """
        self.graph = graph
        n = len(graph)
        if n < 3:
            raise ValueError("Graph must have at least 3 vertices for TSP")

        # Step 1: Use low anchor heuristic to find optimal starting configuration
        heuristic = LowAnchorHeuristic()
        initial_cycle = heuristic.solve(graph)

        # Extract starting vertex and key anchors from initial cycle
        start_vertex = initial_cycle[0]
        # Find vertices with highest degree/connectivity as potential anchors
        anchor_candidates = self._identify_anchor_candidates(graph, initial_cycle)

        # Step 2: Apply Christofides-like construction with anchor bias
        cycle = self._christofides_with_anchor_bias(graph, start_vertex, anchor_candidates)

        return cycle

    def evaluate(self, cycle):
        """
        Evaluate the cost of a given cycle.

        Args:
            cycle: List of vertices representing the cycle.

        Returns:
            float: Total weight/cost of the cycle.
        """
        if self.graph is None:
            raise ValueError("Graph not set. Call solve() first.")
        return calculate_cycle_cost(cycle, self.graph)

    def get_name(self):
        """
        Get the name of the heuristic.

        Returns:
            str: Name of the heuristic.
        """
        return "Hybrid Anchor Established"

    def _identify_anchor_candidates(self, graph, initial_cycle):
        """Identify potential anchor vertices from initial cycle based on connectivity."""
        n = len(graph)
        # Use vertices that appear to be structural points in the initial cycle
        # For simplicity, select vertices with above-average degree
        degrees = [sum(1 for j in range(n) if graph[i][j] < float('inf') and i != j) for i in range(n)]
        avg_degree = sum(degrees) / n

        anchor_candidates = [i for i in range(n) if degrees[i] > avg_degree]
        # Ensure we have at least 2 anchors
        if len(anchor_candidates) < 2:
            # Fallback: use first and last vertices from initial cycle
            anchor_candidates = [initial_cycle[0], initial_cycle[-2]]  # -2 to avoid duplicate with start

        return anchor_candidates[:min(4, len(anchor_candidates))]  # Limit to 4 anchors

    def _christofides_with_anchor_bias(self, graph, start_vertex, anchor_candidates):
        """
        Modified Christofides algorithm that biases towards anchor candidates.
        """
        n = len(graph)

        # Step 1: Find MST with anchor bias (prefer edges connected to anchors)
        mst_edges = self._find_biased_mst(graph, anchor_candidates)

        # Step 2: Find vertices with odd degree in MST
        odd_degree_vertices = self._find_odd_degree_vertices(mst_edges, n)

        # Step 3: Find minimum weight perfect matching on odd degree vertices
        matching_edges = self._find_minimum_weight_perfect_matching(graph, odd_degree_vertices)

        # Step 4: Combine MST and matching to form Eulerian multigraph
        eulerian_edges = mst_edges + matching_edges

        # Step 5: Find Eulerian tour starting from our preferred start vertex
        eulerian_tour = self._find_eulerian_tour(eulerian_edges, n, start_vertex)

        # Step 6: Convert to Hamiltonian cycle
        hamiltonian_cycle = self._convert_to_hamiltonian_cycle(eulerian_tour, n, start_vertex)

        return hamiltonian_cycle

    def _find_biased_mst(self, graph, anchor_candidates):
        """
        Find MST with bias towards anchor candidates.
        Edges connected to anchors get priority in case of weight ties.
        """
        n = len(graph)
        visited = [False] * n
        mst_edges = []
        anchor_set = set(anchor_candidates)

        # Priority queue: (weight, vertex, parent, anchor_bonus)
        pq = [(0, 0, -1, 0)]

        while pq:
            weight, u, parent, _ = heapq.heappop(pq)

            if visited[u]:
                continue

            visited[u] = True

            if parent != -1:
                mst_edges.append((parent, u, weight))

            # Add all adjacent vertices to priority queue with anchor bias
            for v in range(n):
                if not visited[v] and graph[u][v] != float('inf'):
                    # Add small bonus for edges connected to anchors
                    anchor_bonus = 0.001 if v in anchor_set else 0
                    effective_weight = graph[u][v] - anchor_bonus
                    heapq.heappush(pq, (effective_weight, v, u, anchor_bonus))

        return mst_edges

    @staticmethod
    def _find_odd_degree_vertices(mst_edges, n):
        """Find vertices with odd degree in the MST."""
        degree = [0] * n
        for u, v, _ in mst_edges:
            degree[u] += 1
            degree[v] += 1
        return [i for i in range(n) if degree[i] % 2 == 1]

    @staticmethod
    def _find_minimum_weight_perfect_matching(graph, odd_vertices):
        """Find minimum weight perfect matching using greedy approach."""
        if len(odd_vertices) % 2 != 0:
            raise ValueError("Number of odd degree vertices must be even")

        if len(odd_vertices) == 0:
            return []

        matching_edges = []
        vertices = odd_vertices.copy()

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

    @staticmethod
    def _find_eulerian_tour(edges, n, start_vertex):
        """Find Eulerian tour using Hierholzer's algorithm starting from specified vertex."""
        if not edges:
            return [start_vertex]

        # Build adjacency list
        adj = defaultdict(list)
        for u, v, _ in edges:
            adj[u].append(v)
            adj[v].append(u)

        # Hierholzer's algorithm starting from specified vertex
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

    @staticmethod
    def _convert_to_hamiltonian_cycle(eulerian_tour, n, start_vertex):
        """Convert Eulerian tour to Hamiltonian cycle."""
        if not eulerian_tour:
            return list(range(n)) + [start_vertex]

        visited = set()
        hamiltonian_cycle = []

        for vertex in eulerian_tour:
            if vertex not in visited:
                hamiltonian_cycle.append(vertex)
                visited.add(vertex)

        # Ensure all vertices are included
        for vertex in range(n):
            if vertex not in visited:
                hamiltonian_cycle.append(vertex)
                visited.add(vertex)

        # Complete the cycle
        if hamiltonian_cycle and hamiltonian_cycle[-1] != start_vertex:
            hamiltonian_cycle.append(start_vertex)

        return hamiltonian_cycle