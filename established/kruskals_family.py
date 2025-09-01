from graph_generator import calculate_cycle_cost
from utils.base_heuristics import EstablishedHeuristic

class KruskalsFamily(EstablishedHeuristic):
    """
    Implementation of Kruskal's Family TSP heuristics.
    """

    def __init__(self):
        self.graph = None

    def solve(self, graph):
        """
        Pure greedy edge-based TSP solver (v3).

        Args:
            graph: Adjacency matrix represented as a list of lists with edge weights

        Returns:
            list: Cycle path including return to start
        """
        self.graph = graph
        n = len(graph)
        if n <= 1:
            return [0]
        if n == 2:
            return [0, 1, 0]

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

        # Select edges
        for weight, u, v in edges:
            if degree[u] >= 2 or degree[v] >= 2:
                continue

            if len(selected_edges) == n - 1:
                endpoints = [i for i in range(n) if degree[i] == 1]
                if len(endpoints) == 2 and {u, v} == set(endpoints):
                    selected_edges.append((u, v, weight))
                    degree[u] += 1
                    degree[v] += 1
                    break
            else:
                if find(u) != find(v):
                    selected_edges.append((u, v, weight))
                    degree[u] += 1
                    degree[v] += 1
                    union(u, v)

        # Build adjacency list
        adj_list = [[] for _ in range(n)]
        for u, v, weight in selected_edges:
            adj_list[u].append(v)
            adj_list[v].append(u)

        # Traverse the cycle
        start = 0
        path = []
        current = start
        prev = -1

        for _ in range(n):
            path.append(current)
            next_vertex = None
            for neighbor in adj_list[current]:
                if neighbor != prev:
                    next_vertex = neighbor
                    break
            prev = current
            current = next_vertex

        path.append(start)
        return path

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
        return "Kruskal's Family"