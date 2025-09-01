from graph_generator import calculate_cycle_cost
from utils.base_heuristics import EstablishedHeuristic

class PrimsTSP(EstablishedHeuristic):
    """
    Implementation of Prim's TSP heuristic.
    """

    def __init__(self):
        self.graph = None

    def solve(self, graph):
        """
        Best Prim's TSP: Apply Prim's TSP from all vertices, return best result

        Args:
            graph: 2D list/array representing adjacency matrix of a complete graph

        Returns:
            list: Best cycle including return to start
        """
        self.graph = graph
        n = len(graph)
        best_cycle = None
        best_cost = float('inf')

        for start in range(n):
            cycle, cost = self._prims_tsp(graph, start)
            if cost < best_cost:
                best_cost = cost
                best_cycle = cycle

        return best_cycle

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
        return "Prim's TSP"

    def _prims_tsp(self, graph, start_vertex):
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
        return cycle, self._calculate_cycle_cost(cycle, graph)

    @staticmethod
    def _calculate_cycle_cost(cycle, graph):
        """Calculate total cost of a cycle"""
        if len(cycle) < 2:
            return 0
        total = 0
        for i in range(len(cycle) - 1):
            total += graph[cycle[i]][cycle[i + 1]]
        return total