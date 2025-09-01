from graph_generator import calculate_cycle_cost
from utils.base_heuristics import EstablishedHeuristic

class NearestNeighbor(EstablishedHeuristic):
    """
    Implementation of the Nearest Neighbor algorithm for TSP.
    """

    def __init__(self):
        self.graph = None

    def solve(self, graph):
        """
        Applies nearest neighbor algorithm to all vertices and returns the cycle with the lowest total weight.

        Args:
            graph: 2D list/array representing adjacency matrix of a complete graph

        Returns:
            list: Best cycle path including return to start
        """
        self.graph = graph
        vertices_count = len(graph)
        best_cycle = None
        best_weight = float('inf')
        best_start_vertex = None

        # Try nearest neighbor algorithm for each vertex as starting point
        for vertex in range(vertices_count):
            try:
                cycle, weight = self._nearest_neighbor_tsp(graph, vertex)

                # Keep track of the best cycle found so far
                if weight < best_weight:
                    best_weight = weight
                    best_cycle = cycle
                    best_start_vertex = vertex

            except (ValueError, IndexError) as e:
                # Skip vertices that cause errors
                continue

        if best_cycle is None:
            raise ValueError("No valid cycle could be constructed from any starting vertex")

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
        return "Nearest Neighbor"

    def _nearest_neighbor_tsp(self, graph, start_vertex):
        """
        Implements the nearest neighbor algorithm for TSP starting from a given vertex.

        Args:
            graph: 2D list/array representing adjacency matrix of a complete graph
            start_vertex: Starting vertex for the nearest neighbor algorithm

        Returns:
            tuple: (cycle_path, total_weight) where cycle_path includes return to start
        """
        vertices_count = len(graph)

        # Handle edge cases
        if vertices_count <= 1:
            return [0], 0
        if vertices_count == 2:
            return [0, 1, 0], graph[0][1] * 2

        # Initialize
        visited = set([start_vertex])
        path = [start_vertex]
        current_vertex = start_vertex
        total_weight = 0

        # Visit all remaining vertices using nearest neighbor strategy
        while len(visited) < vertices_count:
            next_vertex = None
            lowest_weight = float("inf")

            # Find the nearest unvisited neighbor
            for i in range(vertices_count):
                if i not in visited and graph[current_vertex][i] < lowest_weight:
                    next_vertex = i
                    lowest_weight = graph[current_vertex][i]

            # If we can't find a next vertex, something is wrong
            if next_vertex is None:
                break

            # Move to the nearest neighbor
            visited.add(next_vertex)
            path.append(next_vertex)
            total_weight += lowest_weight
            current_vertex = next_vertex

        # Complete the cycle by returning to start
        total_weight += graph[current_vertex][start_vertex]
        path.append(start_vertex)

        return path, total_weight