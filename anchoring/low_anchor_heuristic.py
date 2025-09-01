from utils.base_heuristics import AnchoringHeuristic
from graph_generator import calculate_cycle_cost

class LowAnchorHeuristic(AnchoringHeuristic):
    """
    Low Anchor Heuristic implementation.
    Uses two lowest-weight edges from each vertex as anchors.
    """

    def __init__(self):
        self.graph = None

    def solve(self, graph):
        """
        Solve TSP using low anchor heuristic.

        Args:
            graph: 2D list/array representing adjacency matrix of a complete graph

        Returns:
            list: Hamiltonian cycle as a list of vertices, including return to start
        """
        self.graph = graph
        cycle, _ = self.best_anchor_heuristic(graph)
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
        return "LowAnchorHeuristic"

    def construct_greedy_cycle(self, start, anchor1, anchor2):
        """Constructs a greedy cycle given a start vertex and 2 anchor points. Ensures Hamiltonian cycle."""
        vertices_count = len(self.graph)

        # Initialize with only the start point in the visited set
        visited = set([start])
        path = [start]
        current_vertex = start
        total_weight = 0

        # First, we need to visit anchor1
        if anchor1 not in visited:
            path.append(anchor1)
            visited.add(anchor1)
            total_weight += self.graph[current_vertex][anchor1]
            current_vertex = anchor1

        # Visit all remaining vertices except anchor2
        while len(visited) < vertices_count - 1:  # -1 because we'll add anchor2 last
            next_vertex = None
            lowest_weight = float("inf")

            for i in range(vertices_count):
                if i not in visited and i != anchor2 and self.graph[current_vertex][i] < lowest_weight:
                    next_vertex = i
                    lowest_weight = self.graph[current_vertex][i]

            # If we can't find a next vertex, break the loop
            if next_vertex is None:
                break

            visited.add(next_vertex)
            path.append(next_vertex)
            total_weight += lowest_weight
            current_vertex = next_vertex

        # Now add anchor2 if it's not already visited
        if anchor2 not in visited:
            path.append(anchor2)
            visited.add(anchor2)
            total_weight += self.graph[current_vertex][anchor2]
            current_vertex = anchor2

        # Complete the cycle by returning to start
        total_weight += self.graph[current_vertex][start]
        path.append(start)

        return path, calculate_cycle_cost(path, self.graph)

    def low_anchor_heuristic(self, vertex):
        def find_two_lowest_indices(values, vertex):
            if len(values) < 2:
                raise ValueError("List must contain at least two elements.")
            sorted_indices = sorted((i for i in range(len(values)) if i != vertex), key=lambda i: values[i])
            return sorted_indices[:2]
        anchors = find_two_lowest_indices(self.graph[vertex], vertex)

        cycle_1, lowest_weight_1 = self.construct_greedy_cycle(vertex, anchors[0], anchors[1])
        cycle_2, lowest_weight_2 = self.construct_greedy_cycle(vertex, anchors[1], anchors[0])

        if lowest_weight_1 < lowest_weight_2:
            return cycle_1, lowest_weight_1
        return cycle_2, lowest_weight_2

    def best_anchor_heuristic(self, graph):
        """
        Applies low_anchor_heuristic to all vertices and returns the cycle with the lowest total weight.

        Args:
            graph: 2D list/array representing adjacency matrix of a complete graph

        Returns:
            tuple: (best_cycle, best_weight) where best_cycle is the path and best_weight is the total weight
        """
        vertices_count = len(graph)
        best_cycle = None
        best_weight = float('inf')
        best_start_vertex = None

        # Try low_anchor_heuristic for each vertex as starting point
        for vertex in range(vertices_count):
            try:
                cycle, weight = self.low_anchor_heuristic(vertex)

                # Keep track of the best cycle found so far
                if weight < best_weight:
                    best_weight = weight
                    best_cycle = cycle
                    best_start_vertex = vertex

            except (ValueError, IndexError) as e:
                # Skip vertices that cause errors (e.g., if graph has < 3 vertices)
                continue

        if best_cycle is None:
            raise ValueError("No valid cycle could be constructed from any starting vertex")

        return best_cycle, best_weight