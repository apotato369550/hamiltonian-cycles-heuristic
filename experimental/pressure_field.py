import math
from graph_generator import calculate_cycle_cost
from utils.base_heuristics import ExperimentalHeuristic

class PressureField(ExperimentalHeuristic):
    """
    Implementation of Pressure Field Navigation TSP heuristic.
    """

    def __init__(self):
        self.graph = None

    def solve(self, graph):
        """
        Solve TSP using Pressure Field Navigation algorithm.

        Args:
            graph: 2D adjacency matrix representing distances between vertices

        Returns:
            list: Path including return to start
        """
        self.graph = graph
        path, _ = self._pressure_field_navigation(graph, 0)
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
        return "Pressure Field"

    @staticmethod
    def _calculate_isolation_risk(graph, vertex, unvisited):
        """Calculate how isolated a vertex is based on distances to other unvisited vertices."""
        if len(unvisited) <= 1:
            return 0

        distances = [graph[vertex][other] for other in unvisited if other != vertex]
        if not distances:
            return 0

        # Higher isolation risk for vertices with larger minimum distances to others
        min_distance = min(distances)
        avg_distance = sum(distances) / len(distances)

        # Isolation risk increases with both minimum and average distances
        return min_distance * 0.6 + avg_distance * 0.4

    @staticmethod
    def _calculate_pressure_field(graph, source, target, unvisited, pressure_shadows):
        """Calculate the pressure field strength from source vertex affecting target vertex."""
        if source == target:
            return 0

        distance = graph[source][target]
        if distance == 0:
            return float('inf')  # Avoid division by zero

        # Base pressure inversely proportional to distance
        base_pressure = 1.0 / distance

        # Boost pressure based on isolation risk of the source
        isolation_boost = 1.0 + PressureField._calculate_isolation_risk(graph, source, unvisited) * 0.1

        # Apply pressure shadows (recently visited areas have reduced influence)
        shadow_factor = 1.0
        if source in pressure_shadows:
            shadow_factor = max(0.3, 1.0 - pressure_shadows[source] * 0.7)

        return base_pressure * isolation_boost * shadow_factor

    @staticmethod
    def _calculate_pressure_gradient(graph, current, unvisited, pressure_shadows):
        """Calculate the pressure gradient for each possible next vertex."""
        gradients = {}

        for candidate in unvisited:
            if candidate == current:
                continue

            total_pressure = 0

            # Calculate combined pressure from all unvisited vertices
            for source in unvisited:
                if source != candidate:
                    pressure = PressureField._calculate_pressure_field(graph, source, candidate, unvisited, pressure_shadows)
                    total_pressure += pressure

            # Add momentum bonus - prefer directions that maintain good progress
            # This is based on how well this move positions us relative to remaining vertices
            if len(unvisited) > 2:
                future_potential = 0
                remaining_after_move = [v for v in unvisited if v != candidate]

                for future_vertex in remaining_after_move:
                    future_distance = graph[candidate][future_vertex]
                    if future_distance > 0:
                        future_potential += 1.0 / future_distance

                momentum_bonus = future_potential * 0.2
                total_pressure += momentum_bonus

            gradients[candidate] = total_pressure

        return gradients

    @staticmethod
    def _pressure_field_navigation(graph, start_vertex=None):
        """
        Solve TSP using Pressure Field Navigation algorithm.
        """
        vertices_count = len(graph)
        if vertices_count == 0:
            return [], 0

        if start_vertex is None:
            start_vertex = 0

        # Initialize
        visited = set([start_vertex])
        unvisited = set(range(vertices_count)) - visited
        path = [start_vertex]
        current_vertex = start_vertex
        total_weight = 0

        # Pressure shadows: tracks recently visited vertices and their influence decay
        pressure_shadows = {}
        shadow_decay_rate = 0.8  # How quickly pressure shadows fade

        # Main construction loop
        while unvisited:
            # Calculate pressure gradients for all unvisited vertices
            gradients = PressureField._calculate_pressure_gradient(graph, current_vertex, unvisited, pressure_shadows)

            if not gradients:
                break

            # Select vertex with highest pressure gradient
            next_vertex = max(gradients.keys(), key=lambda v: gradients[v])

            # Move to next vertex
            edge_weight = graph[current_vertex][next_vertex]
            path.append(next_vertex)
            visited.add(next_vertex)
            unvisited.remove(next_vertex)
            total_weight += edge_weight

            # Update pressure shadows
            pressure_shadows[current_vertex] = 1.0  # Full shadow for just-visited vertex

            # Decay existing shadows
            for vertex in list(pressure_shadows.keys()):
                pressure_shadows[vertex] *= shadow_decay_rate
                if pressure_shadows[vertex] < 0.1:  # Remove very weak shadows
                    del pressure_shadows[vertex]

            current_vertex = next_vertex

        # Complete the cycle by returning to start
        if current_vertex != start_vertex:
            total_weight += graph[current_vertex][start_vertex]
            path.append(start_vertex)

        return path, total_weight