import random
from collections import defaultdict
from typing import List, Tuple, Dict
from utils.base_heuristics import AnchoringHeuristic
from graph_generator import generate_complete_graph, calculate_cycle_cost

class LowAnchorMetaheuristic(AnchoringHeuristic):
    """
    Low Anchor Metaheuristic implementation.
    Extends low anchor heuristic with metaheuristic strategies for starting vertex selection.
    """

    def __init__(self):
        self.graph = None

    def solve(self, graph):
        """
        Solve TSP using low anchor metaheuristic.

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
        return "LowAnchorMetaheuristic"

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

        return path, total_weight

    def low_anchor_heuristic(self, vertex):
        """Your original anchor heuristic implementation"""
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

    def calculate_vertex_total_weight(self, vertex):
        """Calculate the total weight of all edges from a vertex"""
        return sum(self.graph[vertex])

    def rank_vertices_by_weight(self):
        """
        Rank vertices by their total outgoing edge weight.
        Returns dictionary with rankings and sorted vertex lists.
        """
        vertices_count = len(self.graph)
        vertex_weights = [(i, self.calculate_vertex_total_weight(i)) for i in range(vertices_count)]

        # Sort by weight (ascending)
        sorted_by_weight = sorted(vertex_weights, key=lambda x: x[1])

        return {
            'lowest_weight_vertex': sorted_by_weight[0][0],
            'highest_weight_vertex': sorted_by_weight[-1][0],
            'sorted_vertices': [vertex for vertex, weight in sorted_by_weight],
            'vertex_weights': dict(vertex_weights),
            'sorted_vertices_and_weights': sorted_by_weight
        }

    def starting_vertex_metaheuristic(self, num_random_trials=5, verbose=False):
        """
        Test the anchor heuristic with different starting vertex selection strategies:
        1. Lowest total weight vertex
        2. Highest total weight vertex
        3. Random vertex selection (multiple trials)
        4. All vertices (for comparison)

        Returns results dictionary with best cycles and analysis.
        """
        vertices_count = len(self.graph)
        ranking_info = self.rank_vertices_by_weight()

        results = {
            'lowest_weight_start': None,
            'highest_weight_start': None,
            'random_starts': [],
            'all_vertices': [],
            'best_overall': None,
            'ranking_info': ranking_info
        }

        if verbose:
            print(f"Vertex weight rankings:")
            for vertex, weight in ranking_info['vertex_weights'].items():
                print(f"  Vertex {vertex}: Total weight = {weight}")
            print()

        # Test 1: Lowest weight vertex as starting point
        lowest_vertex = ranking_info['lowest_weight_vertex']
        cycle, weight = self.low_anchor_heuristic(lowest_vertex)
        results['lowest_weight_start'] = {
            'vertex': lowest_vertex,
            'cycle': cycle,
            'weight': weight,
            'vertex_total_weight': ranking_info['vertex_weights'][lowest_vertex]
        }

        if verbose:
            print(f"Lowest weight start (vertex {lowest_vertex}): Cycle weight = {weight}")

        # Test 2: Highest weight vertex as starting point
        highest_vertex = ranking_info['highest_weight_vertex']
        cycle, weight = self.low_anchor_heuristic(highest_vertex)
        results['highest_weight_start'] = {
            'vertex': highest_vertex,
            'cycle': cycle,
            'weight': weight,
            'vertex_total_weight': ranking_info['vertex_weights'][highest_vertex]
        }

        if verbose:
            print(f"Highest weight start (vertex {highest_vertex}): Cycle weight = {weight}")

        # Test 3: Random vertex selection
        random_vertices = random.sample(range(vertices_count), min(num_random_trials, vertices_count))
        for vertex in random_vertices:
            cycle, weight = self.low_anchor_heuristic(vertex)
            results['random_starts'].append({
                'vertex': vertex,
                'cycle': cycle,
                'weight': weight,
                'vertex_total_weight': ranking_info['vertex_weights'][vertex]
            })

            if verbose:
                print(f"Random start (vertex {vertex}): Cycle weight = {weight}")

        # Test 4: All vertices for comprehensive comparison
        for vertex in range(vertices_count):
            cycle, weight = self.low_anchor_heuristic(vertex)
            results['all_vertices'].append({
                'vertex': vertex,
                'cycle': cycle,
                'weight': weight,
                'vertex_total_weight': ranking_info['vertex_weights'][vertex]
            })

        # Find best overall result
        all_results = ([results['lowest_weight_start'], results['highest_weight_start']] +
                      results['random_starts'] + results['all_vertices'])

        results['best_overall'] = min(all_results, key=lambda x: x['weight'])

        if verbose:
            print(f"\nBest overall result: Vertex {results['best_overall']['vertex']} "
                  f"with cycle weight {results['best_overall']['weight']}")

        return results

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