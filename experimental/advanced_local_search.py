import random
import time
from graph_generator import calculate_cycle_cost
from utils.base_heuristics import ExperimentalHeuristic
from established.nearest_neighbor import NearestNeighbor

class AdvancedLocalSearch(ExperimentalHeuristic):
    """
    Advanced local search heuristic combining 2-opt, 3-opt, and vertex insertion moves
    with adaptive stopping criteria and tabu search elements.
    """

    def __init__(self, max_iterations=1000, no_improve_limit=50, time_limit=30):
        self.graph = None
        self.max_iterations = max_iterations
        self.no_improve_limit = no_improve_limit
        self.time_limit = time_limit

    def solve(self, graph):
        """
        Solve TSP using advanced local search.

        Args:
            graph: 2D list/array representing adjacency matrix of a complete graph

        Returns:
            list: Hamiltonian cycle as a list of vertices, including return to start
        """
        self.graph = graph
        n = len(graph)
        if n < 3:
            raise ValueError("Graph must have at least 3 vertices for TSP")

        # Start with nearest neighbor solution
        nn_heuristic = NearestNeighbor()
        current_cycle = nn_heuristic.solve(graph)
        current_cost = calculate_cycle_cost(current_cycle, graph)

        best_cycle = current_cycle[:]
        best_cost = current_cost

        # Initialize search parameters
        iteration = 0
        no_improve_count = 0
        start_time = time.time()
        tabu_list = set()
        move_weights = {'2opt': 1.0, '3opt': 1.0, 'insertion': 1.0}

        while (iteration < self.max_iterations and
               no_improve_count < self.no_improve_limit and
               time.time() - start_time < self.time_limit):

            iteration += 1
            improved = False

            # Select move type based on adaptive weights
            move_type = self._select_move_type(move_weights)

            # Generate neighborhood based on selected move
            if move_type == '2opt':
                neighbors = self._generate_2opt_neighbors(current_cycle)
            elif move_type == '3opt':
                neighbors = self._generate_3opt_neighbors(current_cycle)
            else:  # insertion
                neighbors = self._generate_insertion_neighbors(current_cycle)

            # Evaluate neighbors and find best non-tabu move
            best_neighbor = None
            best_neighbor_cost = float('inf')

            for neighbor in neighbors:
                neighbor_tuple = tuple(neighbor[:-1])  # Exclude closing vertex for tabu
                if neighbor_tuple not in tabu_list:
                    cost = calculate_cycle_cost(neighbor, graph)
                    if cost < best_neighbor_cost:
                        best_neighbor = neighbor
                        best_neighbor_cost = cost

            # Apply best move if found
            if best_neighbor and best_neighbor_cost < current_cost:
                current_cycle = best_neighbor
                current_cost = best_neighbor_cost
                tabu_list.add(tuple(current_cycle[:-1]))

                # Update best solution
                if current_cost < best_cost:
                    best_cycle = current_cycle[:]
                    best_cost = current_cost
                    no_improve_count = 0
                    improved = True

                # Adapt move weights based on success
                move_weights[move_type] *= 1.1  # Increase weight for successful move
            else:
                no_improve_count += 1

            # Decay move weights and normalize
            for move in move_weights:
                move_weights[move] *= 0.99
            total_weight = sum(move_weights.values())
            for move in move_weights:
                move_weights[move] /= total_weight

            # Maintain tabu list size
            if len(tabu_list) > 20:
                tabu_list.pop()

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
        return "Advanced Local Search"

    def _select_move_type(self, move_weights):
        """Select move type based on adaptive weights."""
        moves = list(move_weights.keys())
        weights = list(move_weights.values())
        return random.choices(moves, weights=weights, k=1)[0]

    def _generate_2opt_neighbors(self, cycle):
        """Generate 2-opt neighborhood."""
        neighbors = []
        n = len(cycle) - 1  # Exclude closing vertex

        for i in range(1, n - 1):
            for j in range(i + 1, n):
                # Create 2-opt move
                new_cycle = cycle[:i] + cycle[i:j+1][::-1] + cycle[j+1:]
                neighbors.append(new_cycle)

        return neighbors

    def _generate_3opt_neighbors(self, cycle):
        """Generate 3-opt neighborhood (simplified version)."""
        neighbors = []
        n = len(cycle) - 1

        # Sample a subset of possible 3-opt moves for efficiency
        for _ in range(min(50, n * (n-1) * (n-2) // 6)):  # Limit neighborhood size
            i, j, k = sorted(random.sample(range(1, n), 3))

            # Try different 3-opt configurations
            for config in range(8):  # 8 possible ways to reconnect 3 edges
                new_cycle = self._apply_3opt_move(cycle, i, j, k, config)
                if new_cycle:
                    neighbors.append(new_cycle)

        return neighbors

    def _apply_3opt_move(self, cycle, i, j, k, config):
        """Apply a specific 3-opt move configuration."""
        n = len(cycle) - 1
        segments = [cycle[:i], cycle[i:j], cycle[j:k], cycle[k:n]]

        if config == 0:  # Standard order
            return segments[0] + segments[1] + segments[2] + segments[3] + [cycle[0]]
        elif config == 1:  # Reverse segment 1
            return segments[0] + segments[1][::-1] + segments[2] + segments[3] + [cycle[0]]
        elif config == 2:  # Reverse segment 2
            return segments[0] + segments[1] + segments[2][::-1] + segments[3] + [cycle[0]]
        elif config == 3:  # Reverse segment 3
            return segments[0] + segments[1] + segments[2] + segments[3][::-1] + [cycle[0]]
        elif config == 4:  # Swap segments 1 and 2
            return segments[0] + segments[2] + segments[1] + segments[3] + [cycle[0]]
        elif config == 5:  # Swap segments 1 and 3
            return segments[0] + segments[3] + segments[2] + segments[1] + [cycle[0]]
        elif config == 6:  # Swap segments 2 and 3
            return segments[0] + segments[1] + segments[3] + segments[2] + [cycle[0]]
        elif config == 7:  # Reverse all segments
            return segments[0] + segments[1][::-1] + segments[2][::-1] + segments[3][::-1] + [cycle[0]]

    def _generate_insertion_neighbors(self, cycle):
        """Generate vertex insertion neighborhood."""
        neighbors = []
        n = len(cycle) - 1

        # For each vertex, try inserting it at different positions
        for vertex_idx in range(1, n):
            vertex = cycle[vertex_idx]

            # Remove vertex from current position
            temp_cycle = cycle[:vertex_idx] + cycle[vertex_idx+1:]

            # Try inserting at different positions
            for insert_pos in range(1, len(temp_cycle)):
                new_cycle = (temp_cycle[:insert_pos] + [vertex] + temp_cycle[insert_pos:])
                neighbors.append(new_cycle)

        return neighbors