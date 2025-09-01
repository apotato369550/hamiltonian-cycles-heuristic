import random
import heapq
from typing import List, Tuple, Set, Dict, Optional, Union
from utils.base_heuristics import AnchoringHeuristic
from graph_generator import calculate_cycle_cost

class BidirectionalGreedy(AnchoringHeuristic):
    """
    Bidirectional Nearest-Neighbor TSP Solver inspired by the low_anchor_heuristic.

    Key concepts:
    1. Uses entrance and exit vertices to avoid monopolizing cheap edges
    2. Constructs paths from both directions simultaneously
    3. Merges paths optimally to form complete Hamiltonian cycle
    4. Includes local optimization with 2-opt improvements
    """

    def __init__(self):
        self.graph = None

    def solve(self, graph):
        """
        Solve TSP using bidirectional nearest-neighbor approach.

        Args:
            graph: 2D list/array representing adjacency matrix of a complete graph

        Returns:
            list: Hamiltonian cycle as a list of vertices, including return to start
        """
        self.graph = graph
        solver = BidirectionalNearestNeighborSolver(graph)
        cycle, _ = solver.solve(start=0, anchor_strategy="low", local_search=False)
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
        return "BidirectionalGreedy"


class BidirectionalNearestNeighborSolver:
    """
    Bidirectional Nearest-Neighbor TSP Solver inspired by the low_anchor_heuristic.

    Key concepts:
    1. Uses entrance and exit vertices to avoid monopolizing cheap edges
    2. Constructs paths from both directions simultaneously
    3. Merges paths optimally to form complete Hamiltonian cycle
    4. Includes local optimization with 2-opt improvements
    """

    def __init__(self, graph: List[List[Union[int, float]]]):
        """
        Initialize the solver with a graph.

        Args:
            graph: Complete weighted graph as adjacency matrix
        """
        self.graph = graph
        self.n = len(graph)

    def solve(self, start: int = 0, anchor_strategy: str = "low",
              num_attempts: int = 1, local_search: bool = True) -> Tuple[List[int], Union[int, float]]:
        """
        Main solving method using bidirectional nearest-neighbor approach.

        Args:
            start: Starting vertex
            anchor_strategy: Strategy for selecting anchors ("low", "high", "random", "adaptive")
            num_attempts: Number of attempts with different anchor selections
            local_search: Whether to apply local optimization

        Returns:
            Tuple of (cycle, total_weight)
        """
        best_cycle = None
        best_weight = float('inf')

        for attempt in range(num_attempts):
            # 1. Select entrance and exit anchors
            entrance, exit_anchor = self._select_anchors(start, anchor_strategy, attempt)

            # 2. Build bidirectional paths
            cycle, weight = self._build_bidirectional_cycle(start, entrance, exit_anchor)

            # 3. Apply local optimization if requested
            if local_search:
                cycle, weight = self._local_optimization(cycle)

            # 4. Keep track of best solution
            if weight < best_weight:
                best_weight = weight
                best_cycle = cycle

        return best_cycle, best_weight

    def _select_anchors(self, start: int, strategy: str, attempt: int = 0) -> Tuple[int, int]:
        """
        Select entrance and exit anchors based on the specified strategy.

        Args:
            start: Starting vertex
            strategy: Selection strategy
            attempt: Current attempt number (for randomization)

        Returns:
            Tuple of (entrance_anchor, exit_anchor)
        """
        available_vertices = [i for i in range(self.n) if i != start]

        if strategy == "low":
            # Select two vertices with lowest weights from start
            sorted_by_weight = sorted(available_vertices, key=lambda v: self.graph[start][v])
            return sorted_by_weight[0], sorted_by_weight[1]

        elif strategy == "high":
            # Select two vertices with highest weights from start
            sorted_by_weight = sorted(available_vertices, key=lambda v: self.graph[start][v], reverse=True)
            return sorted_by_weight[0], sorted_by_weight[1]

        elif strategy == "random":
            # Randomly select two vertices
            random.seed(attempt)  # Use attempt as seed for reproducibility
            return tuple(random.sample(available_vertices, 2))

        elif strategy == "adaptive":
            # Adaptive strategy based on graph characteristics
            return self._adaptive_anchor_selection(start, available_vertices)

        else:
            raise ValueError(f"Unknown anchor strategy: {strategy}")

    def _adaptive_anchor_selection(self, start: int, available_vertices: List[int]) -> Tuple[int, int]:
        """
        Adaptive anchor selection based on graph structure analysis.

        Args:
            start: Starting vertex
            available_vertices: List of vertices to choose from

        Returns:
            Tuple of (entrance_anchor, exit_anchor)
        """
        # Calculate vertex importance based on connectivity and centrality
        vertex_scores = []

        for v in available_vertices:
            # Distance from start
            start_distance = self.graph[start][v]

            # Average distance to all other vertices (centrality measure)
            avg_distance = sum(self.graph[v][u] for u in range(self.n) if u != v) / (self.n - 1)

            # Variance in distances (connectivity diversity)
            distances = [self.graph[v][u] for u in range(self.n) if u != v]
            mean_dist = sum(distances) / len(distances)
            variance = sum((d - mean_dist) ** 2 for d in distances) / len(distances)

            # Combined score (lower is better for entrance, higher variance is better for diversity)
            score = start_distance * 0.4 + avg_distance * 0.4 + variance * 0.2
            vertex_scores.append((score, v))

        # Sort by score and select strategically
        vertex_scores.sort()

        # Choose entrance as vertex with good balance of proximity and connectivity
        entrance = vertex_scores[len(vertex_scores) // 4][1]  # First quartile

        # Choose exit as vertex that's well-connected but not too close to entrance
        remaining_vertices = [v for score, v in vertex_scores if v != entrance]
        exit_candidates = [(self.graph[entrance][v], v) for v in remaining_vertices]
        exit_candidates.sort(reverse=True)  # Sort by distance from entrance (descending)

        # Select exit from top half of distances to ensure good separation
        exit_anchor = exit_candidates[len(exit_candidates) // 4][1]

        return entrance, exit_anchor

    def _build_bidirectional_cycle(self, start: int, entrance: int, exit_anchor: int) -> Tuple[List[int], Union[int, float]]:
        """
        Build cycle using bidirectional nearest-neighbor approach.

        Args:
            start: Starting vertex
            entrance: Entrance anchor vertex
            exit_anchor: Exit anchor vertex

        Returns:
            Tuple of (cycle, total_weight)
        """
        # Try both orderings of anchors and return the best
        cycle1, weight1 = self._construct_bidirectional_path(start, entrance, exit_anchor)
        cycle2, weight2 = self._construct_bidirectional_path(start, exit_anchor, entrance)

        if weight1 <= weight2:
            return cycle1, weight1
        else:
            return cycle2, weight2

    def _construct_bidirectional_path(self, start: int, first_anchor: int, second_anchor: int) -> Tuple[List[int], Union[int, float]]:
        """
        Construct path using bidirectional approach with specified anchor order.

        Args:
            start: Starting vertex
            first_anchor: First anchor to visit
            second_anchor: Second anchor to visit

        Returns:
            Tuple of (cycle, total_weight)
        """
        visited = {start, first_anchor, second_anchor}
        unvisited = set(range(self.n)) - visited

        # Initialize forward and backward paths
        forward_path = [start, first_anchor]
        backward_path = [second_anchor]  # Will be reversed when merged

        forward_end = first_anchor
        backward_end = second_anchor
        total_weight = self.graph[start][first_anchor]

        # Build paths bidirectionally
        while unvisited:
            forward_next = None
            backward_next = None
            forward_weight = float('inf')
            backward_weight = float('inf')

            # Find best next vertex for forward path
            if unvisited:
                forward_next = min(unvisited, key=lambda v: self.graph[forward_end][v])
                forward_weight = self.graph[forward_end][forward_next]

            # Find best next vertex for backward path
            if unvisited:
                backward_next = min(unvisited, key=lambda v: self.graph[backward_end][v])
                backward_weight = self.graph[backward_end][backward_next]

            # Choose the better option
            if forward_weight <= backward_weight:
                # Extend forward path
                if forward_next is not None:
                    forward_path.append(forward_next)
                    total_weight += forward_weight
                    visited.add(forward_next)
                    unvisited.remove(forward_next)
                    forward_end = forward_next
            else:
                # Extend backward path
                if backward_next is not None:
                    backward_path.append(backward_next)
                    total_weight += backward_weight
                    visited.add(backward_next)
                    unvisited.remove(backward_next)
                    backward_end = backward_next

        # Merge paths and complete cycle
        backward_path.reverse()  # Reverse to get correct order

        # Connect forward end to backward start
        if backward_path:
            total_weight += self.graph[forward_end][backward_path[0]]

        # Connect backward end to start (complete the cycle)
        total_weight += self.graph[backward_end][start]

        # Construct complete cycle
        complete_cycle = forward_path + backward_path + [start]

        return complete_cycle, total_weight

    def _local_optimization(self, cycle: List[int]) -> Tuple[List[int], Union[int, float]]:
        """
        Apply 2-opt local search optimization to improve the cycle.

        Args:
            cycle: Current cycle

        Returns:
            Tuple of (optimized_cycle, total_weight)
        """
        if len(cycle) <= 4:  # Not enough vertices for meaningful 2-opt
            return cycle, self._calculate_cycle_weight(cycle)

        improved = True
        iteration = 0
        max_iterations = min(100, len(cycle) * 2)

        while improved and iteration < max_iterations:
            improved = False
            iteration += 1

            for i in range(len(cycle) - 2):
                for j in range(i + 2, len(cycle) - 1):
                    # Skip if edges are adjacent
                    if j == i + 1:
                        continue

                    # Calculate current edge costs
                    current_cost = (self.graph[cycle[i]][cycle[i + 1]] +
                                  self.graph[cycle[j]][cycle[j + 1]])

                    # Calculate cost after 2-opt swap
                    new_cost = (self.graph[cycle[i]][cycle[j]] +
                              self.graph[cycle[i + 1]][cycle[j + 1]])

                    # If improvement found, perform swap
                    if new_cost < current_cost:
                        cycle[i + 1:j + 1] = cycle[i + 1:j + 1][::-1]
                        improved = True
                        break

            if improved:
                break

        return cycle, self._calculate_cycle_weight(cycle)

    def _calculate_cycle_weight(self, cycle: List[int]) -> Union[int, float]:
        """
        Calculate total weight of a cycle.

        Args:
            cycle: List of vertices in cycle order

        Returns:
            Total weight of the cycle
        """
        if len(cycle) <= 1:
            return 0

        total_weight = 0
        for i in range(len(cycle) - 1):
            total_weight += self.graph[cycle[i]][cycle[i + 1]]

        return total_weight

    def solve_multiple_strategies(self, start: int = 0, local_search: bool = True) -> Dict[str, Tuple[List[int], Union[int, float]]]:
        """
        Solve using multiple anchor strategies and return all results.

        Args:
            start: Starting vertex
            local_search: Whether to apply local optimization

        Returns:
            Dictionary mapping strategy names to (cycle, weight) tuples
        """
        strategies = ["low", "high", "random", "adaptive"]
        results = {}

        for strategy in strategies:
            num_attempts = 3 if strategy == "random" else 1
            cycle, weight = self.solve(start, strategy, num_attempts, local_search)
            results[strategy] = (cycle, weight)

        return results


# Wrapper function to maintain compatibility with existing interface
def bidirectional_nearest_neighbor_tsp(graph: List[List[Union[int, float]]],
                                     start: int = 0,
                                     anchor_strategy: str = "low",
                                     local_search: bool = False) -> Tuple[List[int], Union[int, float]]:
    """
    Bidirectional Nearest-Neighbor TSP solver wrapper function.

    Args:
        graph: Complete weighted graph as adjacency matrix
        start: Starting vertex
        anchor_strategy: Strategy for anchor selection ("low", "high", "random", "adaptive")
        local_search: Whether to apply local optimization

    Returns:
        Tuple of (cycle, total_weight)
    """
    solver = BidirectionalNearestNeighborSolver(graph)
    cycle, weight = solver.solve(start=start, anchor_strategy=anchor_strategy,
                               local_search=local_search)

    return cycle, weight


# Enhanced wrapper function for comprehensive testing
def bidirectional_tsp_comprehensive(graph: List[List[Union[int, float]]],
                                  start: int = 0,
                                  local_search: bool = False) -> Dict[str, Tuple[List[int], Union[int, float]]]:
    """
    Comprehensive bidirectional TSP solver that tries multiple strategies.

    Args:
        graph: Complete weighted graph as adjacency matrix
        start: Starting vertex
        local_search: Whether to apply local optimization

    Returns:
        Dictionary with results from all strategies, plus the best overall result
    """
    solver = BidirectionalNearestNeighborSolver(graph)
    results = solver.solve_multiple_strategies(start=start, local_search=local_search)

    # Find the best result
    best_strategy = min(results.keys(), key=lambda k: results[k][1])
    results["best"] = results[best_strategy]
    results["best_strategy"] = (best_strategy, results[best_strategy])

    return results