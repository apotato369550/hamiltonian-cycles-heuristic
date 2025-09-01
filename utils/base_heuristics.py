from abc import ABC, abstractmethod

class TSPHeuristic(ABC):
    """
    Abstract base class for TSP heuristics.
    """

    @abstractmethod
    def solve(self, graph):
        """
        Solve the TSP for the given graph.

        Args:
            graph: 2D list/array representing adjacency matrix of a complete graph.
                   Use float('inf') for non-existent edges.

        Returns:
            list: Hamiltonian cycle as a list of vertices, including return to start.
        """
        pass

    @abstractmethod
    def evaluate(self, cycle):
        """
        Evaluate the cost of a given cycle.

        Args:
            cycle: List of vertices representing the cycle.

        Returns:
            float: Total weight/cost of the cycle.
        """
        pass

    @abstractmethod
    def get_name(self):
        """
        Get the name of the heuristic.

        Returns:
            str: Name of the heuristic.
        """
        pass

class EstablishedHeuristic(TSPHeuristic):
    """
    Base class for established TSP heuristics (well-known algorithms).
    """
    pass

class ExperimentalHeuristic(TSPHeuristic):
    """
    Base class for experimental TSP heuristics (novel or custom approaches).
    """
    pass

class AnchoringHeuristic(TSPHeuristic):
    """
    Base class for anchoring-based TSP heuristics.
    """
    pass