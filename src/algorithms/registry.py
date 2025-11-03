"""
Algorithm registry system for TSP benchmarking.

This module provides the registry pattern for managing TSP algorithm implementations.
Allows registering, retrieving, and filtering algorithms by name, tags, and constraints.

Usage:
    @register_algorithm(name="nearest_neighbor", tags=["baseline", "greedy"])
    class NearestNeighborAlgorithm(TSPAlgorithm):
        ...

    # Later, retrieve algorithm:
    algo = AlgorithmRegistry.get_algorithm("nearest_neighbor")

    # Filter by tags or applicability:
    baselines = AlgorithmRegistry.list_algorithms(tags=["baseline"])
"""

from typing import Dict, List, Type, Optional, Set
from .base import TSPAlgorithm


class AlgorithmRegistry:
    """
    Singleton registry for all TSP algorithms.

    Maintains a mapping of algorithm names to their implementations,
    supporting registration, retrieval, and filtering by constraints.
    """

    _instance = None
    _algorithms: Dict[str, Dict] = {}  # name -> {class, tags, constraints}

    def __new__(cls):
        """Ensure singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def register(
        cls,
        name: str,
        algorithm_class: Type[TSPAlgorithm],
        tags: Optional[List[str]] = None,
        constraints: Optional[Dict] = None
    ) -> None:
        """
        Register an algorithm in the registry.

        Args:
            name: Unique algorithm name (e.g., 'nearest_neighbor')
            algorithm_class: TSPAlgorithm subclass
            tags: List of tags for filtering (e.g., ['baseline', 'greedy'])
            constraints: Dict of applicability constraints (graph_types, size limits, etc.)

        Raises:
            ValueError: If algorithm name already registered
            TypeError: If algorithm_class is not a TSPAlgorithm subclass
        """
        if name in cls._algorithms:
            raise ValueError(f"Algorithm '{name}' already registered")

        if not issubclass(algorithm_class, TSPAlgorithm):
            raise TypeError(f"{algorithm_class} must be a TSPAlgorithm subclass")

        cls._algorithms[name] = {
            'class': algorithm_class,
            'tags': tags or [],
            'constraints': constraints or {}
        }

    @classmethod
    def get_algorithm(cls, name: str, random_seed: Optional[int] = None) -> TSPAlgorithm:
        """
        Retrieve an algorithm instance by name.

        Args:
            name: Algorithm name
            random_seed: Random seed for reproducibility

        Returns:
            Instance of the algorithm class

        Raises:
            KeyError: If algorithm not found in registry
        """
        if name not in cls._algorithms:
            available = ", ".join(cls._algorithms.keys())
            raise KeyError(
                f"Algorithm '{name}' not registered. "
                f"Available: {available}"
            )

        algorithm_class = cls._algorithms[name]['class']
        return algorithm_class(random_seed=random_seed)

    @classmethod
    def list_algorithms(
        cls,
        tags: Optional[List[str]] = None,
        graph_type: Optional[str] = None,
        graph_size: Optional[int] = None
    ) -> List[str]:
        """
        List registered algorithms, optionally filtering by criteria.

        Args:
            tags: Only return algorithms with ALL these tags
            graph_type: Only return algorithms applicable to this graph type
            graph_size: Only return algorithms that can handle this graph size

        Returns:
            List of algorithm names matching criteria
        """
        matching = []

        for name, info in cls._algorithms.items():
            # Check tags filter
            if tags is not None:
                if not all(tag in info['tags'] for tag in tags):
                    continue

            # Check graph type filter
            if graph_type is not None:
                allowed_types = info['constraints'].get('graph_types', None)
                if allowed_types is not None and graph_type not in allowed_types:
                    continue

            # Check graph size filter
            if graph_size is not None:
                min_size = info['constraints'].get('min_size', None)
                max_size = info['constraints'].get('max_size', None)

                if min_size is not None and graph_size < min_size:
                    continue
                if max_size is not None and graph_size > max_size:
                    continue

            matching.append(name)

        return sorted(matching)

    @classmethod
    def get_tags(cls, name: str) -> List[str]:
        """Get tags for an algorithm."""
        if name not in cls._algorithms:
            raise KeyError(f"Algorithm '{name}' not registered")
        return cls._algorithms[name]['tags']

    @classmethod
    def get_constraints(cls, name: str) -> Dict:
        """Get applicability constraints for an algorithm."""
        if name not in cls._algorithms:
            raise KeyError(f"Algorithm '{name}' not registered")
        return cls._algorithms[name]['constraints']

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """Check if an algorithm is registered."""
        return name in cls._algorithms

    @classmethod
    def clear(cls) -> None:
        """Clear all registered algorithms (useful for testing)."""
        cls._algorithms = {}

    @classmethod
    def count(cls) -> int:
        """Get count of registered algorithms."""
        return len(cls._algorithms)

    @classmethod
    def summary(cls) -> str:
        """Get human-readable summary of all registered algorithms."""
        lines = [f"Algorithm Registry ({len(cls._algorithms)} algorithms):"]

        for name in sorted(cls._algorithms.keys()):
            info = cls._algorithms[name]
            tags_str = ", ".join(info['tags']) if info['tags'] else "(no tags)"
            lines.append(f"  - {name}: {tags_str}")

        return "\n".join(lines)


def register_algorithm(
    name: str,
    tags: Optional[List[str]] = None,
    constraints: Optional[Dict] = None
):
    """
    Decorator for registering an algorithm class.

    Usage:
        @register_algorithm(
            name="nearest_neighbor",
            tags=["baseline", "greedy"],
            constraints={"graph_types": ["euclidean", "metric", "random"]}
        )
        class NearestNeighborAlgorithm(TSPAlgorithm):
            ...

    Args:
        name: Unique algorithm identifier
        tags: List of tags for filtering
        constraints: Dict of applicability constraints

    Returns:
        Decorator function that registers the class and returns it
    """
    def decorator(cls: Type[TSPAlgorithm]) -> Type[TSPAlgorithm]:
        AlgorithmRegistry.register(name, cls, tags, constraints)
        return cls

    return decorator
