"""
Algorithm Benchmarking System for TSP Research Pipeline.

This package provides a comprehensive benchmarking framework for TSP algorithms:
- Unified algorithm interface (base.py)
- Algorithm registry system (registry.py)
- Baseline algorithms (nearest neighbor, greedy, exact)
- Anchor-based algorithms (single, best, multi-anchor variants)
- Tour validation and metrics (validation.py, metrics.py)
- Single-graph and batch benchmarking runners
- Statistical analysis and visualization tools

Main components:
- base: TSPAlgorithm abstract class and TourResult/AlgorithmMetadata dataclasses
- registry: Algorithm registry and decorator for registration
- validation: Tour validation functions
- metrics: Quality metrics computation
- nearest_neighbor: Nearest neighbor algorithm implementations
- greedy: Greedy edge-picking algorithm
- exact: Held-Karp exact algorithm
- single_anchor, best_anchor, multi_anchor: Anchor-based heuristics
- single_benchmark: Single-graph benchmarking runner
- batch_benchmark: Batch benchmarking orchestration
- results_storage: Results persistence and retrieval
- statistics: Statistical analysis tools
- comparison: Algorithm comparison framework
- visualization: Plotting and visualization functions
- reporting: Automated report generation
"""

from .base import TourResult, AlgorithmMetadata, TSPAlgorithm
from .registry import AlgorithmRegistry, register_algorithm

# Import algorithm implementations to trigger @register_algorithm decorators
# This ensures algorithms are registered when the package is imported
from . import nearest_neighbor
from . import nearest_neighbor_adaptive
from . import greedy
from . import exact
from . import single_anchor
from . import single_anchor_v3
from . import best_anchor
from . import multi_anchor

__version__ = "1.0.0"

__all__ = [
    # Core classes and functions
    'TourResult',
    'AlgorithmMetadata',
    'TSPAlgorithm',
    'AlgorithmRegistry',
    'register_algorithm',
]
