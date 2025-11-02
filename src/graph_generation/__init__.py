"""
Graph Generation System for TSP Research Pipeline.

This package provides a comprehensive system for generating, verifying,
storing, and analyzing TSP graph instances across multiple dimensions:
- Euclidean graphs (2D/3D point-based)
- Metric graphs (non-Euclidean but satisfying triangle inequality)
- Random graphs (baseline chaotic instances)

Main components:
- graph_instance: Core data structures
- euclidean_generator: Euclidean graph generation
- metric_generator: Metric and quasi-metric graph generation
- random_generator: Random graph generation
- verification: Property verification system
- storage: Graph persistence and retrieval
- batch_generator: Batch generation pipeline
- visualization: Graph visualization utilities
- collection_analysis: Collection analysis tools
"""

from .graph_instance import GraphInstance, GraphMetadata, GraphProperties, create_graph_instance
from .euclidean_generator import generate_euclidean_graph, EuclideanGraphGenerator
from .metric_generator import generate_metric_graph, generate_quasi_metric_graph
from .random_generator import generate_random_graph
from .verification import GraphVerifier, verify_graph_properties, print_verification_report
from .storage import GraphStorage, save_graph, load_graph
from .batch_generator import BatchGenerator, BatchGenerationConfig, generate_batch_from_config
from .visualization import GraphVisualizer, visualize_graph
from .collection_analysis import CollectionAnalyzer, analyze_collection

__version__ = "1.0.0"

__all__ = [
    # Core classes
    'GraphInstance',
    'GraphMetadata',
    'GraphProperties',
    'create_graph_instance',

    # Generators
    'generate_euclidean_graph',
    'EuclideanGraphGenerator',
    'generate_metric_graph',
    'generate_quasi_metric_graph',
    'generate_random_graph',

    # Verification
    'GraphVerifier',
    'verify_graph_properties',
    'print_verification_report',

    # Storage
    'GraphStorage',
    'save_graph',
    'load_graph',

    # Batch generation
    'BatchGenerator',
    'BatchGenerationConfig',
    'generate_batch_from_config',

    # Visualization
    'GraphVisualizer',
    'visualize_graph',

    # Analysis
    'CollectionAnalyzer',
    'analyze_collection',
]
