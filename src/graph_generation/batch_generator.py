"""
Batch graph generation pipeline.

Provides high-level interface for generating diverse collections of graphs
based on configuration files with progress tracking and error handling.
"""

import json
import yaml
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime
import traceback

from .graph_instance import GraphInstance, create_graph_instance
from .euclidean_generator import generate_euclidean_graph
from .metric_generator import generate_metric_graph, generate_quasi_metric_graph
from .random_generator import generate_random_graph
from .storage import GraphStorage


class BatchGenerationConfig:
    """Configuration for batch graph generation."""

    def __init__(self, config_dict: Dict[str, Any]):
        """
        Initialize from configuration dictionary.

        Args:
            config_dict: Configuration parameters
        """
        self.config = config_dict

        # Extract main parameters
        self.batch_name = config_dict.get('batch_name', 'unnamed_batch')
        self.output_directory = config_dict.get('output_directory', 'data/graphs')
        self.graph_specs = config_dict.get('graphs', [])
        self.verification_mode = config_dict.get('verification_mode', 'fast')  # 'fast', 'full', 'none'
        self.continue_on_error = config_dict.get('continue_on_error', True)
        self.save_failed_graphs = config_dict.get('save_failed_graphs', False)

    @classmethod
    def from_yaml(cls, filepath: str) -> 'BatchGenerationConfig':
        """Load configuration from YAML file."""
        with open(filepath, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(config_dict)

    @classmethod
    def from_json(cls, filepath: str) -> 'BatchGenerationConfig':
        """Load configuration from JSON file."""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        return cls(config_dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return self.config


class BatchGenerator:
    """
    Batch graph generator.

    Generates multiple graphs based on configuration with progress
    tracking, verification, and error handling.
    """

    def __init__(self, config: BatchGenerationConfig):
        """
        Initialize batch generator.

        Args:
            config: Batch generation configuration
        """
        self.config = config
        self.storage = GraphStorage(base_directory=config.output_directory)
        self.results = {
            'generated': [],
            'failed': [],
            'verification_failed': []
        }

    def generate_all(self) -> Dict[str, Any]:
        """
        Generate all graphs specified in configuration.

        Returns:
            Summary report dictionary
        """
        print(f"Starting batch generation: {self.config.batch_name}")
        print(f"Total specifications: {len(self.config.graph_specs)}")
        print("=" * 70)

        start_time = datetime.utcnow()
        total_graphs = 0
        generated_graphs = []

        for spec_idx, spec in enumerate(self.config.graph_specs, 1):
            print(f"\nProcessing specification {spec_idx}/{len(self.config.graph_specs)}")
            print(f"  Type: {spec.get('type')}")
            print(f"  Sizes: {spec.get('sizes')}")
            print(f"  Instances per size: {spec.get('instances_per_size', 1)}")

            spec_graphs = self._generate_from_spec(spec)
            generated_graphs.extend(spec_graphs)
            total_graphs += len(spec_graphs)

            print(f"  Generated: {len(spec_graphs)} graphs")

        # Save batch
        print("\n" + "=" * 70)
        print("Saving batch...")

        if generated_graphs:
            manifest = self.storage.save_batch(
                graphs=generated_graphs,
                batch_name=self.config.batch_name,
                create_manifest=True
            )
        else:
            manifest = {'num_graphs': 0}

        # Create summary report
        end_time = datetime.utcnow()
        duration = (end_time - start_time).total_seconds()

        report = {
            'batch_name': self.config.batch_name,
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'duration_seconds': duration,
            'total_generated': len(generated_graphs),
            'total_failed': len(self.results['failed']),
            'verification_failed': len(self.results['verification_failed']),
            'graphs_by_type': self._summarize_by_type(generated_graphs),
            'graphs_by_size': self._summarize_by_size(generated_graphs),
            'property_distribution': self._summarize_properties(generated_graphs),
            'failed_details': self.results['failed']
        }

        # Save report
        report_path = Path(self.config.output_directory) / self.config.batch_name / 'generation_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        # Print summary
        self._print_summary(report)

        return report

    def _generate_from_spec(self, spec: Dict[str, Any]) -> List[GraphInstance]:
        """Generate graphs from a single specification."""
        graph_type = spec['type']
        sizes = spec.get('sizes', [])
        instances_per_size = spec.get('instances_per_size', 1)
        seed_start = spec.get('seed_start', 1000)
        params = spec.get('parameters', {})

        generated = []

        for size in sizes:
            for instance_idx in range(instances_per_size):
                seed = seed_start + instance_idx

                try:
                    graph = self._generate_single_graph(
                        graph_type=graph_type,
                        size=size,
                        seed=seed,
                        params=params
                    )

                    if graph is not None:
                        generated.append(graph)
                        self.results['generated'].append({
                            'type': graph_type,
                            'size': size,
                            'seed': seed,
                            'id': graph.id
                        })

                except Exception as e:
                    error_info = {
                        'type': graph_type,
                        'size': size,
                        'seed': seed,
                        'error': str(e),
                        'traceback': traceback.format_exc()
                    }
                    self.results['failed'].append(error_info)

                    print(f"    ERROR: Failed to generate {graph_type} graph (size={size}, seed={seed})")
                    print(f"           {str(e)}")

                    if not self.config.continue_on_error:
                        raise

        return generated

    def _generate_single_graph(
        self,
        graph_type: str,
        size: int,
        seed: int,
        params: Dict[str, Any]
    ) -> Optional[GraphInstance]:
        """Generate a single graph instance."""

        # Generate adjacency matrix based on type
        if graph_type == 'euclidean':
            adjacency_matrix, coordinates = self._generate_euclidean(size, seed, params)
            generation_params = {
                'dimensions': params.get('dimensions', 2),
                'weight_range': params.get('weight_range'),
                'distribution': params.get('distribution', 'uniform'),
                **params
            }

        elif graph_type == 'metric':
            adjacency_matrix = self._generate_metric(size, seed, params)
            coordinates = None
            generation_params = {
                'strategy': params.get('strategy', 'mst'),
                'weight_range': params.get('weight_range', (1.0, 100.0)),
                **params
            }

        elif graph_type == 'quasi_metric':
            adjacency_matrix = self._generate_quasi_metric(size, seed, params)
            coordinates = None
            generation_params = {
                'asymmetry_factor': params.get('asymmetry_factor', 0.2),
                'weight_range': params.get('weight_range', (1.0, 100.0)),
                **params
            }

        elif graph_type == 'random':
            adjacency_matrix = self._generate_random(size, seed, params)
            coordinates = None
            generation_params = {
                'distribution': params.get('distribution', 'uniform'),
                'weight_range': params.get('weight_range', (1.0, 100.0)),
                'is_symmetric': params.get('is_symmetric', True),
                **params
            }

        else:
            raise ValueError(f"Unknown graph type: {graph_type}")

        # Create graph instance with verification
        verify = (self.config.verification_mode == 'full')
        graph = create_graph_instance(
            adjacency_matrix=adjacency_matrix,
            graph_type=graph_type,
            generation_params=generation_params,
            random_seed=seed,
            coordinates=coordinates,
            verify=verify
        )

        # Additional verification check
        if self.config.verification_mode != 'none':
            if not self._verify_graph(graph):
                self.results['verification_failed'].append({
                    'type': graph_type,
                    'size': size,
                    'seed': seed,
                    'id': graph.id
                })
                if not self.config.save_failed_graphs:
                    return None

        return graph

    def _generate_euclidean(
        self,
        size: int,
        seed: int,
        params: Dict[str, Any]
    ) -> tuple:
        """Generate Euclidean graph."""
        return generate_euclidean_graph(
            num_vertices=size,
            dimensions=params.get('dimensions', 2),
            coord_bounds=params.get('coord_bounds', (0.0, 100.0)),
            weight_range=params.get('weight_range'),
            distribution=params.get('distribution', 'uniform'),
            distribution_params=params.get('distribution_params'),
            random_seed=seed
        )

    def _generate_metric(
        self,
        size: int,
        seed: int,
        params: Dict[str, Any]
    ) -> List[List[float]]:
        """Generate metric graph."""
        return generate_metric_graph(
            num_vertices=size,
            weight_range=params.get('weight_range', (1.0, 100.0)),
            strategy=params.get('strategy', 'mst'),
            metric_strictness=params.get('metric_strictness', 1.0),
            is_symmetric=params.get('is_symmetric', True),
            distribution=params.get('distribution', 'uniform'),
            random_seed=seed
        )

    def _generate_quasi_metric(
        self,
        size: int,
        seed: int,
        params: Dict[str, Any]
    ) -> List[List[float]]:
        """Generate quasi-metric graph."""
        return generate_quasi_metric_graph(
            num_vertices=size,
            weight_range=params.get('weight_range', (1.0, 100.0)),
            asymmetry_factor=params.get('asymmetry_factor', 0.2),
            random_seed=seed
        )

    def _generate_random(
        self,
        size: int,
        seed: int,
        params: Dict[str, Any]
    ) -> List[List[float]]:
        """Generate random graph."""
        return generate_random_graph(
            num_vertices=size,
            weight_range=params.get('weight_range', (1.0, 100.0)),
            distribution=params.get('distribution', 'uniform'),
            is_symmetric=params.get('is_symmetric', True),
            distribution_params=params.get('distribution_params'),
            random_seed=seed
        )

    def _verify_graph(self, graph: GraphInstance) -> bool:
        """Verify a generated graph."""
        from .verification import GraphVerifier

        verifier = GraphVerifier(fast_mode=(self.config.verification_mode == 'fast'))
        results = verifier.verify_all(graph.adjacency_matrix, graph.coordinates)

        # Check for critical failures
        for result in results:
            if not result.passed:
                if result.property_name in ['symmetry', 'euclidean_distances']:
                    # Critical failures
                    print(f"    WARNING: Graph {graph.id} failed {result.property_name} verification")
                    return False

        return True

    def _summarize_by_type(self, graphs: List[GraphInstance]) -> Dict[str, int]:
        """Summarize graphs by type."""
        by_type = {}
        for graph in graphs:
            graph_type = graph.metadata.graph_type
            by_type[graph_type] = by_type.get(graph_type, 0) + 1
        return by_type

    def _summarize_by_size(self, graphs: List[GraphInstance]) -> Dict[int, int]:
        """Summarize graphs by size."""
        by_size = {}
        for graph in graphs:
            size = graph.metadata.size
            by_size[size] = by_size.get(size, 0) + 1
        return dict(sorted(by_size.items()))

    def _summarize_properties(self, graphs: List[GraphInstance]) -> Dict[str, Any]:
        """Summarize graph properties."""
        if not graphs:
            return {}

        metric_count = sum(1 for g in graphs if g.properties.is_metric)
        symmetric_count = sum(1 for g in graphs if g.properties.is_symmetric)

        metricity_scores = [
            g.properties.metricity_score
            for g in graphs
            if g.properties.metricity_score is not None
        ]

        return {
            'metric_graphs': metric_count,
            'metric_percentage': metric_count / len(graphs) * 100,
            'symmetric_graphs': symmetric_count,
            'symmetric_percentage': symmetric_count / len(graphs) * 100,
            'avg_metricity_score': sum(metricity_scores) / len(metricity_scores) if metricity_scores else None
        }

    def _print_summary(self, report: Dict[str, Any]) -> None:
        """Print generation summary."""
        print("\n" + "=" * 70)
        print("BATCH GENERATION SUMMARY")
        print("=" * 70)
        print(f"Batch: {report['batch_name']}")
        print(f"Duration: {report['duration_seconds']:.2f} seconds")
        print(f"\nGeneration Results:")
        print(f"  Total Generated: {report['total_generated']}")
        print(f"  Total Failed: {report['total_failed']}")
        print(f"  Verification Failed: {report['verification_failed']}")

        print(f"\nGraphs by Type:")
        for graph_type, count in report['graphs_by_type'].items():
            print(f"  {graph_type}: {count}")

        print(f"\nGraphs by Size:")
        for size, count in report['graphs_by_size'].items():
            print(f"  {size} vertices: {count}")

        print(f"\nProperty Distribution:")
        props = report['property_distribution']
        if props:
            print(f"  Metric: {props.get('metric_graphs', 0)} ({props.get('metric_percentage', 0):.1f}%)")
            print(f"  Symmetric: {props.get('symmetric_graphs', 0)} ({props.get('symmetric_percentage', 0):.1f}%)")

        print("=" * 70)


def generate_batch_from_config(config_file: str) -> Dict[str, Any]:
    """
    Convenience function to generate a batch from configuration file.

    Args:
        config_file: Path to YAML or JSON configuration file

    Returns:
        Generation report
    """
    if config_file.endswith('.yaml') or config_file.endswith('.yml'):
        config = BatchGenerationConfig.from_yaml(config_file)
    else:
        config = BatchGenerationConfig.from_json(config_file)

    generator = BatchGenerator(config)
    return generator.generate_all()
