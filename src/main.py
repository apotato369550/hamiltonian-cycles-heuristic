"""
Main entry point for graph generation system.

Demonstrates the complete graph generation pipeline with examples
of all major functionality.
"""

import sys
from pathlib import Path

# Add graph_generation to path
sys.path.insert(0, str(Path(__file__).parent))

from graph_generation import (
    generate_euclidean_graph,
    generate_metric_graph,
    generate_random_graph,
    create_graph_instance,
    GraphStorage,
    BatchGenerator,
    BatchGenerationConfig,
    visualize_graph,
    analyze_collection,
    print_verification_report,
    GraphVerifier
)


def demo_single_graph_generation():
    """Demonstrate generating and working with a single graph."""
    print("=" * 70)
    print("DEMO 1: Single Graph Generation")
    print("=" * 70)

    # Generate a Euclidean graph
    print("\n1. Generating Euclidean graph...")
    matrix, coords = generate_euclidean_graph(
        num_vertices=10,
        dimensions=2,
        weight_range=(1.0, 100.0),
        distribution='uniform',
        random_seed=42
    )

    # Create graph instance with verification
    print("2. Creating graph instance with verification...")
    graph = create_graph_instance(
        adjacency_matrix=matrix,
        graph_type='euclidean',
        generation_params={
            'dimensions': 2,
            'weight_range': (1.0, 100.0),
            'distribution': 'uniform'
        },
        random_seed=42,
        coordinates=coords,
        verify=True
    )

    # Print summary
    print("\n3. Graph Summary:")
    print(graph.summary())

    # Verify properties
    print("\n4. Detailed Verification:")
    verifier = GraphVerifier(fast_mode=False)
    results = verifier.verify_all(graph.adjacency_matrix, graph.coordinates)
    print_verification_report(results)

    # Save graph
    print("\n5. Saving graph...")
    storage = GraphStorage()
    filepath = storage.save_graph(graph, subdirectory='demo')
    print(f"   Saved to: {filepath}")

    # Visualize (if matplotlib available)
    try:
        print("\n6. Creating visualizations...")
        viz_files = visualize_graph(graph, output_dir='visualizations/demo')
        print(f"   Created {len(viz_files)} visualizations")
        for f in viz_files:
            print(f"   - {f}")
    except Exception as e:
        print(f"   Visualization skipped: {e}")

    return graph


def demo_multiple_graph_types():
    """Demonstrate generating different graph types."""
    print("\n" + "=" * 70)
    print("DEMO 2: Multiple Graph Types")
    print("=" * 70)

    graphs = []

    # Euclidean graph
    print("\n1. Euclidean Graph (2D, clustered):")
    matrix, coords = generate_euclidean_graph(
        num_vertices=15,
        dimensions=2,
        distribution='clustered',
        distribution_params={'num_clusters': 3},
        random_seed=100
    )
    graph = create_graph_instance(matrix, 'euclidean', {}, 100, coords)
    print(f"   ID: {graph.id}, Metric: {graph.properties.is_metric}")
    graphs.append(graph)

    # Metric graph
    print("\n2. Metric Graph (non-Euclidean):")
    matrix = generate_metric_graph(
        num_vertices=15,
        weight_range=(1.0, 100.0),
        strategy='mst',
        random_seed=101
    )
    graph = create_graph_instance(matrix, 'metric', {'strategy': 'mst'}, 101)
    print(f"   ID: {graph.id}, Metric: {graph.properties.is_metric}")
    graphs.append(graph)

    # Random graph
    print("\n3. Random Graph (uniform distribution):")
    matrix = generate_random_graph(
        num_vertices=15,
        distribution='uniform',
        random_seed=102
    )
    graph = create_graph_instance(matrix, 'random', {'distribution': 'uniform'}, 102)
    print(f"   ID: {graph.id}, Metric: {graph.properties.is_metric}")
    print(f"   Metricity Score: {graph.properties.metricity_score:.2%}")
    graphs.append(graph)

    # Save all graphs
    print("\n4. Saving graphs...")
    storage = GraphStorage()
    manifest = storage.save_batch(graphs, 'demo_types')
    print(f"   Saved {manifest['num_graphs']} graphs to batch 'demo_types'")

    return graphs


def demo_batch_generation():
    """Demonstrate batch generation from configuration."""
    print("\n" + "=" * 70)
    print("DEMO 3: Batch Generation from Configuration")
    print("=" * 70)

    # Create a simple configuration
    config_dict = {
        'batch_name': 'demo_batch',
        'output_directory': 'data/graphs',
        'verification_mode': 'fast',
        'continue_on_error': True,
        'graphs': [
            {
                'type': 'euclidean',
                'sizes': [10, 20],
                'instances_per_size': 2,
                'seed_start': 1000,
                'parameters': {
                    'dimensions': 2,
                    'weight_range': [1.0, 100.0],
                    'distribution': 'uniform'
                }
            },
            {
                'type': 'metric',
                'sizes': [10, 20],
                'instances_per_size': 2,
                'seed_start': 2000,
                'parameters': {
                    'weight_range': [1.0, 100.0],
                    'strategy': 'mst'
                }
            },
            {
                'type': 'random',
                'sizes': [10, 20],
                'instances_per_size': 2,
                'seed_start': 3000,
                'parameters': {
                    'weight_range': [1.0, 100.0],
                    'distribution': 'uniform',
                    'is_symmetric': True
                }
            }
        ]
    }

    print("\n1. Configuration:")
    print(f"   Batch name: {config_dict['batch_name']}")
    print(f"   Graph types: {len(config_dict['graphs'])}")
    print(f"   Total graphs to generate: 12")

    print("\n2. Starting batch generation...")
    config = BatchGenerationConfig(config_dict)
    generator = BatchGenerator(config)
    report = generator.generate_all()

    print(f"\n3. Generation complete!")
    print(f"   Generated: {report['total_generated']}")
    print(f"   Failed: {report['total_failed']}")

    return report


def demo_collection_analysis():
    """Demonstrate collection analysis."""
    print("\n" + "=" * 70)
    print("DEMO 4: Collection Analysis")
    print("=" * 70)

    print("\n1. Analyzing graph collection...")
    analysis = analyze_collection(
        batch_name='demo_batch',
        create_visualizations=False  # Set to True if matplotlib available
    )

    if 'error' in analysis:
        print(f"   Error: {analysis['error']}")
        return

    print("\n2. Summary:")
    summary = analysis['summary']
    print(f"   Total graphs: {summary['total_graphs']}")
    print(f"   Unique types: {summary['unique_types']}")
    print(f"   Size range: {summary['size_range']}")

    print("\n3. Coverage:")
    coverage = analysis['coverage']
    print(f"   Total combinations: {coverage['total_combinations']}")
    print(f"   Instances per combination: {coverage['min_instances']}-{coverage['max_instances']}")

    print("\n4. Property Distribution:")
    props = analysis['property_distribution']
    print(f"   Metric graphs: {props['metric_count']} ({props['metric_percentage']:.1f}%)")
    print(f"   Symmetric graphs: {props['symmetric_count']} ({props['symmetric_percentage']:.1f}%)")

    print("\n5. Diversity Metrics:")
    diversity = analysis['diversity_metrics']
    print(f"   Diversity score: {diversity['diversity_score']:.2f}")
    print(f"   Avg pairwise distance: {diversity['avg_pairwise_distance']:.2f}")

    return analysis


def demo_storage_queries():
    """Demonstrate storage and querying."""
    print("\n" + "=" * 70)
    print("DEMO 5: Storage and Querying")
    print("=" * 70)

    storage = GraphStorage()

    # Get storage stats
    print("\n1. Storage Statistics:")
    stats = storage.get_storage_stats()
    print(f"   Total graphs: {stats['total_graphs']}")
    print(f"   Storage size: {stats['total_size_mb']:.2f} MB")
    print(f"   By type: {stats['by_type']}")

    # Find graphs by criteria
    print("\n2. Querying graphs:")

    # Find all Euclidean graphs
    euclidean_graphs = storage.find_graphs(graph_type='euclidean')
    print(f"   Euclidean graphs: {len(euclidean_graphs)}")

    # Find graphs in size range
    medium_graphs = storage.find_graphs(size_range=(15, 25))
    print(f"   Graphs with 15-25 vertices: {len(medium_graphs)}")

    # Find metric graphs
    metric_graphs = storage.find_graphs(is_metric=True)
    print(f"   Metric graphs: {len(metric_graphs)}")

    # List batches
    print("\n3. Available batches:")
    batches = storage.list_batches()
    for batch in batches:
        print(f"   - {batch}")


def main():
    """Run all demonstrations."""
    print("\n" + "=" * 70)
    print("GRAPH GENERATION SYSTEM - DEMONSTRATION")
    print("=" * 70)

    try:
        # Demo 1: Single graph
        demo_single_graph_generation()

        # Demo 2: Multiple types
        demo_multiple_graph_types()

        # Demo 3: Batch generation
        demo_batch_generation()

        # Demo 4: Collection analysis
        demo_collection_analysis()

        # Demo 5: Storage queries
        demo_storage_queries()

        print("\n" + "=" * 70)
        print("ALL DEMONSTRATIONS COMPLETE")
        print("=" * 70)

    except Exception as e:
        print(f"\n\nError during demonstration: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
