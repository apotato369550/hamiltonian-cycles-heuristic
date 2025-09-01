#!/usr/bin/env python3
"""
Comprehensive CLI interface for TSP Heuristic Algorithms

This CLI provides a unified interface to run, compare, generate graphs, and visualize
various TSP heuristic algorithms including established, experimental, and anchoring-based approaches.
"""

import argparse
import sys
import json
import csv
import time
from typing import List, Dict, Any, Optional
from pathlib import Path

# Import graph generation utilities
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.graph_generator import generate_complete_graph, analyze_graph_properties, calculate_cycle_cost

# Import established algorithms
from established.christofides import Christofides
from established.nearest_neighbor import NearestNeighbor
from established.prims import PrimsTSP
from established.kruskals import KruskalsGreedy
from established.kruskals_family import KruskalsFamily

# Import experimental algorithms
from experimental.hamiltonian import Hamiltonian
from experimental.pressure_field import PressureField
from experimental.hybrid_anchor_established import HybridAnchorEstablished
from experimental.advanced_local_search import AdvancedLocalSearch

# Import anchoring algorithms
from anchoring.low_anchor_heuristic import LowAnchorHeuristic
from anchoring.low_anchor_metaheuristic import LowAnchorMetaheuristic
from anchoring.anchor_heuristic_family import AnchorHeuristicFamily
from anchoring.hamiltonian_improved import HamiltonianImproved
from anchoring.bidirectional_greedy import BidirectionalGreedy
from anchoring.hamiltonian_anchor import HamiltonianAnchor

# Import simulator for visualization
from cli.simulator import TSPStepByStepVisualizer

# Algorithm registry
ALGORITHMS = {
    'established': {
        'christofides': Christofides,
        'nearest_neighbor': NearestNeighbor,
        'prims': PrimsTSP,
        'kruskals': KruskalsGreedy,
        'kruskals_family': KruskalsFamily,
    },
    'experimental': {
        'hamiltonian': Hamiltonian,
        'pressure_field': PressureField,
        'hybrid_anchor_established': HybridAnchorEstablished,
        'advanced_local_search': AdvancedLocalSearch,
    },
    'anchoring': {
        'low_anchor_heuristic': LowAnchorHeuristic,
        'low_anchor_metaheuristic': LowAnchorMetaheuristic,
        'anchor_heuristic_family': AnchorHeuristicFamily,
        'hamiltonian_improved': HamiltonianImproved,
        'bidirectional_greedy': BidirectionalGreedy,
        'hamiltonian_anchor': HamiltonianAnchor,
    }
}

def get_algorithm_instance(algorithm_name: str):
    """Get algorithm instance by name."""
    for category, algorithms in ALGORITHMS.items():
        if algorithm_name in algorithms:
            return algorithms[algorithm_name]()
    raise ValueError(f"Unknown algorithm: {algorithm_name}")

def get_available_algorithms() -> List[str]:
    """Get list of all available algorithm names."""
    algorithms = []
    for category, algos in ALGORITHMS.items():
        algorithms.extend(algos.keys())
    return algorithms

def run_single_algorithm(args):
    """Run a single algorithm on a generated graph."""
    print(f"[INFO] Running {args.algorithm} algorithm...")
    print(f"   Graph size: {args.vertices} vertices")
    print(f"   Weight range: {args.min_weight}-{args.max_weight}")
    print(f"   Metric graph: {args.metric}")
    print(f"   Seed: {args.seed}")

    # Generate graph
    graph = generate_complete_graph(
        vertices=args.vertices,
        weight_range=(args.min_weight, args.max_weight),
        metric=args.metric,
        seed=args.seed
    )

    # Get algorithm instance
    algorithm = get_algorithm_instance(args.algorithm)

    # Run algorithm
    start_time = time.time()
    try:
        cycle = algorithm.solve(graph)
        weight = algorithm.evaluate(cycle)
        execution_time = time.time() - start_time

        # Display results
        print("\n[SUCCESS] Algorithm completed successfully!")
        print(f"   Cycle: {' -> '.join(map(str, cycle))}")
        print(f"   Total weight: {weight}")
        print(f"   Execution time: {execution_time:.3f}s")

        # Analyze graph properties
        if args.show_stats:
            graph_stats = analyze_graph_properties(graph)
            print("\n[STATS] Graph Statistics:")
            print(f"   Vertices: {graph_stats['vertices']}")
            print(f"   Edges: {graph_stats['edges']}")
            print(f"   Weight range: {graph_stats['weight_range']}")
            print(f"   Weight mean: {graph_stats['weight_mean']:.2f}")
            print(f"   Weight std: {graph_stats['weight_std']:.2f}")
            print(f"   Is metric: {graph_stats['is_metric']}")
            if not graph_stats['is_metric']:
                print(f"   Triangle violations: {graph_stats['triangle_violations']}")

        # Save results if requested
        if args.output:
            result_data = {
                'algorithm': args.algorithm,
                'vertices': args.vertices,
                'weight_range': [args.min_weight, args.max_weight],
                'metric': args.metric,
                'seed': args.seed,
                'cycle': cycle,
                'weight': weight,
                'execution_time': execution_time,
                'graph_stats': analyze_graph_properties(graph) if args.show_stats else None
            }

            if args.output_format == 'json':
                with open(args.output, 'w') as f:
                    json.dump(result_data, f, indent=2)
                print(f"[SAVE] Results saved to {args.output}")
            elif args.output_format == 'csv':
                with open(args.output, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['Algorithm', 'Vertices', 'Weight', 'Execution Time', 'Cycle'])
                    writer.writerow([args.algorithm, args.vertices, weight, execution_time, ' -> '.join(map(str, cycle))])
                print(f"[SAVE] Results saved to {args.output}")

    except Exception as e:
        print(f"[ERROR] Error running algorithm: {e}")
        sys.exit(1)

def run_comparison(args):
    """Compare multiple algorithms on the same graph."""
    print(f"[INFO] Comparing algorithms on {args.vertices}-vertex graph...")
    print(f"   Algorithms: {', '.join(args.algorithms)}")
    print(f"   Weight range: {args.min_weight}-{args.max_weight}")
    print(f"   Metric graph: {args.metric}")
    print(f"   Seed: {args.seed}")
    print(f"   Runs per algorithm: {args.runs}")

    # Generate graph
    graph = generate_complete_graph(
        vertices=args.vertices,
        weight_range=(args.min_weight, args.max_weight),
        metric=args.metric,
        seed=args.seed
    )

    results = []

    for algorithm_name in args.algorithms:
        print(f"\n[RUN] Running {algorithm_name}...")
        algorithm = get_algorithm_instance(algorithm_name)

        algorithm_results = []
        for run in range(args.runs):
            start_time = time.time()
            try:
                cycle = algorithm.solve(graph)
                weight = algorithm.evaluate(cycle)
                execution_time = time.time() - start_time
                algorithm_results.append({
                    'run': run + 1,
                    'cycle': cycle,
                    'weight': weight,
                    'execution_time': execution_time
                })
                print(f"   Run {run + 1}: Weight = {weight}, Time = {execution_time:.3f}s")
            except Exception as e:
                print(f"   Run {run + 1}: Error - {e}")
                algorithm_results.append({
                    'run': run + 1,
                    'error': str(e)
                })

        # Calculate statistics
        successful_runs = [r for r in algorithm_results if 'weight' in r]
        if successful_runs:
            weights = [r['weight'] for r in successful_runs]
            times = [r['execution_time'] for r in successful_runs]

            stats = {
                'algorithm': algorithm_name,
                'successful_runs': len(successful_runs),
                'total_runs': args.runs,
                'best_weight': min(weights),
                'worst_weight': max(weights),
                'avg_weight': sum(weights) / len(weights),
                'avg_time': sum(times) / len(times),
                'results': algorithm_results
            }
            results.append(stats)

    # Display comparison results
    print("\n[RESULTS] Comparison Results:")
    print("-" * 80)
    print(f"{'Algorithm':<15} {'Success':<8} {'Best':<10} {'Worst':<10} {'Avg':<10} {'Time':<8}")
    print("-" * 80)

    for result in sorted(results, key=lambda x: x['avg_weight']):
        print(f"{result['algorithm']:<15} "
              f"{result['successful_runs']}/{result['total_runs']:<8} "
              f"{result['best_weight']:<10.2f} "
              f"{result['worst_weight']:<10.2f} "
              f"{result['avg_weight']:<10.2f} "
              f"{result['avg_time']:<8.3f}")
    # Save results if requested
    if args.output:
        output_data = {
            'comparison_config': {
                'vertices': args.vertices,
                'weight_range': [args.min_weight, args.max_weight],
                'metric': args.metric,
                'seed': args.seed,
                'algorithms': args.algorithms,
                'runs': args.runs
            },
            'results': results,
            'graph_stats': analyze_graph_properties(graph) if args.show_stats else None
        }

        if args.output_format == 'json':
            with open(args.output, 'w') as f:
                json.dump(output_data, f, indent=2)
            print(f"\n[SAVE] Comparison results saved to {args.output}")
        elif args.output_format == 'csv':
            with open(args.output, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Algorithm', 'Successful Runs', 'Best Weight', 'Worst Weight', 'Avg Weight', 'Avg Time'])
                for result in results:
                    writer.writerow([
                        result['algorithm'],
                        result['successful_runs'],
                        result['best_weight'],
                        result['worst_weight'],
                        result['avg_weight'],
                        result['avg_time']
                    ])
            print(f"\n[SAVE] Comparison results saved to {args.output}")

def generate_graph(args):
    """Generate and save graphs."""
    print(f"[INFO] Generating {args.vertices}-vertex graph...")
    print(f"   Weight range: {args.min_weight}-{args.max_weight}")
    print(f"   Metric graph: {args.metric}")
    print(f"   Seed: {args.seed}")
    print(f"   Output format: {args.format}")

    # Generate graph
    graph = generate_complete_graph(
        vertices=args.vertices,
        weight_range=(args.min_weight, args.max_weight),
        metric=args.metric,
        seed=args.seed
    )

    # Analyze graph
    stats = analyze_graph_properties(graph)

    # Display graph info
    print("\n[STATS] Generated Graph:")
    print(f"   Vertices: {stats['vertices']}")
    print(f"   Edges: {stats['edges']}")
    print(f"   Weight range: {stats['weight_range']}")
    print(f"   Weight mean: {stats['weight_mean']:.2f}")
    print(f"   Weight std: {stats['weight_std']:.2f}")
    print(f"   Is metric: {stats['is_metric']}")
    if not stats['is_metric']:
        print(f"   Triangle violations: {stats['triangle_violations']}")

    # Save graph
    if args.output:
        if args.format == 'json':
            output_data = {
                'metadata': {
                    'vertices': args.vertices,
                    'weight_range': [args.min_weight, args.max_weight],
                    'metric': args.metric,
                    'seed': args.seed,
                    'generated_at': time.time()
                },
                'statistics': stats,
                'adjacency_matrix': graph
            }
            with open(args.output, 'w') as f:
                json.dump(output_data, f, indent=2)
        elif args.format == 'csv':
            with open(args.output, 'w', newline='') as f:
                writer = csv.writer(f)
                # Write metadata
                writer.writerow(['Vertices', args.vertices])
                writer.writerow(['Weight Range', f"{args.min_weight}-{args.max_weight}"])
                writer.writerow(['Metric', args.metric])
                writer.writerow(['Seed', args.seed])
                writer.writerow([])
                # Write adjacency matrix
                writer.writerow(['Adjacency Matrix'] + list(range(args.vertices)))
                for i, row in enumerate(graph):
                    writer.writerow([i] + row)

        print(f"[SAVE] Graph saved to {args.output}")

    # Display sample of graph if requested
    if args.show_sample:
        print("\n[SAMPLE] Sample adjacency matrix (first 5x5):")
        for i in range(min(5, args.vertices)):
            row = [graph[i][j] for j in range(min(5, args.vertices))]
            print(f"   {i}: {row}")

def run_visualization(args):
    """Run interactive visualization."""
    print("[INFO] Starting TSP Algorithm Visualizer...")
    print(f"   Graph size: {args.vertices} vertices")
    print(f"   Weight range: {args.min_weight}-{args.max_weight}")
    print(f"   Metric graph: {args.metric}")
    print(f"   Seed: {args.seed}")

    # Create visualizer
    visualizer = TSPStepByStepVisualizer(num_vertices=args.vertices)

    # Generate graph
    if args.metric:
        graph = visualizer.generate_euclidean_graph(seed=args.seed)
        graph_type = "Euclidean Distance"
    else:
        graph = visualizer.generate_random_graph(seed=args.seed)
        graph_type = "Random Weights"

    visualizer.create_networkx_graph()

    print(f"\n[STATS] Generated {graph_type} Graph ({args.vertices} vertices)")
    print("\nAdjacency Matrix (first 5x5):")
    for i in range(min(5, args.vertices)):
        row = [graph[i][j] for j in range(min(5, args.vertices))]
        print(f"  {i}: {row}")

    # Run visualization
    try:
        result = visualizer.run_complete_simulation(use_euclidean=args.metric, seed=args.seed)
        print("\n[SUCCESS] Visualization completed!")
        if result:
            print(f"   Final cycle: {' -> '.join(map(str, result['best_cycle']))}")
            print(f"   Total weight: {result['best_weight']}")
    except Exception as e:
        print(f"[ERROR] Error during visualization: {e}")
        sys.exit(1)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="TSP Heuristic Algorithms CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run single algorithm
  python cli/cli.py run christofides --vertices 10 --metric

  # Run anchoring algorithm
  python cli/cli.py run low_anchor_heuristic --vertices 12 --metric

  # Compare multiple algorithms
  python cli/cli.py compare nearest_neighbor christofides --vertices 15 --runs 3

  # Compare with anchoring algorithms
  python cli/cli.py compare nearest_neighbor low_anchor_heuristic anchor_heuristic_family --vertices 15 --runs 3

  # Generate and save graph
  python cli/cli.py generate --vertices 20 --output graph.json

  # Run interactive visualization
  python cli/cli.py visualize --vertices 6 --metric

Available algorithms:
  Established: christofides, nearest_neighbor, prims, kruskals, kruskals_family
  Experimental: hamiltonian, pressure_field, hybrid_anchor_established, advanced_local_search
  Anchoring: low_anchor_heuristic, low_anchor_metaheuristic, anchor_heuristic_family, hamiltonian_improved, bidirectional_greedy, hamiltonian_anchor
        """
    )

    parser.add_argument('--version', action='version', version='TSP CLI v1.0.0')

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Run command
    run_parser = subparsers.add_parser('run', help='Run a single algorithm')
    run_parser.add_argument('algorithm', choices=get_available_algorithms(),
                           help='Algorithm to run')
    run_parser.add_argument('--vertices', type=int, default=10,
                           help='Number of vertices in the graph (default: 10)')
    run_parser.add_argument('--min-weight', type=int, default=1,
                           help='Minimum edge weight (default: 1)')
    run_parser.add_argument('--max-weight', type=int, default=100,
                           help='Maximum edge weight (default: 100)')
    run_parser.add_argument('--metric', action='store_true',
                           help='Generate metric graph (satisfies triangle inequality)')
    run_parser.add_argument('--seed', type=int,
                           help='Random seed for reproducible results')
    run_parser.add_argument('--show-stats', action='store_true',
                           help='Show detailed graph statistics')
    run_parser.add_argument('--output', type=str,
                           help='Output file path for results')
    run_parser.add_argument('--output-format', choices=['json', 'csv'], default='json',
                           help='Output format (default: json)')

    # Compare command
    compare_parser = subparsers.add_parser('compare', help='Compare multiple algorithms')
    compare_parser.add_argument('algorithms', nargs='+', choices=get_available_algorithms(),
                               help='Algorithms to compare')
    compare_parser.add_argument('--vertices', type=int, default=15,
                               help='Number of vertices in the graph (default: 15)')
    compare_parser.add_argument('--min-weight', type=int, default=1,
                               help='Minimum edge weight (default: 1)')
    compare_parser.add_argument('--max-weight', type=int, default=100,
                               help='Maximum edge weight (default: 100)')
    compare_parser.add_argument('--metric', action='store_true',
                               help='Generate metric graph (satisfies triangle inequality)')
    compare_parser.add_argument('--seed', type=int,
                               help='Random seed for reproducible results')
    compare_parser.add_argument('--runs', type=int, default=1,
                               help='Number of runs per algorithm (default: 1)')
    compare_parser.add_argument('--show-stats', action='store_true',
                               help='Show detailed graph statistics')
    compare_parser.add_argument('--output', type=str,
                               help='Output file path for comparison results')
    compare_parser.add_argument('--output-format', choices=['json', 'csv'], default='json',
                               help='Output format (default: json)')

    # Generate command
    generate_parser = subparsers.add_parser('generate', help='Generate and save graphs')
    generate_parser.add_argument('--vertices', type=int, default=10,
                                help='Number of vertices in the graph (default: 10)')
    generate_parser.add_argument('--min-weight', type=int, default=1,
                                help='Minimum edge weight (default: 1)')
    generate_parser.add_argument('--max-weight', type=int, default=100,
                                help='Maximum edge weight (default: 100)')
    generate_parser.add_argument('--metric', action='store_true',
                                help='Generate metric graph (satisfies triangle inequality)')
    generate_parser.add_argument('--seed', type=int,
                                help='Random seed for reproducible results')
    generate_parser.add_argument('--output', type=str,
                                help='Output file path for the generated graph')
    generate_parser.add_argument('--format', choices=['json', 'csv'], default='json',
                                help='Output format (default: json)')
    generate_parser.add_argument('--show-sample', action='store_true',
                                help='Show sample of the adjacency matrix')

    # Visualize command
    visualize_parser = subparsers.add_parser('visualize', help='Run interactive visualization')
    visualize_parser.add_argument('--vertices', type=int, default=6,
                                 help='Number of vertices in the graph (default: 6)')
    visualize_parser.add_argument('--min-weight', type=int, default=5,
                                 help='Minimum edge weight (default: 5)')
    visualize_parser.add_argument('--max-weight', type=int, default=25,
                                 help='Maximum edge weight (default: 25)')
    visualize_parser.add_argument('--metric', action='store_true',
                                 help='Generate metric graph (satisfies triangle inequality)')
    visualize_parser.add_argument('--seed', type=int, default=42,
                                 help='Random seed for reproducible results (default: 42)')

    # Parse arguments
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Execute command
    try:
        if args.command == 'run':
            run_single_algorithm(args)
        elif args.command == 'compare':
            run_comparison(args)
        elif args.command == 'generate':
            generate_graph(args)
        elif args.command == 'visualize':
            run_visualization(args)
    except KeyboardInterrupt:
        print("\n[WARNING] Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()