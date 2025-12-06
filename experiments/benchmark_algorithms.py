#!/usr/bin/env python3
"""
Standalone Algorithm Benchmarking Script

Simple script to test TSP algorithms on generated graphs without running the full pipeline.
Configure algorithms, graph types, and sizes to benchmark.

Usage:
    python experiments/benchmark_algorithms.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import time
import numpy as np
from typing import List, Dict, Any
import matplotlib.pyplot as plt
from src.algorithms.registry import AlgorithmRegistry
from src.graph_generation.euclidean_generator import EuclideanGraphGenerator
from src.graph_generation.metric_generator import MetricGraphGenerator
from src.graph_generation.random_generator import RandomGraphGenerator


# ===== CONFIGURATION =====
# Edit these settings to customize your benchmarking

# Which algorithms to test (see available with list_algorithms())
# Available: nearest_neighbor_random, nearest_neighbor_best, greedy_edge,
#            held_karp_exact, single_anchor_v1, single_anchor_v2,
#            best_anchor_exhaustive, multi_anchor_random, multi_anchor_distributed
ALGORITHMS = [
    'nearest_neighbor_random',
    'nearest_neighbor_adaptive',
    'nearest_neighbor_best',
    'single_anchor_v1',
    'single_anchor_v2',
    'single_anchor_v3',
    'greedy_edge',
    'best_anchor_exhaustive',
]

# Graph configurations: (type, size)
# Types: 'euclidean', 'metric', 'random'
GRAPH_CONFIGS = [
    ('random', 10),
    ('random', 20),
    ('random', 50),
    ('random', 100),
    ('random', 150),
    ('random', 250),
    ('random', 500),
]

# Number of graphs to generate per configuration
GRAPHS_PER_CONFIG = 3

# Random seed for reproducibility
SEED = 42

# Show plots?
SHOW_PLOTS = True

# ===== END CONFIGURATION =====


def generate_graph(graph_type: str, size: int, seed: int):
    """Generate a graph of specified type and size."""
    if graph_type == 'euclidean':
        gen = EuclideanGraphGenerator(random_seed=seed)
        adjacency_matrix, coordinates = gen.generate(num_vertices=size)
        return adjacency_matrix
    elif graph_type == 'metric':
        gen = MetricGraphGenerator(random_seed=seed)
        adjacency_matrix = gen.generate(num_vertices=size, strategy='completion')
        return adjacency_matrix
    elif graph_type == 'random':
        gen = RandomGraphGenerator(random_seed=seed)
        adjacency_matrix = gen.generate(num_vertices=size)
        return adjacency_matrix
    else:
        raise ValueError(f"Unknown graph type: {graph_type}")


def benchmark_algorithms():
    """Run benchmarking experiments."""
    print("=" * 80)
    print("TSP ALGORITHM BENCHMARKING")
    print("=" * 80)
    print(f"\nAlgorithms: {', '.join(ALGORITHMS)}")
    print(f"Graph configurations: {len(GRAPH_CONFIGS)}")
    print(f"Graphs per configuration: {GRAPHS_PER_CONFIG}")
    print(f"Total runs: {len(ALGORITHMS) * len(GRAPH_CONFIGS) * GRAPHS_PER_CONFIG}")
    print(f"Random seed: {SEED}\n")

    # Storage for results
    results = []

    # Run benchmarks
    for graph_type, size in GRAPH_CONFIGS:
        print(f"\n{'='*80}")
        print(f"Graph Type: {graph_type.upper()}, Size: {size}")
        print(f"{'='*80}")

        # Generate graphs for this configuration
        graphs = []
        for i in range(GRAPHS_PER_CONFIG):
            graph = generate_graph(graph_type, size, SEED + i)
            graphs.append(graph)

        # Test each algorithm
        for algo_name in ALGORITHMS:
            print(f"\n--- {algo_name} ---")

            times = []
            qualities = []

            for i, graph in enumerate(graphs, 1):
                try:
                    # Create algorithm with same seed as graph generation for fair comparison
                    algo_with_seed = AlgorithmRegistry.get_algorithm(algo_name, random_seed=SEED + i - 1)
                    result = algo_with_seed.solve(graph)

                    if result.success:
                        times.append(result.runtime)
                        qualities.append(result.weight)

                        print(f"  Graph {i}: quality={result.weight:.2f}, time={result.runtime:.4f}s")
                    else:
                        print(f"  Graph {i}: FAILED - {result.error_message}")

                except Exception as e:
                    print(f"  Graph {i}: FAILED - {e}")
                    continue

            if times:
                avg_time = np.mean(times)
                avg_quality = np.mean(qualities)
                std_quality = np.std(qualities)

                print(f"  Average: quality={avg_quality:.2f} (±{std_quality:.2f}), time={avg_time:.4f}s")

                results.append({
                    'algorithm': algo_name,
                    'graph_type': graph_type,
                    'graph_size': size,
                    'avg_quality': avg_quality,
                    'std_quality': std_quality,
                    'avg_time': avg_time,
                    'qualities': qualities,
                    'times': times
                })

    return results


def display_summary(results: List[Dict[str, Any]]):
    """Display summary table of results."""
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'Algorithm':<20} {'Graph Type':<15} {'Size':<6} {'Avg Quality':<15} {'Avg Time (s)'}")
    print("-" * 80)

    for r in results:
        print(f"{r['algorithm']:<20} {r['graph_type']:<15} {r['graph_size']:<6} "
              f"{r['avg_quality']:>7.2f} ±{r['std_quality']:>5.2f}  {r['avg_time']:>10.4f}")


def plot_results(results: List[Dict[str, Any]]):
    """Create visualization of benchmark results."""
    if not results:
        return

    # Group by graph configuration
    configs = list(set((r['graph_type'], r['graph_size']) for r in results))
    configs.sort()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Quality comparison
    ax1 = axes[0]
    width = 0.8 / len(ALGORITHMS)
    x = np.arange(len(configs))

    for i, algo in enumerate(ALGORITHMS):
        qualities = []
        errors = []
        for config in configs:
            matches = [r for r in results if r['algorithm'] == algo and
                      (r['graph_type'], r['graph_size']) == config]
            if matches:
                qualities.append(matches[0]['avg_quality'])
                errors.append(matches[0]['std_quality'])
            else:
                qualities.append(0)
                errors.append(0)

        ax1.bar(x + i * width, qualities, width, label=algo, yerr=errors, capsize=3)

    ax1.set_xlabel('Graph Configuration')
    ax1.set_ylabel('Tour Quality (lower is better)')
    ax1.set_title('Algorithm Performance Comparison')
    ax1.set_xticks(x + width * (len(ALGORITHMS) - 1) / 2)
    ax1.set_xticklabels([f"{t}\nn={s}" for t, s in configs])
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    # Plot 2: Runtime comparison
    ax2 = axes[1]
    for i, algo in enumerate(ALGORITHMS):
        times = []
        for config in configs:
            matches = [r for r in results if r['algorithm'] == algo and
                      (r['graph_type'], r['graph_size']) == config]
            if matches:
                times.append(matches[0]['avg_time'])
            else:
                times.append(0)

        ax2.bar(x + i * width, times, width, label=algo)

    ax2.set_xlabel('Graph Configuration')
    ax2.set_ylabel('Runtime (seconds)')
    ax2.set_title('Algorithm Runtime Comparison')
    ax2.set_xticks(x + width * (len(ALGORITHMS) - 1) / 2)
    ax2.set_xticklabels([f"{t}\nn={s}" for t, s in configs])
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    # Save plot
    output_path = os.path.join(os.path.dirname(__file__), 'benchmark_results.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")

    if SHOW_PLOTS:
        plt.show()


def main():
    """Main entry point."""
    print("\nAvailable algorithms:", AlgorithmRegistry.list_algorithms())
    print()

    # Run benchmarks
    results = benchmark_algorithms()

    # Display results
    display_summary(results)

    # Create visualizations
    plot_results(results)

    print("\n" + "=" * 80)
    print("Benchmarking complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
