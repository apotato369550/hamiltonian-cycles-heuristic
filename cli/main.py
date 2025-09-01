#!/usr/bin/env python3
"""
Legacy CLI interface for TSP Heuristic Algorithms

This file provides backward compatibility and simple test functions.
For the full-featured CLI, use cli/cli.py instead.
"""

import sys
import os
from collections import defaultdict

# Add parent directory to path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.graph_generator import generate_complete_graph
from utils import print_graph, run_base_benchmark, greedy_algorithm, find_optimal_cycle_held_karp
from anchoring.low_anchor_heuristic import LowAnchorHeuristic
from anchoring.hamiltonian_anchor import HamiltonianAnchor
from anchoring.hamiltonian_improved import HamiltonianImproved
from anchoring.bidirectional_greedy import BidirectionalGreedy
from kruskals_greedy_family import greedy_edge_tsp, greedy_edge_tsp_v2, greedy_edge_tsp_v3
from anchoring.anchor_heuristic_family import AnchorHeuristicFamily
from pressure_field_heuristic import pressure_field_heuristic
from christofides_algorithm import christofides_algorithm
from prims_tsp import prims_tsp, best_prims_tsp, prims_anchoring_tsp, best_prims_anchoring_tsp
from github_heuristic import heuristic_path
from nearest_neighbor import nearest_neighbor_tsp, best_nearest_neighbor

from anchoring.low_anchor_metaheuristic import LowAnchorMetaheuristic

def base_heuristic_test(num_graphs=3, num_vertices=9, weight_range=(1, 100), seed_base=100):
    all_weights = defaultdict(list)

    for g in range(num_graphs):
        print(f"\n--- Graph {g+1} ---")
        graph = generate_complete_graph(num_vertices, weight_range=weight_range, seed=seed_base + g)
        print_graph(graph)

        for v in range(num_vertices):
            results = run_base_benchmark(graph, v)
            for method, (_, weight) in results.items():
                all_weights[method].append(weight)

    # Compute and print average weights
    print("\n--- AVERAGE RESULTS ACROSS ALL TESTS ---")
    for method, weights in all_weights.items():
        avg = sum(weights) / len(weights)
        print(f"{method}: Average Weight = {avg:.2f} over {len(weights)} runs")

def improved_base_heuristic_test(num_graphs=3, num_vertices=15, weight_range=(1, 100), seed_base=200):
    """
    Args:
        num_graphs (int): Number of graphs to generate.
        num_vertices (int): Number of vertices per graph.
        weight_range (tuple): Range of edge weights.
        seed_base (int): Seed base for reproducibility.
    """
    all_weights = defaultdict(list)

    for g in range(num_graphs):
        print(f"\n--- Multi-Anchor Graph {g+1} ---")
        graph = generate_complete_graph(num_vertices, weight_range=weight_range, seed=seed_base + g)
        # print_graph(graph)

        for v in range(num_vertices):
            anchors = [v, (v + 5) % num_vertices]  
            # example: 3 anchors spread apart
            # anchors = [v]
            results = {}

            # Run optimal
            # results["optimal_held_karp"] = find_optimal_cycle_held_karp(graph, v)
            
            # Run greedy
            # work on this VVV
            ranking_info = rank_vertices_by_weight(graph)
            highest_v = ranking_info["highest_weight_vertex"]
            # print(f"Vertex weights and shit: ")
            # print(ranking_info["sorted_vertices_and_weights"])


            # Run your multi-anchor heuristic
            results["nearest_neighbor"] = nearest_neighbor_tsp(graph, v)
            results["best_nearest_neighbor"] = best_nearest_neighbor(graph)
            results["single_anchor"] = low_anchor_heuristic(graph, v)
            results["best_anchor"] = best_anchor_heuristic(graph)
            results["github_heuristic"] = heuristic_path(graph, v)
            results["prims_tsp"] = prims_tsp(graph, v)
            results["best_prims_tsp"] = best_prims_tsp(graph)
            results["prims_anchor_tsp"] = prims_anchoring_tsp(graph, v)
            results["best_prims_anchor_tsp"] = best_prims_anchoring_tsp(graph)
            # renamed to kruskal's TSP (from greedy_v3)
            results["kruskals_tsp"] = greedy_edge_tsp_v3(graph)
            '''
            results["high_anchor_heuristic"] = low_anchor_heuristic(graph, highest_v)
            results["best_2_anchor"] = best_multi_anchor_heuristic(graph, 2)
            results["best_2_anchor_early_exit"] = best_multi_anchor_heuristic(graph, 2, early_exit=True)
            results["best_3_anchor"] = best_multi_anchor_heuristic(graph, 3)
            results["multi_anchor"] = hamiltonian_cycle_heuristic(
                graph,
                start=v,
                anchors=anchors,
                max_depth=-1,
                early_exit=False
            )
            results["greedy_v1"] = greedy_edge_tsp(graph, v)
            results["greedy_v2"] = greedy_edge_tsp_v2(graph, v)
            results["pressure_field_heuristic"] = pressure_field_heuristic(graph)
            results["bidirectional_greedy"] = bidirectional_nearest_neighbor_tsp(graph, v)
            results["adaptive_anchor"] = adaptive_anchor_heuristic(graph, v)
            results["multi_anchor_v2"] = multi_anchor_heuristic(graph, v)
            results["smart_anchor"] = smart_anchor_heuristic(graph, v)
            results["hybrid_anchor"] = hybrid_anchor_heuristic(graph, v)
            results["insertion_anchor"] = insertion_anchor_heuristic(graph, v)
            results["probablistic_anchor"] = probabilistic_anchor_heuristic(graph, v)
            
            results["hamiltonian_improved"] = hamiltonian_cycle_heuristic_improved(
                graph,
                start=v,
                anchors=anchors,
                max_depth=-1,
                early_exit=False
            )
            '''


            for method, (_, weight) in results.items():
                all_weights[method].append(weight)

    # Compute and print average weights
    print("\n--- AVERAGE RESULTS ACROSS MULTI-ANCHOR TESTS ---")
    for method, weights in all_weights.items():
        avg = sum(weights) / len(weights)
        print(f"{method}: Average Weight = {avg:.2f} over {len(weights)} runs")


def metric_algorithms_test(num_graphs=3, num_vertices=15, weight_range=(1, 100), seed_base=200):
    all_weights = defaultdict(list)

    for g in range(num_graphs):
        # print(f"\n--- Metric Graph {g+1} ---")
        graph = generate_complete_graph(num_vertices, weight_range=weight_range, seed=seed_base + g, metric=True)
        # print_graph(graph)

        for v in range(num_vertices):
            anchors = [v, (v + 5) % num_vertices]  
            # example: 3 anchors spread apart
            # anchors = [v]
            results = {}

            # Run optimal
            # results["optimal_held_karp"] = find_optimal_cycle_held_karp(graph, v)
            
            # Run greedy
            # work on this VVV
            ranking_info = rank_vertices_by_weight(graph)
            highest_v = ranking_info["highest_weight_vertex"]
            # print(f"Vertex weights and shit: ")
            # print(ranking_info["sorted_vertices_and_weights"])


            results["nearest_neighbor"] = greedy_algorithm(graph, v)
            results["single_anchor"] = low_anchor_heuristic(graph, v)
            results["best_anchor"] = best_anchor_heuristic(graph)
            results["kruskals_tsp"] = greedy_edge_tsp_v3(graph)
            results["high_anchor_heuristic"] = low_anchor_heuristic(graph, highest_v)
            results["christofides_algorithm"] = christofides_algorithm(graph)

            for method, (_, weight) in results.items():
                all_weights[method].append(weight)

    # Compute and print average weights
    print("\n--- AVERAGE RESULTS ACROSS MULTI-ANCHOR TESTS ---")
    for method, weights in all_weights.items():
        avg = sum(weights) / len(weights)
        print(f"{method}: Average Weight = {avg:.2f} over {len(weights)} runs")

def main():
    """Legacy main function - use cli/cli.py for full functionality."""
    print("🔄 TSP Heuristic Algorithms - Legacy Interface")
    print("=" * 50)
    print("For the full-featured CLI with advanced options, use:")
    print("  python cli/cli.py --help")
    print()
    print("Available commands:")
    print("  • Run single algorithm: python cli/cli.py run <algorithm>")
    print("  • Compare algorithms:   python cli/cli.py compare <alg1> <alg2>")
    print("  • Generate graphs:      python cli/cli.py generate")
    print("  • Interactive visualization: python cli/cli.py visualize")
    print()

    # Run legacy tests
    choice = input("Run legacy benchmark tests? (y/N): ").strip().lower()
    if choice == 'y':
        print("\n🏃 Running legacy benchmark tests...")
        improved_base_heuristic_test(num_graphs=3, num_vertices=20, weight_range=(1, 100), seed_base=42)
    else:
        print("💡 Use the new CLI for better functionality!")

def show_cli_usage():
    """Display usage information for the new CLI."""
    print("""
🎯 TSP Heuristic Algorithms CLI Usage
=====================================

The new CLI (cli/cli.py) provides comprehensive functionality:

BASIC USAGE:
  # Run a single algorithm
  python cli/cli.py run christofides --vertices 10 --metric

  # Compare multiple algorithms
  python cli/cli.py compare nearest_neighbor christofides hamiltonian --vertices 15 --runs 3

  # Generate and save graphs
  python cli/cli.py generate --vertices 20 --output graph.json --metric

  # Interactive visualization
  python cli/cli.py visualize --vertices 6 --metric

AVAILABLE ALGORITHMS:
  Established: christofides, nearest_neighbor, prims, kruskals, kruskals_family
  Experimental: hamiltonian, pressure_field

OPTIONS:
  --vertices N        Number of vertices (default: varies by command)
  --min-weight N      Minimum edge weight (default: 1)
  --max-weight N      Maximum edge weight (default: 100)
  --metric            Generate metric graphs (satisfy triangle inequality)
  --seed N            Random seed for reproducibility
  --output FILE       Save results to file
  --show-stats        Display detailed statistics
  --runs N            Number of runs for comparison (default: 1)

EXAMPLES:
  # Quick test with default settings
  python cli/cli.py run nearest_neighbor

  # Advanced comparison with custom settings
  python cli/cli.py compare christofides hamiltonian --vertices 25 --metric --runs 5 --output results.json

  # Generate large graph for testing
  python cli/cli.py generate --vertices 50 --metric --seed 123 --output large_graph.json

For more help: python cli/cli.py --help
    """)

# Legacy functions remain for backward compatibility
# (base_heuristic_test, improved_base_heuristic_test, metric_algorithms_test)

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--show-cli":
        show_cli_usage()
    else:
        main()
