from collections import defaultdict
from utils import generate_complete_graph, print_graph, run_base_benchmark, greedy_algorithm, find_optimal_cycle_held_karp, low_anchor_heuristic
from hamiltonian import hamiltonian_cycle_heuristic
from hamiltonian_improved import hamiltonian_cycle_heuristic_improved
from bidirectional_greedy import bidirectional_nearest_neighbor_tsp, bidirectional_tsp_comprehensive

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

def multi_anchor_heuristic_test(num_graphs=3, num_vertices=15, weight_range=(1, 100), seed_base=200):
    """
    Benchmarks the multi-anchor heuristic against greedy and Held-Karp (optimal).

    Runs the benchmark on multiple graphs, each with multiple starting vertices.

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
        print_graph(graph)

        for v in range(num_vertices):
            anchors = [v, (v + 5) % num_vertices, (v + 10) % num_vertices, (v + 15) % num_vertices]  
            # example: 3 anchors spread apart
            # anchors = [v]
            results = {}

            # Run optimal
            # results["optimal_held_karp"] = find_optimal_cycle_held_karp(graph, v)
            
            # Run greedy
            results["greedy"] = greedy_algorithm(graph, v)

            # Run your multi-anchor heuristic
            results["multi_anchor"] = hamiltonian_cycle_heuristic(
                graph,
                start=v,
                anchors=anchors,
                max_depth=-1,
                early_exit=False
            )
            results["hamiltonian_improved"] = hamiltonian_cycle_heuristic_improved(
                graph,
                start=v,
                anchors=anchors,
                max_depth=-1,
                early_exit=False
            )
            results["single_anchor"] = low_anchor_heuristic(graph, v)
            results["bidirectional_greedy"] = bidirectional_nearest_neighbor_tsp(graph, v)
            results["bidirectional_greedy_comprehensive"] = bidirectional_tsp_comprehensive(graph, v)

            for method, (_, weight) in results.items():
                all_weights[method].append(weight)

    # Compute and print average weights
    print("\n--- AVERAGE RESULTS ACROSS MULTI-ANCHOR TESTS ---")
    for method, weights in all_weights.items():
        avg = sum(weights) / len(weights)
        print(f"{method}: Average Weight = {avg:.2f} over {len(weights)} runs")

def main():
    # base_heuristic_test()
    multi_anchor_heuristic_test(num_graphs=3, num_vertices=20, weight_range=(1, 100), seed_base=42069)


# 1 graph, 1 anchor, 2 permutations
# 1 graph, 2 anchors, 4 permutations
# 1 graph, 3 anchors, but only 2 permutations when there should be at least 
# problem is: hamiltonian.py is not doing permutations properly

if __name__ == "__main__":
    main()
