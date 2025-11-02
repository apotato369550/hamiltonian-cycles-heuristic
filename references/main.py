from collections import defaultdict
from graph_generator import generate_complete_graph
from utils import print_graph, run_base_benchmark, greedy_algorithm, find_optimal_cycle_held_karp
from low_anchor_heuristic import low_anchor_heuristic, best_anchor_heuristic
from hamiltonian import hamiltonian_cycle_heuristic, best_multi_anchor_heuristic
from hamiltonian_improved import hamiltonian_cycle_heuristic_improved
from bidirectional_greedy import bidirectional_nearest_neighbor_tsp
from kruskals_greedy_family import greedy_edge_tsp, greedy_edge_tsp_v2, greedy_edge_tsp_v3
from anchor_heuristic_family import adaptive_anchor_heuristic, multi_anchor_heuristic, smart_anchor_heuristic, hybrid_anchor_heuristic, insertion_anchor_heuristic, probabilistic_anchor_heuristic
from pressure_field_heuristic import pressure_field_heuristic
from christofides_algorithm import christofides_algorithm
from prims_tsp import prims_tsp, best_prims_tsp, prims_anchoring_tsp, best_prims_anchoring_tsp
from github_heuristic import heuristic_path
from nearest_neighbor import nearest_neighbor_tsp, best_nearest_neighbor

from low_anchor_metaheuristic import rank_vertices_by_weight

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
    # base_heuristic_test()
    improved_base_heuristic_test(num_graphs=5, num_vertices=50, weight_range=(1, 400), seed_base=696969)
    # metric_algorithms_test(num_graphs=5, num_vertices=30, weight_range=(1, 400), seed_base=696969)



# 1 graph, 1 anchor, 2 permutations
# 1 graph, 2 anchors, 4 permutations
# 1 graph, 3 anchors, but only 2 permutations when there should be at least 
# problem is: hamiltonian.py is not doing permutations properly

if __name__ == "__main__":
    main()
