from collections import defaultdict
from utils import generate_complete_graph, print_graph, run_base_benchmark

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

def multi_anchor_heuristic_test():
    # Placeholder for multi-anchor heuristic test
    # test optimal (held-karp) vs anchor heuristic vs regular greedy
    # test on graphs with 20 vertices, let's try 5 of them
    # benchmark the total weights and stuff

    pass

def main():
    base_heuristic_test()
    multi_anchor_heuristic_test()



if __name__ == "__main__":
    main()
