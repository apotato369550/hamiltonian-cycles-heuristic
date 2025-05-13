from collections import defaultdict
from utils import generate_complete_graph, print_graph, run_benchmark

def main():
    num_graphs = 3
    num_vertices = 9
    weight_range = (1, 100)
    seed_base = 100  # We'll vary this per graph for uniqueness

    all_weights = defaultdict(list)

    for g in range(num_graphs):
        print(f"\n--- Graph {g+1} ---")
        graph = generate_complete_graph(num_vertices, weight_range=weight_range, seed=seed_base + g)
        print_graph(graph)

        for v in range(num_vertices):
            results = run_benchmark(graph, v)
            for method, (_, weight) in results.items():
                all_weights[method].append(weight)

    # Compute and print average weights
    print("\n--- AVERAGE RESULTS ACROSS ALL TESTS ---")
    for method, weights in all_weights.items():
        avg = sum(weights) / len(weights)
        print(f"{method}: Average Weight = {avg:.2f} over {len(weights)} runs")

if __name__ == "__main__":
    main()
