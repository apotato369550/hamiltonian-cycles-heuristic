from utils import generate_complete_graph, print_graph, run_benchmark

def display_menu():
    print("Menu goes here.")

def main():
    print("--- WELCOME TO HAMILTONIAN CYCLES HEURISTIC ---")
    # display_menu()
    graph = generate_complete_graph(6, (1, 16), replace=True, strict=True, seed=42069)
    print_graph(graph)
    print(run_benchmark(graph, 0))

if __name__ == "__main__":
    main()