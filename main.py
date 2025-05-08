from utils import generate_complete_graph, print_graph

def display_menu():
    print("Menu goes here.")

def main():
    print("--- WELCOME TO HAMILTONIAN CYCLES HEURISTIC ---")
    # display_menu()
    graph = generate_complete_graph(6, (1, 16), replace=True, strict=True)
    print_graph(graph)

if __name__ == "__main__":
    main()