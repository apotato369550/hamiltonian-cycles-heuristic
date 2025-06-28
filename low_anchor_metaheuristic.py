import random
from collections import defaultdict
from typing import List, Tuple, Dict
from graph_generator import generate_complete_graph

def construct_greedy_cycle(graph, start, anchor1, anchor2):
    """Constructs a greedy cycle given a start vertex and 2 anchor points. Ensures Hamiltonian cycle."""
    vertices_count = len(graph)
    
    # Initialize with only the start point in the visited set
    visited = set([start])
    path = [start]
    current_vertex = start
    total_weight = 0
    
    # First, we need to visit anchor1
    if anchor1 not in visited:
        path.append(anchor1)
        visited.add(anchor1)
        total_weight += graph[current_vertex][anchor1]
        current_vertex = anchor1
    
    # Visit all remaining vertices except anchor2
    while len(visited) < vertices_count - 1:  # -1 because we'll add anchor2 last
        next_vertex = None
        lowest_weight = float("inf")
        
        for i in range(vertices_count):
            if i not in visited and i != anchor2 and graph[current_vertex][i] < lowest_weight:
                next_vertex = i
                lowest_weight = graph[current_vertex][i]
        
        # If we can't find a next vertex, break the loop
        if next_vertex is None:
            break
            
        visited.add(next_vertex)
        path.append(next_vertex)
        total_weight += lowest_weight
        current_vertex = next_vertex
    
    # Now add anchor2 if it's not already visited
    if anchor2 not in visited:
        path.append(anchor2)
        visited.add(anchor2)
        total_weight += graph[current_vertex][anchor2]
        current_vertex = anchor2
    
    # Complete the cycle by returning to start
    total_weight += graph[current_vertex][start]
    path.append(start)
    
    return path, total_weight

def low_anchor_heuristic(graph, vertex):
    """Your original anchor heuristic implementation"""
    def find_two_lowest_indices(values, vertex):
        if len(values) < 2:
            raise ValueError("List must contain at least two elements.")
        sorted_indices = sorted((i for i in range(len(values)) if i != vertex), key=lambda i: values[i])
        return sorted_indices[:2]
    
    anchors = find_two_lowest_indices(graph[vertex], vertex)
    
    cycle_1, lowest_weight_1 = construct_greedy_cycle(graph, vertex, anchors[0], anchors[1])
    cycle_2, lowest_weight_2 = construct_greedy_cycle(graph, vertex, anchors[1], anchors[0])

    if lowest_weight_1 < lowest_weight_2:
        return cycle_1, lowest_weight_1 
    return cycle_2, lowest_weight_2

def calculate_vertex_total_weight(graph, vertex):
    """Calculate the total weight of all edges from a vertex"""
    return sum(graph[vertex])

def rank_vertices_by_weight(graph):
    """
    Rank vertices by their total outgoing edge weight.
    Returns dictionary with rankings and sorted vertex lists.
    """
    vertices_count = len(graph)
    vertex_weights = [(i, calculate_vertex_total_weight(graph, i)) for i in range(vertices_count)]
    
    # Sort by weight (ascending)
    sorted_by_weight = sorted(vertex_weights, key=lambda x: x[1])
    
    return {
        'lowest_weight_vertex': sorted_by_weight[0][0],
        'highest_weight_vertex': sorted_by_weight[-1][0],
        'sorted_vertices': [vertex for vertex, weight in sorted_by_weight],
        'vertex_weights': dict(vertex_weights),
        'sorted_vertices_and_weights': sorted_by_weight
    }

def starting_vertex_metaheuristic(graph, num_random_trials=5, verbose=False):
    """
    Test the anchor heuristic with different starting vertex selection strategies:
    1. Lowest total weight vertex
    2. Highest total weight vertex  
    3. Random vertex selection (multiple trials)
    4. All vertices (for comparison)
    
    Returns results dictionary with best cycles and analysis.
    """
    vertices_count = len(graph)
    ranking_info = rank_vertices_by_weight(graph)
    
    results = {
        'lowest_weight_start': None,
        'highest_weight_start': None,
        'random_starts': [],
        'all_vertices': [],
        'best_overall': None,
        'ranking_info': ranking_info
    }
    
    if verbose:
        print(f"Vertex weight rankings:")
        for vertex, weight in ranking_info['vertex_weights'].items():
            print(f"  Vertex {vertex}: Total weight = {weight}")
        print()
    
    # Test 1: Lowest weight vertex as starting point
    lowest_vertex = ranking_info['lowest_weight_vertex']
    cycle, weight = low_anchor_heuristic(graph, lowest_vertex)
    results['lowest_weight_start'] = {
        'vertex': lowest_vertex,
        'cycle': cycle,
        'weight': weight,
        'vertex_total_weight': ranking_info['vertex_weights'][lowest_vertex]
    }
    
    if verbose:
        print(f"Lowest weight start (vertex {lowest_vertex}): Cycle weight = {weight}")
    
    # Test 2: Highest weight vertex as starting point
    highest_vertex = ranking_info['highest_weight_vertex']
    cycle, weight = low_anchor_heuristic(graph, highest_vertex)
    results['highest_weight_start'] = {
        'vertex': highest_vertex,
        'cycle': cycle,
        'weight': weight,
        'vertex_total_weight': ranking_info['vertex_weights'][highest_vertex]
    }
    
    if verbose:
        print(f"Highest weight start (vertex {highest_vertex}): Cycle weight = {weight}")
    
    # Test 3: Random vertex selection
    random_vertices = random.sample(range(vertices_count), min(num_random_trials, vertices_count))
    for vertex in random_vertices:
        cycle, weight = low_anchor_heuristic(graph, vertex)
        results['random_starts'].append({
            'vertex': vertex,
            'cycle': cycle,
            'weight': weight,
            'vertex_total_weight': ranking_info['vertex_weights'][vertex]
        })
        
        if verbose:
            print(f"Random start (vertex {vertex}): Cycle weight = {weight}")
    
    # Test 4: All vertices for comprehensive comparison
    for vertex in range(vertices_count):
        cycle, weight = low_anchor_heuristic(graph, vertex)
        results['all_vertices'].append({
            'vertex': vertex,
            'cycle': cycle,
            'weight': weight,
            'vertex_total_weight': ranking_info['vertex_weights'][vertex]
        })
    
    # Find best overall result
    all_results = ([results['lowest_weight_start'], results['highest_weight_start']] + 
                  results['random_starts'] + results['all_vertices'])
    
    results['best_overall'] = min(all_results, key=lambda x: x['weight'])
    
    if verbose:
        print(f"\nBest overall result: Vertex {results['best_overall']['vertex']} "
              f"with cycle weight {results['best_overall']['weight']}")
    
    return results

def analyze_starting_vertex_impact(results, verbose=True):
    """
    Analyze the results to determine if starting vertex selection matters.
    """
    analysis = {}
    
    # Get weights from all vertex tests
    all_weights = [result['weight'] for result in results['all_vertices']]
    vertex_weights = [result['vertex_total_weight'] for result in results['all_vertices']]
    
    analysis['cycle_weight_stats'] = {
        'min': min(all_weights),
        'max': max(all_weights),
        'range': max(all_weights) - min(all_weights),
        'avg': sum(all_weights) / len(all_weights),
        'variance': sum((w - sum(all_weights)/len(all_weights))**2 for w in all_weights) / len(all_weights)
    }
    
    # Compare strategy performance
    lowest_start_weight = results['lowest_weight_start']['weight']
    highest_start_weight = results['highest_weight_start']['weight']
    random_weights = [r['weight'] for r in results['random_starts']]
    
    analysis['strategy_comparison'] = {
        'lowest_weight_start': lowest_start_weight,
        'highest_weight_start': highest_start_weight,
        'random_avg': sum(random_weights) / len(random_weights) if random_weights else 0,
        'random_best': min(random_weights) if random_weights else float('inf'),
        'overall_best': results['best_overall']['weight']
    }
    
    # Determine if starting vertex matters
    weight_range = analysis['cycle_weight_stats']['range']
    analysis['starting_vertex_matters'] = weight_range > 0
    analysis['impact_magnitude'] = weight_range / analysis['cycle_weight_stats']['min'] * 100  # Percentage
    
    if verbose:
        print("\n" + "="*60)
        print("STARTING VERTEX IMPACT ANALYSIS")
        print("="*60)
        print(f"Cycle weight range: {weight_range:.2f}")
        print(f"Best cycle weight: {analysis['cycle_weight_stats']['min']:.2f}")
        print(f"Worst cycle weight: {analysis['cycle_weight_stats']['max']:.2f}")
        print(f"Average cycle weight: {analysis['cycle_weight_stats']['avg']:.2f}")
        print(f"Impact magnitude: {analysis['impact_magnitude']:.2f}%")
        print()
        print("Strategy Performance:")
        print(f"  Lowest weight vertex start: {lowest_start_weight:.2f}")
        print(f"  Highest weight vertex start: {highest_start_weight:.2f}")
        print(f"  Random selection average: {analysis['strategy_comparison']['random_avg']:.2f}")
        print(f"  Random selection best: {analysis['strategy_comparison']['random_best']:.2f}")
        print(f"  Overall best: {analysis['strategy_comparison']['overall_best']:.2f}")
        print()
        if analysis['starting_vertex_matters']:
            print("✓ CONCLUSION: Starting vertex selection DOES matter!")
            print(f"  The choice of starting vertex can impact cycle weight by up to {analysis['impact_magnitude']:.1f}%")
        else:
            print("✗ CONCLUSION: Starting vertex selection does NOT matter.")
            print("  All starting vertices produce the same cycle weight.")
    
    return analysis

def starting_vertex_selection_test(num_graphs=3, num_vertices=15, weight_range=(1, 100), seed_base=300):
    """
    Test the starting vertex selection metaheuristic across multiple graphs.
    
    This function follows the same format as multi_anchor_heuristic_test() in main.py
    to test whether starting vertex selection matters for the anchor heuristic.
    
    Args:
        num_graphs (int): Number of graphs to generate and test.
        num_vertices (int): Number of vertices per graph.
        weight_range (tuple): Range of edge weights (min, max).
        seed_base (int): Base seed for reproducible graph generation.
    """
    all_weights = defaultdict(list)
    all_impact_data = []
    
    print(f"\n{'='*80}")
    print(f"STARTING VERTEX SELECTION METAHEURISTIC TEST")
    print(f"Testing {num_graphs} graphs with {num_vertices} vertices each")
    print(f"Weight range: {weight_range}, Seed base: {seed_base}")
    print(f"{'='*80}")

    for g in range(num_graphs):
        print(f"\n--- Starting Vertex Test Graph {g+1} ---")
        graph = generate_complete_graph(num_vertices, weight_range=weight_range, seed=seed_base + g)
        
        # Print graph info (first few rows for brevity)
        print(f"Graph {g+1} (showing first 3 rows):")
        for i in range(min(3, len(graph))):
            row_str = str(graph[i][:min(8, len(graph[i]))])
            if len(graph[i]) > 8:
                row_str = row_str[:-1] + ", ...]"
            print(f"  Row {i}: {row_str}")
        
        # Run the starting vertex metaheuristic
        results = starting_vertex_metaheuristic(graph, num_random_trials=5, verbose=False)
        
        # Analyze the impact for this graph
        analysis = analyze_starting_vertex_impact(results, verbose=False)
        all_impact_data.append(analysis)
        
        # Collect results for averaging
        all_weights['lowest_weight_start'].append(results['lowest_weight_start']['weight'])
        all_weights['highest_weight_start'].append(results['highest_weight_start']['weight'])
        
        # Average random results for this graph
        random_avg = sum(r['weight'] for r in results['random_starts']) / len(results['random_starts'])
        all_weights['random_avg'].append(random_avg)
        all_weights['random_best'].append(min(r['weight'] for r in results['random_starts']))
        
        # Best overall for this graph
        all_weights['best_overall'].append(results['best_overall']['weight'])
        
        # Store all individual vertex results
        for vertex_result in results['all_vertices']:
            all_weights['all_vertices_individual'].append(vertex_result['weight'])
        
        # Print summary for this graph
        print(f"  Graph {g+1} Results:")
        print(f"    Lowest weight start: {results['lowest_weight_start']['weight']:.2f} (vertex {results['lowest_weight_start']['vertex']})")
        print(f"    Highest weight start: {results['highest_weight_start']['weight']:.2f} (vertex {results['highest_weight_start']['vertex']})")
        print(f"    Random average: {random_avg:.2f}")
        print(f"    Best overall: {results['best_overall']['weight']:.2f} (vertex {results['best_overall']['vertex']})")
        print(f"    Impact range: {analysis['cycle_weight_stats']['range']:.2f} ({analysis['impact_magnitude']:.1f}%)")

    # Compute and print comprehensive average results
    print(f"\n{'='*80}")
    print("COMPREHENSIVE AVERAGE RESULTS ACROSS ALL STARTING VERTEX TESTS")
    print(f"{'='*80}")
    
    for method, weights in all_weights.items():
        if method == 'all_vertices_individual':
            continue  # Skip individual results for summary
        avg = sum(weights) / len(weights)
        std_dev = (sum((w - avg) ** 2 for w in weights) / len(weights)) ** 0.5
        print(f"{method:20}: Average Weight = {avg:7.2f} +- {std_dev:5.2f} over {len(weights):3d} runs")
    
    # Analyze overall impact across all graphs
    print(f"\n{'='*80}")
    print("STARTING VERTEX IMPACT ANALYSIS SUMMARY")
    print(f"{'='*80}")
    
    impact_ranges = [analysis['cycle_weight_stats']['range'] for analysis in all_impact_data]
    impact_percentages = [analysis['impact_magnitude'] for analysis in all_impact_data]
    
    avg_range = sum(impact_ranges) / len(impact_ranges)
    avg_percentage = sum(impact_percentages) / len(impact_percentages)
    
    graphs_with_impact = sum(1 for analysis in all_impact_data if analysis['starting_vertex_matters'])
    
    print(f"Graphs showing starting vertex impact: {graphs_with_impact}/{num_graphs} ({graphs_with_impact/num_graphs*100:.1f}%)")
    print(f"Average impact range: {avg_range:.2f}")
    print(f"Average impact percentage: {avg_percentage:.2f}%")
    print(f"Maximum impact seen: {max(impact_percentages):.2f}%")
    print(f"Minimum impact seen: {min(impact_percentages):.2f}%")
    
    # Determine overall conclusion
    if graphs_with_impact > 0:
        print(f"\n OVERALL CONCLUSION: Starting vertex selection DOES matter!")
        print(f"  - {graphs_with_impact} out of {num_graphs} graphs showed variation in cycle weights")
        print(f"  - Average impact magnitude: {avg_percentage:.1f}%")
        print(f"  - This suggests that starting vertex selection can be used to improve the anchor heuristic")
    else:
        print(f"\n OVERALL CONCLUSION: Starting vertex selection does NOT matter.")
        print(f"  - All graphs produced identical results regardless of starting vertex")
    
    return all_weights, all_impact_data

def print_starting_vertex_weights(all_weights):
    """
    Print the all_weights dictionary in a clean, readable format.
    Similar to how results are printed in main.py
    """
    print(f"\n{'='*60}")
    print("DETAILED WEIGHT RESULTS")
    print(f"{'='*60}")
    
    for method, weights in all_weights.items():
        if method == 'all_vertices_individual':
            print(f"\n{method}:")
            print(f"  Total individual tests: {len(weights)}")
            print(f"  Min weight: {min(weights):.2f}")
            print(f"  Max weight: {max(weights):.2f}")
            print(f"  Average: {sum(weights)/len(weights):.2f}")
            continue
            
        print(f"\n{method}:")
        print(f"  Values: {[f'{w:.2f}' for w in weights]}")
        print(f"  Count: {len(weights)}")
        print(f"  Average: {sum(weights)/len(weights):.2f}")
        print(f"  Min: {min(weights):.2f}")
        print(f"  Max: {max(weights):.2f}")

def main():
    """Main function to run the starting vertex selection tests"""
    print("Starting Vertex Selection Metaheuristic Testing")
    
    # Test with different configurations
    print("\n" + "="*100)
    print("TEST 1: Small graphs for detailed analysis")
    all_weights_small, impact_data_small = starting_vertex_selection_test(
        num_graphs=3, 
        num_vertices=8, 
        weight_range=(1, 50), 
        seed_base=42069
    )
    print_starting_vertex_weights(all_weights_small)
    
    print("\n" + "="*100)
    print("TEST 2: Medium graphs for statistical significance")
    all_weights_medium, impact_data_medium = starting_vertex_selection_test(
        num_graphs=5, 
        num_vertices=15, 
        weight_range=(1, 100), 
        seed_base=42069
    )
    print_starting_vertex_weights(all_weights_medium)
    
    print("\n" + "="*100)
    print("TEST 3: Larger graphs for real-world applicability")
    all_weights_large, impact_data_large = starting_vertex_selection_test(
        num_graphs=3, 
        num_vertices=25, 
        weight_range=(1, 200), 
        seed_base=42069
    )
    print_starting_vertex_weights(all_weights_large)

    print("\n" + "="*100)
    print("TEST 3: GIANT graphs for fun <3")
    all_weights_extra_large, impact_data_extra_large = starting_vertex_selection_test(
        num_graphs=3, 
        num_vertices=100, 
        weight_range=(1, 400), 
        seed_base=42069
    )
    print_starting_vertex_weights(all_weights_extra_large)

if __name__ == "__main__":
    main()