import random
import statistics
from collections import defaultdict
from typing import List, Tuple, Dict, Optional
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
    """Core anchor heuristic implementation"""
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

def calculate_vertex_statistics(graph, vertex):
    """
    Calculate comprehensive statistics for a vertex's edges.
    
    Args:
        graph: Adjacency matrix representation of the graph
        vertex: The vertex to analyze
        
    Returns:
        Dictionary containing total weight, variance, std deviation, min, max, and mean
    """
    edges = [graph[vertex][i] for i in range(len(graph)) if i != vertex]
    
    if not edges:
        return {
            'total_weight': 0,
            'variance': 0,
            'std_deviation': 0,
            'mean': 0,
            'min_edge': 0,
            'max_edge': 0,
            'edge_count': 0
        }
    
    total_weight = sum(edges)
    mean_weight = total_weight / len(edges)
    variance = statistics.variance(edges) if len(edges) > 1 else 0
    std_deviation = statistics.stdev(edges) if len(edges) > 1 else 0
    
    return {
        'total_weight': total_weight,
        'variance': variance,
        'std_deviation': std_deviation,
        'mean': mean_weight,
        'min_edge': min(edges),
        'max_edge': max(edges),
        'edge_count': len(edges)
    }

def rank_vertices_by_weight_and_variance(graph):
    """
    Rank vertices by total weight and variance using different combination strategies.
    
    Args:
        graph: Adjacency matrix representation of the graph
        
    Returns:
        Dictionary containing various rankings and vertex statistics
    """
    vertices_count = len(graph)
    vertex_stats = {}
    
    # Calculate statistics for all vertices
    for vertex in range(vertices_count):
        vertex_stats[vertex] = calculate_vertex_statistics(graph, vertex)
    
    # Create different ranking strategies
    rankings = {}
    
    # Strategy 1: Pure weight ranking (highest first)
    weight_ranked = sorted(vertex_stats.items(), key=lambda x: x[1]['total_weight'], reverse=True)
    rankings['by_weight'] = [vertex for vertex, stats in weight_ranked]
    
    # Strategy 2: Pure variance ranking (highest first)
    variance_ranked = sorted(vertex_stats.items(), key=lambda x: x[1]['variance'], reverse=True)
    rankings['by_variance'] = [vertex for vertex, stats in variance_ranked]
    
    # Strategy 3: Weighted sum (weight + variance)
    weighted_sum_ranked = sorted(vertex_stats.items(), 
                                key=lambda x: x[1]['total_weight'] + x[1]['variance'], 
                                reverse=True)
    rankings['by_weight_plus_variance'] = [vertex for vertex, stats in weighted_sum_ranked]
    
    # Strategy 4: Normalized weighted sum
    max_weight = max(stats['total_weight'] for stats in vertex_stats.values())
    max_variance = max(stats['variance'] for stats in vertex_stats.values())
    
    if max_weight > 0 and max_variance > 0:
        normalized_scores = {}
        for vertex, stats in vertex_stats.items():
            normalized_weight = stats['total_weight'] / max_weight
            normalized_variance = stats['variance'] / max_variance
            normalized_scores[vertex] = normalized_weight + normalized_variance
        
        normalized_ranked = sorted(normalized_scores.items(), key=lambda x: x[1], reverse=True)
        rankings['by_normalized_sum'] = [vertex for vertex, score in normalized_ranked]
    else:
        rankings['by_normalized_sum'] = rankings['by_weight_plus_variance']
    
    # Strategy 5: Product of weight and variance
    product_ranked = sorted(vertex_stats.items(), 
                           key=lambda x: x[1]['total_weight'] * x[1]['variance'], 
                           reverse=True)
    rankings['by_weight_times_variance'] = [vertex for vertex, stats in product_ranked]
    
    # Strategy 6: Weight-to-variance ratio (high weight, high variance preference)
    ratio_scores = {}
    for vertex, stats in vertex_stats.items():
        if stats['variance'] > 0:
            # Higher weight and higher variance both contribute positively
            ratio_scores[vertex] = stats['total_weight'] * (1 + stats['variance'] / stats['mean'])
        else:
            ratio_scores[vertex] = stats['total_weight']
    
    ratio_ranked = sorted(ratio_scores.items(), key=lambda x: x[1], reverse=True)
    rankings['by_weight_variance_ratio'] = [vertex for vertex, score in ratio_ranked]
    
    return {
        'vertex_statistics': vertex_stats,
        'rankings': rankings,
        'best_vertices': {
            'highest_weight': rankings['by_weight'][0],
            'highest_variance': rankings['by_variance'][0],
            'best_weight_plus_variance': rankings['by_weight_plus_variance'][0],
            'best_normalized': rankings['by_normalized_sum'][0],
            'best_product': rankings['by_weight_times_variance'][0],
            'best_ratio': rankings['by_weight_variance_ratio'][0]
        }
    }

def weight_variance_metaheuristic(graph, strategy='weight_plus_variance', num_random_trials=5, verbose=False):
    """
    Test the anchor heuristic with weight-variance based starting vertex selection.
    
    Args:
        graph: Adjacency matrix representation of the graph
        strategy: Strategy for combining weight and variance ('weight_plus_variance', 
                 'normalized_sum', 'product', 'ratio', 'weight_only', 'variance_only')
        num_random_trials: Number of random vertex trials for comparison
        verbose: Whether to print detailed output
        
    Returns:
        Dictionary containing results and analysis
    """
    ranking_info = rank_vertices_by_weight_and_variance(graph)
    strategy_map = {
        'weight_plus_variance': 'best_weight_plus_variance',
        'normalized_sum': 'best_normalized',
        'product': 'best_product',
        'ratio': 'best_ratio',
        'weight_only': 'highest_weight',
        'variance_only': 'highest_variance'
    }
    
    if strategy not in strategy_map:
        raise ValueError(f"Unknown strategy: {strategy}. Available: {list(strategy_map.keys())}")
    
    results = {
        'strategy_used': strategy,
        'selected_vertex': None,
        'strategy_result': None,
        'random_trials': [],
        'all_vertices': [],
        'best_overall': None,
        'ranking_info': ranking_info
    }
    
    # Test the selected strategy
    selected_vertex = ranking_info['best_vertices'][strategy_map[strategy]]
    cycle, weight = low_anchor_heuristic(graph, selected_vertex)
    vertex_stats = ranking_info['vertex_statistics'][selected_vertex]
    
    results['selected_vertex'] = selected_vertex
    results['strategy_result'] = {
        'vertex': selected_vertex,
        'cycle': cycle,
        'weight': weight,
        'vertex_stats': vertex_stats
    }
    
    if verbose:
        print(f"Strategy '{strategy}' selected vertex {selected_vertex}")
        print(f"  Total weight: {vertex_stats['total_weight']:.2f}")
        print(f"  Variance: {vertex_stats['variance']:.2f}")
        print(f"  Cycle weight: {weight:.2f}")
    
    # Random trials for comparison
    vertices_count = len(graph)
    random_vertices = random.sample(range(vertices_count), min(num_random_trials, vertices_count))
    
    for vertex in random_vertices:
        cycle, weight = low_anchor_heuristic(graph, vertex)
        results['random_trials'].append({
            'vertex': vertex,
            'cycle': cycle,
            'weight': weight,
            'vertex_stats': ranking_info['vertex_statistics'][vertex]
        })
        
        if verbose:
            print(f"Random vertex {vertex}: Cycle weight = {weight:.2f}")
    
    # Test all vertices for comprehensive comparison
    for vertex in range(vertices_count):
        cycle, weight = low_anchor_heuristic(graph, vertex)
        results['all_vertices'].append({
            'vertex': vertex,
            'cycle': cycle,
            'weight': weight,
            'vertex_stats': ranking_info['vertex_statistics'][vertex]
        })
    
    # Find best overall result
    all_results = [results['strategy_result']] + results['random_trials'] + results['all_vertices']
    results['best_overall'] = min(all_results, key=lambda x: x['weight'])
    
    if verbose:
        print(f"Best overall: Vertex {results['best_overall']['vertex']} with weight {results['best_overall']['weight']:.2f}")
    
    return results

def analyze_weight_variance_impact(results, verbose=True):
    """
    Analyze the effectiveness of weight-variance based vertex selection.
    
    Args:
        results: Results from weight_variance_metaheuristic
        verbose: Whether to print detailed analysis
        
    Returns:
        Dictionary containing analysis metrics
    """
    analysis = {}
    
    # Extract performance data
    strategy_weight = results['strategy_result']['weight']
    random_weights = [r['weight'] for r in results['random_trials']]
    all_weights = [r['weight'] for r in results['all_vertices']]
    
    # Calculate statistics
    analysis['performance_stats'] = {
        'strategy_weight': strategy_weight,
        'random_avg': sum(random_weights) / len(random_weights) if random_weights else 0,
        'random_best': min(random_weights) if random_weights else float('inf'),
        'random_worst': max(random_weights) if random_weights else 0,
        'all_vertices_avg': sum(all_weights) / len(all_weights),
        'all_vertices_best': min(all_weights),
        'all_vertices_worst': max(all_weights),
        'best_overall': results['best_overall']['weight']
    }
    
    # Performance comparison
    random_avg = analysis['performance_stats']['random_avg']
    all_avg = analysis['performance_stats']['all_vertices_avg']
    
    analysis['strategy_effectiveness'] = {
        'vs_random_improvement': ((random_avg - strategy_weight) / random_avg * 100) if random_avg > 0 else 0,
        'vs_average_improvement': ((all_avg - strategy_weight) / all_avg * 100) if all_avg > 0 else 0,
        'is_optimal': strategy_weight == analysis['performance_stats']['best_overall'],
        'rank_among_all': sorted(all_weights).index(strategy_weight) + 1,
        'percentile': (1 - (sorted(all_weights).index(strategy_weight) / len(all_weights))) * 100
    }
    
    # Vertex characteristics analysis
    selected_vertex = results['selected_vertex']
    selected_stats = results['strategy_result']['vertex_stats']
    
    analysis['vertex_characteristics'] = {
        'selected_vertex': selected_vertex,
        'total_weight': selected_stats['total_weight'],
        'variance': selected_stats['variance'],
        'weight_rank': results['ranking_info']['rankings']['by_weight'].index(selected_vertex) + 1,
        'variance_rank': results['ranking_info']['rankings']['by_variance'].index(selected_vertex) + 1
    }
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"WEIGHT-VARIANCE METAHEURISTIC ANALYSIS")
        print(f"Strategy: {results['strategy_used']}")
        print(f"{'='*70}")
        
        print(f"Selected Vertex: {selected_vertex}")
        print(f"  Total Weight: {selected_stats['total_weight']:.2f} (rank #{analysis['vertex_characteristics']['weight_rank']})")
        print(f"  Variance: {selected_stats['variance']:.2f} (rank #{analysis['vertex_characteristics']['variance_rank']})")
        print(f"  Mean Edge Weight: {selected_stats['mean']:.2f}")
        print(f"  Edge Range: {selected_stats['min_edge']:.2f} - {selected_stats['max_edge']:.2f}")
        
        print(f"\nPerformance Results:")
        print(f"  Strategy Result: {strategy_weight:.2f}")
        print(f"  Random Average: {random_avg:.2f}")
        print(f"  All Vertices Average: {all_avg:.2f}")
        print(f"  Best Overall: {analysis['performance_stats']['best_overall']:.2f}")
        
        print(f"\nStrategy Effectiveness:")
        print(f"  vs Random Improvement: {analysis['strategy_effectiveness']['vs_random_improvement']:+.2f}%")
        print(f"  vs Average Improvement: {analysis['strategy_effectiveness']['vs_average_improvement']:+.2f}%")
        print(f"  Rank among all vertices: #{analysis['strategy_effectiveness']['rank_among_all']}/{len(all_weights)}")
        print(f"  Percentile: {analysis['strategy_effectiveness']['percentile']:.1f}th")
        print(f"  Is Optimal: {'Yes' if analysis['strategy_effectiveness']['is_optimal'] else 'No'}")
    
    return analysis

def weight_variance_strategy_comparison_test(num_graphs=3, num_vertices=15, weight_range=(1, 100), seed_base=500):
    """
    Compare different weight-variance combination strategies across multiple graphs.
    
    Args:
        num_graphs: Number of graphs to test
        num_vertices: Number of vertices per graph
        weight_range: Range of edge weights (min, max)
        seed_base: Base seed for reproducible results
        
    Returns:
        Comprehensive results comparing all strategies
    """
    strategies = ['weight_plus_variance', 'normalized_sum', 'product', 'ratio', 'weight_only', 'variance_only']
    strategy_results = defaultdict(list)
    all_analyses = []
    
    print(f"\n{'='*90}")
    print(f"WEIGHT-VARIANCE STRATEGY COMPARISON TEST")
    print(f"Testing {len(strategies)} strategies across {num_graphs} graphs")
    print(f"Graphs: {num_vertices} vertices, weight range: {weight_range}, seed base: {seed_base}")
    print(f"{'='*90}")
    
    for g in range(num_graphs):
        print(f"\n--- Graph {g+1} ---")
        graph = generate_complete_graph(num_vertices, weight_range=weight_range, seed=seed_base + g)
        
        # Print brief graph info
        print(f"Graph {g+1} adjacency matrix preview (first 3 rows):")
        for i in range(min(3, len(graph))):
            row_preview = str(graph[i][:min(6, len(graph[i]))])
            if len(graph[i]) > 6:
                row_preview = row_preview[:-1] + ", ...]"
            print(f"  Row {i}: {row_preview}")
        
        graph_results = {}
        
        # Test each strategy on this graph
        for strategy in strategies:
            results = weight_variance_metaheuristic(graph, strategy=strategy, num_random_trials=3, verbose=False)
            analysis = analyze_weight_variance_impact(results, verbose=False)
            
            graph_results[strategy] = {
                'weight': results['strategy_result']['weight'],
                'vertex': results['strategy_result']['vertex'],
                'vertex_stats': results['strategy_result']['vertex_stats'],
                'analysis': analysis
            }
            
            strategy_results[strategy].append(results['strategy_result']['weight'])
        
        # Print results for this graph
        print(f"\nGraph {g+1} Results:")
        for strategy in strategies:
            result = graph_results[strategy]
            print(f"  {strategy:20}: Weight = {result['weight']:7.2f}, Vertex = {result['vertex']:2d}, "
                  f"Total Weight = {result['vertex_stats']['total_weight']:6.1f}, "
                  f"Variance = {result['vertex_stats']['variance']:6.1f}")
        
        # Find best strategy for this graph
        best_strategy = min(graph_results.items(), key=lambda x: x[1]['weight'])
        print(f"  Best for Graph {g+1}: {best_strategy[0]} (weight: {best_strategy[1]['weight']:.2f})")
        
        all_analyses.append(graph_results)
    
    # Compute overall statistics
    print(f"\n{'='*90}")
    print("OVERALL STRATEGY COMPARISON RESULTS")
    print(f"{'='*90}")
    
    strategy_stats = {}
    for strategy in strategies:
        weights = strategy_results[strategy]
        avg_weight = sum(weights) / len(weights)
        std_dev = (sum((w - avg_weight) ** 2 for w in weights) / len(weights)) ** 0.5
        
        strategy_stats[strategy] = {
            'avg_weight': avg_weight,
            'std_dev': std_dev,
            'best_weight': min(weights),
            'worst_weight': max(weights),
            'weights': weights
        }
        
        print(f"{strategy:20}: Avg = {avg_weight:7.2f} +- {std_dev:5.2f}, "
              f"Best = {min(weights):7.2f}, Worst = {max(weights):7.2f}")
    
    # Rank strategies by average performance
    ranked_strategies = sorted(strategy_stats.items(), key=lambda x: x[1]['avg_weight'])
    
    print(f"\nStrategy Ranking (by average weight):")
    for i, (strategy, stats) in enumerate(ranked_strategies, 1):
        print(f"  {i}. {strategy:20}: {stats['avg_weight']:.2f}")
    
    # Count wins
    wins = defaultdict(int)
    for graph_results in all_analyses:
        best_strategy = min(graph_results.items(), key=lambda x: x[1]['weight'])[0]
        wins[best_strategy] += 1
    
    print(f"\nStrategy Wins (best performance per graph):")
    for strategy in sorted(wins.keys(), key=lambda x: wins[x], reverse=True):
        print(f"  {strategy:20}: {wins[strategy]}/{num_graphs} graphs")
    
    return strategy_results, all_analyses, strategy_stats

def print_detailed_strategy_results(strategy_results, strategy_stats):
    """Print detailed results in a clean format"""
    print(f"\n{'='*70}")
    print("DETAILED STRATEGY RESULTS")
    print(f"{'='*70}")
    
    for strategy, weights in strategy_results.items():
        stats = strategy_stats[strategy]
        print(f"\n{strategy}:")
        print(f"  Individual Results: {[f'{w:.2f}' for w in weights]}")
        print(f"  Average: {stats['avg_weight']:.2f}")
        print(f"  Standard Deviation: {stats['std_dev']:.2f}")
        print(f"  Range: {stats['best_weight']:.2f} - {stats['worst_weight']:.2f}")

def main():
    """Main function to run weight-variance metaheuristic tests"""
    print("Weight-Variance Metaheuristic Testing for TSP Anchor Selection")
    
    # Test 1: Small graphs for detailed analysis
    print("\n" + "="*100)
    print("TEST 1: Small graphs - detailed strategy comparison")
    results_small, analyses_small, stats_small = weight_variance_strategy_comparison_test(
        num_graphs=15, 
        num_vertices=8, 
        weight_range=(1, 50), 
        seed_base=42069
    )
    print_detailed_strategy_results(results_small, stats_small)
    
    # Test 2: Medium graphs for statistical significance
    print("\n" + "="*100)
    print("TEST 2: Medium graphs - statistical significance")
    results_medium, analyses_medium, stats_medium = weight_variance_strategy_comparison_test(
        num_graphs=10, 
        num_vertices=15, 
        weight_range=(1, 100), 
        seed_base=12345
    )
    print_detailed_strategy_results(results_medium, stats_medium)
    
    # Test 3: Larger graphs for real-world applicability
    print("\n" + "="*100)
    print("TEST 3: Large graphs - real-world applicability")
    results_large, analyses_large, stats_large = weight_variance_strategy_comparison_test(
        num_graphs=10, 
        num_vertices=25, 
        weight_range=(1, 200), 
        seed_base=54321
    )
    print_detailed_strategy_results(results_large, stats_large)

    print("\n" + "="*100)
    print("TEST 3: Extra-large graphs - for the funsies :))")
    results_extra_large, analyses_extra_large, stats_extra_large = weight_variance_strategy_comparison_test(
        num_graphs=10, 
        num_vertices=100, 
        weight_range=(1, 400), 
        seed_base=13579
    )
    print_detailed_strategy_results(results_extra_large, stats_extra_large)

if __name__ == "__main__":
    main()