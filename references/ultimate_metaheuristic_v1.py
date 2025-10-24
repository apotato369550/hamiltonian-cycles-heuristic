import random
import statistics
import math
from collections import defaultdict
from typing import List, Tuple, Dict, Optional, Union
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

def calculate_comprehensive_vertex_statistics(graph, vertex):
    """
    Calculate comprehensive statistics for a vertex's edges.
    
    Args:
        graph: Adjacency matrix representation of the graph
        vertex: The vertex to analyze
        
    Returns:
        Dictionary containing all statistical measures
    """
    edges = [graph[vertex][i] for i in range(len(graph)) if i != vertex]
    
    if not edges:
        return {
            'total_weight': 0, 'variance': 0, 'std_deviation': 0, 'mean': 0,
            'min_edge': 0, 'max_edge': 0, 'edge_count': 0, 'range': 0,
            'median': 0, 'q1': 0, 'q3': 0, 'iqr': 0, 'skewness': 0,
            'coefficient_of_variation': 0, 'mean_abs_deviation': 0
        }
    
    # Basic statistics
    total_weight = sum(edges)
    mean_weight = total_weight / len(edges)
    variance = statistics.variance(edges) if len(edges) > 1 else 0
    std_deviation = statistics.stdev(edges) if len(edges) > 1 else 0
    min_edge = min(edges)
    max_edge = max(edges)
    edge_range = max_edge - min_edge
    
    # Advanced statistics
    median = statistics.median(edges)
    
    # Quartiles and IQR
    sorted_edges = sorted(edges)
    n = len(sorted_edges)
    q1 = statistics.median(sorted_edges[:n//2]) if n > 1 else sorted_edges[0]
    q3 = statistics.median(sorted_edges[(n+1)//2:]) if n > 1 else sorted_edges[0]
    iqr = q3 - q1
    
    # Coefficient of variation
    cv = (std_deviation / mean_weight) if mean_weight > 0 else 0
    
    # Mean absolute deviation
    mad = sum(abs(x - mean_weight) for x in edges) / len(edges)
    
    # Skewness (using sample skewness formula)
    if len(edges) > 2 and std_deviation > 0:
        skewness = (sum((x - mean_weight) ** 3 for x in edges) / len(edges)) / (std_deviation ** 3)
    else:
        skewness = 0
    
    return {
        'total_weight': total_weight,
        'variance': variance,
        'std_deviation': std_deviation,
        'mean': mean_weight,
        'min_edge': min_edge,
        'max_edge': max_edge,
        'edge_count': len(edges),
        'range': edge_range,
        'median': median,
        'q1': q1,
        'q3': q3,
        'iqr': iqr,
        'skewness': skewness,
        'coefficient_of_variation': cv,
        'mean_abs_deviation': mad
    }

def create_advanced_vertex_rankings(graph):
    """
    Create comprehensive rankings using all statistical measures.
    
    Args:
        graph: Adjacency matrix representation of the graph
        
    Returns:
        Dictionary containing various rankings and vertex statistics
    """
    vertices_count = len(graph)
    vertex_stats = {}
    
    # Calculate comprehensive statistics for all vertices
    for vertex in range(vertices_count):
        vertex_stats[vertex] = calculate_comprehensive_vertex_statistics(graph, vertex)
    
    # Extract all statistical measures for normalization
    measures = ['total_weight', 'variance', 'std_deviation', 'mean', 'range', 
               'median', 'iqr', 'coefficient_of_variation', 'mean_abs_deviation']
    
    # Calculate min/max for normalization
    measure_ranges = {}
    for measure in measures:
        values = [vertex_stats[v][measure] for v in range(vertices_count)]
        measure_ranges[measure] = {
            'min': min(values),
            'max': max(values),
            'range': max(values) - min(values)
        }
    
    # Create various ranking strategies
    rankings = {}
    
    # Pure statistical measure rankings (highest first for most measures)
    for measure in measures:
        rankings[f'by_{measure}'] = sorted(
            range(vertices_count), 
            key=lambda v: vertex_stats[v][measure], 
            reverse=True
        )
    
    # Special cases - some measures might be better when low
    rankings['by_mean_low'] = sorted(
        range(vertices_count), 
        key=lambda v: vertex_stats[v]['mean']
    )
    
    rankings['by_std_deviation_low'] = sorted(
        range(vertices_count), 
        key=lambda v: vertex_stats[v]['std_deviation']
    )
    
    # Advanced combination strategies
    
    # 1. Weight + Variance + Mean + StdDev (equal weights)
    rankings['by_quad_sum'] = sorted(
        range(vertices_count),
        key=lambda v: (vertex_stats[v]['total_weight'] + vertex_stats[v]['variance'] + 
                      vertex_stats[v]['mean'] + vertex_stats[v]['std_deviation']),
        reverse=True
    )
    
    # 2. Normalized quad sum
    normalized_scores = {}
    for vertex in range(vertices_count):
        score = 0
        for measure in ['total_weight', 'variance', 'mean', 'std_deviation']:
            if measure_ranges[measure]['range'] > 0:
                normalized_val = ((vertex_stats[vertex][measure] - measure_ranges[measure]['min']) / 
                                measure_ranges[measure]['range'])
                score += normalized_val
        normalized_scores[vertex] = score
    
    rankings['by_normalized_quad'] = sorted(
        normalized_scores.keys(), 
        key=lambda v: normalized_scores[v], 
        reverse=True
    )
    
    # 3. Weighted combinations (favoring different aspects)
    
    # Variance-focused combination
    variance_focused_scores = {}
    for vertex in range(vertices_count):
        stats = vertex_stats[vertex]
        # Weight variance heavily, with supporting weight and mean
        score = (stats['variance'] * 0.5 + stats['total_weight'] * 0.3 + 
                stats['mean'] * 0.2)
        variance_focused_scores[vertex] = score
    
    rankings['by_variance_focused'] = sorted(
        variance_focused_scores.keys(),
        key=lambda v: variance_focused_scores[v],
        reverse=True
    )
    
    # Mean-focused combination
    mean_focused_scores = {}
    for vertex in range(vertices_count):
        stats = vertex_stats[vertex]
        # High mean with high variance indicates good anchoring potential
        score = (stats['mean'] * 0.4 + stats['variance'] * 0.3 + 
                stats['total_weight'] * 0.3)
        mean_focused_scores[vertex] = score
    
    rankings['by_mean_focused'] = sorted(
        mean_focused_scores.keys(),
        key=lambda v: mean_focused_scores[v],
        reverse=True
    )
    
    # 4. Coefficient of Variation focused (high CV indicates high relative variability)
    cv_focused_scores = {}
    for vertex in range(vertices_count):
        stats = vertex_stats[vertex]
        # High CV with high total weight
        score = (stats['coefficient_of_variation'] * 0.4 + stats['total_weight'] * 0.6)
        cv_focused_scores[vertex] = score
    
    rankings['by_cv_focused'] = sorted(
        cv_focused_scores.keys(),
        key=lambda v: cv_focused_scores[v],
        reverse=True
    )
    
    # 5. Range-based strategy (high range indicates diverse edge weights)
    range_focused_scores = {}
    for vertex in range(vertices_count):
        stats = vertex_stats[vertex]
        # High range with high total weight
        score = (stats['range'] * 0.5 + stats['total_weight'] * 0.5)
        range_focused_scores[vertex] = score
    
    rankings['by_range_focused'] = sorted(
        range_focused_scores.keys(),
        key=lambda v: range_focused_scores[v],
        reverse=True
    )
    
    # 6. Multi-objective Pareto-style approach
    # Find vertices that are in top 25% for multiple measures
    pareto_scores = {}
    key_measures = ['total_weight', 'variance', 'mean', 'std_deviation']
    
    for vertex in range(vertices_count):
        score = 0
        for measure in key_measures:
            vertex_rank = rankings[f'by_{measure}'].index(vertex)
            # Top 25% gets points
            if vertex_rank < vertices_count * 0.25:
                score += (vertices_count * 0.25 - vertex_rank) / (vertices_count * 0.25)
        pareto_scores[vertex] = score
    
    rankings['by_pareto_multi'] = sorted(
        pareto_scores.keys(),
        key=lambda v: pareto_scores[v],
        reverse=True
    )
    
    # 7. Hybrid statistical approach (combining multiple statistical insights)
    hybrid_scores = {}
    for vertex in range(vertices_count):
        stats = vertex_stats[vertex]
        # Complex formula considering multiple factors
        base_score = stats['total_weight'] * 0.3
        variability_score = (stats['variance'] + stats['std_deviation'] + stats['range']) * 0.2
        central_tendency_score = stats['mean'] * 0.2
        distribution_score = (stats['coefficient_of_variation'] + stats['iqr']) * 0.15
        outlier_potential = abs(stats['skewness']) * 0.15  # Skewed distributions might be good
        
        hybrid_scores[vertex] = (base_score + variability_score + central_tendency_score + 
                               distribution_score + outlier_potential)
    
    rankings['by_hybrid_statistical'] = sorted(
        hybrid_scores.keys(),
        key=lambda v: hybrid_scores[v],
        reverse=True
    )
    
    return {
        'vertex_statistics': vertex_stats,
        'rankings': rankings,
        'measure_ranges': measure_ranges,
        'normalized_scores': normalized_scores,
        'variance_focused_scores': variance_focused_scores,
        'mean_focused_scores': mean_focused_scores,
        'cv_focused_scores': cv_focused_scores,
        'range_focused_scores': range_focused_scores,
        'pareto_scores': pareto_scores,
        'hybrid_scores': hybrid_scores
    }

def advanced_statistical_metaheuristic(graph, strategy='hybrid_statistical', num_random_trials=5, verbose=False):
    """
    Test the anchor heuristic with advanced statistical starting vertex selection.
    
    Available strategies:
    - Pure measures: 'total_weight', 'variance', 'std_deviation', 'mean', 'range', 
                    'median', 'iqr', 'coefficient_of_variation', 'mean_abs_deviation'
    - Low variants: 'mean_low', 'std_deviation_low'
    - Combinations: 'quad_sum', 'normalized_quad', 'variance_focused', 'mean_focused',
                   'cv_focused', 'range_focused', 'pareto_multi', 'hybrid_statistical'
    
    Args:
        graph: Adjacency matrix representation of the graph
        strategy: Strategy name for vertex selection
        num_random_trials: Number of random vertex trials for comparison
        verbose: Whether to print detailed output
        
    Returns:
        Dictionary containing results and analysis
    """
    ranking_info = create_advanced_vertex_rankings(graph)
    
    # Map strategy names to ranking keys
    strategy_key_map = {}
    for key in ranking_info['rankings'].keys():
        if key.startswith('by_'):
            clean_name = key[3:]  # Remove 'by_' prefix
            strategy_key_map[clean_name] = key
    
    if strategy not in strategy_key_map:
        available_strategies = list(strategy_key_map.keys())
        raise ValueError(f"Unknown strategy: {strategy}. Available: {available_strategies}")
    
    ranking_key = strategy_key_map[strategy]
    selected_vertex = ranking_info['rankings'][ranking_key][0]  # Top-ranked vertex
    
    results = {
        'strategy_used': strategy,
        'selected_vertex': selected_vertex,
        'strategy_result': None,
        'random_trials': [],
        'all_vertices': [],
        'best_overall': None,
        'ranking_info': ranking_info
    }
    
    # Test the selected strategy
    cycle, weight = low_anchor_heuristic(graph, selected_vertex)
    vertex_stats = ranking_info['vertex_statistics'][selected_vertex]
    
    results['strategy_result'] = {
        'vertex': selected_vertex,
        'cycle': cycle,
        'weight': weight,
        'vertex_stats': vertex_stats
    }
    
    if verbose:
        print(f"Strategy '{strategy}' selected vertex {selected_vertex}")
        print(f"  Total weight: {vertex_stats['total_weight']:.2f}")
        print(f"  Mean: {vertex_stats['mean']:.2f}")
        print(f"  Std deviation: {vertex_stats['std_deviation']:.2f}")
        print(f"  Variance: {vertex_stats['variance']:.2f}")
        print(f"  Coefficient of variation: {vertex_stats['coefficient_of_variation']:.2f}")
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

def analyze_advanced_statistical_impact(results, verbose=True):
    """
    Analyze the effectiveness of advanced statistical vertex selection.
    
    Args:
        results: Results from advanced_statistical_metaheuristic
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
    
    # Comprehensive vertex characteristics analysis
    selected_vertex = results['selected_vertex']
    selected_stats = results['strategy_result']['vertex_stats']
    
    analysis['vertex_characteristics'] = {
        'selected_vertex': selected_vertex,
        'comprehensive_stats': selected_stats
    }
    
    # Add rankings for key measures
    ranking_info = results['ranking_info']
    key_measures = ['total_weight', 'variance', 'std_deviation', 'mean', 'coefficient_of_variation']
    
    for measure in key_measures:
        rank_key = f'by_{measure}'
        if rank_key in ranking_info['rankings']:
            rank = ranking_info['rankings'][rank_key].index(selected_vertex) + 1
            analysis['vertex_characteristics'][f'{measure}_rank'] = rank
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"ADVANCED STATISTICAL METAHEURISTIC ANALYSIS")
        print(f"Strategy: {results['strategy_used']}")
        print(f"{'='*80}")
        
        print(f"Selected Vertex: {selected_vertex}")
        print(f"Comprehensive Statistics:")
        for stat_name, value in selected_stats.items():
            if stat_name in ['total_weight', 'variance', 'std_deviation', 'mean', 
                           'coefficient_of_variation', 'range', 'median']:
                rank_key = f'{stat_name}_rank'
                rank_info = f" (rank #{analysis['vertex_characteristics'].get(rank_key, 'N/A')})" if rank_key in analysis['vertex_characteristics'] else ""
                print(f"  {stat_name.replace('_', ' ').title()}: {value:.2f}{rank_info}")
        
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

def comprehensive_statistical_strategy_test(num_graphs=5, num_vertices=15, weight_range=(1, 100), seed_base=1000):
    """
    Compare all advanced statistical strategies across multiple graphs.
    
    Args:
        num_graphs: Number of graphs to test
        num_vertices: Number of vertices per graph
        weight_range: Range of edge weights (min, max)
        seed_base: Base seed for reproducible results
        
    Returns:
        Comprehensive results comparing all strategies
    """
    # Define all strategies to test
    basic_strategies = ['total_weight', 'variance', 'std_deviation', 'mean', 'range', 
                       'coefficient_of_variation', 'mean_abs_deviation']
    
    combination_strategies = ['quad_sum', 'normalized_quad', 'variance_focused', 
                            'mean_focused', 'cv_focused', 'range_focused', 
                            'pareto_multi', 'hybrid_statistical']
    
    all_strategies = basic_strategies + combination_strategies
    
    strategy_results = defaultdict(list)
    all_analyses = []
    
    print(f"\n{'='*100}")
    print(f"COMPREHENSIVE ADVANCED STATISTICAL STRATEGY TEST")
    print(f"Testing {len(all_strategies)} strategies across {num_graphs} graphs")
    print(f"Graphs: {num_vertices} vertices, weight range: {weight_range}, seed base: {seed_base}")
    print(f"{'='*100}")
    
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
        for strategy in all_strategies:
            try:
                results = advanced_statistical_metaheuristic(graph, strategy=strategy, 
                                                           num_random_trials=3, verbose=False)
                analysis = analyze_advanced_statistical_impact(results, verbose=False)
                
                graph_results[strategy] = {
                    'weight': results['strategy_result']['weight'],
                    'vertex': results['strategy_result']['vertex'],
                    'vertex_stats': results['strategy_result']['vertex_stats'],
                    'analysis': analysis
                }
                
                strategy_results[strategy].append(results['strategy_result']['weight'])
                
            except Exception as e:
                print(f"  Error with strategy {strategy}: {e}")
                continue
        
        # Print results for this graph (top 5 performers)
        sorted_results = sorted(graph_results.items(), key=lambda x: x[1]['weight'])
        print(f"\nGraph {g+1} Results (Top 5):")
        for i, (strategy, result) in enumerate(sorted_results[:5]):
            stats = result['vertex_stats']
            print(f"  {i+1:2d}. {strategy:20}: Weight = {result['weight']:7.2f}, "
                  f"Vertex = {result['vertex']:2d}, Mean = {stats['mean']:6.1f}, "
                  f"StdDev = {stats['std_deviation']:6.1f}")
        
        if sorted_results:
            best_strategy, best_result = sorted_results[0]
            print(f"  Best for Graph {g+1}: {best_strategy} (weight: {best_result['weight']:.2f})")
        
        all_analyses.append(graph_results)
    
    # Compute overall statistics
    print(f"\n{'='*100}")
    print("COMPREHENSIVE STRATEGY COMPARISON RESULTS")
    print(f"{'='*100}")
    
    strategy_stats = {}
    for strategy in all_strategies:
        if strategy in strategy_results and strategy_results[strategy]:
            weights = strategy_results[strategy]
            avg_weight = sum(weights) / len(weights)
            std_dev = (sum((w - avg_weight) ** 2 for w in weights) / len(weights)) ** 0.5
            
            strategy_stats[strategy] = {
                'avg_weight': avg_weight,
                'std_dev': std_dev,
                'best_weight': min(weights),
                'worst_weight': max(weights),
                'weights': weights,
                'success_rate': len(weights) / num_graphs
            }
    
    # Rank strategies by average performance
    ranked_strategies = sorted(strategy_stats.items(), key=lambda x: x[1]['avg_weight'])
    
    print(f"Strategy Ranking (by average weight):")
    for i, (strategy, stats) in enumerate(ranked_strategies[:10], 1):  # Top 10
        print(f"  {i:2d}. {strategy:20}: {stats['avg_weight']:7.2f} +- {stats['std_dev']:5.2f} "
              f"(best: {stats['best_weight']:6.2f}, success: {stats['success_rate']*100:3.0f}%)")
    
    # Count wins
    wins = defaultdict(int)
    for graph_results in all_analyses:
        if graph_results:
            best_strategy = min(graph_results.items(), key=lambda x: x[1]['weight'])[0]
            wins[best_strategy] += 1
    
    print(f"\nStrategy Wins (best performance per graph):")
    sorted_wins = sorted(wins.items(), key=lambda x: x[1], reverse=True)
    for strategy, win_count in sorted_wins[:10]:  # Top 10 winners
        print(f"  {strategy:20}: {win_count:2d}/{num_graphs} graphs ({win_count/num_graphs*100:4.1f}%)")
    
    # Performance category analysis
    print(f"\n{'='*60}")
    print("PERFORMANCE CATEGORY ANALYSIS")
    print(f"{'='*60}")
    
    # Categorize strategies
    basic_performers = [(s, strategy_stats[s]['avg_weight']) for s in basic_strategies if s in strategy_stats]
    combo_performers = [(s, strategy_stats[s]['avg_weight']) for s in combination_strategies if s in strategy_stats]
    
    if basic_performers:
        best_basic = min(basic_performers, key=lambda x: x[1])
        print(f"Best Basic Strategy: {best_basic[0]} (avg: {best_basic[1]:.2f})")
    
    if combo_performers:
        best_combo = min(combo_performers, key=lambda x: x[1])
        print(f"Best Combination Strategy: {best_combo[0]} (avg: {best_combo[1]:.2f})")
    
    return strategy_results, all_analyses, strategy_stats

def print_detailed_advanced_results(strategy_results, strategy_stats):
    # todo
    return