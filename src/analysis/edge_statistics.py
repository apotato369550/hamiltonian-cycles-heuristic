"""
Simple edge statistics for anchor analysis.
This is ALL we need for the core research question.
"""
import numpy as np
from typing import Dict, List, Any


def compute_vertex_edge_stats(graph, vertex: int) -> Dict[str, float]:
    """
    Compute edge statistics for a single vertex.

    Returns dict with:
    - sum_weight, mean_weight, median_weight
    - variance_weight, std_weight
    - min_weight, max_weight, range_weight
    - cv_weight (coefficient of variation)
    - min2_weight (second smallest)
    - anchor_edge_sum (min1 + min2)
    """
    # Get all edge weights from this vertex
    weights = [graph[vertex][neighbor]['weight']
               for neighbor in graph.neighbors(vertex)]
    weights = np.array(weights)
    sorted_weights = np.sort(weights)

    mean_val = np.mean(weights)
    std_val = np.std(weights)
    cv = std_val / mean_val if mean_val > 0 else 0

    return {
        'sum_weight': float(np.sum(weights)),
        'mean_weight': float(mean_val),
        'median_weight': float(np.median(weights)),
        'variance_weight': float(np.var(weights)),
        'std_weight': float(std_val),
        'min_weight': float(sorted_weights[0]),
        'max_weight': float(sorted_weights[-1]),
        'range_weight': float(sorted_weights[-1] - sorted_weights[0]),
        'cv_weight': float(cv),
        'min2_weight': float(sorted_weights[1] if len(sorted_weights) > 1 else sorted_weights[0]),
        'anchor_edge_sum': float(sorted_weights[0] + (sorted_weights[1] if len(sorted_weights) > 1 else sorted_weights[0])),
    }


def compute_all_vertex_stats(graph) -> List[Dict[str, Any]]:
    """Compute edge statistics for all vertices in a graph."""
    results = []
    for vertex in graph.nodes():
        stats = compute_vertex_edge_stats(graph, vertex)
        stats['vertex_id'] = vertex
        results.append(stats)
    return results
