"""
Graph generator module - re-exports functions from utils.graph_generator
for backward compatibility with existing imports.
"""

from utils.graph_generator import (
    generate_complete_graph,
    analyze_graph_properties,
    calculate_cycle_cost,
    verify_triangle_inequality,
    test_improved_generator
)

# Re-export all functions for backward compatibility
__all__ = [
    'generate_complete_graph',
    'analyze_graph_properties',
    'calculate_cycle_cost',
    'verify_triangle_inequality',
    'test_improved_generator'
]