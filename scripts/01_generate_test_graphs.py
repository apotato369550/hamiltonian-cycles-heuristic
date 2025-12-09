"""
Phase 1: Generate test graphs for anchor analysis.
Creates 100 graphs: 25 Euclidean, 25 metric, 25 random, 25 quasi-metric.
"""
import os
import sys
import json
import pickle
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.graph_generation import (
    EuclideanGraphGenerator,
    generate_metric_graph,
    generate_random_graph,
    generate_quasi_metric_graph,
)

def generate_test_graphs():
    """Generate 100 diverse test graphs."""
    output_dir = Path(__file__).parent.parent / "data" / "anchor_analysis" / "graphs"
    output_dir.mkdir(parents=True, exist_ok=True)

    graphs_metadata = []
    graph_id = 0

    configs = [
        ("euclidean", 25, "euclidean"),
        ("metric", 25, "metric"),
        ("random", 25, "random"),
        ("quasi-metric", 25, "quasi-metric"),
    ]

    for graph_type_name, count, graph_type_func in configs:
        print(f"\nGenerating {count} {graph_type_name} graphs...")

        for i in range(count):
            # Vary graph size between 20-50 vertices
            num_vertices = 20 + (i % 30)

            try:
                # Generate graph based on type
                if graph_type_func == "euclidean":
                    gen = EuclideanGraphGenerator(
                        num_vertices=num_vertices,
                        weight_range=(1.0, 100.0),
                        seed=1000 + graph_id,
                    )
                    graph = gen.generate()
                elif graph_type_func == "metric":
                    graph = generate_metric_graph(
                        num_vertices=num_vertices,
                        weight_range=(1.0, 100.0),
                        random_seed=1000 + graph_id,
                    )
                elif graph_type_func == "random":
                    graph = generate_random_graph(
                        num_vertices=num_vertices,
                        weight_range=(1.0, 100.0),
                        random_seed=1000 + graph_id,
                    )
                elif graph_type_func == "quasi-metric":
                    graph = generate_quasi_metric_graph(
                        num_vertices=num_vertices,
                        weight_range=(1.0, 100.0),
                        random_seed=1000 + graph_id,
                    )

                # Save graph
                graph_path = output_dir / f"graph_{graph_id:03d}_{graph_type_name}.pkl"
                with open(graph_path, "wb") as f:
                    pickle.dump(graph, f)

                # Record metadata
                graphs_metadata.append({
                    "graph_id": graph_id,
                    "type": graph_type_name,
                    "num_vertices": num_vertices,
                    "filename": graph_path.name,
                })

                if (i + 1) % 5 == 0:
                    print(f"  Generated {i + 1}/{count} graphs")

                graph_id += 1

            except Exception as e:
                print(f"  ⚠️  Failed to generate graph {i}: {e}")
                continue

    # Save metadata
    metadata_path = output_dir / "graphs_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(graphs_metadata, f, indent=2)

    print(f"\n✅ Generated {len(graphs_metadata)} graphs")
    print(f"   Saved to: {output_dir}")
    print(f"   Metadata: {metadata_path}")

    return graphs_metadata


if __name__ == "__main__":
    generate_test_graphs()
