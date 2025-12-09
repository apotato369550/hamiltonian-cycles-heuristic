"""
Phase 2: Compute anchor quality for all graphs.
For each graph, run single_anchor from every vertex and record tour weights.
"""
import os
import sys
import json
import pickle
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.algorithms import get_algorithm

def compute_anchor_quality_for_graph(graph, graph_id, graph_type):
    """Run single_anchor from each vertex and return quality DataFrame."""
    results = []

    algorithm = get_algorithm("single_anchor")

    for vertex in graph.nodes():
        try:
            tour, weight = algorithm(graph, start_vertex=vertex)

            results.append({
                "graph_id": graph_id,
                "graph_type": graph_type,
                "vertex_id": vertex,
                "tour_weight": weight,
            })
        except Exception as e:
            print(f"    ⚠️  Failed to compute anchor quality for vertex {vertex}: {e}")
            continue

    df = pd.DataFrame(results)
    if len(df) > 0:
        df["rank"] = df["tour_weight"].rank()
        df["percentile"] = (df["tour_weight"].rank(pct=True) * 100).round(2)

    return df


def main():
    """Compute anchor quality for all generated graphs."""
    graphs_dir = Path(__file__).parent.parent / "data" / "anchor_analysis" / "graphs"
    output_dir = Path(__file__).parent.parent / "data" / "anchor_analysis"

    # Load metadata
    metadata_path = graphs_dir / "graphs_metadata.json"
    with open(metadata_path, "r") as f:
        graphs_metadata = json.load(f)

    all_results = []

    print(f"\nComputing anchor quality for {len(graphs_metadata)} graphs...\n")

    for i, graph_meta in enumerate(graphs_metadata):
        graph_id = graph_meta["graph_id"]
        graph_type = graph_meta["type"]
        filename = graph_meta["filename"]

        graph_path = graphs_dir / filename

        try:
            with open(graph_path, "rb") as f:
                graph = pickle.load(f)

            df = compute_anchor_quality_for_graph(graph, graph_id, graph_type)
            all_results.append(df)

            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{len(graphs_metadata)} graphs")

        except Exception as e:
            print(f"  ⚠️  Failed to process graph {graph_id}: {e}")
            continue

    if all_results:
        combined_df = pd.concat(all_results, ignore_index=True)
        output_path = output_dir / "anchor_quality.csv"
        combined_df.to_csv(output_path, index=False)

        print(f"\n✅ Computed anchor quality for {len(combined_df)} vertices")
        print(f"   Saved to: {output_path}")
        print(f"\n   Summary:")
        print(f"   - Total vertices tested: {len(combined_df)}")
        print(f"   - Graphs processed: {combined_df['graph_id'].nunique()}")
        print(f"   - Graph types: {combined_df['graph_type'].unique().tolist()}")
    else:
        print("❌ No results generated!")


if __name__ == "__main__":
    main()
