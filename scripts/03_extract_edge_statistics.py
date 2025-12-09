"""
Phase 3: Extract simple edge statistics for all vertices.
Computes: sum, mean, median, variance, std, min, max, range, cv, min2, anchor_sum.
"""
import os
import sys
import json
import pickle
import pandas as pd
from pathlib import Path
import networkx as nx

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis.edge_statistics import compute_all_vertex_stats


def main():
    """Extract edge statistics for all vertices in all graphs."""
    graphs_dir = Path(__file__).parent.parent / "data" / "anchor_analysis" / "graphs"
    output_dir = Path(__file__).parent.parent / "data" / "anchor_analysis"

    # Load metadata
    metadata_path = graphs_dir / "graphs_metadata.json"
    with open(metadata_path, "r") as f:
        graphs_metadata = json.load(f)

    all_stats = []

    print(f"\nExtracting edge statistics for {len(graphs_metadata)} graphs...\n")

    for i, graph_meta in enumerate(graphs_metadata):
        graph_id = graph_meta["graph_id"]
        graph_type = graph_meta["type"]
        filename = graph_meta["filename"]

        graph_path = graphs_dir / filename

        try:
            with open(graph_path, "rb") as f:
                graph = pickle.load(f)

            # Compute edge statistics for all vertices
            # Convert adjacency matrix to NetworkX graph for feature extraction
            G = nx.Graph()
            n = len(graph)
            for i in range(n):
                for j in range(n):
                    if i != j:
                        G.add_edge(i, j, weight=graph[i][j])

            stats = compute_all_vertex_stats(G)

            # Add graph metadata
            for stat in stats:
                stat["graph_id"] = graph_id
                stat["graph_type"] = graph_type

            all_stats.extend(stats)

            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{len(graphs_metadata)} graphs")

        except Exception as e:
            print(f"  ⚠️  Failed to process graph {graph_id}: {e}")
            continue

    if all_stats:
        df = pd.DataFrame(all_stats)

        # Reorder columns for readability
        cols = ["graph_id", "graph_type", "vertex_id"]
        stat_cols = [c for c in df.columns if c not in cols]
        df = df[cols + stat_cols]

        output_path = output_dir / "vertex_statistics.csv"
        df.to_csv(output_path, index=False)

        print(f"\n✅ Extracted edge statistics for {len(df)} vertices")
        print(f"   Saved to: {output_path}")
        print(f"\n   Statistics columns: {stat_cols}")
        print(f"   Rows: {len(df)}")
        print(f"   Graphs: {df['graph_id'].nunique()}")
    else:
        print("❌ No statistics generated!")


if __name__ == "__main__":
    main()
