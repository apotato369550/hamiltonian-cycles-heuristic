"""
Phase 8: Practical validation.
Does predicting anchors actually improve tour quality?
Compare predicted vs random vs best vs other algorithms.
"""
import sys
import json
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LinearRegression

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.algorithms import get_algorithm


def main():
    """Validate predictions on held-out test data."""
    data_dir = Path(__file__).parent.parent / "data" / "anchor_analysis"
    graphs_dir = data_dir / "graphs"
    output_dir = Path(__file__).parent.parent / "results" / "anchor_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("\nLoading data files...\n")

    stats_df = pd.read_csv(data_dir / "vertex_statistics.csv")
    quality_df = pd.read_csv(data_dir / "anchor_quality.csv")

    merged = pd.merge(stats_df, quality_df, on=["graph_id", "vertex_id"])

    print(f"✅ Loaded {len(merged)} vertex-quality pairs\n")

    # Train a simple predictor model
    print("Training prediction model...\n")

    feature_cols = ["sum_weight", "variance_weight"]
    X = merged[feature_cols]
    y = merged["percentile"]

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, merged.index, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    print(f"✅ Model trained (R² = {model.score(X_test, y_test):.4f})\n")

    # Get test graphs
    metadata_path = graphs_dir / "graphs_metadata.json"
    with open(metadata_path, "r") as f:
        graphs_metadata = json.load(f)

    test_graph_ids = merged.loc[idx_test, "graph_id"].unique()[:5]  # Test on first 5

    print(f"Testing practical validation on {len(test_graph_ids)} graphs...\n")

    results = []

    single_anchor = get_algorithm("single_anchor")
    nearest_neighbor = get_algorithm("nearest_neighbor")

    for graph_id in test_graph_ids:
        graph_meta = graphs_metadata[graph_id]
        graph_path = graphs_dir / graph_meta["filename"]

        with open(graph_path, "rb") as f:
            graph = pickle.load(f)

        # Get ground truth (best anchor)
        test_quality = quality_df[quality_df["graph_id"] == graph_id]
        best_vertex = test_quality.loc[test_quality["tour_weight"].idxmin(), "vertex_id"]
        best_tour_weight = test_quality["tour_weight"].min()

        # Get predicted anchor
        test_stats = stats_df[stats_df["graph_id"] == graph_id]
        predictions = model.predict(test_stats[feature_cols].values)
        predicted_vertex = test_stats.iloc[np.argmax(predictions)]["vertex_id"]

        # Random anchor
        random_vertex = np.random.choice(list(graph.nodes()))

        # Test all three
        tour_best, weight_best = single_anchor(graph, start_vertex=int(best_vertex))
        tour_pred, weight_pred = single_anchor(graph, start_vertex=int(predicted_vertex))
        tour_rand, weight_rand = single_anchor(graph, start_vertex=int(random_vertex))
        tour_nn, weight_nn = nearest_neighbor(graph)

        # Calculate improvements
        improvement_pred = ((weight_rand - weight_pred) / weight_rand) * 100
        improvement_best = ((weight_rand - weight_best) / weight_rand) * 100

        results.append(
            {
                "graph_id": graph_id,
                "graph_type": graph_meta["type"],
                "best_vertex": int(best_vertex),
                "predicted_vertex": int(predicted_vertex),
                "random_vertex": int(random_vertex),
                "weight_best": weight_best,
                "weight_predicted": weight_pred,
                "weight_random": weight_rand,
                "weight_nearest_neighbor": weight_nn,
                "improvement_pred_vs_rand": improvement_pred,
                "improvement_best_vs_rand": improvement_best,
                "optimality_gap_pred": ((weight_pred - weight_best) / weight_best) * 100,
            }
        )

        print(f"Graph {graph_id} ({graph_meta['type']}):")
        print(f"  Best anchor tour:      {weight_best:.2f}")
        print(f"  Predicted anchor tour: {weight_pred:.2f} ({results[-1]['optimality_gap_pred']:.2f}% from best)")
        print(f"  Random anchor tour:    {weight_rand:.2f}")
        print(f"  Nearest neighbor:      {weight_nn:.2f}")
        print(f"  Predicted improvement: {improvement_pred:.2f}% vs random")
        print()

    # Save results
    results_df = pd.DataFrame(results)
    results_path = output_dir / "practical_validation_results.csv"
    results_df.to_csv(results_path, index=False)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Average improvement (predicted vs random): {results_df['improvement_pred_vs_rand'].mean():.2f}%")
    print(f"Average gap from best: {results_df['optimality_gap_pred'].mean():.2f}%")
    print(f"✅ Results saved to: {results_path}\n")

    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Comparison of tour weights
    ax = axes[0]
    x = np.arange(len(results_df))
    width = 0.2

    ax.bar(x - width * 1.5, results_df["weight_best"], width, label="Best", alpha=0.7)
    ax.bar(x - width / 2, results_df["weight_predicted"], width, label="Predicted", alpha=0.7)
    ax.bar(x + width / 2, results_df["weight_random"], width, label="Random", alpha=0.7)
    ax.bar(x + width * 1.5, results_df["weight_nearest_neighbor"], width, label="Nearest Neighbor", alpha=0.7)

    ax.set_ylabel("Tour Weight")
    ax.set_title("Tour Quality Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels([f"G{gid}" for gid in results_df["graph_id"]])
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # Improvement vs random
    ax = axes[1]
    ax.bar(x, results_df["improvement_pred_vs_rand"], alpha=0.7, color="green")
    ax.set_ylabel("Improvement (%)")
    ax.set_title("Predicted Anchor Improvement vs Random")
    ax.set_xticks(x)
    ax.set_xticklabels([f"G{gid}" for gid in results_df["graph_id"]])
    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plot_path = output_dir / "practical_validation.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"✅ Plot saved to: {plot_path}\n")

    plt.close()

    print(f"✅ All practical validation results saved to: {output_dir}")


if __name__ == "__main__":
    main()
