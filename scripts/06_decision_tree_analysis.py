"""
Phase 6: Decision tree analysis for interpretability.
Train shallow decision tree to see which features split first.
"""
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    """Train and visualize decision tree model."""
    data_dir = Path(__file__).parent.parent / "data" / "anchor_analysis"
    output_dir = Path(__file__).parent.parent / "results" / "anchor_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("\nLoading data files...\n")

    stats_df = pd.read_csv(data_dir / "vertex_statistics.csv")
    quality_df = pd.read_csv(data_dir / "anchor_quality.csv")

    # Merge
    merged = pd.merge(stats_df, quality_df, on=["graph_id", "vertex_id"])

    print(f"✅ Loaded {len(merged)} vertex-quality pairs\n")

    # Identify feature columns
    feature_cols = [
        c
        for c in stats_df.columns
        if c not in ["graph_id", "graph_type", "vertex_id"]
    ]

    X = merged[feature_cols]
    y = merged["percentile"]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train shallow decision tree
    print("Training decision tree (max_depth=5)...\n")

    dt = DecisionTreeRegressor(max_depth=5, random_state=42, min_samples_split=5)
    dt.fit(X_train, y_train)

    # Evaluate
    y_pred_train = dt.predict(X_train)
    y_pred_test = dt.predict(X_test)

    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))

    print(f"✅ Decision Tree Performance:")
    print(f"   R² (train): {r2_train:.4f}")
    print(f"   R² (test):  {r2_test:.4f}")
    print(f"   RMSE (test): {rmse_test:.4f}\n")

    # Feature importance
    importances = dt.feature_importances_
    importance_df = pd.DataFrame(
        {"feature": feature_cols, "importance": importances}
    ).sort_values("importance", ascending=False)

    print("Feature Importance Ranking:")
    print(importance_df.head(10).to_string(index=False))
    print()

    # Save importance
    importance_path = output_dir / "tree_feature_importance.csv"
    importance_df.to_csv(importance_path, index=False)
    print(f"✅ Feature importance saved to: {importance_path}\n")

    # Visualize tree
    print("Creating decision tree visualization...")

    fig, ax = plt.subplots(figsize=(20, 12))
    plot_tree(
        dt,
        feature_names=feature_cols,
        ax=ax,
        filled=True,
        rounded=True,
        fontsize=10,
    )
    plt.tight_layout()

    tree_path = output_dir / "decision_tree_visualization.png"
    plt.savefig(tree_path, dpi=150, bbox_inches="tight")
    print(f"✅ Tree visualization saved to: {tree_path}\n")

    plt.close()

    # Visualize feature importance
    fig, ax = plt.subplots(figsize=(10, 6))

    top_n = 12
    top_importance = importance_df.head(top_n)

    ax.barh(range(len(top_importance)), top_importance["importance"], alpha=0.7)
    ax.set_yticks(range(len(top_importance)))
    ax.set_yticklabels(top_importance["feature"])
    ax.set_xlabel("Feature Importance")
    ax.set_title("Decision Tree Feature Importance (Top 12)")
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    importance_plot_path = output_dir / "tree_feature_importance.png"
    plt.savefig(importance_plot_path, dpi=150, bbox_inches="tight")
    print(f"✅ Feature importance plot saved to: {importance_plot_path}\n")

    plt.close()

    # Save summary
    summary_path = output_dir / "tree_summary.txt"
    with open(summary_path, "w") as f:
        f.write("DECISION TREE ANALYSIS\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Tree Configuration:\n")
        f.write(f"  Max depth: 5\n")
        f.write(f"  Min samples split: 5\n")
        f.write(f"  Number of leaves: {dt.get_n_leaves()}\n\n")

        f.write(f"Performance:\n")
        f.write(f"  R² (train): {r2_train:.4f}\n")
        f.write(f"  R² (test): {r2_test:.4f}\n")
        f.write(f"  RMSE (test): {rmse_test:.4f}\n\n")

        f.write(f"Feature Importance Ranking:\n")
        for idx, row in importance_df.iterrows():
            f.write(f"  {row['feature']}: {row['importance']:.4f}\n")

    print(f"✅ Summary saved to: {summary_path}")
    print(f"\n✅ All decision tree results saved to: {output_dir}")


if __name__ == "__main__":
    main()
