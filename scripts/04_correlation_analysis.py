"""
Phase 4: Correlation analysis.
Compute correlation between each edge statistic and anchor quality (percentile rank).
"""
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis.anchor_analysis import correlation_analysis


def main():
    """Run correlation analysis."""
    data_dir = Path(__file__).parent.parent / "data" / "anchor_analysis"
    output_dir = Path(__file__).parent.parent / "results" / "anchor_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("\nLoading data files...\n")

    stats_df = pd.read_csv(data_dir / "vertex_statistics.csv")
    quality_df = pd.read_csv(data_dir / "anchor_quality.csv")

    print(f"✅ Loaded {len(stats_df)} vertex statistics")
    print(f"✅ Loaded {len(quality_df)} anchor quality records")

    # Run correlation analysis
    print("\nComputing correlations...\n")

    corr_df = correlation_analysis(stats_df, quality_df)

    # Save results
    output_path = output_dir / "correlations.csv"
    corr_df.to_csv(output_path, index=False)

    print(f"✅ Correlation analysis complete")
    print(f"   Saved to: {output_path}\n")

    print("Top 10 most correlated features:")
    print(corr_df.head(10).to_string(index=False))

    # Create visualization
    print("\n\nCreating visualization...")

    fig, ax = plt.subplots(figsize=(10, 8))

    # Bar plot of absolute correlations
    plot_df = corr_df.head(15).copy()
    colors = ["green" if x > 0 else "red" for x in plot_df["correlation"]]

    ax.barh(range(len(plot_df)), plot_df["correlation"], color=colors, alpha=0.7)
    ax.set_yticks(range(len(plot_df)))
    ax.set_yticklabels(plot_df["feature"])
    ax.set_xlabel("Pearson Correlation with Anchor Quality (Percentile)")
    ax.set_title("Feature Correlation with Anchor Quality")
    ax.axvline(x=0, color="black", linestyle="-", linewidth=0.5)
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plot_path = output_dir / "correlations_plot.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"✅ Plot saved to: {plot_path}")

    plt.close()

    # Create scatter plots for top features
    print("\nCreating scatter plots for top 4 features...")

    merged = pd.merge(stats_df, quality_df, on=["graph_id", "vertex_id"])

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for plot_idx, (idx, row) in enumerate(corr_df.head(4).iterrows()):
        feature = row["feature"]
        corr = row["correlation"]

        ax = axes[plot_idx]
        ax.scatter(merged[feature], merged["percentile"], alpha=0.5, s=30)
        ax.set_xlabel(feature)
        ax.set_ylabel("Anchor Quality (Percentile)")
        ax.set_title(f"r = {corr:.3f}")
        ax.grid(alpha=0.3)

    plt.tight_layout()
    scatter_path = output_dir / "top_features_scatter.png"
    plt.savefig(scatter_path, dpi=150, bbox_inches="tight")
    print(f"✅ Scatter plots saved to: {scatter_path}")

    plt.close()

    print(f"\n✅ All results saved to: {output_dir}")


if __name__ == "__main__":
    main()
