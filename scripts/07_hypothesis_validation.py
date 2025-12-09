"""
Phase 7: Hypothesis validation using statistical tests.
Test:
  1. Do high-weight vertices make better anchors?
  2. Do high-variance vertices make better anchors?
  3. Do high-weight + high-variance vertices make best anchors?
"""
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats as sp_stats
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    """Validate research hypotheses."""
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

    results = []

    # Hypothesis 1: High total weight → better anchors
    print("=" * 60)
    print("HYPOTHESIS 1: High Total Weight → Better Anchors")
    print("=" * 60)

    merged["weight_quartile"] = pd.qcut(
        merged["sum_weight"], q=4, labels=["Q1 (Low)", "Q2", "Q3", "Q4 (High)"]
    )

    h1_results = merged.groupby("weight_quartile")["percentile"].agg(
        ["mean", "std", "count"]
    )
    print(h1_results)

    # ANOVA test
    groups = [group["percentile"].values for name, group in merged.groupby("weight_quartile")]
    f_stat, p_value = sp_stats.f_oneway(*groups)

    print(f"\nANOVA Test:")
    print(f"  F-statistic: {f_stat:.4f}")
    print(f"  P-value: {p_value:.6f}")

    if p_value < 0.05:
        print(f"  ✅ SIGNIFICANT: Total weight matters (p < 0.05)")
    else:
        print(f"  ❌ NOT SIGNIFICANT: Total weight doesn't matter (p >= 0.05)")

    results.append(
        {
            "hypothesis": "High Weight → Better Anchors",
            "test": "ANOVA",
            "f_statistic": f_stat,
            "p_value": p_value,
            "significant": p_value < 0.05,
        }
    )

    # Hypothesis 2: High variance → better anchors
    print("\n" + "=" * 60)
    print("HYPOTHESIS 2: High Variance → Better Anchors")
    print("=" * 60)

    merged["variance_quartile"] = pd.qcut(
        merged["variance_weight"], q=4, labels=["Q1 (Low)", "Q2", "Q3", "Q4 (High)"]
    )

    h2_results = merged.groupby("variance_quartile")["percentile"].agg(
        ["mean", "std", "count"]
    )
    print(h2_results)

    # ANOVA test
    groups = [group["percentile"].values for name, group in merged.groupby("variance_quartile")]
    f_stat, p_value = sp_stats.f_oneway(*groups)

    print(f"\nANOVA Test:")
    print(f"  F-statistic: {f_stat:.4f}")
    print(f"  P-value: {p_value:.6f}")

    if p_value < 0.05:
        print(f"  ✅ SIGNIFICANT: Variance matters (p < 0.05)")
    else:
        print(f"  ❌ NOT SIGNIFICANT: Variance doesn't matter (p >= 0.05)")

    results.append(
        {
            "hypothesis": "High Variance → Better Anchors",
            "test": "ANOVA",
            "f_statistic": f_stat,
            "p_value": p_value,
            "significant": p_value < 0.05,
        }
    )

    # Hypothesis 3: High weight + high variance → best anchors
    print("\n" + "=" * 60)
    print("HYPOTHESIS 3: High Weight + High Variance → Best Anchors")
    print("=" * 60)

    merged["weight_group"] = merged["sum_weight"] > merged["sum_weight"].median()
    merged["variance_group"] = merged["variance_weight"] > merged["variance_weight"].median()

    h3_results = merged.groupby(["weight_group", "variance_group"])["percentile"].agg(
        ["mean", "std", "count"]
    )

    h3_results.index = [
        "Low Weight, Low Variance",
        "Low Weight, High Variance",
        "High Weight, Low Variance",
        "High Weight, High Variance",
    ]

    print(h3_results)

    best_cell = h3_results.loc["High Weight, High Variance", "mean"]
    worst_cell = h3_results.loc["Low Weight, Low Variance", "mean"]
    improvement = ((best_cell - worst_cell) / worst_cell) * 100

    print(f"\nImprovement (High-High vs Low-Low): {improvement:.2f}%")

    if improvement > 0:
        print(f"  ✅ High weight + high variance vertices ARE better anchors")
    else:
        print(f"  ❌ High weight + high variance vertices are NOT better anchors")

    results.append(
        {
            "hypothesis": "High Weight + High Variance → Best Anchors",
            "test": "Group Comparison",
            "improvement_percent": improvement,
            "significant": improvement > 0,
        }
    )

    # Save results
    results_df = pd.DataFrame(results)
    results_path = output_dir / "hypothesis_test_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\n✅ Results saved to: {results_path}")

    # Create visualizations
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # H1: Weight quartiles
    ax = axes[0]
    h1_results["mean"].plot(kind="bar", ax=ax, color="steelblue", alpha=0.7)
    ax.set_ylabel("Mean Anchor Quality (Percentile)")
    ax.set_title("H1: Weight Quartiles")
    ax.set_xlabel("Weight Quartile")
    ax.grid(axis="y", alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    # H2: Variance quartiles
    ax = axes[1]
    h2_results["mean"].plot(kind="bar", ax=ax, color="coral", alpha=0.7)
    ax.set_ylabel("Mean Anchor Quality (Percentile)")
    ax.set_title("H2: Variance Quartiles")
    ax.set_xlabel("Variance Quartile")
    ax.grid(axis="y", alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    # H3: 2x2 grouping
    ax = axes[2]
    h3_results["mean"].plot(kind="bar", ax=ax, color="green", alpha=0.7)
    ax.set_ylabel("Mean Anchor Quality (Percentile)")
    ax.set_title("H3: Weight + Variance Combination")
    ax.set_xlabel("Group")
    ax.grid(axis="y", alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

    plt.tight_layout()
    plot_path = output_dir / "hypothesis_validation.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"✅ Plot saved to: {plot_path}\n")

    plt.close()

    print(f"✅ All hypothesis validation results saved to: {output_dir}")


if __name__ == "__main__":
    main()
