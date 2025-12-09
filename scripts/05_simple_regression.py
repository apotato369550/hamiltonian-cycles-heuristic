"""
Phase 5: Simple linear regression models.
Test which features best predict anchor quality.
"""
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis.anchor_analysis import simple_regression


def main():
    """Train and evaluate simple regression models."""
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

    y = merged["percentile"]

    # Test different models
    models = [
        ("sum_weight_only", ["sum_weight"]),
        ("variance_weight_only", ["variance_weight"]),
        ("sum_and_variance", ["sum_weight", "variance_weight"]),
        ("all_features", feature_cols),
    ]

    results = []

    print("Training models...\n")

    for model_name, features in models:
        print(f"  {model_name}:")
        print(f"    Features: {features}")

        X = merged[features]

        try:
            result = simple_regression(X, y)

            results.append(
                {
                    "model": model_name,
                    "features": ",".join(features),
                    "num_features": len(features),
                    "r2_train": result["r2_train"],
                    "r2_test": result["r2_test"],
                    "rmse_test": result["rmse_test"],
                    "coefficients": result["coefficients"],
                    "intercept": result["intercept"],
                }
            )

            print(f"    R² (train): {result['r2_train']:.4f}")
            print(f"    R² (test):  {result['r2_test']:.4f}")
            print(f"    RMSE (test): {result['rmse_test']:.4f}\n")

        except Exception as e:
            print(f"    ❌ Failed: {e}\n")

    # Create comparison table
    comparison_df = pd.DataFrame(
        [
            {
                "Model": r["model"],
                "Num Features": r["num_features"],
                "R² Train": f"{r['r2_train']:.4f}",
                "R² Test": f"{r['r2_test']:.4f}",
                "RMSE Test": f"{r['rmse_test']:.4f}",
            }
            for r in results
        ]
    )

    print("=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)
    print(comparison_df.to_string(index=False))
    print("=" * 60)

    # Save results
    results_path = output_dir / "regression_results.txt"
    with open(results_path, "w") as f:
        f.write("MODEL COMPARISON\n")
        f.write("=" * 60 + "\n")
        f.write(comparison_df.to_string(index=False))
        f.write("\n" + "=" * 60 + "\n\n")

        # Write detailed results
        for r in results:
            f.write(f"Model: {r['model']}\n")
            f.write(f"Features: {r['features']}\n")
            f.write(f"R² (train): {r['r2_train']:.4f}\n")
            f.write(f"R² (test): {r['r2_test']:.4f}\n")
            f.write(f"RMSE (test): {r['rmse_test']:.4f}\n")
            f.write(f"Intercept: {r['intercept']:.4f}\n")
            f.write("Coefficients:\n")
            for feature, coef in r["coefficients"].items():
                f.write(f"  {feature}: {coef:.6f}\n")
            f.write("\n")

    print(f"\n✅ Results saved to: {results_path}")

    # Visualize model comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    model_names = comparison_df["Model"].values
    r2_test = [float(x) for x in comparison_df["R² Test"].values]
    rmse_test = [float(x) for x in comparison_df["RMSE Test"].values]

    ax1.bar(range(len(model_names)), r2_test, alpha=0.7, color="steelblue")
    ax1.set_xticks(range(len(model_names)))
    ax1.set_xticklabels(model_names, rotation=45, ha="right")
    ax1.set_ylabel("R² (Test Set)")
    ax1.set_title("Model R² Comparison")
    ax1.grid(axis="y", alpha=0.3)

    ax2.bar(range(len(model_names)), rmse_test, alpha=0.7, color="coral")
    ax2.set_xticks(range(len(model_names)))
    ax2.set_xticklabels(model_names, rotation=45, ha="right")
    ax2.set_ylabel("RMSE (Test Set)")
    ax2.set_title("Model RMSE Comparison")
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plot_path = output_dir / "model_comparison.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"✅ Plot saved to: {plot_path}")

    plt.close()

    print(f"\n✅ All regression results saved to: {output_dir}")


if __name__ == "__main__":
    main()
