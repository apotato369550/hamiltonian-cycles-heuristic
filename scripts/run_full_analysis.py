"""
Master script: Run all 8 phases of anchor statistics analysis.
"""
import subprocess
import sys
from pathlib import Path

def run_script(script_name, description):
    """Run a single analysis script."""
    script_path = Path(__file__).parent / script_name

    print("\n" + "=" * 70)
    print(f"PHASE: {description}")
    print("=" * 70)

    try:
        result = subprocess.run([sys.executable, str(script_path)], check=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed with error code {e.returncode}")
        return False

def main():
    """Run all phases."""
    print("\n" + "=" * 70)
    print("ANCHOR STATISTICS ANALYSIS - FULL PIPELINE")
    print("=" * 70)

    phases = [
        ("01_generate_test_graphs.py", "Phase 1: Generate Test Data (100 graphs)"),
        ("02_compute_anchor_quality.py", "Phase 2: Compute Anchor Quality"),
        ("03_extract_edge_statistics.py", "Phase 3: Extract Edge Statistics"),
        ("04_correlation_analysis.py", "Phase 4: Correlation Analysis"),
        ("05_simple_regression.py", "Phase 5: Simple Linear Regression"),
        ("06_decision_tree_analysis.py", "Phase 6: Decision Tree Analysis"),
        ("07_hypothesis_validation.py", "Phase 7: Hypothesis Validation"),
        ("08_practical_validation.py", "Phase 8: Practical Validation"),
    ]

    results = []

    for script, description in phases:
        success = run_script(script, description)
        results.append((description, success))

        if not success:
            print(f"\n⚠️  Stopping at {description}")
            break

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for description, success in results:
        status = "✅" if success else "❌"
        print(f"{status} {description}")

    print("=" * 70)

    all_passed = all(success for _, success in results)
    if all_passed:
        print("\n🎉 ALL PHASES COMPLETED SUCCESSFULLY")
        print("\nResults saved to: results/anchor_analysis/")
        print("\nKey files:")
        print("  - correlations.csv - Feature correlations with anchor quality")
        print("  - regression_results.txt - Model comparison and coefficients")
        print("  - tree_feature_importance.csv - Feature importance ranking")
        print("  - hypothesis_test_results.csv - Statistical hypothesis tests")
        print("  - practical_validation_results.csv - Real-world validation")
        print("\nVisualizations:")
        print("  - correlations_plot.png")
        print("  - model_comparison.png")
        print("  - decision_tree_visualization.png")
        print("  - hypothesis_validation.png")
        print("  - practical_validation.png")
    else:
        print("\n❌ PIPELINE INCOMPLETE - See errors above")

    print("=" * 70)

if __name__ == "__main__":
    main()
