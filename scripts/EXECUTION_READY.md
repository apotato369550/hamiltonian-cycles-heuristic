# Anchor Statistics Analysis - READY TO EXECUTE

**Status:** ✅ All scripts created and ready to run
**Date:** December 9, 2025
**Total Code:** 1,126 lines across 9 analysis scripts

---

## What Was Created

### 8 Analysis Phases (1,126 lines of code)

| Phase | Script | Lines | Purpose |
|-------|--------|-------|---------|
| 1 | `01_generate_test_graphs.py` | 86 | Generate 100 diverse test graphs |
| 2 | `02_compute_anchor_quality.py` | 95 | Run anchor heuristic from each vertex |
| 3 | `03_extract_edge_statistics.py` | 80 | Compute 11 edge statistics per vertex |
| 4 | `04_correlation_analysis.py` | 100 | Correlate statistics with anchor quality |
| 5 | `05_simple_regression.py` | 158 | Train regression models to predict quality |
| 6 | `06_decision_tree_analysis.py` | 147 | Identify most important features |
| 7 | `07_hypothesis_validation.py` | 199 | Test hypotheses with statistical tests |
| 8 | `08_practical_validation.py` | 177 | Validate predictions on new data |
| - | `run_full_analysis.py` | 84 | Master script to run all phases |

**Total:** 1,126 lines

### Supporting Documentation

- `scripts/README.md` - Comprehensive guide for each phase
- `EXECUTION_READY.md` - This file

---

## How to Run

### Option A: Run Everything

```bash
cd /home/jay/Desktop/Coding\ Stuff/hamiltonian-cycles-heuristic
python scripts/run_full_analysis.py
```

This runs all 8 phases sequentially and reports progress.

### Option B: Run Individual Phases

```bash
# Run each phase separately
python scripts/01_generate_test_graphs.py
python scripts/02_compute_anchor_quality.py
python scripts/03_extract_edge_statistics.py
python scripts/04_correlation_analysis.py
python scripts/05_simple_regression.py
python scripts/06_decision_tree_analysis.py
python scripts/07_hypothesis_validation.py
python scripts/08_practical_validation.py
```

---

## What You'll Get

### Data Files
- `data/anchor_analysis/graphs/` - 100 test graphs
- `data/anchor_analysis/anchor_quality.csv` - Ground truth
- `data/anchor_analysis/vertex_statistics.csv` - Edge statistics

### Results (in `results/anchor_analysis/`)
- `correlations.csv` - Which statistics correlate with anchor quality?
- `regression_results.txt` - Model comparison and coefficients
- `tree_feature_importance.csv` - Feature importance ranking
- `hypothesis_test_results.csv` - Statistical test results
- `practical_validation_results.csv` - Real-world validation

### Visualizations (PNG files in `results/anchor_analysis/`)
- `correlations_plot.png` - Top 15 feature correlations
- `top_features_scatter.png` - Scatter plots of top 4 features
- `model_comparison.png` - Regression model R² and RMSE
- `tree_feature_importance.png` - Decision tree importance
- `decision_tree_visualization.png` - Full tree structure
- `hypothesis_validation.png` - Hypothesis test results
- `practical_validation.png` - Prediction validation results

---

## What Each Phase Does

### Phase 1: Generate Test Graphs
Creates 100 diverse graphs to test on:
- 25 Euclidean (points in 2D space)
- 25 Metric (triangle inequality holds)
- 25 Random (no constraints)
- 25 Quasi-metric (asymmetric)

**Runtime:** ~1-2 minutes
**Output:** 100 pickle files + metadata JSON

---

### Phase 2: Compute Anchor Quality
For each of 100 graphs, runs single_anchor from every vertex.
Records which vertices make good anchors (ground truth).

**Runtime:** ~5-10 minutes (depends on graph size)
**Output:** `anchor_quality.csv` (thousands of rows)

---

### Phase 3: Extract Edge Statistics
Computes 11 simple statistics for each vertex:

```
sum_weight        (total of incident edges)
mean_weight       (average edge weight)
median_weight     (middle edge weight)
variance_weight   (spread of edge weights)
std_weight        (standard deviation)
min_weight        (shortest edge)
max_weight        (longest edge)
range_weight      (max - min)
cv_weight         (coefficient of variation)
min2_weight       (second shortest edge)
anchor_edge_sum   (sum of two shortest)
```

**Runtime:** ~2-3 minutes
**Output:** `vertex_statistics.csv` (thousands of rows)

---

### Phase 4: Correlation Analysis
Answers: Which statistics correlate with anchor quality?

Computes Pearson r for each feature vs percentile rank.

**Runtime:** ~30 seconds
**Output:** `correlations.csv` ranked by strength

---

### Phase 5: Simple Linear Regression
Trains 4 regression models:
1. Just sum_weight
2. Just variance_weight
3. Both together
4. All 11 features

Shows which features matter most.

**Runtime:** ~1 minute
**Output:** Model comparison + coefficients

---

### Phase 6: Decision Tree Analysis
Trains shallow decision tree for interpretability.

Shows: Which feature is the most important first split?

**Runtime:** ~1 minute
**Output:** Tree visualization + feature importance

---

### Phase 7: Hypothesis Validation
Tests 3 specific hypotheses with ANOVA:

**H1:** High total weight → better anchors?
**H2:** High variance → better anchors?
**H3:** High weight + high variance → best?

**Runtime:** ~1 minute
**Output:** Statistical test results with p-values

---

### Phase 8: Practical Validation
Tests prediction model on held-out graphs.

Compares:
- Best anchor (exhaustive search)
- Predicted anchor (from regression)
- Random anchor
- Nearest neighbor

**Runtime:** ~2-3 minutes
**Output:** Practical validation results CSV

---

## Total Runtime Estimate

- Phase 1: ~2 minutes
- Phase 2: ~8 minutes (could be longer)
- Phase 3: ~3 minutes
- Phases 4-8: ~5 minutes total

**Total: 15-20 minutes for full pipeline**

---

## What You're Testing

Your hypothesis from June 24:

> "High-weight, high-variance vertices make better anchors because:
> 1. High weight removes expensive edges early
> 2. High variance means some edges are cheap
> 3. This combination exploits greedy construction"

Expected results IF hypothesis is correct:
- sum_weight shows positive correlation with anchor quality
- variance_weight shows positive correlation
- Both together show even stronger correlation
- Predicted anchors beat random on new graphs

---

## After Running

1. Check `results/anchor_analysis/correlations.csv`
   - Are any |r| values > 0.3?
   - Is sum_weight positive?
   - Is variance_weight positive?

2. Check `results/anchor_analysis/hypothesis_test_results.csv`
   - H1 p-value < 0.05? (weight matters?)
   - H2 p-value < 0.05? (variance matters?)
   - H3 improvement > 0? (both better?)

3. Check `results/anchor_analysis/practical_validation_results.csv`
   - Average improvement vs random > 0?
   - How close to best anchor?

4. Look at visualizations
   - Do patterns support hypothesis?
   - Any surprising results?

---

## Important Notes

✅ **No test scripts run** - You handle that part
✅ **Simple, focused code** - No over-engineering
✅ **All results saved** - To `results/anchor_analysis/`
✅ **Clear documentation** - Each script is readable
✅ **Reproducible** - Using fixed random seeds

---

## Next Steps After Execution

1. Review all results in `results/anchor_analysis/`
2. Check `scripts/README.md` for interpretation guide
3. Write up findings
4. Decide: hypothesis validated or refuted?
5. Plan next steps if needed

---

## Questions to Answer

After running, you'll be able to answer:

1. Do edge statistics predict anchor quality? (R² = ?)
2. Which statistic matters most? (Feature importance ranking)
3. Does high weight help? (H1 p-value)
4. Does high variance help? (H2 p-value)
5. Do both together help? (H3 improvement %)
6. Can we predict anchors in practice? (Improvement vs random)

---

**You're ready to run. Go test your hypothesis!**
