# Anchor Statistics Analysis Scripts

Complete pipeline for analyzing what makes a vertex a good anchor using simple edge statistics.

---

## Quick Start

Run the entire analysis pipeline:

```bash
python scripts/run_full_analysis.py
```

Or run individual phases:

```bash
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

## What Each Script Does

### Phase 1: Generate Test Graphs
**File:** `01_generate_test_graphs.py`

Generates 100 diverse test graphs:
- 25 Euclidean graphs (random points in 2D space)
- 25 Metric graphs (triangle inequality holds)
- 25 Random graphs (no constraints)
- 25 Quasi-metric graphs (asymmetric)

Each graph has 20-50 vertices and edge weights in range [1.0, 100.0].

**Output:** `data/anchor_analysis/graphs/`

---

### Phase 2: Compute Anchor Quality
**File:** `02_compute_anchor_quality.py`

For each graph, runs the single_anchor heuristic from every vertex as a starting point.

Records:
- Which vertex was the starting point
- The resulting tour weight
- Quality ranking (percentile)

This is the "ground truth" for what makes a good anchor.

**Output:** `data/anchor_analysis/anchor_quality.csv`

---

### Phase 3: Extract Edge Statistics
**File:** `03_extract_edge_statistics.py`

For each vertex in each graph, computes simple edge statistics:

| Statistic | Meaning |
|-----------|---------|
| sum_weight | Total of all incident edges |
| mean_weight | Average edge weight |
| median_weight | Middle edge weight |
| variance_weight | Variance of edge weights |
| std_weight | Standard deviation |
| min_weight | Shortest edge (nearest neighbor) |
| max_weight | Longest edge (farthest vertex) |
| range_weight | max - min |
| cv_weight | Coefficient of variation (std/mean) |
| min2_weight | Second shortest edge |
| anchor_edge_sum | Sum of two shortest edges |

**Output:** `data/anchor_analysis/vertex_statistics.csv`

---

### Phase 4: Correlation Analysis
**File:** `04_correlation_analysis.py`

Computes Pearson correlation between each statistic and anchor quality.

Tests: Which features most strongly predict whether a vertex is a good anchor?

**Output:**
- `correlations.csv` - Feature correlations ranked
- `correlations_plot.png` - Bar chart of top 15 correlations
- `top_features_scatter.png` - Scatter plots of top 4 features

---

### Phase 5: Simple Linear Regression
**File:** `05_simple_regression.py`

Trains four regression models to predict anchor quality:

1. **sum_weight_only** - Just total edge weight
2. **variance_weight_only** - Just variance
3. **sum_and_variance** - Both together
4. **all_features** - All 11 statistics

Compares R² and RMSE to see which features matter most.

**Output:**
- `regression_results.txt` - Model comparison and coefficients
- `model_comparison.png` - R² and RMSE comparison chart

---

### Phase 6: Decision Tree Analysis
**File:** `06_decision_tree_analysis.py`

Trains a shallow decision tree (max depth 5) for interpretability.

Shows:
- Which feature is the first split (most important)?
- Feature importance ranking
- Visual tree structure

**Output:**
- `tree_feature_importance.csv` - Feature importance scores
- `decision_tree_visualization.png` - The tree structure
- `tree_feature_importance.png` - Importance bar chart
- `tree_summary.txt` - Detailed tree analysis

---

### Phase 7: Hypothesis Validation
**File:** `07_hypothesis_validation.py`

Tests three specific hypotheses with statistical tests:

**H1: High total weight → better anchors?**
- Groups vertices into weight quartiles
- ANOVA test to see if groups differ significantly

**H2: High variance → better anchors?**
- Groups vertices into variance quartiles
- ANOVA test

**H3: High weight + high variance → best anchors?**
- 2x2 grouping: low/high weight × low/high variance
- Tests if the "high-high" group is best

**Output:**
- `hypothesis_test_results.csv` - Statistical test results
- `hypothesis_validation.png` - Visual comparison of groups

---

### Phase 8: Practical Validation
**File:** `08_practical_validation.py`

Tests the prediction model on held-out graphs:

Compares:
- Best anchor (from exhaustive search)
- Predicted anchor (from regression model)
- Random anchor
- Nearest neighbor algorithm

Shows: Does predicting anchors actually improve tour quality?

**Output:**
- `practical_validation_results.csv` - Detailed results per graph
- `practical_validation.png` - Tour weight and improvement comparison

---

## All Output Files

All results saved to `results/anchor_analysis/`:

### Data Files
- `correlations.csv` - Feature correlations
- `regression_results.txt` - Model comparison
- `tree_feature_importance.csv` - Tree feature importance
- `hypothesis_test_results.csv` - Hypothesis test results
- `practical_validation_results.csv` - Practical validation results

### Visualizations
- `correlations_plot.png` - Top 15 feature correlations
- `top_features_scatter.png` - Scatter plots of top 4 features
- `model_comparison.png` - Regression model comparison
- `tree_feature_importance.png` - Tree feature importance bar chart
- `decision_tree_visualization.png` - Full tree structure
- `hypothesis_validation.png` - Hypothesis test visualizations
- `practical_validation.png` - Practical validation results

---

## Data Files Generated

During execution, the pipeline creates:

### Input Data
- `data/anchor_analysis/graphs/` - 100 test graphs (pickle files)
- `data/anchor_analysis/graphs/graphs_metadata.json` - Graph metadata

### Intermediate Data
- `data/anchor_analysis/anchor_quality.csv` - Ground truth anchor quality
- `data/anchor_analysis/vertex_statistics.csv` - Edge statistics for all vertices

---

## Interpreting Results

### Key Questions to Answer

1. **Do edge statistics predict anchor quality?**
   - Look at `correlations.csv` - are any |r| > 0.3?
   - Look at `regression_results.txt` - what's the R² value?

2. **Which statistics matter most?**
   - Look at `tree_feature_importance.csv` - what's ranked first?
   - Look at regression coefficients - which have largest magnitude?

3. **Does high weight make good anchors?**
   - Look at `hypothesis_test_results.csv` for H1
   - What's the p-value? (< 0.05 = significant)

4. **Does high variance make good anchors?**
   - Look at `hypothesis_test_results.csv` for H2
   - What's the p-value?

5. **Can we predict good anchors in practice?**
   - Look at `practical_validation_results.csv`
   - How much improvement vs random anchor?

---

## Troubleshooting

### "Module not found: src.algorithms"
Make sure you're running from the project root:
```bash
cd /path/to/hamiltonian-cycles-heuristic
python scripts/run_full_analysis.py
```

### "Graph generation failed"
Check that graph generators are installed:
```bash
pip install -r requirements.txt
```

### Out of memory during Phase 2
Reduce the number of graphs by editing the configs in `01_generate_test_graphs.py`.

---

## Understanding the Research

This analysis tests whether the hypothesis from June 24, 2025 is true:

> "High-weight, high-variance vertices make better anchors because they have some very cheap edges despite high total weight, which exploits the greedy construction process."

The pipeline systematically:
1. Generates diverse test cases
2. Measures anchor quality empirically
3. Extracts simple statistics
4. Tests statistical relationships
5. Validates predictions on new data

If the hypothesis is correct, you should see:
- Positive correlation between total weight and anchor quality
- Positive correlation between variance and anchor quality
- Both combined perform even better
- Predictions outperform random on new graphs
