# Plan: Back to Basics - Anchor Statistics Analysis

**Created:** December 9, 2025
**Purpose:** Steer the project back to investigating what makes a good anchor using SIMPLE edge statistics

---

## The Core Question

**What vertex-level edge statistics predict whether a vertex will be a good anchor?**

Specifically, we want to test Jay's hypothesis:
- High total weight vertices tend to be better anchors (removes expensive edges from the equation early)
- High variance vertices may be even better (they have some very cheap edges despite high total weight)

---

## What We're NOT Doing

- NOT using MST features, topological features, centrality measures, etc.
- NOT building complex ML pipelines with 50+ features
- NOT over-engineering

We are ONLY analyzing edge statistics for each vertex:
- Sum (total weight)
- Mean
- Median
- Mode (if applicable)
- Variance
- Standard deviation
- Min edge weight
- Max edge weight
- Range (max - min)
- Coefficient of variation (std/mean)

That's it. 10-ish features. All directly computed from a vertex's incident edges.

---

## Phase 1: Generate Test Data

**Goal:** Create a diverse set of graphs to test on

**Tasks:**
1. Generate 100 graphs total:
   - 25 Euclidean (random points in 2D space)
   - 25 Metric (triangle inequality holds)
   - 25 Random (no constraints)
   - 25 Quasi-metric (asymmetric, if working)

2. Each graph should have:
   - 20-50 vertices (small enough for exhaustive anchor testing)
   - Complete (all pairs connected)
   - Varied weight distributions

3. Use existing graph generation from `src/graph_generation/`

**Output:** 100 graphs saved to `data/anchor_analysis/graphs/`

---

## Phase 2: Compute Anchor Quality (Ground Truth)

**Goal:** For each graph, determine which vertices are good/bad anchors

**Tasks:**
1. For each graph, for each vertex:
   - Run single-anchor heuristic starting from that vertex
   - Record the resulting tour weight

2. Rank vertices by tour quality (best = rank 1)

3. Store results:
   - `graph_id, vertex_id, tour_weight, rank, percentile`

**Output:** `data/anchor_analysis/anchor_quality.csv`

**Note:** This uses existing algorithm code from `src/algorithms/`

---

## Phase 3: Extract SIMPLE Edge Statistics

**Goal:** For each vertex, compute ONLY edge-based statistics

**Tasks:**
1. For each graph, for each vertex, compute:
   ```
   - sum_weight: sum of all incident edge weights
   - mean_weight: average edge weight
   - median_weight: median edge weight
   - variance_weight: variance of edge weights
   - std_weight: standard deviation
   - min_weight: smallest edge (nearest neighbor distance)
   - max_weight: largest edge (farthest vertex)
   - range_weight: max - min
   - cv_weight: coefficient of variation (std/mean)
   - min2_weight: second smallest edge
   - min1_plus_min2: sum of two cheapest edges (anchor edges)
   ```

2. For asymmetric graphs, compute separately for incoming/outgoing

**Output:** `data/anchor_analysis/vertex_statistics.csv`

---

## Phase 4: Correlation Analysis

**Goal:** Which statistics correlate with anchor quality?

**Tasks:**
1. Merge `anchor_quality.csv` and `vertex_statistics.csv`

2. Compute Pearson correlation between each statistic and anchor quality (percentile rank)

3. Create correlation matrix visualization

4. Identify:
   - Which features have |r| > 0.3?
   - Does high total weight correlate with good anchor quality?
   - Does high variance correlate with good anchor quality?
   - Is there a combination (high weight + high variance) that's even better?

**Output:**
- Correlation coefficients table
- Scatter plots: each statistic vs anchor quality
- `results/anchor_analysis/correlation_report.md`

---

## Phase 5: Simple Linear Regression

**Goal:** Can we predict anchor quality from edge statistics?

**Tasks:**
1. Split data: 80% train, 20% test (stratify by graph type)

2. Train simple linear regression models:
   - Model A: sum_weight only
   - Model B: variance_weight only
   - Model C: sum_weight + variance_weight
   - Model D: all statistics

3. Evaluate using R-squared and RMSE

4. Extract and interpret coefficients:
   - Which features have positive/negative coefficients?
   - Do the coefficients support the hypothesis?

**Output:**
- Model comparison table
- Coefficient interpretation
- `results/anchor_analysis/regression_report.md`

---

## Phase 6: Decision Tree Analysis

**Goal:** Validate findings with interpretable tree-based model

**Tasks:**
1. Train a shallow decision tree (max depth 3-5)

2. Visualize the tree:
   - What's the first split? (Most important feature)
   - What thresholds does it use?

3. Extract feature importance ranking

4. Compare to linear regression findings:
   - Do both methods agree on important features?

**Output:**
- Decision tree visualization
- Feature importance comparison

---

## Phase 7: Hypothesis Validation

**Goal:** Directly test Jay's hypothesis

**Hypothesis 1:** High total weight vertices are better anchors
- Group vertices by total weight quartile
- Compare average anchor quality across quartiles
- Statistical test (ANOVA or t-test)

**Hypothesis 2:** High variance vertices are better anchors
- Group vertices by variance quartile
- Compare average anchor quality across quartiles
- Statistical test

**Hypothesis 3:** High weight + high variance is best
- Create 2x2 grouping: low/high weight x low/high variance
- Compare average anchor quality in each cell
- Does the high-high cell outperform others?

**Output:**
- Statistical test results
- Boxplots by group
- Clear YES/NO answer to each hypothesis
- `results/anchor_analysis/hypothesis_test_results.md`

---

## Phase 8: Practical Validation

**Goal:** Does predicting anchors improve tour quality?

**Tasks:**
1. For each test graph:
   - Compute the "predicted best anchor" using the regression model
   - Run anchor heuristic from predicted best
   - Compare to random anchor, low-weight anchor, and true best anchor

2. Measure improvement:
   - Predicted vs random: how much better?
   - Predicted vs best: how close?

3. Also compare to Kruskal's greedy (the benchmark Jay mentioned)

**Output:**
- Performance comparison table
- `results/anchor_analysis/practical_validation.md`

---

## Implementation Notes

### What to use from existing code:
- `src/graph_generation/` - Graph generators (already built)
- `src/algorithms/single_anchor.py` - Single anchor heuristic (already built)
- `src/algorithms/` - Other algorithms for comparison (already built)

### What to write new:
- Simple script to compute edge statistics (maybe 50 lines of Python)
- Analysis notebook or script (straightforward pandas/sklearn)

### What to IGNORE:
- The complex feature extraction in `src/features/` (overkill for this)
- The ML pipeline in `src/ml/` (too complex for our simple analysis)
- The orchestration in `src/pipeline/` (manual is fine for this)

---

## Success Criteria

1. Clear answer to: "Does high total weight predict good anchors?" (with correlation coefficient)
2. Clear answer to: "Does high variance predict good anchors?" (with correlation coefficient)
3. Clear answer to: "Does combining them help?" (with regression results)
4. Practical demonstration: predicted anchors perform X% better than random
5. Single document summarizing findings (could become thesis chapter)

---

## Estimated Effort

- Phase 1: 30 minutes (graph generation is built)
- Phase 2: 1 hour (anchor quality computation)
- Phase 3: 30 minutes (simple statistics)
- Phase 4: 1-2 hours (correlation analysis)
- Phase 5: 1-2 hours (regression)
- Phase 6: 1 hour (decision tree)
- Phase 7: 2 hours (hypothesis testing)
- Phase 8: 1-2 hours (practical validation)

**Total: 1-2 focused days of work**

---

## How to Proceed

1. Create `experiments/anchor_statistics_analysis.py` script
2. Or create `notebooks/anchor_statistics_analysis.ipynb` notebook
3. Run phases sequentially, documenting as you go
4. Results go in `results/anchor_analysis/`

The existing infrastructure is overkill. We just need:
- NumPy for statistics
- Pandas for data manipulation
- Matplotlib/Seaborn for visualization
- Scikit-learn for simple regression

---

## Final Note

Jay - your intuition from June was good. The project got complicated because Claude (previous instances) kept adding complexity. But your core question is simple:

**Do high-weight, high-variance vertices make good anchors?**

Let's answer that directly.
