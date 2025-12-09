# Anchor-Based TSP Heuristic Research

**Simplified:** December 9, 2025

---

## What This Project Is

Investigating what makes a vertex a good "anchor" (starting point) for TSP heuristics.

**Core Hypothesis:** High-weight, high-variance vertices make better anchors because:
1. Starting from a "heavy" vertex removes expensive edges from consideration early
2. High variance means some edges are much cheaper than average
3. This combination allows greedy construction to find lower-cost tours

## Quick Answer to Your Question

Can we predict which vertices are good anchors from simple edge statistics (sum, mean, variance, std, min, max, coefficient of variation)?

**Status:** Testing this now with simple correlation analysis and linear regression.

---

## Project Structure

```
src/
├── graph_generation/    # Generate test graphs (Euclidean, metric, random)
│   ├── euclidean_generator.py
│   ├── metric_generator.py
│   ├── random_generator.py
│   ├── batch_generator.py
│   ├── storage.py
│   ├── verification.py
│   ├── visualization.py
│   └── collection_analysis.py
│
├── algorithms/          # TSP heuristics (your core implementations)
│   ├── base.py          # Algorithm interface
│   ├── registry.py      # Algorithm registry
│   ├── validation.py    # Tour validation
│   ├── metrics.py       # Tour quality metrics
│   ├── nearest_neighbor.py
│   ├── greedy.py
│   ├── single_anchor.py     # Your anchor heuristic
│   ├── best_anchor.py       # Exhaustive anchor search
│   ├── multi_anchor.py
│   └── exact.py             # Held-Karp for small graphs
│
├── analysis/            # NEW: Simple statistics and analysis
│   ├── edge_statistics.py   # Compute vertex edge stats
│   └── anchor_analysis.py   # Correlation and regression
│
└── tests/
    ├── test_graph_generators.py      # KEEP
    └── test_phase2_algorithms.py     # KEEP

results/                # Where analysis outputs go

notebooks/             # Jupyter notebooks for exploration

archive/v1_overengineered/  # Archived complex code (features/, ml/, pipeline/)
```

---

## How to Run Analysis

1. **Generate test graphs** (if needed):
   ```bash
   python -c "from src.graph_generation import EuclideanGraphGenerator; ..."
   ```

2. **Run analysis manually** or in a notebook:
   ```python
   from src.algorithms import get_algorithm
   from src.analysis import compute_all_vertex_stats, compute_anchor_quality, correlation_analysis

   # Load graph
   # Compute stats
   # Compute anchor quality
   # Run correlation analysis
   ```

3. **Results** go to `results/` directory

---

## Key Files

- `src/algorithms/single_anchor.py` - The anchor-based heuristic
- `src/algorithms/best_anchor.py` - Exhaustive search for ground truth
- `src/analysis/edge_statistics.py` - Compute vertex edge statistics
- `src/analysis/anchor_analysis.py` - Correlation and regression utilities

---

## Dependencies

```
numpy, scipy          # Numerical computation
networkx              # Graph structures
pandas                # Data manipulation
scikit-learn          # Simple regression
matplotlib, seaborn   # Visualization
```

See `requirements.txt` for exact versions.

---

## Archive

The complex over-engineered version (features system, ML pipelines, orchestration) has been archived in `archive/v1_overengineered/`.

This included:
- `src/features/` - 4,760 lines of feature extraction (not needed)
- `src/ml/` - 5,449 lines of ML pipeline (not needed)
- `src/pipeline/` - 5,196 lines of orchestration (not needed)
- Old guides and test files

You can reference this code if needed, but the project now focuses on **simple, direct analysis** of the core question.

---

## Research Timeline

- **May 2025:** Initial anchor heuristic idea
- **June 2025:** Key insight about high-weight, high-variance vertices
- **November 2025:** Overengineered with complex features/ML/pipeline
- **December 9, 2025:** Back to basics - testing the original hypothesis directly

---

## Success Criteria

1. Clear correlation coefficient between edge statistics and anchor quality
2. Answer to: "Does high total weight predict good anchors?" (YES/NO)
3. Answer to: "Does high variance predict good anchors?" (YES/NO)
4. Practical demonstration: predicted anchors perform better than random
5. Single research document ready for thesis chapter

---

## Notes for Future Work

- This simplified codebase is intentionally lean
- Focus on answering the core question before adding complexity
- If you need sophisticated analysis later, the archived code is still available
- Keep things simple - the best research is clear research
