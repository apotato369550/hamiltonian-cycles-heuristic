# Anchor-Based Heuristics for TSP

A research project investigating what makes a vertex a good starting point ("anchor") for greedy TSP heuristics.

## The Core Idea

The **anchor heuristic** works like this:
1. Choose a starting vertex (the "anchor")
2. Fix its two cheapest edges as entry and exit points in the tour
3. Greedily complete the tour from there

**Key observation:** Different starting vertices produce very different tour qualities.

**Research question:** Can we predict which vertices will produce good tours from simple edge statistics?

---

## Hypothesis

High-weight, high-variance vertices make better anchors:
- **High weight:** Removes expensive edges early, constraining the greedy process
- **High variance:** Despite high total weight, some edges are much cheaper than average

Example:
- Vertex A: edges {1, 1, 99, 99} — sum=200, variance=2400
- Vertex B: edges {45, 55, 60, 40} — sum=200, variance=56.5

Vertex A has higher variance. Despite both having the same total weight, A's cheap edges are exploitable by the anchor heuristic.

---

## Quick Start

### Installation

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Running Analysis

```python
from src.graph_generation import EuclideanGraphGenerator
from src.algorithms import get_algorithm
from src.analysis import compute_all_vertex_stats, compute_anchor_quality, correlation_analysis
import pandas as pd

# Generate a test graph
gen = EuclideanGraphGenerator(num_vertices=30, weight_range=(1, 100))
graph = gen.generate()

# Get the anchor algorithm
algorithm = get_algorithm('single_anchor')

# Compute anchor quality (which starting vertex is best?)
quality_df = compute_anchor_quality(graph, algorithm)

# Compute edge statistics for all vertices
stats_df = pd.DataFrame(compute_all_vertex_stats(graph))

# Find correlations
corr_df = correlation_analysis(stats_df, quality_df)
print(corr_df)
```

---

## Project Structure

```
src/
├── graph_generation/   # Generate test graphs
├── algorithms/         # TSP heuristics (nearest-neighbor, anchor-based, greedy, exact)
├── analysis/          # Simple statistics and correlation analysis
└── tests/             # Unit tests

results/               # Where analysis outputs go
notebooks/            # Jupyter notebooks for exploration
```

---

## Key Files

| File | Purpose |
|------|---------|
| `src/algorithms/single_anchor.py` | The anchor-based heuristic |
| `src/algorithms/best_anchor.py` | Exhaustive anchor search (ground truth) |
| `src/analysis/edge_statistics.py` | Compute vertex edge statistics |
| `src/analysis/anchor_analysis.py` | Correlation and regression utilities |

---

## Project Status

- [x] Graph generation (Euclidean, metric, random)
- [x] Algorithm implementations (anchor, nearest-neighbor, greedy, exact)
- [x] Edge statistics computation
- [ ] Correlation analysis (testing hypothesis)
- [ ] Regression modeling
- [ ] Practical validation

---

## Dependencies

- **numpy, scipy** — Numerical computation
- **networkx** — Graph structures and algorithms
- **pandas** — Data manipulation
- **scikit-learn** — Regression models
- **matplotlib, seaborn** — Visualization

See `requirements.txt` for exact versions.

---

## Archive

This project was previously much larger (23,000+ lines). The over-engineered components (feature extraction, ML pipelines, orchestration) have been archived in `archive/v1_overengineered/` for reference.

The current focus is on **directly answering the core research question** with simple, clear analysis.

---

## Contact

John Andre Yap — University of San Carlos

Started: May 2025
Last updated: December 9, 2025
