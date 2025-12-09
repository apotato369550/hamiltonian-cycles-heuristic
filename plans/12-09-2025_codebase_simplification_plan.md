# Plan: Codebase Simplification

**Created:** December 9, 2025
**Purpose:** Strip away unnecessary complexity and refocus the project

---

## Current State (The Problem)

The codebase has **70 Python files** with significant bloat:

| Module | Lines of Code | Verdict |
|--------|---------------|---------|
| `src/features/` | 4,760 | ARCHIVE - Only need ~100 lines for edge stats |
| `src/ml/` | 5,449 | ARCHIVE - Only need simple sklearn calls |
| `src/pipeline/` | 5,196 | ARCHIVE - Manual scripts are fine |
| `src/graph_generation/` | 4,118 | KEEP - Actually useful |
| `src/algorithms/` | 3,126 | KEEP - Core heuristics |
| `src/tests/` | 6,284 | PARTIAL - Keep relevant tests |

**~15,400 lines of code** in features/ml/pipeline that don't serve the core research question.

---

## Target State (The Solution)

A lean codebase focused on:
1. Generating graphs
2. Running anchor heuristics
3. Computing simple edge statistics
4. Basic analysis (correlation, regression)

---

## Phase 1: Create Archive Structure

**Goal:** Preserve work without cluttering the main codebase

**Tasks:**

1. Create archive directory structure:
```
archive/
├── v1_overengineered/
│   ├── src/
│   │   ├── features/       <- Move entire features/ here
│   │   ├── ml/             <- Move entire ml/ here
│   │   └── pipeline/       <- Move entire pipeline/ here
│   ├── tests/
│   │   ├── test_phase3_features.py
│   │   ├── test_phase4_ml.py
│   │   └── test_phase5_pipeline.py
│   ├── guides/             <- Move guides/ here
│   ├── config/             <- Move complex configs here
│   └── experiments/        <- Move experiment runner here
```

2. Add `archive/v1_overengineered/README.md` explaining what this was

**Why archive instead of delete:**
- You spent months on this - don't throw it away
- Might be useful reference later
- Shows progression of the project

---

## Phase 2: Clean Up Source Directory

**Goal:** `src/` contains only what's needed

**After cleanup, src/ should contain:**
```
src/
├── __init__.py
├── main.py                 <- Keep, but simplify
├── graph_generation/       <- KEEP AS-IS (4,118 lines - useful)
│   ├── __init__.py
│   ├── euclidean_generator.py
│   ├── metric_generator.py
│   ├── random_generator.py
│   ├── graph_instance.py
│   ├── verification.py
│   ├── storage.py
│   ├── batch_generator.py
│   ├── visualization.py
│   ├── collection_analysis.py
│   └── CLAUDE.md
├── algorithms/             <- KEEP AS-IS (3,126 lines - core heuristics)
│   ├── __init__.py
│   ├── base.py
│   ├── registry.py
│   ├── validation.py
│   ├── metrics.py
│   ├── nearest_neighbor.py
│   ├── greedy.py
│   ├── single_anchor.py
│   ├── best_anchor.py
│   ├── multi_anchor.py
│   ├── exact.py            <- Held-Karp for small graphs
│   └── CLAUDE.md
├── analysis/               <- NEW: Simple analysis utilities
│   ├── __init__.py
│   ├── edge_statistics.py  <- ~100 lines: compute vertex edge stats
│   └── anchor_analysis.py  <- ~150 lines: correlation, regression
└── tests/
    ├── __init__.py
    ├── test_graph_generators.py    <- KEEP
    └── test_phase2_algorithms.py   <- KEEP
```

**Files to REMOVE from src/ (move to archive):**
- `src/features/` (entire directory)
- `src/ml/` (entire directory)
- `src/pipeline/` (entire directory)
- `src/tests/test_phase3_features.py`
- `src/tests/test_phase4_ml.py`
- `src/tests/test_phase5_pipeline.py`

---

## Phase 3: Create Simple Analysis Module

**Goal:** Replace 15,000+ lines with ~250 focused lines

### File: `src/analysis/edge_statistics.py` (~100 lines)

```python
"""
Simple edge statistics for anchor analysis.
This is ALL we need for the core research question.
"""
import numpy as np
from typing import Dict, List, Any

def compute_vertex_edge_stats(graph, vertex: int) -> Dict[str, float]:
    """
    Compute edge statistics for a single vertex.

    Returns dict with:
    - sum_weight, mean_weight, median_weight
    - variance_weight, std_weight
    - min_weight, max_weight, range_weight
    - cv_weight (coefficient of variation)
    - min2_weight (second smallest)
    - anchor_edge_sum (min1 + min2)
    """
    # Get all edge weights from this vertex
    weights = [graph[vertex][neighbor]['weight']
               for neighbor in graph.neighbors(vertex)]
    weights = np.array(weights)
    sorted_weights = np.sort(weights)

    return {
        'sum_weight': np.sum(weights),
        'mean_weight': np.mean(weights),
        'median_weight': np.median(weights),
        'variance_weight': np.var(weights),
        'std_weight': np.std(weights),
        'min_weight': sorted_weights[0],
        'max_weight': sorted_weights[-1],
        'range_weight': sorted_weights[-1] - sorted_weights[0],
        'cv_weight': np.std(weights) / np.mean(weights) if np.mean(weights) > 0 else 0,
        'min2_weight': sorted_weights[1] if len(sorted_weights) > 1 else sorted_weights[0],
        'anchor_edge_sum': sorted_weights[0] + sorted_weights[1] if len(sorted_weights) > 1 else sorted_weights[0] * 2,
    }

def compute_all_vertex_stats(graph) -> List[Dict[str, Any]]:
    """Compute edge statistics for all vertices in a graph."""
    results = []
    for vertex in graph.nodes():
        stats = compute_vertex_edge_stats(graph, vertex)
        stats['vertex_id'] = vertex
        results.append(stats)
    return results
```

### File: `src/analysis/anchor_analysis.py` (~150 lines)

```python
"""
Simple analysis utilities for anchor quality prediction.
Uses pandas and sklearn - no complex abstractions.
"""
import pandas as pd
import numpy as np
from scipy import stats

def compute_anchor_quality(graph, algorithm_func) -> pd.DataFrame:
    """
    Run anchor algorithm from each vertex and return quality scores.
    """
    results = []
    for vertex in graph.nodes():
        tour, weight = algorithm_func(graph, start_vertex=vertex)
        results.append({
            'vertex_id': vertex,
            'tour_weight': weight,
        })

    df = pd.DataFrame(results)
    df['rank'] = df['tour_weight'].rank()
    df['percentile'] = df['tour_weight'].rank(pct=True) * 100
    return df

def correlation_analysis(stats_df: pd.DataFrame, quality_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute correlation between each statistic and anchor quality.
    """
    merged = pd.merge(stats_df, quality_df, on='vertex_id')

    stat_cols = [c for c in stats_df.columns if c != 'vertex_id']
    results = []

    for col in stat_cols:
        r, p = stats.pearsonr(merged[col], merged['percentile'])
        results.append({
            'feature': col,
            'correlation': r,
            'p_value': p,
            'abs_correlation': abs(r)
        })

    return pd.DataFrame(results).sort_values('abs_correlation', ascending=False)

def simple_regression(X: pd.DataFrame, y: pd.Series):
    """
    Train simple linear regression and return coefficients.
    """
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score, mean_squared_error

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    return {
        'model': model,
        'coefficients': dict(zip(X.columns, model.coef_)),
        'intercept': model.intercept_,
        'r2_train': model.score(X_train, y_train),
        'r2_test': r2_score(y_test, y_pred),
        'rmse_test': np.sqrt(mean_squared_error(y_test, y_pred))
    }
```

---

## Phase 4: Simplify Root Directory

**Goal:** Clean up clutter at project root

**Current root (messy):**
```
├── CHANGELOG.md
├── CLAUDE.md           <- 30,930 bytes (way too long)
├── README.md           <- 13,170 bytes (too long)
├── config/
├── data/
├── docs/
├── experiments/
├── guides/             <- Move to archive
├── plans/
├── references/
├── visualizations/
├── archive/
└── src/
```

**Target root (clean):**
```
├── README.md           <- Simplified (~200 lines)
├── CLAUDE.md           <- Simplified (~150 lines)
├── requirements.txt
├── src/
├── data/
├── results/            <- New: for analysis outputs
├── notebooks/          <- New: for Jupyter analysis
├── references/
├── archive/            <- All the old complex stuff
└── plans/              <- Keep for now
```

**Directories to move to archive:**
- `guides/` -> `archive/v1_overengineered/guides/`
- `experiments/` -> `archive/v1_overengineered/experiments/`
- `config/` -> `archive/v1_overengineered/config/`
- `docs/` -> `archive/v1_overengineered/docs/`

---

## Phase 5: Rewrite CLAUDE.md

**Goal:** Simple context file that reflects actual project scope

**New CLAUDE.md (~150 lines):**

```markdown
# Anchor-Based TSP Heuristic Research

## What This Project Is

Investigating what makes a vertex a good "anchor" (starting point) for TSP heuristics.

**Core Hypothesis:** High-weight, high-variance vertices make better anchors because:
1. Starting from a "heavy" vertex removes expensive edges from consideration early
2. High variance means some edges are much cheaper than average

## Project Structure

```
src/
├── graph_generation/   # Generate test graphs (Euclidean, metric, random)
├── algorithms/         # TSP heuristics (nearest-neighbor, anchor-based, greedy)
├── analysis/           # Simple edge statistics and correlation analysis
└── tests/              # Unit tests for graph generation and algorithms
```

## How to Run

1. Generate graphs: `python -m src.graph_generation.batch_generator`
2. Run analysis: `python notebooks/anchor_analysis.ipynb`

## Key Files

- `src/algorithms/single_anchor.py` - The anchor-based heuristic
- `src/algorithms/best_anchor.py` - Exhaustive anchor search (ground truth)
- `src/analysis/edge_statistics.py` - Compute vertex edge statistics
- `src/analysis/anchor_analysis.py` - Correlation and regression

## Dependencies

- numpy, scipy - Numerical computation
- networkx - Graph structures
- pandas - Data manipulation
- sklearn - Simple regression
- matplotlib - Visualization

## Archive

Previous over-engineered version is preserved in `archive/v1_overengineered/`.
This included complex ML pipelines, feature extractors, and orchestration
that weren't needed for the core research question.
```

---

## Phase 6: Rewrite README.md

**Goal:** Clear, honest project description

**New README.md (~100 lines):**

```markdown
# Anchor-Based Heuristics for TSP

A research project investigating what makes a vertex a good starting point
for greedy TSP heuristics.

## The Idea

The "anchor" heuristic fixes the two cheapest edges from a starting vertex,
then greedily completes the tour. Different starting vertices produce
different tour qualities.

**Research Question:** Can we predict which vertices will produce good tours
based on simple edge statistics (sum, variance, min, max)?

## Quick Start

```bash
# Setup
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run analysis
jupyter notebook notebooks/anchor_analysis.ipynb
```

## Project Status

- [x] Graph generation (Euclidean, metric, random)
- [x] Anchor heuristic implementation
- [ ] Edge statistics analysis
- [ ] Correlation with anchor quality
- [ ] Simple regression model

## Author

John Andre Yap - University of San Carlos
Started: May 2025
```

---

## Phase 7: Update .gitignore

**Add to .gitignore:**
```
# Archive (too large for git)
archive/v1_overengineered/

# Data and results
data/
results/

# Notebooks checkpoints
.ipynb_checkpoints/
```

---

## Execution Order

1. **Create archive structure** (Phase 1)
   - `mkdir -p archive/v1_overengineered/src`
   - Move directories

2. **Move files to archive** (Phase 2)
   - `mv src/features archive/v1_overengineered/src/`
   - `mv src/ml archive/v1_overengineered/src/`
   - `mv src/pipeline archive/v1_overengineered/src/`
   - Move test files
   - Move guides, experiments, config, docs

3. **Create new analysis module** (Phase 3)
   - `mkdir src/analysis`
   - Create `edge_statistics.py`
   - Create `anchor_analysis.py`

4. **Create new directories** (Phase 4)
   - `mkdir results`
   - `mkdir notebooks`

5. **Rewrite documentation** (Phase 5, 6)
   - Backup old CLAUDE.md and README.md to archive
   - Write new simplified versions

6. **Update .gitignore** (Phase 7)

7. **Commit changes**
   - `git add -A`
   - `git commit -m "Simplify codebase: archive over-engineered components"`

---

## What You Keep

- **Graph generation** - 4,118 lines of working code
- **Algorithms** - 3,126 lines of working heuristics
- **Core tests** - ~500 lines of relevant tests
- **Your notes** - my_notes.md stays

## What You Archive

- **Features system** - 4,760 lines
- **ML pipeline** - 5,449 lines
- **Orchestration** - 5,196 lines
- **Complex tests** - ~5,800 lines
- **Guides** - The metaprompts that drove the over-engineering

## What You Gain

- Clear project structure
- Focused codebase (~7,500 lines instead of ~23,000)
- Documentation that matches reality
- Mental clarity

---

## Final Notes

This simplification doesn't mean the archived work was wasted. You:
- Learned about ML pipelines, feature engineering, orchestration
- Have reference code if you need it later
- Can point to it as "exploration" in your thesis

But for answering your core question - "what makes a good anchor?" - you don't need any of it. You need graphs, algorithms, and basic statistics.

Keep it simple.
