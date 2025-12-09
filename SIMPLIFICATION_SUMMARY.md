# Codebase Simplification Summary

**Date:** December 9, 2025
**Action:** Archived over-engineered components and refocused project

---

## What Changed

### Before
- **23,000+ lines of code** across 70 Python files
- Complex modules: features (4,760 lines), ml (5,449 lines), pipeline (5,196 lines)
- Elaborate testing suite, guides, and orchestration
- Project sprawl across multiple concepts

### After
- **~9,700 lines of code** across 31 Python files
- Focused modules: graph_generation, algorithms, analysis
- Simple, direct testing
- Clear single research question

**Reduction:** 13,300 lines removed (58% less code)

---

## Directories Changed

| Directory | Action | Why |
|-----------|--------|-----|
| `src/features/` | → Archived | Complex feature extraction not needed |
| `src/ml/` | → Archived | ML pipeline replaces with simple sklearn calls |
| `src/pipeline/` | → Archived | Orchestration unnecessary for core question |
| `guides/` | → Archived | Metaprompts that drove over-engineering |
| `experiments/` | → Archived | Will create simple scripts instead |
| `config/` | → Archived | Manual configuration sufficient |
| `docs/` | → Archived | Documentation simplified in CLAUDE.md/README.md |
| `src/analysis/` | ✨ NEW | Simple statistics and analysis module |
| `results/` | ✨ NEW | Where analysis outputs go |
| `notebooks/` | ✨ NEW | Jupyter for exploration |

---

## New Analysis Module

Created `src/analysis/` with two simple utilities:

### `edge_statistics.py` (~100 lines)
Compute per-vertex edge statistics:
- sum_weight, mean_weight, median_weight
- variance_weight, std_weight
- min_weight, max_weight, range_weight
- coefficient_variation (std/mean)
- min2_weight, anchor_edge_sum

### `anchor_analysis.py` (~150 lines)
Simple analysis functions:
- `compute_anchor_quality()` - Run anchor heuristic from each vertex
- `correlation_analysis()` - Compute feature-target correlations
- `simple_regression()` - Train linear regression model

**Total new code:** ~250 lines

---

## What Stayed

✅ `src/graph_generation/` - 4,118 lines (working, essential)
✅ `src/algorithms/` - 3,126 lines (your heuristics, essential)
✅ `src/tests/test_graph_generators.py` - Essential tests
✅ `src/tests/test_phase2_algorithms.py` - Essential tests
✅ `my_notes.md` - Your research journey
✅ `requirements.txt` - Dependencies

---

## Documentation Updates

### CLAUDE.md
- **Before:** 30,930 bytes, 900+ lines describing complex pipeline
- **After:** 4,953 bytes, 151 lines focusing on research question
- New sections clarify what's in archive and why

### README.md
- **Before:** 13,170 bytes, generic project template
- **After:** 3,881 bytes, concrete research focus
- Clear hypothesis and quick-start code

### .gitignore
- Added `archive/v1_overengineered/` (too large for git)
- Added `data/` and `results/` directories

---

## File Counts

| Component | Before | After | Change |
|-----------|--------|-------|--------|
| Python files | 70 | 31 | -39 |
| Lines of code | 23,000+ | 9,700 | -13,300 (-58%) |
| Main directories | 12 | 6 | -6 |
| Test files | 5 | 2 | -3 |

---

## Archive Location

Everything archived in: `archive/v1_overengineered/`

If you need:
- **Feature extraction code:** See `archive/v1_overengineered/src/features/`
- **ML pipelines:** See `archive/v1_overengineered/src/ml/`
- **Orchestration patterns:** See `archive/v1_overengineered/src/pipeline/`
- **Original guides:** See `archive/v1_overengineered/guides/`

---

## Next Steps

1. ✅ **Simplification complete**
2. ⏭️ **Execute anchor statistics analysis plan** (Phase 1-8)
   - Generate test graphs
   - Compute anchor quality
   - Extract edge statistics
   - Run correlation analysis
   - Train simple regression
   - Validate hypotheses

---

## Key Takeaway

This simplification represents a deliberate refocus on the **core research question**:

> Can we predict which vertices make good anchors from simple edge statistics?

Not:
- Can we build a perfect ML system?
- Can we extract every possible feature?
- Can we orchestrate experiments elegantly?

Just:
- Is the hypothesis true?
- What statistics matter?
- How good is the prediction?

**Simpler codebase = clearer thinking = faster progress.**
