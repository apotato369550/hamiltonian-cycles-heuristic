# Directives for Claude Instances

This file contains important context and directives for Claude Code instances working on this TSP graph generation project. Last updated: 10-30-2025

---

## Project Overview

**Project Name:** Hamiltonian Cycles Heuristic - TSP Graph Generation System

**Purpose:** Generate diverse graph instances for Traveling Salesman Problem (TSP) research

**Architecture:** Modular Python package with:
- Multiple graph generators (Euclidean, Metric, Random)
- Graph verification system
- Batch generation pipeline
- Visualization utilities
- Comprehensive test suite (34 tests, all passing)

---

## Critical Technical Principles

### 1. Euclidean Property Must Be Preserved

**PRINCIPLE:** Edge weights in Euclidean graphs MUST equal geometric distances from coordinates.

**Context:** In previous work (10-29-2025), the system was using weight scaling which broke this property.

**Correct Approach:**
- Scale COORDINATES, not weights
- Coordinate scaling multiplies all distances by a constant factor
- This preserves the Euclidean property and geometric relationships

**Implementation:**
- See `src/graph_generation/euclidean_generator.py` - `_scale_coordinates()` method
- Uses centroid-based scaling
- Handles edge cases (coincident points, identical distances)

**Limitation to Remember:**
- Coordinate scaling can match the MAXIMUM distance in a target range
- Cannot independently set MINIMUM distance (mathematical limitation)
- For narrow weight ranges, the minimum may not equal target_min

### 2. Quasi-Metrics Require Directional Understanding

**PRINCIPLE:** Quasi-metrics (asymmetric metrics) satisfy triangle inequality ONLY for forward paths.

**Constraint:** `d(x,z) ≤ d(x,y) + d(y,z)` for all x,y,z

**What NOT to Check:**
- `d(j,k) ≤ d(i,j) + d(i,k)` (requires going backwards)
- Not all six permutations of triangle inequalities

**Current Verifier Issue:**
- `src/graph_generation/verification.py` checks all six permutations
- Causes false positives for quasi-metric graphs
- Workaround: Use metricity_score >= 0.9 instead of passed=True
- TODO: Update `_check_triplet()` to handle asymmetric graphs correctly

**When Working on Verification:**
- Remember that asymmetric graphs need special handling
- A matrix[i][j] can differ from matrix[j][i]
- Only check paths that follow directed edges

### 3. MST vs Completion Strategies for Metric Graphs

**MST Strategy:**
- Generates tree edges with specified weights
- Computes shortest paths (which SUM tree edges)
- Results in WIDE distribution of final weights
- **USE FOR:** Normal metric graphs, when distribution spread is acceptable
- **AVOID:** Narrow weight ranges (creates very wide output distribution)

**Completion Strategy:**
- Samples ALL edge weights directly from specified range
- Uses Floyd-Warshall only to REDUCE weights (no summing)
- Keeps weights WITHIN original range
- **USE FOR:** Narrow weight ranges, controlled distributions
- **BEST FOR:** Quasi-metric generation with narrow ranges

**Example:**
- MST with range (10.0, 10.01) produces std dev ~14.6 ❌
- Completion with range (10.0, 10.01) produces std dev ~0.05 ✅

---

## Current Known Issues

### ~~Issue #1: Verifier Checks Invalid Constraints for Asymmetric Graphs~~ ✅ FIXED (10-30-2025)

**Status:** RESOLVED

**Solution Implemented:**
- Added `symmetric: bool = True` parameter to `verify_metricity()` and `_check_triplet()`
- Auto-detects symmetry when parameter is None
- For asymmetric graphs, only checks valid forward-path constraints
- See `docs/10-30-2025_change.md` for details

**Usage:**
```python
# For asymmetric/quasi-metric graphs
result = verifier.verify_metricity(matrix, symmetric=False)

# For symmetric graphs or auto-detect (default)
result = verifier.verify_metricity(matrix)  # Auto-detects
```

---

## Test Suite Status

**All 34 Tests Passing:** ✅

**Test Coverage:**
- 10 Euclidean generator tests
- 6 Metric generator tests
- 3 Quasi-metric generator tests
- 7 Random generator tests
- 3 Edge case tests
- 2 Consistency tests
- 3 Performance benchmarks

**Recent Test Changes:**

**10-30-2025:**
1. `test_metricity()` (QuasiMetric) - Now uses `symmetric=False` parameter and expects `passed=True` (line 253)

**10-29-2025:**
1. `test_weight_scaling()` - Changed to only verify max distance (line 76)
2. `test_very_narrow_weight_range()` - Now uses completion strategy (line 344)
3. `test_metricity()` (QuasiMetric) - Initially used metricity_score >= 0.9 (superseded by 10-30 fix)

**When Modifying Tests:**
- Run full test suite: `python src/tests/test_graph_generators.py`
- Verify all 34 tests pass before committing
- Add documentation explaining any changes to test expectations

---

## File Structure

```
src/
  graph_generation/
    __init__.py
    graph_instance.py        - Core data structure
    euclidean_generator.py   - Euclidean TSP graphs
    metric_generator.py      - Metric/quasi-metric graphs
    random_generator.py      - Random baseline graphs
    verification.py          - Property verification
    storage.py              - Persistence
    batch_generator.py      - Batch pipeline
    visualization.py        - Visualization utilities
    collection_analysis.py  - Collection analysis
  tests/
    test_graph_generators.py - Comprehensive test suite
  main.py                    - Demo script

docs/
  10-29-2025_change.md      - Bug fixes: Euclidean scaling, quasi-metrics, test updates
  10-30-2025_change.md      - Asymmetric verification & storage query fixes
  README.md                 - Documentation index

config/
  example_batch_config.yaml - Example configuration

CLAUDE.md                   - This file
```

---

## Common Tasks and How to Approach Them

### Task: Fix a Graph Generator Bug

1. **Identify the bug:**
   - Run tests: `python src/tests/test_graph_generators.py`
   - Check which test(s) fail
   - Read the test to understand expected behavior

2. **Understand the principle:**
   - Is it about Euclidean property? → See "Critical Principle #1"
   - Is it about triangle inequality? → See "Critical Principle #2"
   - Is it about weight ranges? → See "Critical Principle #3"

3. **Check past solutions:**
   - Review `docs/10-29-2025_change.md` for similar issues
   - Look at implementation in relevant generator file

4. **Test your fix:**
   - Run specific test: `python -m pytest src/tests/test_graph_generators.py::TestClass::test_name`
   - Run full suite: `python src/tests/test_graph_generators.py`
   - Verify no regressions

### Task: Improve the Verifier

1. **Current State:**
   - Checks all six permutations of triangle inequalities
   - Works fine for symmetric graphs
   - Over-checks for asymmetric graphs

2. **Proposed Improvement:**
   - Add `symmetric=True` parameter to `verify_metricity()`
   - For symmetric: keep current behavior
   - For asymmetric: only check forward path constraints

3. **Testing:**
   - Update `test_metricity()` for QuasiMetric to use `passed=True` once fixed
   - Should still pass with same random_seed=42

### Task: Add a New Generator Type

1. **Choose your strategy:**
   - Based on constraints needed (Euclidean? Metric? Random?)
   - Consider: weight range, distribution shape, symmetry

2. **Implement the generator:**
   - Create new class in appropriate file or new file
   - Follow the pattern of existing generators
   - Implement `generate()` method returning (adjacency_matrix, optional_metadata)

3. **Add tests:**
   - Create test class in `test_graph_generators.py`
   - Test: basic generation, determinism (same seed = same graph), properties
   - Add edge cases (single vertex, identical weights, etc.)
   - Run full suite and verify all 34 tests still pass

4. **Verify properties:**
   - Use `GraphVerifier` to check properties
   - Document any expected violations (e.g., non-metric random graphs)

---

## Environment Setup

**Python Version:** 3.8+ (tested with venv)

**Key Dependencies:**
- numpy - Numerical computations
- scipy - Scientific algorithms
- matplotlib - Visualization
- pyyaml - Configuration
- pytest - Testing

**Installation:**
```bash
python -m venv venv
venv/Scripts/activate
pip install -r requirements.txt
```

**Note:** On MSYS2/MinGW environments, you may need to use conda or standard Python from python.org due to wheel compatibility issues.

---

## Performance Considerations

**Generation Times (from benchmarks):**
- Euclidean 100-vertex: ~0.5ms
- Metric 100-vertex: ~2ms (MST strategy) or ~5ms (completion)
- Random 100-vertex: ~0.3ms
- Batch of 100 graphs: ~0.5 seconds

**Memory:**
- 100-vertex graph: ~80KB (dense matrix)
- Batch of 1000 graphs: ~80MB

**Verification:**
- Fast mode (sampling): ~10ms for 1000 triplets
- Exhaustive mode: O(n³) - ~6 seconds for 100 vertices

---

## Debugging Guide

**Print Debug Info:**
```python
from graph_generation.verification import GraphVerifier

verifier = GraphVerifier(fast_mode=False)
result = verifier.verify_metricity(matrix)
print(f"Score: {result.details['metricity_score']}")
print(f"Violations: {result.details['violations']}")
print(f"Errors: {result.errors[:5]}")  # First 5 errors
```

**Check if Matrix is Metric:**
```python
# All triangle inequalities should pass
passed_all = result.details['violations'] == 0
passed_some = result.details['metricity_score'] >= 0.95
```

**Check Euclidean Property:**
```python
matrix, coords = generate_euclidean_graph(...)
result = verifier.verify_euclidean_distances(matrix, coords)
print(f"Is Euclidean: {result.passed}")
print(f"Mismatches: {result.details['mismatch_count']}")
```

---

## Communication Guidelines

**When Creating Issues/TODOs:**
- Reference the change log: "See docs/10-29-2025_change.md for context"
- Cite the principle violated: "Violates Critical Principle #2 about quasi-metrics"
- Link to relevant code: Include file paths and line numbers

**When Documenting Changes:**
- Create new entry in `docs/` with format: `MM-dd-yyyy_change.md`
- Update this file (CLAUDE.md) with new principles or known issues
- Add comments to code explaining WHY, not just WHAT

---

## Questions to Ask Yourself

1. **Does this preserve the Euclidean property?**
   - If working with Euclidean graphs, answer must be YES
   - If scaling weights, you're doing it WRONG

2. **Does this assume symmetry?**
   - For asymmetric graphs (quasi-metrics), verify only forward-path constraints
   - Don't check all six permutations

3. **What's the distribution impact?**
   - MST strategy creates wide distributions (path sums)
   - Completion strategy keeps narrow distributions
   - Choose appropriate strategy for your use case

4. **Have the tests been run?**
   - All 34 tests should pass
   - Document why if you intentionally change test expectations

---

## Final Notes

This project has reached a stable state after 10-29-2025 bug fixes. The focus should now be on:

1. **Enhancing the verifier** to properly handle asymmetric graphs
2. **Expanding generator types** (e.g., geometric layouts, real-world TSP instances)
3. **Improving visualization** (network plots, heat maps, etc.)
4. **Scaling to larger graphs** (optimization for n > 1000 vertices)

All work should reference the critical principles, particularly around Euclidean property preservation and quasi-metric constraints.

---

**Document Version:** 1.0
**Last Updated:** 10-29-2025
**Maintained By:** Claude Code instances
