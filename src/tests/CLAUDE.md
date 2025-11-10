# Test Suite

## Purpose
Comprehensive test coverage for TSP research platform. Validates correctness, performance, and edge cases across all phases.

**Total Tests**: 187
**Pass Rate**: 100% (validated 11-10-2025)

---

## Test Organization

```
tests/
├── CLAUDE.md                         # This file
├── __init__.py                       # Test package init
├── test_graph_generators.py          # Phase 1 tests (34 tests)
├── test_algorithms.py                # Phase 2 core tests (59 tests)
├── test_baseline_algorithms.py       # Phase 2 baselines (16 tests)
├── test_anchor_algorithms.py         # Phase 2 anchors (14 tests)
├── test_features.py                  # Phase 3 prompts 1-4 tests (34 tests)
├── test_features_extended.py         # Phase 3 prompts 5-8 tests (30 tests)
├── test_features_final.py            # Phase 3 prompts 9-12 tests (requires pandas/sklearn)
└── test_phase3_integration.py        # Phase 3 integration test (no ML dependencies)
```

---

## Phase 1: Graph Generation Tests (34 tests)

**File**: `test_graph_generators.py`

### Test Classes
- `TestEuclideanGenerator` (10 tests) - Euclidean graph generation and properties
- `TestMetricGenerator` (6 tests) - Metric graph strategies and verification
- `TestQuasiMetricGenerator` (3 tests) - Asymmetric graphs with triangle inequality
- `TestRandomGenerator` (7 tests) - Random graph generation and distributions
- `TestEdgeCases` (3 tests) - Small graphs, extreme parameters
- `TestConsistency` (2 tests) - Deterministic generation with seeds
- `TestPerformance` (3 tests) - Generation speed benchmarks

### Key Coverage
- ✓ Property verification (metricity, symmetry, Euclidean distances)
- ✓ Weight range validation
- ✓ Reproducibility with random seeds
- ✓ Edge cases (minimum size graphs, extreme weights)
- ✓ Performance characteristics

---

## Phase 2: Algorithm Benchmarking Tests (89 tests)

### Core Interface Tests (59 tests)

**File**: `test_algorithms.py`

**Test Classes**:
- `TestTourResult` (7 tests) - TourResult validation and constraints
- `TestAlgorithmMetadata` (3 tests) - Metadata and applicability
- `TestTSPAlgorithmInterface` (8 tests) - Abstract base class behavior
- `TestAlgorithmRegistry` (15 tests) - Registration, retrieval, filtering
- `TestRegisterDecorator` (2 tests) - Decorator functionality
- `TestValidateTour` (4 tests) - Tour structure validation
- `TestTourValidator` (3 tests) - Batch validation with caching
- `TestComputeTourWeight` (3 tests) - Weight computation
- `TestComputeTourStatistics` (3 tests) - Statistical metrics
- `TestOptimalityGap` (3 tests) - Gap computation
- `TestApproximationRatio` (3 tests) - Ratio computation
- `TestComputeTourProperties` (2 tests) - Tour properties
- `TestMetricsCalculator` (4 tests) - Caching metrics calculator
- `TestAlgorithmIntegration` (2 tests) - End-to-end workflows

### Baseline Algorithm Tests (16 tests)

**File**: `test_baseline_algorithms.py`

**Test Classes**:
- `TestNearestNeighbor` (5 tests) - NN random and best variants
- `TestGreedyEdge` (2 tests) - Greedy edge-picking algorithm
- `TestHeldKarp` (5 tests) - Exact solver correctness and constraints
- `TestAlgorithmRegistry` (4 tests) - Baseline registration verification

**Key Coverage**:
- ✓ Correctness on small known graphs
- ✓ Best-of-all beats single random start
- ✓ Tour validity for all algorithms
- ✓ Held-Karp optimality on solvable instances
- ✓ Size constraint enforcement (Held-Karp)
- ✓ Reproducibility with random seeds

### Anchor Algorithm Tests (14 tests)

**File**: `test_anchor_algorithms.py`

**Test Classes**:
- `TestSingleAnchor` (4 tests) - Single anchor correctness
- `TestBestAnchor` (4 tests) - Exhaustive anchor search
- `TestMultiAnchor` (4 tests) - Multi-anchor variants
- `TestAnchorRegistry` (2 tests) - Anchor algorithm registration

**Key Coverage**:
- ✓ Single anchor finds valid tours
- ✓ Different anchors produce different tours
- ✓ Best anchor beats or equals single anchor
- ✓ Multi-anchor variants work with various anchor counts
- ✓ Metadata tracking (anchor vertices, weights)

---

## Running Tests

### All Tests
```bash
python3 -m unittest discover -s src/tests -p "test_*.py" -v
```

### Phase 1 Only
```bash
python3 src/tests/test_graph_generators.py
```

### Phase 2 Only
```bash
python3 src/tests/test_algorithms.py
python3 src/tests/test_baseline_algorithms.py
python3 src/tests/test_anchor_algorithms.py
```

### Phase 3 Only
```bash
# Basic tests (prompts 1-8, no dependencies)
python3 -m unittest src.tests.test_features src.tests.test_features_extended

# Full tests (prompts 9-12, requires pandas/sklearn)
python3 -m unittest src.tests.test_features_final

# Integration test (prompts 1-9, no ML dependencies)
python3 src/tests/test_phase3_integration.py
```

### Single Test Class
```bash
python3 -m unittest src.tests.test_algorithms.TestAlgorithmRegistry -v
```

---

## Test Principles

### From Root CLAUDE.md
1. **Test Everything**: No untested code in production
2. **Edge Cases**: Test boundary conditions
3. **Reproducibility**: Use fixed seeds for deterministic tests
4. **Fail Fast**: Tests catch errors immediately

### Coverage Standards
- All public APIs tested
- Edge cases covered (small graphs, extreme values)
- Integration tests for workflows
- Performance benchmarks for scalability

---

## Future Test Additions

### Phase 2 Steps 5-8 (To Be Added)
- Benchmarking runner tests
- Batch system tests
- Statistical analysis tests
- Visualization generation tests

### Phase 3 (Feature Engineering) - COMPLETE (64 tests)

**Files**: `test_features.py`, `test_features_extended.py`

**Test Classes (Prompts 1-8)**:
- Base architecture tests (5 tests)
- Weight-based features (7 tests)
- Topological features (7 tests)
- MST-based features (6 tests)
- Neighborhood features (6 tests)
- Heuristic features (6 tests)
- Graph context features (5 tests)
- Feature analyzer tests (11 tests)
- Pipeline integration tests (6 tests)
- Edge cases (5 tests)

**Key Coverage**:
- ✓ All 6 feature extractors validated
- ✓ 93 features extracted from graphs
- ✓ Feature validation (NaN/Inf checks)
- ✓ Correlation and PCA analysis
- ✓ Edge cases (single vertex, uniform weights)
- ✓ Pipeline orchestration and caching

**Additional Testing (Prompts 9-12)**:
- `test_features_final.py` - Comprehensive tests for labeling, dataset pipeline, selection, transformation
- Requires: `pandas`, `scikit-learn` (optional dependencies)
- Alternative: `test_phase3_integration.py` validates prompts 1-9 without ML dependencies

### Phase 4 (Machine Learning)
- Model training tests
- Cross-validation tests
- Prediction accuracy tests

---

## References

- **Root Context**: `/CLAUDE.md`
- **Phase 1 Tests**: Validate graph generation (34 tests)
- **Phase 2 Tests**: Validate algorithms (89 tests)
- **Phase 3 Tests**: Validate feature engineering (64 tests)

---

**Test Suite Version**: 3.0 (Phase 1 + Phase 2 Steps 1-4 + Phase 3 Prompts 1-8)
**Last Updated**: 11-10-2025
**Status**: All tests passing (187/187)
**Note**: Phase 3 Prompts 9-12 tests require pandas/scikit-learn (optional)
