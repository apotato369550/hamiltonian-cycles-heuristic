# Test Suite

## Purpose
Comprehensive test coverage for TSP research platform. Validates correctness, performance, and edge cases across all phases.

**Total Tests**: 234+
**Pass Rate**: 100% (validated 11-13-2025)
**Organization**: One consolidated file per phase + integration tests

---

## Test Organization

**Updated 11-13-2025:** Tests consolidated to one file per phase for better maintainability.

```
tests/
├── CLAUDE.md                         # This file
├── __init__.py                       # Test package init
├── test_graph_generators.py          # Phase 1 (34 tests) ✅ No changes
├── test_phase2_algorithms.py         # Phase 2 (89 tests) ⭐ NEW - Consolidated
├── test_phase3_features.py           # Phase 3 (111 tests) ⭐ NEW - Consolidated
└── test_phase3_integration.py        # Phase 3 integration (quick smoke test)
```

**Legacy files removed** (11-13-2025):
- ~~test_algorithms.py~~ → Merged into test_phase2_algorithms.py
- ~~test_baseline_algorithms.py~~ → Merged into test_phase2_algorithms.py
- ~~test_anchor_algorithms.py~~ → Merged into test_phase2_algorithms.py
- ~~test_features.py~~ → Merged into test_phase3_features.py
- ~~test_features_extended.py~~ → Merged into test_phase3_features.py
- ~~test_features_final.py~~ → Merged into test_phase3_features.py

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

**File**: `test_phase2_algorithms.py` (consolidated 11-13-2025)

**Test Organization**:
- Core Interface Tests (59 tests) - TourResult, registry, validation, metrics
- Baseline Algorithm Tests (16 tests) - Nearest Neighbor, Greedy, Held-Karp
- Anchor Algorithm Tests (14 tests) - Single Anchor (v1/v2), Best Anchor, Multi-Anchor

**Test Classes**:

### Core Interface (59 tests)
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

### Baseline Algorithms (16 tests)
- `TestNearestNeighbor` (5 tests) - NN random and best variants
- `TestGreedyEdge` (2 tests) - Greedy edge-picking algorithm
- `TestHeldKarp` (5 tests) - Exact solver correctness and constraints
- `TestAlgorithmRegistry` (4 tests) - Baseline registration verification

### Anchor Algorithms (14 tests)
- `TestSingleAnchor` (4 tests) - Single anchor v1 and v2 correctness
- `TestBestAnchor` (4 tests) - Exhaustive anchor search
- `TestMultiAnchor` (4 tests) - Multi-anchor variants
- `TestAnchorRegistry` (2 tests) - Anchor algorithm registration

**Key Coverage**:
- ✓ All algorithm interfaces and contracts
- ✓ Tour validity for all algorithms
- ✓ Registry auto-registration (as of 11-13-2025)
- ✓ Baseline correctness and optimality
- ✓ Anchor-based heuristics and metadata
- ✓ Reproducibility with random seeds

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

### Phase 2 Only (Updated 11-13-2025)
```bash
# Consolidated file (89 tests)
python3 src/tests/test_phase2_algorithms.py

# Or via unittest
python3 -m unittest src.tests.test_phase2_algorithms -v
```

### Phase 3 Only (Updated 11-13-2025)
```bash
# Consolidated file (111 tests, requires pandas/sklearn)
python3 src/tests/test_phase3_features.py

# Or via unittest
python3 -m unittest src.tests.test_phase3_features -v

# Integration test (quick smoke test, no ML dependencies)
python3 src/tests/test_phase3_integration.py
```

### Single Test Class
```bash
# Example: Test algorithm registry
python3 -m unittest src.tests.test_phase2_algorithms.TestAlgorithmRegistry -v

# Example: Test weight features
python3 -m unittest src.tests.test_phase3_features.TestWeightFeatureExtractor -v
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

### Phase 3 (Feature Engineering) - COMPLETE (111 tests)

**File**: `test_phase3_features.py` (consolidated 11-13-2025)

**Test Organization**:
- Prompts 1-4 Tests (34 tests) - Base architecture, weight, topological, MST features
- Prompts 5-8 Tests (30 tests) - Neighborhood, heuristic, graph context, analyzer
- Prompts 9-12 Tests (47 tests) - Labeling, dataset pipeline, selection, transformation

**Test Classes (Prompts 1-12)**:
- Base architecture (5 tests)
- Weight-based features (7 tests)
- Topological features (7 tests)
- MST-based features (6 tests)
- Neighborhood features (6 tests)
- Heuristic features (6 tests)
- Graph context features (5 tests)
- Feature analyzer (11 tests)
- Anchor quality labeling (10 tests)
- Feature dataset pipeline (9 tests)
- Feature selection (11 tests)
- Feature transformation (17 tests)
- Pipeline integration (6 tests)
- Edge cases (5 tests)

**Key Coverage**:
- ✓ All 6 feature extractors (93 features total)
- ✓ Feature validation (NaN/Inf checks)
- ✓ Correlation and PCA analysis
- ✓ Anchor quality labeling (5 strategies)
- ✓ End-to-end dataset pipeline
- ✓ Feature selection methods
- ✓ Feature transformations
- ✓ Edge cases (single vertex, uniform weights)

**Dependencies**:
- Prompts 1-9: No dependencies required
- Prompts 10-12: Requires `pandas` and `scikit-learn`
- Alternative: `test_phase3_integration.py` for quick validation without ML dependencies

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

**Test Suite Version**: 4.0 (Consolidated test organization)
**Last Updated**: 11-13-2025
**Status**: All tests passing (234+/234+)
**Note**: Phase 3 Prompts 10-12 require pandas/scikit-learn (optional)

**Changelog**:
- v4.0 (11-13-2025): Consolidated tests to one file per phase, algorithm auto-registration fix
- v3.0 (11-10-2025): Added Phase 3 Prompts 1-8 tests
- v2.0 (11-05-2025): Added Phase 2 tests
- v1.0 (10-29-2025): Initial Phase 1 tests
