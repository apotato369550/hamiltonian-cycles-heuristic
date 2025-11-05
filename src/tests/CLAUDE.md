# Test Suite

## Purpose
Comprehensive test coverage for TSP research platform. Validates correctness, performance, and edge cases across all phases.

**Total Tests**: 123
**Pass Rate**: 100% (validated 11-05-2025)

---

## Test Organization

```
tests/
├── CLAUDE.md                         # This file
├── __init__.py                       # Test package init
├── test_graph_generators.py          # Phase 1 tests (34 tests)
├── test_algorithms.py                # Phase 2 core tests (59 tests)
├── test_baseline_algorithms.py       # Phase 2 baselines (16 tests)
└── test_anchor_algorithms.py         # Phase 2 anchors (14 tests)
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

### Phase 3 (Feature Engineering)
- Feature extraction tests
- Feature validation tests
- Normalization tests

### Phase 4 (Machine Learning)
- Model training tests
- Cross-validation tests
- Prediction accuracy tests

---

## References

- **Root Context**: `/CLAUDE.md`
- **Phase 1 Tests**: Validate graph generation (34 tests)
- **Phase 2 Tests**: Validate algorithms (89 tests)

---

**Test Suite Version**: 2.0 (Phase 1 + Phase 2 Steps 1-4)
**Last Updated**: 11-05-2025
**Status**: All tests passing (123/123)
