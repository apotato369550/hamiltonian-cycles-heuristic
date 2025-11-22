# Test Suite

## Purpose
Comprehensive test coverage for TSP research platform. Validates correctness, performance, and edge cases across all phases.

**Total Tests**: 375+
**Pass Rate**: 100% (validated 11-22-2025)
**Organization**: One consolidated file per phase for better maintainability

---

## **IMPORTANT: Keeping Tests Current**

**⚠️ CRITICAL REMINDER:** Whenever you work on a new phase or perform a significant refactor, you MUST:

1. **Create or update test scripts** for the modified/new code
2. **Run all existing tests** to ensure no regressions
3. **Update this CLAUDE.md** to document the new tests
4. **Consolidate tests** into one file per phase (don't scatter tests across multiple files)
5. **Remove obsolete tests** that no longer apply

**Test-Driven Development:**
- Write tests BEFORE or DURING implementation, not after
- Each new feature should have corresponding tests
- Aim for high coverage of edge cases and error conditions
- Document any known limitations or untested scenarios

**Quality Gate:** All tests must pass before marking a phase as complete.

---

## Test Organization

**Updated 11-22-2025:** Tests consolidated to one file per phase. Phase 5 tests added.

```
tests/
├── CLAUDE.md                         # This file
├── __init__.py                       # Test package init
├── test_graph_generators.py          # Phase 1 (34 tests) ✅ No changes
├── test_phase2_algorithms.py         # Phase 2 (89 tests) ⭐ Consolidated
├── test_phase3_features.py           # Phase 3 (111 tests) ⭐ Consolidated
├── test_phase4_ml.py                 # Phase 4 (96 tests) ⭐ Prompts 1-8
└── test_phase5_pipeline.py           # Phase 5 (45 tests) ⭐ NEW - Prompts 1-4
```

**Legacy files removed** (11-13-2025):
- ~~test_algorithms.py~~ → Merged into test_phase2_algorithms.py
- ~~test_baseline_algorithms.py~~ → Merged into test_phase2_algorithms.py
- ~~test_anchor_algorithms.py~~ → Merged into test_phase2_algorithms.py
- ~~test_features.py~~ → Merged into test_phase3_features.py
- ~~test_features_extended.py~~ → Merged into test_phase3_features.py
- ~~test_features_final.py~~ → Merged into test_phase3_features.py
- ~~test_phase3_integration.py~~ → Removed (11-17-2025) to reduce confusion

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

### Phase 3 Only (Updated 11-17-2025)
```bash
# Consolidated file (111 tests, requires pandas/sklearn)
python3 src/tests/test_phase3_features.py

# Or via unittest
python3 -m unittest src.tests.test_phase3_features -v
```

**Note:** Phase 3 tests require pandas and scikit-learn for full coverage (Prompts 10-12).
Install with: `pip install pandas scikit-learn`

### Phase 4 Only (NEW - 11-17-2025)
```bash
# Requires pandas and scikit-learn
python3 src/tests/test_phase4_ml.py

# Or via unittest
python3 -m unittest src.tests.test_phase4_ml -v
```

**Note:** Phase 4 tests require pandas and scikit-learn.
Install with: `pip install pandas scikit-learn`

### Single Test Class
```bash
# Example: Test algorithm registry
python3 -m unittest src.tests.test_phase2_algorithms.TestAlgorithmRegistry -v

# Example: Test weight features
python3 -m unittest src.tests.test_phase3_features.TestWeightFeatureExtractor -v

# Example: Test linear regression
python3 -m unittest src.tests.test_phase4_ml.TestLinearRegressionModel -v
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
- Prompts 1-9: Core dependencies only (numpy, scipy)
- Prompts 10-12: Requires `pandas` and `scikit-learn` for full testing

---

## Phase 4: Machine Learning Tests (Prompts 1-4)

**File**: `test_phase4_ml.py` (NEW - Added 11-17-2025)

**Status**: Prompts 1-4 implemented and tested

### Test Classes (5 classes, comprehensive coverage)

**Dataset Preparation Tests** (`TestDatasetPreparator` - 7 tests):
- Preparator initialization
- Missing value handling (mean/median imputation, removal)
- Constant feature removal
- Outlier handling (clipping, removal)
- Metadata extraction

**Train/Test Splitting Tests** (`TestTrainTestSplitter` - 6 tests):
- Random split (baseline)
- Graph-based split (no graph in multiple sets)
- Stratified graph split (balanced by graph type)
- Graph-type holdout (train on some types, test on others)
- Size-based holdout (train on small, test on large)
- Split summary generation

**Linear Regression Tests** (`TestLinearRegressionModel` - 9 tests):
- OLS fitting and prediction
- Ridge regression (L2 regularization)
- Lasso regression (L1 regularization)
- ElasticNet (L1 + L2 regularization)
- Coefficient extraction
- Feature importance extraction
- Model evaluation (R², MAE, RMSE)
- Diagnostic information (residuals)
- Error handling (predict before fit)

**Tree Model Tests** (`TestTreeBasedModel` - 5 tests):
- Decision tree fitting and prediction
- Random forest ensemble
- Gradient boosting
- Feature importance extraction
- Non-linearity handling (vs linear models)

**Integration Tests** (`TestModelIntegration` - 1 test):
- Complete pipeline: data prep → split → train → evaluate
- Tests both linear and tree models
- Validates end-to-end workflow

**Total Phase 4 Tests**: 28 tests (Prompts 1-4 only)

**Key Coverage**:
- ✓ Dataset preparation (missing values, outliers, constant features)
- ✓ 5 splitting strategies (random, graph-based, stratified, holdout x2)
- ✓ 4 linear models (OLS, Ridge, Lasso, ElasticNet)
- ✓ 3 tree models (Decision Tree, Random Forest, Gradient Boosting)
- ✓ Model evaluation metrics (R², MAE, RMSE)
- ✓ Feature importance extraction
- ✓ End-to-end integration

**Dependencies**:
- Requires: pandas, scikit-learn
- Install with: `pip install pandas scikit-learn`

---

## Phase 5: Pipeline Integration Tests (Prompts 1-4)

**File**: `test_phase5_pipeline.py` (NEW - Added 11-22-2025)

**Status**: Prompts 1-4 implemented and tested

### Test Classes (9 classes, 45 tests total)

**Pipeline Orchestration Tests** (`TestPipelineStage`, `TestPipelineOrchestrator` - 9 tests):
- Stage initialization and execution
- Input validation
- Stage failure handling
- Multi-stage pipeline execution
- Pipeline stops on failure
- Manifest generation

**Configuration Tests** (`TestExperimentConfig`, `TestConfigValidation` - 10 tests):
- Config initialization and serialization
- YAML load/save roundtrip
- Validation of all config sections
- Error catching (missing name, invalid types, bad ratios)
- Template generation

**Experiment Tracking Tests** (`TestExperimentTracker`, `TestExperimentRegistry` - 11 tests):
- Tracker initialization and directory creation
- Experiment lifecycle (pending → running → completed/failed)
- Metadata saving
- Registry persistence
- Querying by status and name
- Experiment ID generation
- Summary statistics

**Reproducibility Tests** (`TestSeedManager`, `TestEnvironmentInfo`, `TestReproducibilityManager` - 15 tests):
- Seed manager initialization
- Global seed setting (Python random, NumPy)
- Stage-specific seed derivation
- Graph/model seed generation
- Environment capture
- Git commit tracking
- Environment verification

**Total Phase 5 Tests**: 45 tests (Prompts 1-4 complete)

**Key Coverage**:
- ✓ Pipeline orchestration with resumability
- ✓ YAML configuration management and validation
- ✓ Experiment tracking and metadata
- ✓ Experiment registry with querying
- ✓ Deterministic seed propagation
- ✓ Environment tracking (Python, packages, OS)
- ✓ Git versioning and reproducibility checks

**Dependencies**:
- Core dependencies only (no ML packages required)
- Uses Python stdlib + numpy

**Run Phase 5 Tests**:
```bash
python3 -m unittest src.tests.test_phase5_pipeline -v
```

---

## References

- **Root Context**: `/CLAUDE.md`
- **Phase 1 Tests**: Validate graph generation (34 tests)
- **Phase 2 Tests**: Validate algorithms (89 tests)
- **Phase 3 Tests**: Validate feature engineering (111 tests)
- **Phase 4 Tests**: Validate ML models (96 tests for Prompts 1-8)
- **Phase 5 Tests**: Validate pipeline integration (45 tests for Prompts 1-4)

---

**Test Suite Version**: 6.0 (Phase 5 Prompts 1-4 added)
**Last Updated**: 11-22-2025
**Status**: All tests passing (375+/375+)
**Note**: Phase 3 and Phase 4 require pandas/scikit-learn

**Changelog**:
- v6.0 (11-22-2025): Added Phase 5 Prompts 1-4 tests (45 tests) - Pipeline integration
- v5.0 (11-17-2025): Added Phase 4 Prompts 1-8 tests (96 tests), fixed test bugs
- v4.1 (11-17-2025): Removed test_phase3_integration.py to reduce confusion, labeling bug fixes
- v4.0 (11-13-2025): Consolidated tests to one file per phase, algorithm auto-registration fix
- v3.0 (11-10-2025): Added Phase 3 Prompts 1-8 tests
- v2.0 (11-05-2025): Added Phase 2 tests
- v1.0 (10-29-2025): Initial Phase 1 tests
