# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [Unreleased]

### Added
- Created `requirements.txt` with all project dependencies (2025-11-28)
- Created `CHANGELOG.md` to track code updates (2025-11-28)

### Changed
- Updated root `CLAUDE.md` to reflect Phase 5 actual completion status (2025-11-28)
- Updated `README.md` to reflect current research platform status (2025-11-28)
- Updated `src/pipeline/CLAUDE.md` to document Prompts 5-8 implementation (2025-11-28)

### Fixed
- Documentation drift between actual implementation and CLAUDE.md files (2025-11-28)

---

## [Phase 5 Prompts 5-8] - 2025-11-27

### Added
- **Prompt 5 (Validation)**: `src/pipeline/validation.py` (352 lines)
  - `StageValidator` for validating pipeline stage outputs
  - Graph generation output validation
  - Benchmarking output validation
  - Feature extraction output validation
  - Model training output validation
  - `ValidationError` exception class

- **Prompt 6 (Profiling)**: `src/pipeline/profiling.py` (366 lines)
  - `PerformanceMonitor` for tracking runtime and memory
  - `PerformanceMetrics` dataclass for performance data
  - `RuntimeProfiler` for detailed profiling
  - `@profile_stage` decorator for automatic profiling
  - Per-graph and per-algorithm profiling support

- **Prompt 7 (Parallel)**: `src/pipeline/parallel.py` (346 lines)
  - `ParallelExecutor` for parallelizing graph/algorithm operations
  - `ParallelConfig` for configuring parallelization
  - `ResourceManager` for managing CPU/memory resources
  - `create_parallel_executor` factory function
  - Support for multiprocessing-based parallelization

- **Prompt 8 (Error Handling)**: `src/pipeline/error_handling.py` (360 lines)
  - `ErrorHandler` for robust error handling
  - `ErrorRecord` for tracking failures
  - `Checkpoint` system for resumability
  - `@retry_with_backoff` decorator for retrying operations
  - `@try_continue` decorator for graceful failure handling
  - `@graceful_degradation` decorator for feature degradation

### Changed
- Updated `src/pipeline/__init__.py` to export all Prompts 5-8 components

---

## [Phase 5 Prompts 1-4] - 2025-11-22

### Added
- **Prompt 1 (Orchestrator)**: `src/pipeline/orchestrator.py` (363 lines)
  - `PipelineStage` abstraction for individual stages
  - `PipelineOrchestrator` for multi-stage coordination
  - `StageResult` and `StageStatus` for execution tracking
  - Modular, idempotent, resumable stage execution

- **Prompt 2 (Configuration)**: `src/pipeline/config.py` (406 lines)
  - `ExperimentConfig` for YAML-based configuration
  - `ConfigValidator` with comprehensive validation
  - Sub-configs: `GraphGenConfig`, `BenchmarkConfig`, `FeatureConfig`, `ModelConfig`
  - Template generation and config validation

- **Prompt 3 (Tracking)**: `src/pipeline/tracking.py` (398 lines)
  - `ExperimentTracker` for tracking individual experiments
  - `ExperimentRegistry` for managing multiple experiments
  - `ExperimentMetadata` with complete experiment info
  - `ExperimentStatus` enum (PENDING, RUNNING, COMPLETED, FAILED, CANCELLED)
  - Directory structure creation and metadata persistence

- **Prompt 4 (Reproducibility)**: `src/pipeline/reproducibility.py` (390 lines)
  - `ReproducibilityManager` for ensuring reproducibility
  - `SeedManager` for deterministic seed propagation
  - `EnvironmentInfo` for tracking Python/package versions
  - Git commit and diff tracking
  - Environment verification and reproducibility checking

### Added
- Test suite: `src/tests/test_phase5_pipeline.py` (45 tests for Prompts 1-4)
  - 9 test classes covering all Prompt 1-4 functionality
  - All tests passing (100% pass rate)

### Added
- Documentation: `src/pipeline/CLAUDE.md`
  - Complete architecture documentation
  - Usage examples for all components
  - Integration patterns with Phases 1-4

---

## [Phase 4 Complete] - 2025-11-22

### Added
- **Prompt 1 (Dataset)**: `src/ml/dataset.py`
  - `MLDataset` class for managing features and labels
  - Dataset statistics and validation
  - Train/test/validation splitting

- **Prompt 2 (Splitting)**: Split strategies in `src/ml/dataset.py`
  - Random split
  - Stratified split (by graph type)
  - Graph-based split (no data leakage)
  - Temporal split
  - Custom split support

- **Prompt 3 (Linear Models)**: Linear regression in `src/ml/models.py`
  - Ordinary Least Squares (OLS)
  - Ridge regression
  - Lasso regression
  - ElasticNet regression

- **Prompt 4 (Tree Models)**: Tree-based models in `src/ml/models.py`
  - Decision Tree regressor
  - Random Forest regressor
  - Gradient Boosting regressor

- **Prompt 5 (Evaluation)**: `src/ml/evaluation.py`
  - `ModelEvaluator` for comprehensive evaluation
  - `ModelComparison` for comparing multiple models
  - Metrics: R², MAE, RMSE, residual analysis
  - Feature importance extraction

- **Prompt 6 (Cross-Validation)**: `src/ml/cross_validation.py`
  - K-fold cross-validation
  - Stratified K-fold
  - Group K-fold (by graph)
  - Nested cross-validation for hyperparameter tuning

- **Prompt 7 (Tuning)**: `src/ml/tuning.py`
  - `HyperparameterTuner` for grid/random search
  - Grid search with cross-validation
  - Random search with cross-validation
  - Best model selection and retraining

- **Prompt 8 (Feature Engineering)**: `src/ml/feature_engineering.py`
  - Feature scaling (StandardScaler, MinMaxScaler, RobustScaler)
  - Non-linear transformations (log, sqrt, polynomial)
  - Interaction features
  - PCA dimensionality reduction

### Added
- Test suite: `src/tests/test_phase4_ml.py` (96 tests)
  - All ML functionality comprehensively tested
  - All tests passing (100% pass rate)

### Added
- Documentation: `src/ml/CLAUDE.md`
  - Complete ML pipeline documentation
  - Usage examples and integration patterns

---

## [Phase 3 Complete] - 2025-11-10

### Added
- Feature extraction system (all 12 prompts)
- 6 feature extractors: weight-based, topological, MST-based, neighborhood, heuristic, graph context
- `FeatureAnalyzer` toolkit for feature analysis
- `AnchorQualityLabeler` with 5 labeling strategies
- `DatasetPipeline` for end-to-end dataset generation
- Feature selection and transformation utilities
- Test suite: `src/tests/test_phase3_features.py` (111 tests)
- Documentation: `src/features/CLAUDE.md`

---

## [Phase 2 Complete] - 2025-11-05

### Added
- Algorithm benchmarking system (Prompts 1-5)
- Unified algorithm interface (`TSPAlgorithm` base class)
- Algorithm registry with auto-registration
- 8 algorithms: Nearest Neighbor (2 variants), Greedy Edge, Held-Karp, Single Anchor (2 variants), Best Anchor, Multi-Anchor (2 variants)
- Tour validation and quality metrics
- Test suite: `src/tests/test_phase2_algorithms.py` (89 tests)
- Documentation: `src/algorithms/CLAUDE.md`

---

## [Phase 1 Complete] - 2025-10-30

### Added
- Graph generation system (complete specification)
- Euclidean, metric, quasi-metric, and random graph generators
- Graph property verification (metricity, symmetry, Euclidean distances)
- Batch generation pipeline with YAML configuration
- Storage and retrieval system
- Visualization utilities
- Test suite: `src/tests/test_graph_generators.py` (34 tests)
- Documentation: `src/graph_generation/CLAUDE.md`

### Fixed
- Euclidean property preservation in coordinate scaling (2025-10-29)
- Quasi-metric triangle inequality verification (2025-10-29)
- Metric graph completion strategy for narrow weight ranges (2025-10-29)

---

## Initial Commit - 2025-05-02

### Added
- Project structure and initial concept
- Basic anchor-based heuristic ideas from Discrete Math II class
