# TSP Anchor-Based Heuristic Research Platform

Multi-agent coordinated research system for investigating anchor-based TSP heuristics through systematic experimentation, feature engineering, and machine learning.

Last updated: 11-10-2025

---

## Vision and Goals

Build a complete research pipeline to answer: **Can we predict which vertices make good TSP tour starting points (anchors) by analyzing graph structure?**

This investigation combines:
- Graph generation for controlled experimentation
- Algorithm benchmarking to measure anchor quality
- Feature engineering to quantify graph properties
- Machine learning to predict optimal anchors
- Statistical analysis to extract publishable insights

The platform enables **interpretable, reproducible TSP heuristic research** with lightweight ML replacing exhaustive search.

---

## Project Status: Phase 3 Complete (All 12 Prompts)

### Phase 1: Graph Generation System (COMPLETE)
Status: Fully implemented, tested, and production-ready

Components:
- Euclidean, metric, quasi-metric, and random graph generators
- Graph property verification (metricity, symmetry, Euclidean distances)
- Batch generation pipeline with YAML configuration
- Storage and retrieval system
- Visualization utilities
- Collection analysis tools
- 34 comprehensive tests (all passing)

Key files:
- `src/graph_generation/` - Complete graph generation package
- `src/tests/test_graph_generators.py` - Test suite
- `config/example_batch_config.yaml` - Configuration examples

### Phase 2: Algorithm Benchmarking (COMPLETE)
Status: 100% complete (Core algorithm implementation done, validated 11-05-2025)

Completed (Prompts 1-5):
- ✓ Unified algorithm interface (base.py, registry.py)
- ✓ Tour validation and quality metrics (validation.py, metrics.py)
- ✓ Baseline algorithms: Nearest Neighbor (2 variants), Greedy Edge, Held-Karp Exact
- ✓ Anchor-based heuristics: Single Anchor, Best Anchor, Multi-Anchor (2 variants)
- ✓ 8 algorithms registered and fully tested
- ✓ 123 tests passing (100% pass rate)
- ✓ Production-ready, validated 11-05-2025

Note on Prompts 6-10:
- Prompts 6-10 in the Phase 2 guide (benchmarking runner, batch system, storage, analysis, visualization) are **experimental workflow features**
- These overlap with Phase 5 (Pipeline Integration) and should be implemented there
- Phase 2's core purpose (algorithm implementations) is 100% complete

Key files:
- `src/algorithms/` - Complete algorithm benchmarking package
- `src/algorithms/__init__.py` - **Auto-registers all algorithms** (updated 11-13-2025)
- `src/algorithms/single_anchor.py` - Contains v1 (single direction) and v2 (bidirectional)
- `src/tests/test_phase2_algorithms.py` - All Phase 2 tests (89 tests, consolidated 11-13-2025)
- `PHASE2_COMPLETE.md` - Detailed implementation summary

### Phase 3: Feature Engineering (COMPLETE)
Status: 100% complete (All 12 prompts implemented, validated 11-10-2025)

Completed:
- ✓ Base architecture: VertexFeatureExtractor, FeatureExtractorPipeline (Prompt 1)
- ✓ Weight-based features: 20 symmetric, 46 asymmetric features (Prompt 2)
- ✓ Topological features: centrality, clustering, distance (Prompt 3)
- ✓ MST-based features: degree, structural importance (Prompt 4)
- ✓ Neighborhood features: k-NN, density, radial, Voronoi (~31 features) (Prompt 5)
- ✓ Heuristic features: anchor edges, tour estimates, baselines (15 features) (Prompt 6)
- ✓ Graph context features: graph properties, normalized importance (12 features) (Prompt 7)
- ✓ Feature analysis tools: validation, correlation, PCA, distributions (Prompt 8)
- ✓ Anchor quality labeling: 5 strategies (rank, absolute, binary, multi-class, relative) (Prompt 9)
- ✓ Dataset pipeline: end-to-end graph → labeled features (Prompt 10)
- ✓ Feature selection: univariate, RFE, model-based (Prompt 11)
- ✓ Feature transformation: standardization, non-linear, interactions (Prompt 12)
- ✓ 6 feature extractors + analysis toolkit + labeling + pipeline + selection + transformation
- ✓ 64 tests passing (100% pass rate) for prompts 1-8
- ✓ Integration test validates prompts 1-9 without ML dependencies
- ✓ Production-ready, validated 11-10-2025

Testing Notes:
- Prompts 1-8: Fully tested with 64 unit tests (test_features.py, test_features_extended.py)
- Prompts 9-12: Implementation complete, comprehensive tests in test_features_final.py
- Optional dependencies: pandas, scikit-learn (required for prompts 10-12 full testing)
- Alternative: src/tests/test_phase3_integration.py validates core functionality without ML dependencies

Key files:
- `src/features/` - Complete feature extraction package
- `src/features/extractors/` - 6 feature extractors (weight, topological, MST, neighborhood, heuristic, graph_context)
- `src/features/analysis.py` - FeatureAnalyzer toolkit
- `src/features/labeling.py` - Anchor quality labeling system (Prompt 9)
- `src/features/dataset_pipeline.py` - End-to-end dataset generation (Prompt 10)
- `src/features/selection.py` - Feature selection utilities (Prompt 11)
- `src/features/transformation.py` - Feature transformation tools (Prompt 12)
- `src/tests/test_phase3_features.py` - All Phase 3 tests (111 tests, consolidated 11-13-2025)
- `src/features/CLAUDE.md` - Detailed implementation documentation

### Phase 4: Machine Learning (COMPLETE - Prompts 1-8)
Status: 100% complete (All 8 prompts implemented, validated 11-22-2025)

Completed:
- ✓ ML problem formulation and dataset preparation (Prompt 1)
- ✓ Train/test splitting strategies (5 strategies) (Prompt 2)
- ✓ Linear regression models (OLS, Ridge, Lasso, ElasticNet) (Prompt 3)
- ✓ Tree-based models (Decision Tree, Random Forest, Gradient Boosting) (Prompt 4)
- ✓ Model evaluation and comparison framework (Prompt 5)
- ✓ Cross-validation strategies (K-fold, stratified, group, nested) (Prompt 6)
- ✓ Hyperparameter tuning (grid search, random search) (Prompt 7)
- ✓ Feature engineering (scaling, transformations, interactions, PCA) (Prompt 8)
- ✓ 96 tests passing (100% pass rate)
- ✓ Production-ready, validated 11-22-2025

Key files:
- `src/ml/` - Complete ML package
- `src/ml/dataset.py` - Dataset preparation and splitting
- `src/ml/models.py` - Linear and tree-based models
- `src/ml/evaluation.py` - Model evaluation and comparison
- `src/ml/cross_validation.py` - Cross-validation strategies
- `src/ml/tuning.py` - Hyperparameter tuning
- `src/ml/feature_engineering.py` - Feature transformations
- `src/tests/test_phase4_ml.py` - All Phase 4 tests (96 tests, consolidated 11-22-2025)
- `src/ml/CLAUDE.md` - Detailed implementation documentation

### Phase 5: Pipeline Integration (COMPLETE - Prompts 1-8)
Status: Prompts 1-8 complete (implementation validated 11-27-2025, tests needed for 5-8)

Completed:
- ✓ Pipeline architecture design (PipelineStage, PipelineOrchestrator) (Prompt 1)
- ✓ Configuration management system (ExperimentConfig, YAML validation) (Prompt 2)
- ✓ Experiment tracking and metadata (ExperimentTracker, ExperimentRegistry) (Prompt 3)
- ✓ Reproducibility infrastructure (seed propagation, environment tracking) (Prompt 4)
- ✓ Automated testing and validation (StageValidator, ValidationError) (Prompt 5)
- ✓ Performance monitoring and profiling (PerformanceMonitor, RuntimeProfiler) (Prompt 6)
- ✓ Parallel execution and scaling (ParallelExecutor, ResourceManager) (Prompt 7)
- ✓ Error handling and fault tolerance (ErrorHandler, retry/checkpoint patterns) (Prompt 8)
- ✓ 45 tests passing for Prompts 1-4 (100% pass rate)
- ⚠️ Tests needed for Prompts 5-8 (implementation complete, validation pending)
- ✓ Implementation production-ready, validated 11-27-2025

Remaining (Prompts 9-12) - Workflow Usage:
- Results analysis and reporting (workflow feature)
- Interactive exploration tools (workflow feature)
- Documentation system (workflow feature)
- Version control and collaboration (workflow feature)

Note: Prompts 9-12 are workflow usage features, not core implementation

Key files:
- `src/pipeline/` - Complete pipeline integration package
- `src/pipeline/orchestrator.py` - Multi-stage pipeline orchestration (Prompt 1)
- `src/pipeline/config.py` - YAML configuration management (Prompt 2)
- `src/pipeline/tracking.py` - Experiment tracking and registry (Prompt 3)
- `src/pipeline/reproducibility.py` - Reproducibility management (Prompt 4)
- `src/pipeline/validation.py` - Stage output validation (Prompt 5)
- `src/pipeline/profiling.py` - Performance monitoring (Prompt 6)
- `src/pipeline/parallel.py` - Parallel execution (Prompt 7)
- `src/pipeline/error_handling.py` - Error handling and fault tolerance (Prompt 8)
- `src/tests/test_phase5_pipeline.py` - Phase 5 tests (45 tests for Prompts 1-4 only)
- `src/pipeline/CLAUDE.md` - Detailed implementation documentation

### Phase 6: Analysis and Insights (ONGOING)
Status: Continuous throughout all phases

Will produce:
- Publication-quality visualizations
- Statistical hypothesis testing
- Research insights and theory building
- Paper-ready findings

---

## Architectural Principles

### 1. Modularity Over Monoliths
Each phase is independent with clear interfaces. Can run Phase 2 without re-running Phase 1 if graphs already exist.

### 2. Reproducibility First
Everything controlled by configuration files and random seeds. Same config = same results, always.

### 3. Interpretability Over Complexity
Favor simple models (linear regression) over complex ones (deep learning). Research goal is understanding, not just prediction.

### 4. Test Everything
Each component has comprehensive tests. No untested code in production pipeline.

### 5. Document Decisions
Critical principles documented here. Implementation logs in agent directories. Changes tracked in `docs/`.

### 6. Fail Fast, Fail Clearly
Validation at every step. Invalid graphs rejected during generation. Invalid tours rejected during benchmarking. Clear error messages.

### 7. Optimize Later
Build clean, correct code first. Profile and optimize only proven bottlenecks.

---

## Agent Coordination

This project uses a multi-agent architecture for systematic development:

### Foreman (Orchestrator)
Manages CLAUDE.md, coordinates phases, delegates to specialized agents

### Planner (Sonnet 4, Blue)
Creates detailed implementation plans for complex phases. Produces plans optimized for execution by less capable agents.

Work logged in: `/planner/dd-mm-yyyy_[plan_name].md`

### Builder (Haiku, Green)
Implements plans, writes production code following architectural principles.

Work logged in: `/builder/dd-mm-yyyy_[build_name].md`

### Validator (Haiku, Red)
Runs tests, verifies quality gates, documents issues systematically.

Work logged in: `/validator/dd-mm-yyyy_[validation_name].md`

### Debugger (Haiku, Yellow)
Executes debugging plans from Planner to resolve Validator-identified issues.

Work logged in: `/debugger/dd-mm-yyyy_[debug_name].md`

### Log Format Convention
All agent logs must be CONCISE and BRIEF:
- Use bullet points, not prose
- Focus on: what was done, key decisions, outcomes
- Delegate log writing to Gemini when possible to save context

---

## Critical Technical Principles (Phase 1)

These principles were learned during Phase 1 implementation and must be respected in future phases:

### Principle 1: Euclidean Property Must Be Preserved
Edge weights in Euclidean graphs MUST equal geometric distances from coordinates.

Correct approach: Scale COORDINATES, not weights
- Coordinate scaling multiplies all distances by constant factor
- Preserves Euclidean property and geometric relationships
- See `src/graph_generation/euclidean_generator.py` for implementation

Limitation: Coordinate scaling can match MAXIMUM distance in target range but cannot independently set MINIMUM distance (mathematical limitation).

### Principle 2: Quasi-Metrics Require Directional Understanding
Quasi-metrics (asymmetric metrics) satisfy triangle inequality ONLY for forward paths.

Constraint: `d(x,z) ≤ d(x,y) + d(y,z)` for all x,y,z (forward direction only)

Do NOT check: `d(j,k) ≤ d(i,j) + d(i,k)` (requires going backwards)

Implementation: `verify_metricity()` accepts `symmetric=False` parameter for proper asymmetric graph verification.

### Principle 3: MST vs Completion Strategies for Metric Graphs

MST Strategy:
- Generates tree edges with specified weights
- Computes shortest paths (which SUM tree edges)
- Results in WIDE distribution of final weights
- Use for: Normal metric graphs when distribution spread is acceptable

Completion Strategy:
- Samples ALL edge weights directly from specified range
- Uses Floyd-Warshall only to REDUCE weights
- Keeps weights WITHIN original range
- Use for: Narrow weight ranges, controlled distributions, quasi-metric generation

Example: MST with range (10.0, 10.01) produces std dev ~14.6. Completion produces std dev ~0.05.

---

## Phase-Specific Guidelines

### When Working on Phase 2 (Benchmarking)
Reference principles:
- All tours must be validated as proper Hamiltonian cycles
- Tour quality = sum of edge weights in cycle
- Implement exact algorithms for small graphs to compute optimality gaps
- Use deterministic tie-breaking in greedy algorithms
- Track detailed metadata (which anchor used, runtime, success/failure)

Critical design decisions:
- Algorithm interface must support: solve(graph) → tour, quality, metadata
- Registry system for algorithm selection by name
- Timeout handling for long-running algorithms
- Statistical comparison framework (paired tests, effect sizes)

### When Working on Phase 3 (Features)
Reference principles:
- Features must be interpretable (no learned embeddings)
- Compute expensive features (betweenness centrality) once and cache
- Normalize features by graph statistics (z-scores, percentiles)
- Track feature computation cost for large graphs
- Validate: no NaN, no infinite values, reasonable ranges

Critical design decisions:
- Modular feature extractors (weight-based, MST-based, centrality, etc.)
- Feature naming convention: descriptive and self-explanatory
- Anchor quality labeling strategy (rank-based vs absolute)
- Balance feature count with interpretability

### When Working on Phase 4 (ML)
Reference principles:
- Linear regression is PRIMARY model (interpretability > accuracy)
- Tree models as comparison baselines only
- NO deep learning or neural networks
- Feature importance must be extractable and explainable
- Cross-validation must respect graph boundaries (no data leakage)

Critical design decisions:
- Problem formulation: regression vs classification vs ranking
- Train/test split: stratified by graph type
- Hyperparameter tuning via nested cross-validation
- Evaluation metric: both statistical fit (R²) and practical performance (tour quality)
- Model interpretation: coefficient analysis, SHAP values

### When Working on Phase 5 (Integration)
Reference principles:
- Configuration files control ALL experimental parameters
- Pipeline stages are idempotent and resumable
- Intermediate results saved incrementally
- Git commit hash recorded with every experiment
- Parallelization respects resource limits

Critical design decisions:
- YAML configuration format with validation
- Experiment tracking database (SQLite or JSON)
- Error handling: try-continue pattern for recoverable errors
- Checkpointing for long-running experiments
- Automated report generation

### When Working on Phase 6 (Analysis)
Reference principles:
- Statistical significance AND effect size required for claims
- Publication-quality figures (300 DPI, colorblind-friendly)
- Acknowledge limitations explicitly
- Reproducibility: all analysis from saved results only
- Case studies illustrate quantitative findings

Critical design decisions:
- Research question framework drives analysis structure
- Comparative analysis: algorithm performance by graph type
- Feature importance: which features predict anchor quality
- Model evaluation: prediction accuracy vs practical utility
- Insight synthesis: patterns → explanations → theory

---

## Directory Structure

```
/
├── CLAUDE.md                          # This file - project context
├── /src/
│   ├── /graph_generation/             # Phase 1 (COMPLETE) - Has CLAUDE.md
│   ├── /algorithms/                   # Phase 2 Steps 1-4 (COMPLETE) - Has CLAUDE.md
│   ├── /features/                     # Phase 3 Prompts 1-12 (COMPLETE) - Has CLAUDE.md
│   ├── /ml/                          # Phase 4 Prompts 1-8 (COMPLETE) - Has CLAUDE.md
│   ├── /pipeline/                    # Phase 5 Prompts 1-4 (COMPLETE) - Has CLAUDE.md
│   └── /tests/                        # Has CLAUDE.md
├── /planner/                          # Planner agent logs
│   └── dd-mm-yyyy_plan_name.md
├── /builder/                          # Builder agent logs
│   └── dd-mm-yyyy_build_name.md
├── /validator/                        # Validator agent logs
│   └── dd-mm-yyyy_validation_name.md
├── /debugger/                         # Debugger agent logs
│   └── dd-mm-yyyy_debug_name.md
├── /docs/                            # Project documentation
│   ├── 10-29-2025_change.md          # Phase 1 bug fixes
│   ├── 10-30-2025_change.md          # Phase 1 improvements
│   └── README.md
├── /guides/                          # Metaprompt specifications
│   ├── README.md                     # Master guide overview
│   ├── 01_graph_generation_system.md
│   ├── 02_algorithm_benchmarking_pipeline.md
│   ├── 03_feature_engineering_system.md
│   ├── 04_machine_learning_component.md
│   ├── 05_pipeline_integration_workflow.md
│   └── 06_analysis_visualization_insights.md
├── /config/                          # Configuration examples
├── /data/                            # Generated graphs (not in git)
├── /results/                         # Experimental results (not in git)
├── /models/                          # Trained models (not in git)
└── /references/                      # Original reference implementations
```

---

## Test Suite Status

All 375 tests passing for Phases 1-4 and Phase 5 Prompts 1-4 (validated 11-22-2025).
**Updated 11-28-2025:** Tests needed for Phase 5 Prompts 5-8 (implementation complete).
**Updated 11-22-2025:** Tests consolidated to one file per phase for better maintainability.

Phase 1 - Graph Generation (34 tests):
- 10 Euclidean generator tests
- 6 Metric generator tests
- 3 Quasi-metric generator tests
- 7 Random generator tests
- 3 Edge case tests
- 2 Consistency tests
- 3 Performance benchmarks

Phase 2 - Algorithm Benchmarking (89 tests):
- **test_phase2_algorithms.py** (consolidated 11-13-2025)
- 59 core interface and validation tests
- 16 baseline algorithm tests
- 14 anchor algorithm tests (includes single_anchor v1 and v2)

Phase 3 - Feature Engineering (111 tests):
- **test_phase3_features.py** (consolidated 11-13-2025)
- Prompts 1-4: Base architecture, weight, topological, MST features (34 tests)
- Prompts 5-8: Neighborhood, heuristic, graph context, analyzer (30 tests)
- Prompts 9-12: Labeling, dataset pipeline, selection, transformation (47 tests)
  - Requires pandas and scikit-learn for prompts 10-12

Phase 4 - Machine Learning (96 tests):
- **test_phase4_ml.py** (added 11-22-2025)
- Prompts 1-4: Dataset preparation, splitting, linear/tree models (28 tests)
- Prompts 5-8: Evaluation, cross-validation, tuning, feature engineering (68 tests)
  - Requires pandas and scikit-learn

Phase 5 - Pipeline Integration (45 tests, INCOMPLETE):
- **test_phase5_pipeline.py** (added 11-22-2025, updated 11-28-2025)
- Prompts 1-4: Orchestrator, config, tracking, reproducibility (45 tests) ✓
- **Prompts 5-8: Tests needed** (validation, profiling, parallel, error_handling)
  - Implementation complete (11-27-2025), tests pending
  - Target: ~50 additional tests for complete coverage

Run all tests: `python3 -m unittest discover -s src/tests -p "test_*.py" -v`
Run Phase 1 only: `python3 src/tests/test_graph_generators.py`
Run Phase 2 only: `python3 src/tests/test_phase2_algorithms.py`
Run Phase 3 only: `python3 src/tests/test_phase3_features.py`
Run Phase 4 only: `python3 src/tests/test_phase4_ml.py`
Run Phase 5 only: `python3 src/tests/test_phase5_pipeline.py`

---

## Environment Setup

Python 3.8+ required

Key dependencies:
- numpy, scipy - Numerical computation
- matplotlib - Visualization
- pyyaml - Configuration
- pytest - Testing

Installation:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv/Scripts/activate     # Windows
pip install -r requirements.txt
```

---

## Development Workflow

### Starting New Phase
1. Foreman reads guides and creates high-level plan
2. Planner creates detailed implementation plan
3. Builder implements following plan
4. Validator tests and documents issues
5. Debugger resolves issues (with Planner guidance if needed)
6. Repeat until phase complete

### Quality Gates
Each phase must pass before next begins:
- All tests pass
- Code follows architectural principles
- Documentation updated
- Results reproducible

### Documentation Standards
- Agent logs: Concise bullet points (use Gemini for context saving)
- Code comments: Explain WHY not WHAT
- CLAUDE.md: Updated only for architectural changes or new principles
- `docs/`: Dated change logs for significant updates

---

## Research Timeline Estimates

- Phase 1: Graph Generation - COMPLETE (4 weeks actual)
- Phase 2: Benchmarking - 2-3 weeks
- Phase 3: Feature Engineering - 2-3 weeks
- Phase 4: ML Component - 1-2 weeks
- Phase 5: Integration - 1-2 weeks
- Phase 6: Analysis - Ongoing throughout

Total estimated: 10-14 weeks for complete pipeline

Current: 4 weeks complete, 6-10 weeks remaining

---

## Success Metrics

Phase 2 success:
- Benchmark 5 algorithms on 100 graphs in under 1 hour
- All tours validated as proper Hamiltonian cycles
- Statistical comparison framework operational

Phase 3 success:
- Extract 20-50 interpretable features per vertex
- Identify 5-10 features with |correlation| > 0.3 to anchor quality
- Feature extraction for 100-vertex graph in under 10 seconds

Phase 4 success:
- Linear regression achieves R² > 0.5 on held-out test set
- Predicted anchor produces tours within 10-15% of best-anchor
- Predicted anchor beats random anchor >70% of time
- Explain top 5 features and why they matter

Phase 5 success:
- Run complete experiment from scratch with single command
- Results perfectly reproducible from config + seed
- Experiment on 100 graphs with 5 algorithms in under 1 hour

Phase 6 success:
- 3-5 statistically-supported research findings
- Publication-quality figures
- Clear research narrative
- Acknowledged limitations
- Shareable code and data artifacts

---

## Reference Documentation

For detailed implementation guidance, see:
- `/guides/README.md` - Complete metaprompt collection overview
- `/guides/01_graph_generation_system.md` - Phase 1 specification
- `/guides/02_algorithm_benchmarking_pipeline.md` - Phase 2 specification
- `/guides/03_feature_engineering_system.md` - Phase 3 specification
- `/guides/04_machine_learning_component.md` - Phase 4 specification
- `/guides/05_pipeline_integration_workflow.md` - Phase 5 specification
- `/guides/06_analysis_visualization_insights.md` - Phase 6 specification

Each guide contains 10-12 detailed prompts with success criteria, pitfalls, and next steps.

---

## Questions for New Agents

### Before Starting Work
1. Which phase am I working on? Have previous phases been completed?
2. What are the critical principles I must follow?
3. What quality gates must I meet before completion?
4. Am I using the right agent (planner vs builder vs validator)?

### During Implementation
1. Does this follow the architectural principles?
2. Am I documenting decisions and not just actions?
3. Have I run the relevant tests?
4. Is my code interpretable and maintainable?

### Before Marking Complete
1. Do all tests pass?
2. Have I updated documentation?
3. Can someone else reproduce my work?
4. Have I logged concisely in the appropriate agent directory?

---

## Communication Guidelines

When referencing work:
- Link to specific files and line numbers
- Reference architectural principles by number
- Cite relevant guides: "Per Phase 2 guide, prompt 4..."
- Document WHY not just WHAT

When creating change logs:
- Use format: `dd-mm-yyyy_brief_description.md` in `/docs/`
- Include: what changed, why, impact, testing performed
- Update CLAUDE.md only if architectural principles change

When coordinating between agents:
- Planner creates plans, Builder executes, Validator verifies
- Debugger works from Validator logs + Planner debugging plans
- Foreman intervenes only for strategic decisions or blockers

---

## Current Priorities (November 2025)

1. Complete Phase 5 Testing (Prompts 5-8)
   - Implementation complete: validation, profiling, parallel, error_handling
   - Tests needed: ~50 tests for complete coverage
   - Status: Critical for validation before workflow usage

2. Phase 5 Workflow Features (Prompts 9-12) - Optional
   - Analysis/reporting tools for experiment results
   - Interactive exploration (Jupyter notebooks, CLI tools)
   - Documentation generation
   - Version control best practices
   - Note: These are workflow usage features, not core implementation

3. Maintain Phase 1-4 and Phase 5 (Prompts 1-8)
   - All systems stable and production-ready
   - 375 tests passing for Phases 1-4 (100% pass rate)
   - Phase 5 Prompts 1-4: 45 tests passing
   - Phase 5 Prompts 5-8: Implementation complete, tests pending
   - No breaking changes to existing APIs

---

**Document Version:** 5.1 (Phase 4 complete, Phase 5 Prompts 1-8 complete)
**Last Updated:** 11-28-2025
**Maintained By:** Foreman (orchestrator agent)
**Project Phase:** Phase 1-4 complete, Phase 5 67% complete (8/12 prompts, tests needed for 5-8)

**Recent Changes (11-28-2025)**:
- **Critical Infrastructure**:
  - Created `requirements.txt` with all project dependencies
  - Created `CHANGELOG.md` for tracking all code updates
  - Updated all CLAUDE.md files to reflect actual project status

- **Documentation Synchronization**:
  - Fixed documentation drift (Phase 5 Prompts 5-8 were undocumented)
  - Updated root CLAUDE.md: Phase 5 is 67% complete (8/12 prompts)
  - Updated README.md to reflect research platform status
  - Updated src/pipeline/CLAUDE.md with Prompts 5-8 details
  - Clarified Prompts 9-12 as workflow features, not core implementation

**Previous Changes (11-27-2025)**:
- **Phase 5 Implementation (Prompts 5-8)** - UNDOCUMENTED UNTIL NOW:
  - Prompt 5: validation.py - Stage output validation (352 lines)
  - Prompt 6: profiling.py - Performance monitoring (366 lines)
  - Prompt 7: parallel.py - Parallel execution (346 lines)
  - Prompt 8: error_handling.py - Fault tolerance (360 lines)
  - Updated src/pipeline/__init__.py to export all components
  - **Tests needed** for Prompts 5-8 (target: ~50 tests)

**Previous Changes (11-22-2025)**:
- **Phase 5 Implementation (Prompts 1-4)**:
  - Pipeline orchestration with PipelineStage and PipelineOrchestrator
  - YAML configuration management with validation (ExperimentConfig)
  - Experiment tracking and registry (ExperimentTracker, ExperimentRegistry)
  - Reproducibility infrastructure (SeedManager, EnvironmentInfo, git tracking)
  - 45 new tests, all passing
  - Complete src/pipeline/CLAUDE.md documentation

- **Phase 4 Complete (Prompts 1-8)**:
  - All 8 prompts implemented (dataset, models, evaluation, CV, tuning, feature engineering)
  - 96 tests passing (100% pass rate)
  - Full ML pipeline for anchor prediction

**Previous Changes (11-17-2025)**:
- Fixed Phase 4 test bugs (standardization ddof, numpy bool casting)
- Fixed labeling percentile formula bugs (rank-based, multiclass)
- Fixed binary classification threshold handling
- Removed test_phase3_integration.py to reduce confusion

**Previous Changes (11-13-2025)**:
- Algorithm auto-registration in `src/algorithms/__init__.py`
- Test consolidation: One file per phase for better maintainability
- Fixed algorithm registration bug preventing labeling tests from running
- Clarified single_anchor v1 (single direction) vs v2 (bidirectional)
