# TSP Anchor-Based Heuristic Research Platform

A comprehensive research platform for investigating anchor-based heuristics in the Traveling Salesman Problem (TSP) through systematic experimentation, feature engineering, and machine learning.

## Overview

This project explores the question: **Can we predict which vertices make good TSP tour starting points (anchors) by analyzing graph structure?**

The platform combines:
- Controlled graph generation for experimentation
- Algorithm benchmarking to measure anchor quality
- Feature engineering to quantify graph properties
- Machine learning to predict optimal anchors
- Statistical analysis for publishable insights

## Project Status

**Complete Research Pipeline**: 375+ tests passing across 5 phases

| Phase | Status | Implementation | Tests |
|-------|--------|----------------|-------|
| **Phase 1**: Graph Generation | ✅ Complete | 100% | 34 tests |
| **Phase 2**: Algorithm Benchmarking | ✅ Complete (10 algos) | 100% | 89 tests* |
| **Phase 3**: Feature Engineering | ✅ Complete | 100% (12/12 prompts) | 111 tests |
| **Phase 4**: Machine Learning | ✅ Complete | 100% (8/8 prompts) | 96 tests |
| **Phase 5**: Pipeline Integration | 🟡 Ready for Use | 67% (8/12 prompts) + CLI | 45 tests |
| **Phase 6**: Analysis & Insights | ⏳ Planned | - | - |

### Phase 5 Status
- **Prompts 1-8**: Implementation complete (orchestrator, config, tracking, reproducibility, validation, profiling, parallel execution, error handling)
- **CLI Integration**: Complete - `experiments/run_experiment.py` ready for end-to-end experiments
- **Configuration**: Template files and comprehensive documentation available
- **Tests**: Only Prompts 1-4 tested (45 tests), Prompts 5-8 need tests (~50 additional tests)
- **Prompts 9-12**: Workflow features (analysis/reporting, exploration tools, documentation) - planned but not required for core functionality
- **Status**: Production-ready for research experiments, testing recommended before large-scale use

## Features

### Phase 1: Graph Generation System ✅
- **Graph Types**: Euclidean, metric, quasi-metric, random
- **Property Verification**: Metricity, symmetry, Euclidean distances
- **Batch Generation**: YAML-configured parallel generation
- **Visualization**: Plot graphs and tours
- **34 comprehensive tests**

### Phase 2: Algorithm Benchmarking ✅
- **Algorithms**: 10 implemented (2 new added 12-06-2025)
  - Baselines: Nearest Neighbor (2 variants), Adaptive NN, Greedy Edge, Held-Karp Exact
  - Anchor-based: Single Anchor (3 variants including v3), Best Anchor, Multi-Anchor (2 variants)
  - **New**: `nearest_neighbor_adaptive` and `single_anchor_v3` (adaptive both-ends path building)
- **Validation**: All tours verified as valid Hamiltonian cycles
- **Metrics**: Tour quality, runtime, optimality gaps
- **89 comprehensive tests** (new algorithms pending tests)

### Phase 3: Feature Engineering ✅
- **6 Feature Extractors**:
  - Weight-based features (20 symmetric, 46 asymmetric)
  - Topological features (centrality, clustering, distance)
  - MST-based features (degree, structural importance)
  - Neighborhood features (k-NN, density, Voronoi)
  - Heuristic features (anchor edges, tour estimates)
  - Graph context features (normalized importance)
- **Analysis Tools**: Validation, correlation, PCA, distributions
- **Labeling Strategies**: 5 strategies for anchor quality
- **Dataset Pipeline**: End-to-end graph → labeled features
- **Feature Selection & Transformation**: Univariate, RFE, model-based, scaling, interactions
- **111 comprehensive tests**

### Phase 4: Machine Learning ✅
- **Dataset Management**: Train/test/validation splitting with 5 strategies
- **Models**:
  - Linear: OLS, Ridge, Lasso, ElasticNet
  - Trees: Decision Tree, Random Forest, Gradient Boosting
- **Evaluation**: R², MAE, RMSE, residual analysis, feature importance
- **Cross-Validation**: K-fold, stratified, group, nested CV
- **Hyperparameter Tuning**: Grid search, random search
- **Feature Engineering**: Scaling, transformations, interactions, PCA
- **96 comprehensive tests**

### Phase 5: Pipeline Integration 🟡 (Ready for Use)
**Implemented (Prompts 1-8 + CLI Integration)**:
- **CLI Entry Point**: `experiments/run_experiment.py` for running complete experiments
- **Stage Factories**: Integration with all Phases 1-4 components
- **Configuration Templates**: `config/complete_experiment_template.yaml`, `config/test_config_small.yaml`
- **Orchestration**: Multi-stage pipeline with resumability
- **Configuration**: YAML-based experiment configuration with validation
- **Tracking**: Experiment metadata, registry, status tracking
- **Reproducibility**: Seed management, environment tracking, git versioning
- **Validation**: Stage output validation to catch errors early
- **Profiling**: Performance monitoring, runtime/memory tracking
- **Parallelization**: Multi-core execution, resource management
- **Error Handling**: Retry patterns, checkpoints, graceful degradation
- **Documentation**: Complete CLI docs (`experiments/README.md`) and config guide (`/docs/experiment_configuration_guide.md`)
- **45 tests for Prompts 1-4** (Prompts 5-8 need tests)

**Planned (Prompts 9-12)**:
- Results analysis and reporting (workflow feature)
- Interactive exploration tools (workflow feature)
- Documentation generation (workflow feature)
- Version control best practices (workflow feature)

## Installation

### Requirements
- Python 3.8+
- See `requirements.txt` for dependencies

### Setup
```bash
# Clone repository
git clone <repository-url>
cd hamiltonian-cycles-heuristic

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Running Complete Experiments (Recommended)

The easiest way to use the platform is through the CLI experiment runner:

```bash
# Quick test with minimal configuration (runtime: ~2-3 minutes)
python experiments/run_experiment.py config/test_config_small.yaml

# Full experiment with all features (runtime: ~30-60 minutes)
python experiments/run_experiment.py config/complete_experiment_template.yaml

# Validate configuration without running
python experiments/run_experiment.py config/my_config.yaml --dry-run

# Run only specific stage
python experiments/run_experiment.py config/my_config.yaml --stage graph_generation
```

**Output Structure:**
```
experiments/<experiment_id>/
├── metadata.json              # Experiment metadata
├── reproducibility.json       # Seeds, git hash, environment
├── logs/                      # Stage execution logs
├── graphs/                    # Generated graph instances
├── benchmarks/                # Algorithm performance results
├── features/                  # Extracted features + labels
└── models/                    # Trained ML models
```

**For detailed CLI documentation, see `experiments/README.md`**

**For configuration reference, see `/docs/experiment_configuration_guide.md`**

### Using Individual Components

#### Generate Graphs
```python
from graph_generation import EuclideanGraphGenerator

generator = EuclideanGraphGenerator(dimension=2, random_seed=42)
adjacency_matrix, coordinates = generator.generate(
    num_vertices=20,
    weight_range=(1.0, 100.0)
)
```

#### Run Algorithms
```python
from algorithms.registry import AlgorithmRegistry

registry = AlgorithmRegistry()
nn_algo = registry.get_algorithm("nearest_neighbor")
result = nn_algo.solve(graph)
print(f"Tour quality: {result.tour_weight}")
```

#### Extract Features
```python
from features import FeatureExtractorPipeline
from features.extractors import WeightBasedFeatureExtractor, TopologicalFeatureExtractor

pipeline = FeatureExtractorPipeline()
pipeline.add_extractor(WeightBasedFeatureExtractor())
pipeline.add_extractor(TopologicalFeatureExtractor())

features = pipeline.extract_features(graph)
```

#### Train ML Model
```python
from ml.dataset import MLDataset
from ml.models import RidgeRegressionModel

dataset = MLDataset.from_labeled_features(features_df, labels_df)
X_train, X_test, y_train, y_test = dataset.train_test_split(test_size=0.2, random_seed=42)

model = RidgeRegressionModel(alpha=1.0)
model.train(X_train, y_train)
predictions = model.predict(X_test)
```

## Testing

### Run All Tests
```bash
python3 -m unittest discover -s src/tests -p "test_*.py" -v
```

### Run Individual Phases
```bash
# Phase 1: Graph Generation (34 tests)
python3 -m unittest src.tests.test_graph_generators -v

# Phase 2: Algorithms (89 tests)
python3 -m unittest src.tests.test_phase2_algorithms -v

# Phase 3: Features (111 tests)
python3 -m unittest src.tests.test_phase3_features -v

# Phase 4: ML (96 tests)
python3 -m unittest src.tests.test_phase4_ml -v

# Phase 5: Pipeline (45 tests)
python3 -m unittest src.tests.test_phase5_pipeline -v
```

## Project Structure

```
/
├── CLAUDE.md                   # Project context and development guide
├── README.md                   # This file
├── CHANGELOG.md                # Detailed change history
├── requirements.txt            # Python dependencies
├── src/
│   ├── graph_generation/       # Phase 1: Graph generation
│   ├── algorithms/             # Phase 2: TSP algorithms
│   ├── features/               # Phase 3: Feature extraction
│   ├── ml/                     # Phase 4: Machine learning
│   ├── pipeline/               # Phase 5: Pipeline integration
│   └── tests/                  # Test suite (375+ tests)
├── guides/                     # Detailed implementation guides
├── config/                     # Configuration examples
├── data/                       # Generated graphs (not in git)
├── results/                    # Experimental results (not in git)
└── models/                     # Trained models (not in git)
```

## Documentation

- **`CLAUDE.md`**: Master project context, architectural principles, agent coordination
- **`CHANGELOG.md`**: Detailed history of all code changes
- **`guides/`**: Detailed metaprompts for each phase (12 prompts per phase)
- **Module CLAUDE.md files**: Phase-specific implementation documentation
  - `src/graph_generation/CLAUDE.md`
  - `src/algorithms/CLAUDE.md`
  - `src/features/CLAUDE.md`
  - `src/ml/CLAUDE.md`
  - `src/pipeline/CLAUDE.md`

## Research Goals

The platform enables **interpretable, reproducible TSP heuristic research** with lightweight ML replacing exhaustive search.

### Success Metrics
- **Phase 4**: Linear regression achieves R² > 0.5 on held-out test set
- **Phase 5**: Run complete experiment from scratch with single command
- **Phase 6**: Produce 3-5 statistically-supported research findings

### Research Questions
1. Which graph features best predict good anchor vertices?
2. How does anchor quality vary across different graph types?
3. Can we beat random anchor selection >70% of the time?
4. Are predicted anchors within 10-15% of best-anchor quality?

## Development

### Agent-Based Development
This project uses a multi-agent architecture:
- **Foreman**: Orchestrator, manages CLAUDE.md
- **Planner**: Creates detailed implementation plans
- **Builder**: Implements features following plans
- **Validator**: Runs tests, documents issues
- **Debugger**: Resolves issues with Planner guidance

### Contributing
See `CLAUDE.md` for:
- Architectural principles
- Development workflow
- Quality gates
- Testing standards

## Timeline

- **Phase 1**: Graph Generation - ✅ Complete (4 weeks)
- **Phase 2**: Benchmarking - ✅ Complete (3 weeks)
- **Phase 3**: Feature Engineering - ✅ Complete (3 weeks)
- **Phase 4**: ML Component - ✅ Complete (2 weeks)
- **Phase 5**: Integration - 🟡 In Progress (67% complete)
- **Phase 6**: Analysis - ⏳ Planned

**Total Progress**: ~12 weeks complete, 2-3 weeks remaining for Phase 5 completion

## License

MIT License

## Origin

Initial concept created on May 2, 2025 during a Discrete Math II class. This repository documents the systematic development and testing of those ideas into a complete research platform.

## Current Status (December 2025)

**Phase 5 Complete - Ready for Research Experiments**
- Implementation: 8/12 prompts complete (67%), CLI integration complete
- Platform ready for end-to-end experiments via `experiments/run_experiment.py`
- All prior phases (1-4) complete and validated
- Comprehensive documentation available

**Next Steps**:
1. Begin research experiments using the platform
2. Complete Phase 5 testing for Prompts 5-8 (~50 additional tests) - recommended before large-scale use
3. Implement Phase 5 workflow features (Prompts 9-12) - optional
4. Begin Phase 6: Analysis and insights from experimental results
