# Pipeline Integration Module (Phase 5)

**Status:** Prompts 1-4 Complete
**Last Updated:** 11-22-2025
**Test Coverage:** 45 tests, all passing

---

## Overview

This module provides infrastructure for orchestrating multi-stage TSP research experiments with modularity, reproducibility, and observability.

**Key Components:**
- `PipelineOrchestrator`: Coordinates multi-stage pipeline execution
- `ExperimentConfig`: YAML-based configuration management
- `ExperimentTracker`: Tracks experiment metadata and results
- `ReproducibilityManager`: Ensures reproducible experiments

---

## Architecture (Prompt 1)

### Pipeline Stages

Each stage is:
- **Modular**: Can run independently
- **Idempotent**: Same inputs → same outputs
- **Resumable**: Skips if outputs exist (optional)
- **Observable**: Clear logging

```python
from pipeline import PipelineStage, PipelineOrchestrator, StageResult, StageStatus

def my_stage_fn(inputs):
    result = StageResult("my_stage", StageStatus.COMPLETED, datetime.now())
    result.outputs = {'data': process(inputs['input'])}
    return result

stage = PipelineStage(
    name="my_stage",
    execute_fn=my_stage_fn,
    required_inputs=['input'],
    output_keys=['data'],
    skip_if_exists=True  # Resume support
)

orchestrator = PipelineOrchestrator(experiment_dir=Path("experiments/exp1"))
orchestrator.add_stage(stage)
results = orchestrator.run({'input': data})
```

### Data Flow

```
initial_inputs → Stage 1 → outputs → Stage 2 → outputs → Stage 3 → final_outputs
```

Outputs from each stage automatically become inputs to next stage.

---

## Configuration Management (Prompt 2)

### YAML Configuration

```yaml
name: baseline_experiment
description: Compare anchor heuristics
random_seed: 42

graph_generation:
  enabled: true
  graph_types:
    - type: euclidean
      sizes: [20, 50, 100]
      instances_per_size: 10

benchmarking:
  enabled: true
  algorithms:
    - {name: nearest_neighbor}
    - {name: best_anchor}
  timeout_seconds: 300

feature_engineering:
  enabled: true
  feature_groups: [weight_based, topological, mst_based]
  labeling_strategy: rank_based

model_training:
  enabled: true
  models:
    - {type: linear_regression, model_variant: ridge, params: {alpha: 1.0}}
  split_strategy: stratified_graph
```

### Usage

```python
from pipeline import ExperimentConfig, ConfigValidator

# Load and validate
config = ExperimentConfig.from_yaml("config.yaml")
ConfigValidator.validate(config)

# Or create template
ConfigValidator.create_template("template.yaml")
```

### Validation

Catches:
- Missing required fields
- Invalid graph types, algorithms, features
- Split ratios that don't sum to 1.0
- Invalid strategies

---

## Experiment Tracking (Prompt 3)

### Tracking Experiments

```python
from pipeline import ExperimentTracker, ExperimentStatus

tracker = ExperimentTracker(
    experiment_id="exp_123",
    name="baseline_v1",
    description="Testing anchor heuristics",
    config=config.to_dict(),
    output_dir=Path("experiments/baseline_v1")
)

tracker.start()
# ... run experiment ...
tracker.complete(results_summary={'best_r2': 0.85})

# Or on failure
tracker.fail("Algorithm timeout")
```

### Directory Structure

```
experiments/baseline_v1/
  metadata.json          # Experiment metadata
  logs/
    generation.log
    benchmarking.log
    features.log
    training.log
  data/
    graphs/
    benchmarks/
    features/
  models/
  reports/
```

### Experiment Registry

```python
from pipeline import ExperimentRegistry

registry = ExperimentRegistry(Path("experiments/registry.json"))

# Register experiment
registry.register(tracker.metadata)

# Query
completed = registry.list_by_status(ExperimentStatus.COMPLETED)
baseline_exps = registry.list_by_name("baseline")

# Summary
summary = registry.get_summary()
# {'total_experiments': 15, 'by_status': {...}, 'recent_experiments': [...]}
```

---

## Reproducibility (Prompt 4)

### Seed Management

```python
from pipeline import ReproducibilityManager

manager = ReproducibilityManager(master_seed=42)
manager.initialize()  # Sets all random seeds

# Get stage-specific seeds
graph_seed = manager.seed_manager.get_graph_seed(graph_index=5)
split_seed = manager.seed_manager.get_split_seed()
model_seed = manager.seed_manager.get_model_seed(model_index=0)
```

### Environment Tracking

```python
# Capture current environment
env_info = manager.environment_info
print(env_info.get_key_versions())
# {'python': '3.12', 'numpy': '1.26.0', ...}

# Save for reproduction
manager.save_reproducibility_info(Path("experiment/repro.json"))

# Verify environment matches
result = manager.verify_environment(saved_env_info)
# {'environment_matches': True/False, 'differences': {...}}
```

### Git Versioning

```python
# Automatically captures git commit
commit = manager.git_commit  # 'abc123def456...'

# Check reproducibility
check = manager.check_reproducibility()
# {
#   'is_reproducible': True,
#   'warnings': [],
#   'git_commit': 'abc123...',
#   'has_uncommitted_changes': False,
#   'on_tagged_release': True
# }
```

---

## Implemented Components

**Prompt 1 (Orchestrator):**
- `PipelineStage`: Individual stage with execute function
- `PipelineOrchestrator`: Multi-stage coordinator
- `StageResult`: Stage execution results
- `StageStatus`: PENDING, RUNNING, COMPLETED, FAILED, SKIPPED

**Prompt 2 (Configuration):**
- `ExperimentConfig`: Complete experiment configuration
- `ConfigValidator`: Validation with helpful error messages
- `GraphGenConfig`, `BenchmarkConfig`, `FeatureConfig`, `ModelConfig`
- YAML load/save with template generation

**Prompt 3 (Tracking):**
- `ExperimentTracker`: Tracks individual experiment
- `ExperimentRegistry`: Registry of all experiments
- `ExperimentMetadata`: Complete experiment metadata
- `ExperimentStatus`: PENDING, RUNNING, COMPLETED, FAILED, CANCELLED

**Prompt 4 (Reproducibility):**
- `ReproducibilityManager`: Complete reproducibility system
- `SeedManager`: Deterministic seed propagation
- `EnvironmentInfo`: Python/package version tracking
- Git commit and diff capture

---

## Test Coverage (45 Tests)

**Test Classes:**
- `TestPipelineStage` (4 tests)
- `TestPipelineOrchestrator` (5 tests)
- `TestExperimentConfig` (4 tests)
- `TestConfigValidation` (6 tests)
- `TestExperimentTracker` (5 tests)
- `TestExperimentRegistry` (6 tests)
- `TestSeedManager` (6 tests)
- `TestEnvironmentInfo` (3 tests)
- `TestReproducibilityManager` (5 tests)

**Run tests:**
```bash
python3 -m unittest src.tests.test_phase5_pipeline -v
```

---

## Integration with Other Phases

**Phase 1 (Graphs) → Pipeline:**
- Graph generation becomes Stage 1
- Outputs: graph collection saved to disk

**Phase 2 (Benchmarking) → Pipeline:**
- Benchmarking becomes Stage 2
- Inputs: graph collection
- Outputs: algorithm performance database

**Phase 3 (Features) → Pipeline:**
- Feature extraction becomes Stage 3
- Inputs: graphs + benchmark results
- Outputs: ML dataset (features + labels)

**Phase 4 (ML) → Pipeline:**
- Model training becomes Stage 4
- Inputs: ML dataset
- Outputs: trained models + performance reports

**Pipeline orchestrates all phases with:**
- Reproducible seed management
- Configuration-driven execution
- Experiment tracking and metadata
- Resumability on failures

---

## Design Decisions

**Why filesystem as data layer?**
- Simple, no database complexity
- Easy to inspect intermediate results
- Natural resumption (check if files exist)
- Works with version control (git-lfs for data)

**Why YAML for config?**
- Human-readable and editable
- Standard format, widely supported
- Easier to version control than JSON
- Less error-prone than Python code

**Why separate tracking from orchestration?**
- Single Responsibility Principle
- Tracker can be used standalone
- Registry persists across runs
- Enables comparison between experiments

**Why master seed + derived seeds?**
- Single seed controls entire experiment
- Deterministic derivation for each stage
- Easy to re-run with different seed
- Stage seeds independent (no conflicts)

---

## Common Patterns

### Complete Pipeline

```python
# 1. Load config
config = ExperimentConfig.from_yaml("config.yaml")
ConfigValidator.validate(config)

# 2. Set up tracking
registry = ExperimentRegistry(Path("experiments/registry.json"))
exp_id = registry.generate_experiment_id(config.name)

tracker = ExperimentTracker(
    experiment_id=exp_id,
    name=config.name,
    description=config.description,
    config=config.to_dict(),
    output_dir=Path(config.output_dir) / exp_id
)

# 3. Set up reproducibility
repro = ReproducibilityManager(master_seed=config.random_seed)
repro.initialize()
tracker.update_environment(repro.environment_info.to_dict())
tracker.update_git_commit(repro.git_commit)

# 4. Create pipeline stages
orchestrator = PipelineOrchestrator(tracker.output_dir)

if config.graph_generation.enabled:
    orchestrator.add_stage(create_graph_gen_stage(config, repro))
if config.benchmarking.enabled:
    orchestrator.add_stage(create_benchmark_stage(config, repro))
if config.feature_engineering.enabled:
    orchestrator.add_stage(create_feature_stage(config, repro))
if config.model_training.enabled:
    orchestrator.add_stage(create_training_stage(config, repro))

# 5. Run
tracker.start()
try:
    results = orchestrator.run()
    tracker.complete(compute_summary(results))
except Exception as e:
    tracker.fail(str(e))
    raise

# 6. Save artifacts
orchestrator.save_manifest()
repro.save_reproducibility_info(tracker.output_dir / "reproducibility.json")
registry.register(tracker.metadata)
```

---

## Future Work (Prompts 5-12)

**Not Implemented:**
- Prompt 5: Automated testing and validation
- Prompt 6: Performance monitoring and profiling
- Prompt 7: Parallel execution and scaling
- Prompt 8: Error handling and fault tolerance
- Prompt 9: Results analysis and reporting
- Prompt 10: Interactive exploration tools
- Prompt 11: Documentation system
- Prompt 12: Version control and collaboration

These are planned for future phases but not required for basic pipeline functionality.

---

## Version History

**v1.0 - 11-22-2025 (Prompts 1-4 Complete)**
- Pipeline orchestration with stage abstraction
- YAML configuration management and validation
- Experiment tracking and registry
- Reproducibility infrastructure (seeds, environment, git)
- 45 comprehensive tests, all passing

---

**Document Maintained By:** Builder Agent
**Last Review:** 11-22-2025
**Status:** Phase 5 (Prompts 1-4) production-ready
