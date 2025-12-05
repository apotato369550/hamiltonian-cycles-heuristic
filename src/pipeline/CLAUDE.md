# Pipeline Integration Module (Phase 5)

**Status:** Prompts 1-8 Complete (Implementation + CLI Integration), Tests Needed for 5-8
**Last Updated:** 12-05-2025
**Test Coverage:** 45 tests for Prompts 1-4, implementation complete for Prompts 5-8
**CLI:** `experiments/run_experiment.py` - Ready for end-to-end experiments

---

## Overview

This module provides complete infrastructure for orchestrating multi-stage TSP research experiments with modularity, reproducibility, observability, validation, profiling, parallelization, and fault tolerance.

**Key Components (Prompts 1-4):**
- `PipelineOrchestrator`: Coordinates multi-stage pipeline execution
- `ExperimentConfig`: YAML-based configuration management
- `ExperimentTracker`: Tracks experiment metadata and results
- `ReproducibilityManager`: Ensures reproducible experiments

**Additional Components (Prompts 5-8):**
- `StageValidator`: Validates stage outputs to catch errors early
- `PerformanceMonitor`: Tracks runtime and memory usage
- `ParallelExecutor`: Enables multi-core parallel execution
- `ErrorHandler`: Provides robust error handling and fault tolerance

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

**Prompt 5 (Validation):**
- `StageValidator`: Validates pipeline stage outputs
- `ValidationError`: Exception for validation failures
- Validation methods for all pipeline stages (graphs, benchmarks, features, models)

**Prompt 6 (Profiling):**
- `PerformanceMonitor`: Tracks runtime and memory usage
- `PerformanceMetrics`: Performance data storage
- `RuntimeProfiler`: Detailed profiling with statistics
- `@profile_stage`: Decorator for automatic profiling

**Prompt 7 (Parallel):**
- `ParallelExecutor`: Multi-core parallel execution
- `ParallelConfig`: Parallelization configuration
- `ResourceManager`: CPU and memory resource management
- `create_parallel_executor`: Factory function

**Prompt 8 (Error Handling):**
- `ErrorHandler`: Robust error tracking and management
- `ErrorRecord`: Individual error records
- `Checkpoint`: Save/resume for long operations
- `@retry_with_backoff`: Retry decorator with exponential backoff
- `@try_continue`: Continue on failure decorator
- `@graceful_degradation`: Feature degradation decorator

---

## Test Coverage

**Implemented (45 Tests for Prompts 1-4):**
- `TestPipelineStage` (4 tests)
- `TestPipelineOrchestrator` (5 tests)
- `TestExperimentConfig` (4 tests)
- `TestConfigValidation` (6 tests)
- `TestExperimentTracker` (5 tests)
- `TestExperimentRegistry` (6 tests)
- `TestSeedManager` (6 tests)
- `TestEnvironmentInfo` (3 tests)
- `TestReproducibilityManager` (5 tests)

**Needed (Tests for Prompts 5-8):**
- `TestStageValidator` - Stage output validation (target: ~12 tests)
- `TestPerformanceMonitor` - Runtime/memory profiling (target: ~12 tests)
- `TestParallelExecutor` - Parallel execution (target: ~12 tests)
- `TestErrorHandler` - Error handling and checkpoints (target: ~14 tests)

**Total Target:** ~95 tests (45 implemented, 50 needed)

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

## Stage Validation (Prompt 5)

### Validating Stage Outputs

```python
from pipeline import StageValidator, ValidationError

# Validate graph generation output
try:
    report = StageValidator.validate_graph_generation_output(Path("data/graphs"))
    print(f"Found {report['graph_count']} valid graphs")
except ValidationError as e:
    print(f"Validation failed: {e}")

# Validate benchmarking output
report = StageValidator.validate_benchmarking_output(Path("data/benchmarks"))

# Validate feature extraction output
report = StageValidator.validate_feature_extraction_output(Path("data/features"))

# Validate model training output
report = StageValidator.validate_model_training_output(Path("models"))
```

### Validation Checks

**Graph Generation:**
- Directory exists and contains graph files
- Each graph is valid JSON with required fields
- Graph properties match configuration (size, type)
- Metricity and symmetry as expected

**Benchmarking:**
- All tours are valid Hamiltonian cycles
- Tour weights are reasonable (no NaN/inf)
- Algorithm metadata is complete
- No missing algorithm-graph combinations

**Feature Extraction:**
- Feature values in reasonable ranges
- No NaN or infinite values
- Feature counts match expected extractors
- Labels present and valid

**Model Training:**
- Model files exist and loadable
- Performance metrics reasonable
- Training/test splits valid

---

## Performance Monitoring (Prompt 6)

### Runtime and Memory Profiling

```python
from pipeline import PerformanceMonitor, profile_stage

# Manual monitoring
monitor = PerformanceMonitor()
monitor.start("graph_generation")
# ... do work ...
metrics = monitor.stop("graph_generation")
print(f"Duration: {metrics.duration_seconds}s")
print(f"Memory delta: {metrics.memory_delta_mb}MB")

# Decorator-based profiling
@profile_stage
def expensive_operation(data):
    # Automatically profiled
    return process(data)

# Get all metrics
all_metrics = monitor.get_all_metrics()
monitor.save_metrics(Path("experiment/performance.json"))
```

### Runtime Profiler

```python
from pipeline import RuntimeProfiler

# Profile specific code sections
profiler = RuntimeProfiler()
with profiler.profile("feature_extraction"):
    features = extractor.extract(graph)

# Get detailed statistics
stats = profiler.get_statistics()
# Scaling analysis: runtime vs graph size
profiler.analyze_scaling(graph_sizes, runtimes)
```

---

## Parallel Execution (Prompt 7)

### Parallelizing Graph Operations

```python
from pipeline import ParallelExecutor, ParallelConfig, create_parallel_executor

# Configure parallelization
config = ParallelConfig(
    n_workers=4,  # Number of parallel workers
    backend='multiprocessing',
    max_memory_mb=8000,  # Memory limit
    timeout_seconds=300
)

# Create executor
executor = create_parallel_executor(config)

# Parallel graph generation
graphs = executor.parallel_graph_generation(
    generator=euclidean_gen,
    sizes=[20, 50, 100],
    instances_per_size=10
)

# Parallel benchmarking
results = executor.parallel_benchmarking(
    graphs=graphs,
    algorithms=['nearest_neighbor', 'best_anchor']
)

# Parallel feature extraction
features = executor.parallel_feature_extraction(
    graphs=graphs,
    extractors=[weight_extractor, topo_extractor]
)
```

### Resource Management

```python
from pipeline import ResourceManager

# Monitor system resources
manager = ResourceManager(max_memory_mb=8000, max_cpu_percent=80)
if manager.can_start_task(estimated_memory_mb=2000):
    # Safe to start task
    manager.reserve_resources(memory_mb=2000, cpu_percent=25)
    try:
        result = expensive_task()
    finally:
        manager.release_resources(memory_mb=2000, cpu_percent=25)
```

---

## Error Handling and Fault Tolerance (Prompt 8)

### Retry Patterns

```python
from pipeline import retry_with_backoff, try_continue, graceful_degradation

# Retry failed operations
@retry_with_backoff(max_attempts=3, initial_delay=1.0, backoff_factor=2.0)
def unstable_operation():
    # Automatically retried on failure
    return fetch_data()

# Continue on individual failures
@try_continue
def process_graph(graph):
    # Failures logged, execution continues
    return extract_features(graph)

# Gracefully degrade on feature failures
@graceful_degradation(default_value={})
def expensive_feature(graph):
    # Returns default if computation fails/times out
    return compute_betweenness_centrality(graph)
```

### Error Handler and Checkpoints

```python
from pipeline import ErrorHandler, Checkpoint

# Track errors across pipeline
error_handler = ErrorHandler(output_dir=Path("experiment/errors"))

for graph in graphs:
    try:
        result = process_graph(graph)
    except Exception as e:
        error_handler.record_error(
            stage="benchmarking",
            item_id=graph.graph_id,
            error=e
        )

# Get error summary
summary = error_handler.get_summary()
print(f"Total errors: {summary['total_errors']}")
print(f"Recoverable: {summary['recoverable_count']}")

# Checkpointing for long operations
checkpoint = Checkpoint(path=Path("experiment/checkpoint.json"))
checkpoint.save({'processed': 50, 'results': results})

# Resume from checkpoint
if checkpoint.exists():
    state = checkpoint.load()
    start_from = state['processed']
```

---

## End-to-End Usage Examples

### Running Complete Experiments via CLI

The primary way to use the pipeline is through the CLI entry point:

```bash
# Run complete experiment
python experiments/run_experiment.py config/complete_experiment_template.yaml

# Validate configuration without running
python experiments/run_experiment.py config/my_config.yaml --dry-run

# Run single stage
python experiments/run_experiment.py config/my_config.yaml --stage graph_generation

# Quick test with small config
python experiments/run_experiment.py config/test_config_small.yaml
```

**See `experiments/README.md` for complete CLI documentation.**

### Programmatic Pipeline Usage

For custom experiments or research scripts:

```python
from pathlib import Path
from pipeline import (
    ExperimentConfig,
    ExperimentTracker,
    ReproducibilityManager,
    PipelineOrchestrator,
    PipelineStage
)
from pipeline.stages import (
    create_graph_generation_stage,
    create_benchmarking_stage,
    create_feature_extraction_stage,
    create_training_stage
)

# Load configuration
config = ExperimentConfig.from_yaml("config/my_experiment.yaml")

# Initialize experiment tracking
tracker = ExperimentTracker(
    experiment_id="exp_001",
    name=config.experiment.name,
    description=config.experiment.description,
    config=config,
    output_dir=Path("experiments/exp_001")
)
tracker.start()

# Initialize reproducibility
repro = ReproducibilityManager(
    base_seed=config.experiment.random_seed,
    output_dir=tracker.experiment_dir
)
repro.initialize()

# Create orchestrator
orchestrator = PipelineOrchestrator(
    experiment_dir=tracker.experiment_dir,
    registry=tracker.registry
)

# Add stages
if config.graph_generation.enabled:
    orchestrator.add_stage(create_graph_generation_stage(config, repro))

if config.benchmarking.enabled:
    orchestrator.add_stage(create_benchmarking_stage(config, repro))

if config.feature_engineering.enabled:
    orchestrator.add_stage(create_feature_extraction_stage(config, repro))

if config.model_training.enabled:
    orchestrator.add_stage(create_training_stage(config, repro))

# Run pipeline
try:
    results = orchestrator.run()
    tracker.complete(success=True, results={"stages": [r.to_dict() for r in results]})
except Exception as e:
    tracker.complete(success=False, error_message=str(e))
    raise
```

### Custom Stage Integration

Adding custom stages to the pipeline:

```python
from pipeline import PipelineStage, StageResult, StageStatus
from datetime import datetime

def my_custom_analysis(inputs):
    """Custom analysis stage."""
    result = StageResult("custom_analysis", StageStatus.RUNNING, datetime.now())

    try:
        # Get data from previous stages
        graphs = inputs['graphs']
        benchmarks = inputs['benchmarks']

        # Perform analysis
        analysis_results = analyze_performance(graphs, benchmarks)

        # Save outputs
        output_path = inputs['experiment_dir'] / 'analysis' / 'results.json'
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_results(analysis_results, output_path)

        # Mark success
        result.complete(StageStatus.COMPLETED, outputs={'analysis': analysis_results})
        return result

    except Exception as e:
        result.complete(StageStatus.FAILED, error_message=str(e))
        return result

# Create stage
custom_stage = PipelineStage(
    name="custom_analysis",
    execute_fn=my_custom_analysis,
    required_inputs=['graphs', 'benchmarks', 'experiment_dir'],
    output_keys=['analysis']
)

# Add to orchestrator
orchestrator.add_stage(custom_stage)
```

### Configuration Examples

**Minimal Configuration:**

```yaml
experiment:
  name: quick_test
  random_seed: 42

graph_generation:
  enabled: true
  graph_types:
    - type: euclidean
      sizes: [20]
      instances_per_size: 2

benchmarking:
  enabled: true
  algorithms:
    - {name: nearest_neighbor}
```

**Full Configuration:**

See `config/complete_experiment_template.yaml` for all available options.

**Configuration Reference:**

See `/docs/experiment_configuration_guide.md` for complete field documentation.

---

## Future Work (Prompts 9-12)

**Not Implemented (Workflow Features):**
- Prompt 9: Results analysis and reporting
- Prompt 10: Interactive exploration tools
- Prompt 11: Documentation system
- Prompt 12: Version control and collaboration

These are workflow usage features for consuming pipeline outputs, not core pipeline infrastructure.

---

## Version History

**v1.2 - 12-05-2025 (CLI Integration Complete)**
- Added CLI entry point: `experiments/run_experiment.py`
- Added stage factories in `src/pipeline/stages.py` integrating Phases 1-4
- Added "End-to-End Usage Examples" section with CLI and programmatic usage
- Added configuration templates and documentation references
- Platform now ready for end-to-end experiments
- All integration issues between Phase 5 and Phases 1-4 resolved

**v1.1 - 11-28-2025 (Prompts 5-8 Documented)**
- Updated documentation to reflect Prompts 5-8 implementation
- Added usage examples for validation, profiling, parallel execution, error handling
- Clarified test coverage status (45 tests for 1-4, tests needed for 5-8)
- Updated "Future Work" to exclude implemented prompts

**v1.0 - 11-27-2025 (Prompts 5-8 Implementation)**
- Stage output validation (validation.py - 352 lines)
- Performance monitoring and profiling (profiling.py - 366 lines)
- Parallel execution and resource management (parallel.py - 346 lines)
- Error handling and fault tolerance (error_handling.py - 360 lines)
- All components exported in __init__.py
- **Tests needed** for Prompts 5-8

**v0.9 - 11-22-2025 (Prompts 1-4 Complete)**
- Pipeline orchestration with stage abstraction
- YAML configuration management and validation
- Experiment tracking and registry
- Reproducibility infrastructure (seeds, environment, git)
- 45 comprehensive tests, all passing

---

**Document Maintained By:** Builder Agent
**Last Review:** 11-28-2025
**Status:** Phase 5 (Prompts 1-8) implementation complete, tests needed for 5-8
