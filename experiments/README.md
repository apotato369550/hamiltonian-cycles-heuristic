# Experiment Runner

Command-line tool for running complete TSP research experiments.

## Overview

The experiment runner orchestrates multi-stage research pipelines:

1. **Graph Generation**: Generate diverse TSP instances
2. **Benchmarking**: Run algorithms and collect performance data
3. **Feature Extraction**: Extract vertex features and anchor quality labels
4. **Model Training**: Train ML models to predict good anchors
5. **Evaluation**: Test model predictions against baselines

## Usage

### Basic Commands

```bash
# Run complete experiment
python experiments/run_experiment.py config/my_config.yaml

# Validate configuration (no execution)
python experiments/run_experiment.py config/my_config.yaml --dry-run

# Run single stage
python experiments/run_experiment.py config/my_config.yaml --stage graph_generation
```

### Configuration Files

See `config/` directory for examples:
- `complete_experiment_template.yaml` - Full featured configuration
- `test_config_small.yaml` - Quick integration test

For detailed configuration documentation, see `/docs/experiment_configuration_guide.md`.

### Output Structure

```
experiments/<experiment_id>/
├── metadata.json              # Experiment metadata
├── reproducibility.json       # Seeds, git hash, environment
├── logs/                      # Stage execution logs
├── graphs/                    # Generated graph instances
│   ├── batch_manifest.json
│   └── *.json                 # Individual graphs
├── benchmarks/                # Algorithm performance results
│   └── results.db
├── features/                  # Extracted features + labels
│   └── feature_dataset.csv
└── models/                    # Trained ML models
    ├── linear_ridge_0.pkl
    └── random_forest_1.pkl
```

### Stage Selection

Run individual stages when:
- Debugging specific stage failures
- Iterating on stage parameters
- Re-running failed stages

**Important**: Stages have dependencies:
- `benchmarking` requires `graph_generation`
- `feature_extraction` requires `graph_generation` + `benchmarking`
- `training` requires `feature_extraction`

### Reproducibility

Set `random_seed` in config for full reproducibility:

```yaml
experiment:
  random_seed: 42
```

The runner captures:
- Complete configuration
- Git commit hash
- Python/package versions
- Random seeds for each stage

### Experiment Registry

All experiments tracked in `experiments/registry.json`:

```json
{
  "experiments": [
    {
      "experiment_id": "baseline_v1_20251205_120000_abc123",
      "name": "baseline_comparison_v1",
      "status": "completed",
      "start_time": "2025-12-05T12:00:00",
      "duration_seconds": 3842.5
    }
  ]
}
```

### Common Issues

**Import Errors**: Ensure all dependencies installed
```bash
pip install -r requirements.txt
```

**Missing Graphs**: Run graph_generation stage first
```bash
python experiments/run_experiment.py config.yaml --stage graph_generation
```

**Timeout Errors**: Increase timeout in config
```yaml
benchmarking:
  timeout_seconds: 600  # Increase from default 300
```

## Examples

### Quick Test

```bash
python experiments/run_experiment.py config/test_config_small.yaml
```

Runtime: ~2-3 minutes
Graphs: 4 (2 sizes × 2 instances)
Algorithms: 2

### Full Baseline Comparison

```bash
python experiments/run_experiment.py config/complete_experiment_template.yaml
```

Runtime: ~30-60 minutes
Graphs: 150 (multiple types and sizes)
Algorithms: 4

## Development

### Adding New Stages

1. Create stage factory in `src/pipeline/stages.py`
2. Add to runner in `experiments/run_experiment.py`
3. Add config section to template YAML

See existing stages for reference implementation.

## Help

For detailed implementation docs, see:
- `/src/pipeline/CLAUDE.md` - Pipeline architecture
- `/CLAUDE.md` - Project overview
- `/docs/experiment_configuration_guide.md` - Complete configuration reference
- `/guides/05_pipeline_integration_workflow.md` - Implementation guide
