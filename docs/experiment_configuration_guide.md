# Experiment Configuration Guide

**Last Updated:** December 5, 2025
**Version:** 1.0

---

## Overview

This guide provides a complete reference for configuring TSP research experiments using the YAML configuration system. All experiments are defined through YAML files that specify graph generation, algorithm benchmarking, feature extraction, and model training parameters.

## Quick Start

```yaml
experiment:
  name: "my_experiment"
  description: "Brief description"
  random_seed: 42
  output_dir: "experiments/my_exp"

graph_generation:
  enabled: true
  # ... configuration

benchmarking:
  enabled: true
  # ... configuration

feature_extraction:
  enabled: true
  # ... configuration

training:
  enabled: true
  # ... configuration
```

---

## Configuration Structure

### Top-Level Fields

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `experiment` | Object | Yes | - | Experiment metadata and settings |
| `graph_generation` | Object | No | - | Graph generation configuration |
| `benchmarking` | Object | No | - | Algorithm benchmarking configuration |
| `feature_extraction` | Object | No | - | Feature extraction configuration |
| `training` | Object | No | - | Model training configuration |

**Note:** Field aliases are supported for compatibility:
- `feature_extraction` = `feature_engineering`
- `training` = `model_training`

---

## Experiment Section

Top-level experiment metadata.

### Fields

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `name` | String | Yes | - | Experiment name (used for tracking) |
| `description` | String | No | `""` | Human-readable description |
| `random_seed` | Integer | No | `42` | Master random seed for reproducibility |
| `output_dir` | String | No | `"experiments"` | Base output directory |

### Example

```yaml
experiment:
  name: "baseline_comparison_v1"
  description: "Compare anchor-based heuristics to nearest neighbor baseline"
  random_seed: 42
  output_dir: "experiments/baseline_v1"
```

---

## Graph Generation Section

Configures graph instance generation for experiments.

### Fields

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `enabled` | Boolean | No | `true` | Enable graph generation stage |
| `batch_name` | String | No | `"default_batch"` | Batch identifier for organizing graphs |
| `types` | List[Object] | Yes | - | List of graph type specifications |
| `output_dir` | String | No | `"data/graphs"` | Output directory for graphs |
| `save_format` | String | No | `"json"` | Format: `"json"` or `"pickle"` |

**Alias:** `graph_types` can be used instead of `types`

### Graph Type Specification

Each entry in `types` list:

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `type` | String | Yes | - | Graph type: `"euclidean"`, `"metric"`, `"quasi_metric"`, `"random"` |
| `sizes` | List[Integer] | Yes | - | List of graph sizes to generate |
| `instances_per_size` | Integer | Yes | - | Number of instances per size |
| `weight_range` | List[Float] | No | `[1.0, 100.0]` | Min/max edge weights |
| `dimension` | Integer | No | `2` | Euclidean dimension (euclidean graphs only) |
| `strategy` | String | No | `"completion"` | Generation strategy for metric/quasi-metric: `"completion"` or `"mst"` |
| `distribution` | String | No | `"uniform"` | Distribution for random graphs: `"uniform"`, `"normal"`, etc. |

### Examples

#### Euclidean Graphs

```yaml
graph_generation:
  enabled: true
  batch_name: "euclidean_batch_001"
  types:
    - type: "euclidean"
      sizes: [20, 50, 100]
      instances_per_size: 10
      dimension: 2
      weight_range: [1.0, 100.0]
```

#### Mixed Graph Types

```yaml
graph_generation:
  enabled: true
  batch_name: "mixed_graphs"
  types:
    # 2D Euclidean
    - type: "euclidean"
      sizes: [20, 50]
      instances_per_size: 5
      dimension: 2

    # 3D Euclidean
    - type: "euclidean"
      sizes: [20, 50]
      instances_per_size: 5
      dimension: 3

    # Metric graphs
    - type: "metric"
      sizes: [50, 100]
      instances_per_size: 5
      strategy: "completion"
      weight_range: [10.0, 50.0]

    # Quasi-metric (asymmetric)
    - type: "quasi_metric"
      sizes: [30, 50]
      instances_per_size: 3
      strategy: "completion"

    # Random graphs (baseline)
    - type: "random"
      sizes: [20, 50]
      instances_per_size: 5
      distribution: "uniform"
```

---

## Benchmarking Section

Configures algorithm benchmarking on generated graphs.

### Fields

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `enabled` | Boolean | No | `true` | Enable benchmarking stage |
| `algorithms` | List[Object] | Yes | - | List of algorithm specifications |
| `exhaustive_anchors` | Boolean | No | `false` | Test all possible anchors (for anchor-based algorithms) |
| `timeout_seconds` | Float | No | `300.0` | Timeout per algorithm-graph pair |
| `output_dir` | String | No | `"results/benchmarks"` | Output directory |
| `save_format` | String | No | `"json"` | Format: `"json"`, `"csv"`, or `"sqlite"` |

**Alias:** `storage_format` can be used instead of `save_format`

### Algorithm Specification

Each entry in `algorithms` list:

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `name` | String | Yes | - | Algorithm name from registry |
| `params` | Object | No | `{}` | Algorithm-specific parameters |

### Available Algorithms

#### Baseline Algorithms

1. **`nearest_neighbor`** - Nearest neighbor heuristic
   - Params:
     - `strategy`: `"first_vertex"` (default) or `"best_start"` (tries all starting vertices)

2. **`greedy_edge`** - Greedy edge selection
   - No additional params

3. **`held_karp`** - Exact algorithm (Held-Karp dynamic programming)
   - **Warning:** Only use for small graphs (n ≤ 20)
   - No additional params

#### Anchor-Based Heuristics

1. **`single_anchor`** - Single anchor construction
   - Params:
     - `bidirectional`: `true` (use v2 - bidirectional) or `false` (use v1 - single direction)
     - `anchor_vertex`: Integer (specific anchor, optional)

2. **`best_anchor`** - Try all anchors, return best
   - No additional params

3. **`multi_anchor`** - Multi-anchor construction
   - Params:
     - `num_anchors`: Integer (number of anchors)
     - `strategy`: `"distributed"` or `"clustered"`

### Examples

#### Basic Benchmarking

```yaml
benchmarking:
  enabled: true
  algorithms:
    - name: "nearest_neighbor"
      params:
        strategy: "best_start"

    - name: "greedy_edge"
      params: {}

    - name: "single_anchor"
      params:
        bidirectional: true

  timeout_seconds: 300
  save_format: "json"
```

#### Exhaustive Anchor Testing

```yaml
benchmarking:
  enabled: true
  exhaustive_anchors: true  # Test all possible anchors
  algorithms:
    - name: "single_anchor"
      params:
        bidirectional: true

  timeout_seconds: 60
```

#### Comparing Multiple Strategies

```yaml
benchmarking:
  enabled: true
  algorithms:
    # Baseline
    - name: "nearest_neighbor"
      params: {strategy: "best_start"}

    # Anchor variants
    - name: "single_anchor"
      params: {bidirectional: false}  # v1

    - name: "single_anchor"
      params: {bidirectional: true}   # v2

    - name: "best_anchor"
      params: {}

    # Multi-anchor
    - name: "multi_anchor"
      params:
        num_anchors: 3
        strategy: "distributed"

  timeout_seconds: 600
```

---

## Feature Extraction Section

Configures feature extraction for machine learning.

### Fields

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `enabled` | Boolean | No | `true` | Enable feature extraction stage |
| `extractors` | List[String] | Yes | - | List of feature extractor names |
| `labeling_strategy` | String | No | `"rank_based"` | Anchor quality labeling strategy |
| `labeling_params` | Object | No | `{}` | Parameters for labeling strategy |
| `output_dir` | String | No | `"results/features"` | Output directory |
| `save_format` | String | No | `"csv"` | Format: `"csv"` or `"parquet"` |

**Aliases:**
- `extractors` = `feature_groups`
- `save_format` = `output_format`

### Available Feature Extractors

| Extractor | Description | Feature Count |
|-----------|-------------|---------------|
| `weight_based` | Edge weight statistics | 20-46 features (symmetric/asymmetric) |
| `topological` | Centrality, clustering | ~15 features |
| `mst_based` | Minimum spanning tree properties | ~10 features |
| `neighborhood` | k-NN, density, Voronoi | ~31 features |
| `heuristic` | Tour estimates, baselines | ~15 features |
| `graph_context` | Graph-level properties | ~12 features |

### Labeling Strategies

| Strategy | Description | Output |
|----------|-------------|--------|
| `rank_based` | Rank vertices by tour quality | Float (percentile rank) |
| `absolute_percentile` | Top/bottom percentile thresholds | Integer class (0=bad, 1=avg, 2=good) |
| `binary` | Binary good/bad classification | Binary (0=bad, 1=good) |
| `multiclass` | Multiple quality classes | Integer (0-4 classes) |
| `relative_gap` | Gap from optimal | Float (normalized gap) |

### Labeling Parameters

#### `rank_based`
- No additional params

#### `absolute_percentile`
```yaml
labeling_params:
  percentile_top: 20    # Top 20% are "good"
  percentile_bottom: 20 # Bottom 20% are "bad"
```

#### `binary`
```yaml
labeling_params:
  threshold: 10.0  # Percent worse than best
```

#### `multiclass`
```yaml
labeling_params:
  num_classes: 5  # Number of quality classes
```

### Examples

#### Basic Feature Extraction

```yaml
feature_extraction:
  enabled: true
  extractors:
    - weight_based
    - topological
    - mst_based

  labeling_strategy: "rank_based"
  output_format: "csv"
```

#### Full Feature Set

```yaml
feature_extraction:
  enabled: true
  extractors:
    - weight_based
    - topological
    - mst_based
    - neighborhood
    - heuristic
    - graph_context

  labeling_strategy: "absolute_percentile"
  labeling_params:
    percentile_top: 15
    percentile_bottom: 15

  output_format: "csv"
```

#### Minimal Features (Fast)

```yaml
feature_extraction:
  enabled: true
  extractors:
    - weight_based  # Only basic features

  labeling_strategy: "binary"
  labeling_params:
    threshold: 15.0

  output_format: "csv"
```

---

## Training Section

Configures machine learning model training.

### Fields

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `enabled` | Boolean | No | `true` | Enable training stage |
| `models` | List[Object] | Yes | - | List of model specifications |
| `split_strategy` | String | No | `"stratified_graph"` | Data splitting strategy |
| `train_ratio` | Float | No | `0.7` | Training set ratio |
| `val_ratio` | Float | No | `0.15` | Validation set ratio |
| `test_ratio` | Float | No | `0.15` | Test set ratio |
| `output_dir` | String | No | `"models"` | Output directory |
| `save_models` | Boolean | No | `true` | Save trained models to disk |

**Additional fields:**
- `problem_type`: `"regression"`, `"classification"`, or `"ranking"` (default: `"regression"`)
- `test_split`: Alternative to `test_ratio`
- `stratify_by`: Column name for stratification
- `cross_validation`: Cross-validation configuration (optional)

### Model Specification

Each entry in `models` list:

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `type` | String | Yes | - | Model type |
| `model_variant` | String | No | - | Specific variant (for linear models) |
| `params` | Object | No | `{}` | Model hyperparameters |

### Available Models

#### Linear Models (Interpretable)

1. **`linear_regression`** - Ordinary least squares
   - `model_variant`: `"ols"` (default), `"ridge"`, `"lasso"`, `"elasticnet"`
   - Params (Ridge/Lasso/ElasticNet):
     - `alpha`: Regularization strength (default: 1.0)
     - `l1_ratio`: ElasticNet mix (default: 0.5)

2. **`linear_ridge`** - Ridge regression (shorthand)
   - Params:
     - `alpha`: Regularization (default: 1.0)

3. **`linear_lasso`** - Lasso regression (shorthand)
   - Params:
     - `alpha`: Regularization (default: 1.0)

#### Tree-Based Models (Comparison)

1. **`decision_tree`** - Single decision tree
   - Params:
     - `max_depth`: Maximum depth (default: None)
     - `min_samples_split`: Min samples to split (default: 2)

2. **`random_forest`** - Random forest ensemble
   - Params:
     - `n_estimators`: Number of trees (default: 100)
     - `max_depth`: Maximum depth (default: None)
     - `random_state`: Random seed

3. **`gradient_boosting`** - Gradient boosting
   - Params:
     - `n_estimators`: Number of boosting stages (default: 100)
     - `learning_rate`: Learning rate (default: 0.1)
     - `max_depth`: Maximum depth (default: 3)

### Split Strategies

| Strategy | Description |
|----------|-------------|
| `random` | Random split |
| `graph_based` | Split by graph (keeps vertices from same graph together) |
| `stratified_graph` | Stratified by graph type (default) |
| `graph_type_holdout` | Hold out entire graph type for testing |
| `size_holdout` | Hold out specific graph sizes for testing |

### Cross-Validation Configuration

```yaml
cross_validation:
  enabled: true
  n_folds: 5
  strategy: "stratified"  # "kfold", "stratified", or "group"
```

### Examples

#### Simple Linear Models

```yaml
training:
  enabled: true
  models:
    - type: "linear_ridge"
      params:
        alpha: 1.0

    - type: "linear_lasso"
      params:
        alpha: 0.1

  problem_type: "regression"
  test_split: 0.2
  stratify_by: "graph_type"
```

#### Comprehensive Model Comparison

```yaml
training:
  enabled: true
  models:
    # Linear models (interpretable)
    - type: "linear_regression"
      model_variant: "ols"
      params: {}

    - type: "linear_regression"
      model_variant: "ridge"
      params: {alpha: 1.0}

    - type: "linear_regression"
      model_variant: "lasso"
      params: {alpha: 0.1}

    - type: "linear_regression"
      model_variant: "elasticnet"
      params: {alpha: 1.0, l1_ratio: 0.5}

    # Tree models (comparison)
    - type: "decision_tree"
      params: {max_depth: 10}

    - type: "random_forest"
      params:
        n_estimators: 100
        max_depth: 15
        random_state: 42

    - type: "gradient_boosting"
      params:
        n_estimators: 100
        learning_rate: 0.1
        max_depth: 5

  split_strategy: "stratified_graph"
  train_ratio: 0.7
  val_ratio: 0.15
  test_ratio: 0.15
```

#### Hyperparameter Tuning with Cross-Validation

```yaml
training:
  enabled: true
  models:
    - type: "random_forest"
      params:
        n_estimators: 200
        max_depth: 20

  cross_validation:
    enabled: true
    n_folds: 5
    strategy: "group"  # Keep graphs together

  split_strategy: "graph_based"
  test_split: 0.2
```

---

## Complete Configuration Examples

### Small Test Configuration

For quick testing and development:

```yaml
experiment:
  name: "quick_test"
  description: "Fast integration test"
  random_seed: 42
  output_dir: "experiments/test"

graph_generation:
  enabled: true
  batch_name: "test_batch"
  types:
    - type: "euclidean"
      sizes: [10, 15]
      instances_per_size: 2
      dimension: 2

benchmarking:
  enabled: true
  algorithms:
    - name: "nearest_neighbor"
      params: {}
    - name: "single_anchor"
      params: {bidirectional: true}
  timeout_seconds: 60

feature_extraction:
  enabled: true
  extractors:
    - weight_based
    - topological
  labeling_strategy: "rank_based"

training:
  enabled: true
  models:
    - type: "linear_ridge"
      params: {alpha: 1.0}
  test_split: 0.3
```

**Expected runtime:** 2-3 minutes

### Full Baseline Comparison

Production experiment configuration:

```yaml
experiment:
  name: "baseline_comparison_v1"
  description: "Compare anchor-based heuristics to baselines across multiple graph types"
  random_seed: 42
  output_dir: "experiments/baseline_v1"

graph_generation:
  enabled: true
  batch_name: "baseline_graphs_001"
  types:
    - type: "euclidean"
      sizes: [20, 50, 100]
      instances_per_size: 10
      dimension: 2
      weight_range: [1.0, 100.0]

    - type: "metric"
      sizes: [50, 100]
      instances_per_size: 5
      strategy: "completion"
      weight_range: [10.0, 50.0]

    - type: "random"
      sizes: [20, 50]
      instances_per_size: 5

benchmarking:
  enabled: true
  algorithms:
    - name: "nearest_neighbor"
      params: {strategy: "best_start"}
    - name: "greedy_edge"
      params: {}
    - name: "single_anchor"
      params: {bidirectional: true}
    - name: "best_anchor"
      params: {}
  exhaustive_anchors: true
  timeout_seconds: 300

feature_extraction:
  enabled: true
  extractors:
    - weight_based
    - topological
    - mst_based
    - neighborhood
    - heuristic
    - graph_context
  labeling_strategy: "rank_based"
  output_format: "csv"

training:
  enabled: true
  models:
    - type: "linear_ridge"
      params: {alpha: 1.0}
    - type: "linear_lasso"
      params: {alpha: 0.1}
    - type: "random_forest"
      params:
        n_estimators: 100
        max_depth: 10
        random_state: 42
  split_strategy: "stratified_graph"
  train_ratio: 0.7
  val_ratio: 0.15
  test_ratio: 0.15
```

**Expected runtime:** 30-60 minutes

---

## Running Experiments

### Basic Usage

```bash
# Validate configuration
python experiments/run_experiment.py config/my_config.yaml --dry-run

# Run complete experiment
python experiments/run_experiment.py config/my_config.yaml

# Run specific stage only
python experiments/run_experiment.py config/my_config.yaml --stage graph_generation
```

### Stage Dependencies

Stages must be run in order:
1. `graph_generation` (no dependencies)
2. `benchmarking` (requires graphs)
3. `feature_extraction` (requires graphs + benchmarks)
4. `training` (requires features)

### Output Structure

```
experiments/<experiment_id>/
├── metadata.json              # Experiment metadata
├── reproducibility.json       # Seeds, git hash, environment
├── logs/                      # Stage execution logs
├── graphs/                    # Generated graph instances
│   ├── batch_manifest.json
│   └── *.json
├── benchmarks/                # Algorithm results
│   └── results.db
├── features/                  # Feature dataset
│   └── feature_dataset.csv
└── models/                    # Trained models
    ├── linear_ridge_0.pkl
    └── random_forest_1.pkl
```

---

## Tips and Best Practices

### Performance

1. **Start small**: Test with `test_config_small.yaml` first
2. **Disable expensive features**: `heuristic` and `graph_context` are slowest
3. **Limit graph sizes**: Start with n ≤ 50 for development
4. **Use timeout**: Set appropriate `timeout_seconds` for large graphs

### Reproducibility

1. **Always set `random_seed`**: Ensures reproducible results
2. **Document changes**: Update `description` field for each experiment
3. **Use meaningful names**: Name experiments descriptively
4. **Track experiments**: Check `experiments/registry.json`

### Feature Engineering

1. **Start with `weight_based`**: Fastest and most informative
2. **Add `topological` next**: Good accuracy/speed tradeoff
3. **Use `exhaustive_anchors`**: Required for label generation
4. **Choose labeling wisely**: `rank_based` is most flexible

### Model Training

1. **Start with linear**: Ridge/Lasso are fast and interpretable
2. **Use cross-validation**: For hyperparameter tuning
3. **Stratify by graph type**: Ensures balanced test set
4. **Compare to baselines**: Always include nearest_neighbor

---

## Troubleshooting

### Common Errors

**"graph_types cannot be empty"**
- Add at least one graph type specification

**"algorithms cannot be empty"**
- Add at least one algorithm to benchmarking

**"Split ratios must sum to 1.0"**
- Ensure `train_ratio + val_ratio + test_ratio = 1.0`

**Import Errors**
- Install dependencies: `pip install -r requirements.txt`

**Timeout Errors**
- Increase `timeout_seconds` in benchmarking
- Reduce graph sizes for testing

**Memory Errors**
- Reduce `instances_per_size`
- Disable expensive feature extractors
- Process graphs in smaller batches

---

## Reference

### Complete Field Index

Quick reference of all configuration fields:

- **experiment**: name, description, random_seed, output_dir
- **graph_generation**: enabled, batch_name, types, output_dir, save_format
  - **type spec**: type, sizes, instances_per_size, weight_range, dimension, strategy, distribution
- **benchmarking**: enabled, algorithms, exhaustive_anchors, timeout_seconds, output_dir, save_format
  - **algorithm spec**: name, params
- **feature_extraction**: enabled, extractors, labeling_strategy, labeling_params, output_dir, save_format
- **training**: enabled, models, split_strategy, train_ratio, val_ratio, test_ratio, output_dir, save_models
  - **model spec**: type, model_variant, params
  - **Optional**: problem_type, test_split, stratify_by, cross_validation

---

**Document Version:** 1.0
**Last Updated:** December 5, 2025
**Maintained By:** TSP Research Team

