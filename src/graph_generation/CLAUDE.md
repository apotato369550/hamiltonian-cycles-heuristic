# Graph Generation System (Phase 1)

## Purpose
Controlled graph generation for TSP research. Generates diverse graph types with configurable properties for systematic algorithm benchmarking and feature engineering experiments.

**Status**: COMPLETE (validated multiple times, production-ready)
**Test Coverage**: 34 tests, 100% pass rate

---

## Package Structure

```
graph_generation/
├── CLAUDE.md                    # This file - package documentation
├── __init__.py                  # Package initialization
├── euclidean_generator.py       # Euclidean graphs from 2D coordinates
├── metric_generator.py          # Metric graphs (symmetric, triangle inequality)
├── quasi_metric_generator.py    # Quasi-metric (asymmetric triangle inequality)
├── random_generator.py          # Random graphs (no constraints)
├── graph_utils.py               # Utilities (verification, storage, visualization)
└── batch_generator.py           # Batch generation from YAML configs
```

---

## Graph Types

### 1. Euclidean Graphs
**Properties**: Symmetric, metric, derived from 2D coordinates
**Use Case**: Geometric TSP instances, spatial problems
**Generator**: `euclidean_generator.py`
**Key Feature**: Weights equal geometric distances

### 2. Metric Graphs
**Properties**: Symmetric, satisfy triangle inequality
**Use Case**: General TSP with metric guarantee
**Generator**: `metric_generator.py`
**Strategies**: MST-based or completion-based

### 3. Quasi-Metric Graphs
**Properties**: Asymmetric, forward triangle inequality only
**Use Case**: Directed TSP, one-way costs
**Generator**: `quasi_metric_generator.py`

### 4. Random Graphs
**Properties**: No constraints, fully customizable
**Use Case**: Stress testing, worst-case analysis
**Generator**: `random_generator.py`
**Distributions**: Uniform, normal, exponential

---

## Critical Technical Principles

### Principle 1: Euclidean Property Preservation
**Rule**: Scale COORDINATES, not edge weights

```python
# CORRECT - Scale coordinates
coords = coords * scale_factor

# WRONG - Scale weights directly
weights = weights * scale_factor  # Breaks Euclidean property
```

**Rationale**: Coordinate scaling preserves geometric relationships and Euclidean property.

### Principle 2: Quasi-Metric Triangle Inequality
**Rule**: Check forward direction only: d(x,z) ≤ d(x,y) + d(y,z)

**Do NOT check**: d(j,k) ≤ d(i,j) + d(i,k) (requires going backwards)

### Principle 3: MST vs Completion Strategies

**MST Strategy**:
- Wide weight distributions
- Use for: Normal metric graphs

**Completion Strategy**:
- Narrow weight distributions (weights stay in specified range)
- Use for: Controlled distributions, quasi-metrics
- Example: Range (10.0, 10.01) → std dev ~0.05 (vs MST ~14.6)

---

## Usage

### Generate Single Graph
```python
from graph_generation.euclidean_generator import generate_euclidean_graph

graph = generate_euclidean_graph(
    num_vertices=20,
    weight_range=(1.0, 100.0),
    random_seed=42
)
```

### Batch Generation
```python
from graph_generation.batch_generator import generate_batch_from_config

# From YAML config
graphs = generate_batch_from_config("config/my_config.yaml")
```

### Verification
```python
from graph_generation.graph_utils import verify_metricity

is_metric = verify_metricity(adjacency_matrix, symmetric=True)
```

---

## Integration Points

### With Phase 2 (Algorithm Benchmarking)
- Outputs adjacency matrices consumed by algorithms
- Graph metadata used for algorithm applicability filtering
- Saved graphs can be loaded for reproducible benchmarking

### With Phase 3 (Feature Engineering)
- Graph structure used for feature extraction
- Coordinates (Euclidean) enable geometric features
- Graph metadata helps stratify feature analysis

---

## Testing

Run tests: `python3 src/tests/test_graph_generators.py`

Test coverage:
- 10 Euclidean tests
- 6 Metric tests
- 3 Quasi-metric tests
- 7 Random tests
- 3 Edge cases
- 2 Consistency tests
- 3 Performance benchmarks

---

## References

- **Implementation Guide**: `/guides/01_graph_generation_system.md`
- **Root Context**: `/CLAUDE.md`
- **Configuration Examples**: `/config/example_batch_config.yaml`

---

**Package Version**: 1.0.0
**Last Updated**: 10-30-2025
**Status**: Production Ready (Phase 1 complete)
