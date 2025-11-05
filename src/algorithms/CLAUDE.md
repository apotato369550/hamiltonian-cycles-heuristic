# Algorithm Benchmarking System (Phase 2)

## Purpose
Core algorithm benchmarking infrastructure for TSP research platform. Implements unified algorithm interface, tour validation, quality metrics, and 8 TSP algorithms (3 baselines + 5 anchor-based heuristics).

**Status**: Steps 1-4 COMPLETE (validated 11-05-2025)
**Test Coverage**: 89 tests, 100% pass rate

---

## Package Structure

```
algorithms/
├── CLAUDE.md                    # This file - package documentation
├── __init__.py                  # Package initialization, exports
├── base.py                      # Core interfaces and data structures
├── registry.py                  # Algorithm registration system
├── validation.py                # Tour validation functions
├── metrics.py                   # Quality metrics computation
├── nearest_neighbor.py          # NN baseline (2 variants)
├── greedy.py                    # Greedy edge-picking baseline
├── exact.py                     # Held-Karp exact solver
├── single_anchor.py             # Single anchor heuristic
├── best_anchor.py               # Best anchor search
└── multi_anchor.py              # Multi-anchor heuristics (2 variants)
```

---

## Core Components

### 1. Algorithm Interface (base.py)
**Purpose**: Abstract base class and data structures for all TSP algorithms

**Key Classes**:
- `TourResult`: Dataclass for algorithm output
  - Fields: tour, weight, runtime, metadata, success, error_message
  - Validation: Rejects invalid tours at construction
- `AlgorithmMetadata`: Describes algorithm properties and constraints
  - Supports applicability filtering by graph type/size
- `TSPAlgorithm`: Abstract base class
  - Required methods: `solve()`, `get_metadata()`
  - Helper methods: `_compute_tour_weight()`, `_validate_tour_structure()`
  - Random seed support for reproducibility

**Lines**: 244

### 2. Registry System (registry.py)
**Purpose**: Centralized algorithm registration and retrieval

**Key Features**:
- Singleton pattern with `@register_algorithm` decorator
- Filtering by: tags, graph type, graph size
- Automatic instantiation with random seed support
- Type-safe: Validates TSPAlgorithm subclasses at registration

**Usage**:
```python
@register_algorithm("my_algo", tags=["heuristic"], constraints={})
class MyAlgorithm(TSPAlgorithm):
    ...

algo = AlgorithmRegistry.get_algorithm("my_algo", random_seed=42)
result = algo.solve(adjacency_matrix)
```

**Lines**: 216

### 3. Tour Validation (validation.py)
**Purpose**: Comprehensive validation of Hamiltonian cycles

**Key Functions**:
- `validate_tour()`: Checks tour structure, vertex validity, edge existence
- `validate_tour_constraints()`: Validates custom constraints (anchor edges, subpaths)
- `validate_adjacency_matrix()`: Pre-validation of input graphs
- `TourValidator`: Batch validation with optional caching

**Validation Checks**:
- Correct length (n vertices)
- No duplicates
- Valid vertex indices
- All edges exist in graph
- Negative weight detection
- NaN/Inf detection

**Lines**: 273

### 4. Quality Metrics (metrics.py)
**Purpose**: Tour quality measurement and comparative analysis

**Key Functions**:
- `compute_tour_weight()`: Total edge weight
- `compute_tour_statistics()`: Mean, median, std, min, max
- `compute_optimality_gap()`: Percentage above optimal
- `compute_approximation_ratio()`: Heuristic/optimal ratio
- `compute_tour_properties()`: Edge statistics, smoothness
- `compute_relative_performance()`: Performance vs baseline

**Classes**:
- `TourStatistics`: Statistical summary dataclass
- `MetricsCalculator`: Caching layer for expensive computations

**Lines**: 304

---

## Implemented Algorithms

### Baseline Algorithms

#### 1. Nearest Neighbor (nearest_neighbor.py)
**Variants**:
- `nearest_neighbor_random`: Start from random vertex
- `nearest_neighbor_best`: Try all starts, return best

**Complexity**: O(n²) per run, O(n³) for best-of-all
**Features**: Deterministic tie-breaking, reproducible with seeds
**Tags**: `["baseline", "greedy", "nearest_neighbor"]`

#### 2. Greedy Edge (greedy.py)
**Algorithm**: Iteratively add cheapest edges maintaining cycle constraints

**Complexity**: O(n² log n) - edge sorting dominates
**Implementation**: Union-Find for cycle detection, degree tracking
**Features**: Deterministic, good for structured graphs
**Tags**: `["baseline", "greedy"]`
**Note**: May fail on certain graph structures (expected behavior)

#### 3. Held-Karp Exact (exact.py)
**Algorithm**: Dynamic programming optimal solver

**Complexity**: O(n² × 2^n) time, O(n × 2^n) space
**Constraints**: Only n ≤ 20 (practical: n ≤ 18)
**Features**: Timeout protection, deterministic
**Tags**: `["exact", "optimal"]`
**Use**: Compute optimality gaps for small graphs

### Anchor-Based Algorithms (Research Contributions)

#### 4. Single Anchor (single_anchor.py)
**Strategy**: Pre-commit two cheapest edges from anchor vertex, build greedily

**Complexity**: O(n²)
**Parameters**: `anchor_vertex` (which vertex to use)
**Metadata**: Tracks anchor vertex, neighbors, edge weights
**Tags**: `["anchor", "heuristic"]`

#### 5. Best Anchor (best_anchor.py)
**Strategy**: Try single anchor from each vertex, return best

**Complexity**: O(n³) - n calls to O(n²) single anchor
**Research Value**: Identifies optimal anchor for graph
**Metadata**: Best anchor, all weights tried, search time
**Tags**: `["anchor", "search"]`

#### 6. Multi-Anchor (multi_anchor.py)
**Variants**:
- `multi_anchor_random`: Random K-vertex selection
- `multi_anchor_distributed`: Greedy max-distance selection

**Complexity**: O(n²)
**Parameters**: `num_anchors` (how many anchors)
**Strategy**: Build tour through multiple anchors with greedy infill
**Tags**: `["anchor", "multi"]`

---

## Usage Examples

### Basic Usage
```python
from algorithms.registry import AlgorithmRegistry
import algorithms.nearest_neighbor  # Trigger registration

# Get algorithm
nn = AlgorithmRegistry.get_algorithm("nearest_neighbor_best", random_seed=42)

# Create adjacency matrix
matrix = [[0, 10, 20], [10, 0, 15], [20, 15, 0]]

# Solve
result = nn.solve(matrix)

if result.success:
    print(f"Tour: {result.tour}")
    print(f"Weight: {result.weight}")
    print(f"Runtime: {result.runtime}s")
```

### Validation
```python
from algorithms.validation import validate_tour

validation_result = validate_tour(tour, adjacency_matrix)
if not validation_result.valid:
    print(validation_result.summary())
```

### Computing Metrics
```python
from algorithms.metrics import compute_tour_weight, compute_optimality_gap

weight = compute_tour_weight(tour, matrix)
gap = compute_optimality_gap(heuristic_weight, optimal_weight)
```

### List Available Algorithms
```python
# All algorithms
all_algos = AlgorithmRegistry.list_algorithms()

# Filter by tag
baselines = AlgorithmRegistry.list_algorithms(tags=["baseline"])
anchors = AlgorithmRegistry.list_algorithms(tags=["anchor"])

# Filter by graph constraints
small_graph_algos = AlgorithmRegistry.list_algorithms(graph_size=10)
```

---

## Testing

### Test Files
- `../tests/test_algorithms.py` (59 tests) - Core interfaces, validation, metrics
- `../tests/test_baseline_algorithms.py` (16 tests) - Baseline algorithm correctness
- `../tests/test_anchor_algorithms.py` (14 tests) - Anchor algorithm correctness

### Run Tests
```bash
# All Phase 2 tests
python3 src/tests/test_algorithms.py
python3 src/tests/test_baseline_algorithms.py
python3 src/tests/test_anchor_algorithms.py

# Or all at once
python3 -m unittest discover -s src/tests -p "test_*.py"
```

---

## Design Principles

### Followed from Root CLAUDE.md

1. **Modularity**: Each algorithm independent, self-contained
2. **Reproducibility**: Random seed support throughout
3. **Fail Fast**: Validation at construction and runtime
4. **Interpretability**: Clear, readable code over cleverness
5. **Test Everything**: 89 tests covering edge cases

### Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Dataclass-based results | Type-safe, extensible, self-documenting |
| Decorator registration | Clean, Pythonic, minimizes boilerplate |
| Separate validation/metrics | Flexibility for different use cases |
| Optional caching | Performance optimization for batch operations |
| Timeout in Held-Karp | Safety against hanging on large graphs |
| Deterministic tie-breaking | Reproducibility with same inputs |

---

## Integration Points

### With Phase 1 (Graph Generation)
- Consumes adjacency matrices from Phase 1 generators
- Compatible with all graph types (euclidean, metric, quasi-metric, random)
- Uses same metadata patterns

### With Phase 2 Steps 5-8 (To Be Built)
- Step 5: Single-graph runner will use algorithm registry
- Step 6: Batch system will iterate over algorithms
- Step 7: Statistical tools will use metrics.py functions
- Step 8: Visualizations will use TourResult metadata

### With Phase 3 (Feature Engineering)
- Benchmark results will provide anchor quality labels
- Best anchor algorithm identifies ground truth
- Algorithm metadata useful for feature correlation

---

## Performance Characteristics

| Algorithm | Time | Space | Practical Limit |
|-----------|------|-------|-----------------|
| NN Random | O(n²) | O(n) | n ≤ 1000 |
| NN Best | O(n³) | O(n) | n ≤ 500 |
| Greedy | O(n² log n) | O(n) | n ≤ 1000 |
| Held-Karp | O(n² × 2^n) | O(n × 2^n) | n ≤ 18 |
| Single Anchor | O(n²) | O(n) | n ≤ 1000 |
| Best Anchor | O(n³) | O(n) | n ≤ 500 |
| Multi-Anchor | O(n²) | O(n) | n ≤ 1000 |

---

## Extension Points

### Adding New Algorithms
1. Create new file `my_algorithm.py`
2. Subclass `TSPAlgorithm`
3. Implement `solve()` and `get_metadata()`
4. Use `@register_algorithm` decorator
5. Import in `__init__.py`
6. Add tests to `../tests/`

Example:
```python
from .base import TSPAlgorithm, TourResult, AlgorithmMetadata
from .registry import register_algorithm

@register_algorithm("my_algo", tags=["heuristic"])
class MyAlgorithm(TSPAlgorithm):
    def solve(self, adjacency_matrix, **kwargs):
        # Implementation
        return TourResult(tour=..., weight=..., runtime=...)

    def get_metadata(self):
        return AlgorithmMetadata(name="my_algo", version="1.0.0")
```

---

## Known Limitations

1. **Greedy Algorithm**: May not find valid tours on all graph types (Union-Find constraints)
2. **Held-Karp**: Limited to n ≤ 20 due to exponential complexity
3. **Memory**: No streaming support for very large graphs
4. **Asymmetric Graphs**: All algorithms handle asymmetry, but performance may vary

---

## Future Work (Steps 5-8)

### Immediate Next Steps
1. **Single-Graph Benchmarking Runner** - Orchestrate multiple algorithms on one graph
2. **Batch Benchmarking** - Run experiments across graph collections
3. **Statistical Analysis** - Paired tests, effect sizes, significance
4. **Visualization** - Publication-quality comparative plots

### Enhancement Opportunities
- 2-opt local search post-processing
- Christofides algorithm for metric graphs
- Parallel execution for batch benchmarking
- GPU acceleration for large graph batches

---

## References

- **Implementation Guide**: `/guides/02_algorithm_benchmarking_pipeline.md`
- **Implementation Summary**: `/PHASE2_COMPLETE.md`
- **Root Context**: `/CLAUDE.md`
- **Test Suite**: `../tests/test_*.py`

---

**Package Version**: 1.0.0
**Last Updated**: 11-05-2025
**Status**: Production Ready (Steps 1-4 complete, validated)
**Maintainers**: Builder (implementation), Validator (verification)
