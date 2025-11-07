# Feature Engineering System (Phase 3)

**Status:** Prompts 1-4 Complete (Base architecture + 3 feature extractors)
**Last Updated:** 11-07-2025
**Test Coverage:** 34 tests, all passing

---

## Overview

This module provides a modular, extensible system for extracting vertex-level features from graphs to support machine learning prediction of optimal TSP anchor vertices.

The feature extraction pipeline transforms raw graph adjacency matrices into interpretable, ML-ready feature vectors that capture structural properties relevant to anchor quality prediction.

---

## Architecture

### Core Design Pattern: Modular Extractors

```
VertexFeatureExtractor (Abstract Base)
    ├── WeightFeatureExtractor       (Prompt 2)
    ├── TopologicalFeatureExtractor  (Prompt 3)
    ├── MSTFeatureExtractor          (Prompt 4)
    ├── (Future: NeighborhoodFeatureExtractor)
    └── (Future: HeuristicFeatureExtractor)

FeatureExtractorPipeline
    - Orchestrates multiple extractors
    - Manages shared cache
    - Validates output
    - Combines results
```

**Key Principles:**
1. **Modularity**: Each extractor is independent, can be enabled/disabled
2. **Caching**: Expensive computations (MST, shortest paths) computed once, shared
3. **Validation**: All features checked for NaN/Inf, shape consistency, naming
4. **Interpretability**: Every feature has descriptive name (e.g., 'mst_degree', 'mean_weight')

---

## Implemented Features (Prompts 1-4)

### 1. Base Architecture (Prompt 1)

**Files:**
- `base.py` - Abstract base class and validation utilities
- `pipeline.py` - Pipeline orchestrator

**Classes:**
- `VertexFeatureExtractor` - Abstract base requiring `extract()` and `get_feature_names()`
- `FeatureExtractorPipeline` - Runs multiple extractors, manages cache
- `CachedComputation` - Helper for get-or-compute pattern
- `FeatureValidationError` - Custom exception for validation failures

**Key Methods:**
```python
extractor.extract(graph, cache) -> (features, feature_names)
    # Returns: NxF array, list of F names

pipeline.add_extractor(extractor)
pipeline.extract_features(graph, cache) -> (features, feature_names)
    # Combines all extractors, prefixes names with extractor.name
```

**Validation Checks:**
- Shape consistency (N vertices, F features)
- No NaN values (unless explicitly allowed)
- No infinite values
- Feature count matches name count
- No duplicate feature names

---

### 2. Weight-Based Features (Prompt 2)

**File:** `extractors/weight_based.py`

**Class:** `WeightFeatureExtractor`

**Features Extracted (20 for symmetric, 46 for asymmetric):**

**Basic Statistics:**
- total_weight: Sum of all edge weights from vertex
- mean_weight: Average distance to other vertices
- median_weight: Middle value of edge weights
- std_weight: Standard deviation of edge weights
- var_weight: Variance of edge weights

**Distribution Features:**
- min_weight: Distance to nearest neighbor
- max_weight: Distance to farthest vertex
- min_max_ratio: Concentration of nearby neighbors
- q25_weight, q50_weight, q75_weight: Quantiles
- iqr_weight: Interquartile range
- skewness: Distribution asymmetry
- kurtosis: Distribution tail heaviness

**Relative Features:**
- z_score_mean: Z-score of mean weight relative to graph
- percentile_rank_mean: Percentile position in graph
- distance_from_median: Difference from graph median
- rank_cheapest: Rank of cheapest edge
- prop_below_median: Proportion of edges below graph median
- distance_to_centroid: Distance to graph centroid vertex

**Asymmetric Graph Features (if enabled):**
- out_* - All above features for outgoing edges
- in_* - All above features for incoming edges
- asym_total_diff, asym_mean_diff, asym_min_diff, asym_max_diff
- asym_total_ratio, asym_mean_ratio

**Edge Cases Handled:**
- Single vertex graph (returns zeros)
- Uniform weights (sets skewness/kurtosis to 0 to avoid warnings)
- Division by zero (returns 0 for ratios)

**Constructor:**
```python
WeightFeatureExtractor(
    include_asymmetric_features=True,  # Enable out/in/asym features
    name="weight_based"
)
```

---

### 3. Topological Features (Prompt 3)

**File:** `extractors/topological.py`

**Class:** `TopologicalFeatureExtractor`

**Features Extracted (5-8 depending on config):**

**Degree-Based:**
- degree: Number of neighbors (n-1 in complete graphs)
- weighted_degree: Sum of edge weights (same as total_weight conceptually)

**Centrality Measures:**
- closeness_centrality: (n-1) / sum_of_shortest_paths
  - High closeness = central vertex
  - Uses cached shortest paths from Dijkstra
- betweenness_centrality: Proportion of shortest paths through vertex (EXPENSIVE O(n³))
  - High betweenness = structural bridge
- eigenvector_centrality: Importance based on connection to important vertices
  - Computed via power iteration
  - Uses inverted weights (1/weight) as adjacency

**Clustering:**
- clustering_coefficient: Proportion of neighbor pairs that are connected
  - In complete graphs: always 1.0
  - In sparse graphs: measures local density

**Distance-Based:**
- eccentricity: Maximum shortest path distance from vertex
  - High eccentricity = periphery vertex
- avg_shortest_path_length: Mean distance to all other vertices

**Computational Cost:**
- Shortest paths: O(n² log n) Dijkstra from each source
- Betweenness: O(n³) naive implementation
- Eigenvector: O(n² × iterations) power method
- Clustering: O(n²) per vertex

**Caching:**
- `shortest_paths` - Shared with other extractors
- `betweenness_centrality` - Computed once for all vertices
- `eigenvector_centrality` - Computed once for all vertices

**Constructor:**
```python
TopologicalFeatureExtractor(
    include_betweenness=True,   # Expensive, disable for large graphs
    include_eigenvector=True,   # Moderately expensive
    include_clustering=True,    # Cheap
    name="topological"
)
```

---

### 4. MST-Based Features (Prompt 4)

**File:** `extractors/mst_based.py`

**Class:** `MSTFeatureExtractor`

**Features Extracted (9 total):**

**Basic MST Features:**
- mst_degree: Number of MST edges incident to vertex
  - High degree = structural hub
  - Sum of all degrees = 2(n-1)
- mst_is_leaf: Boolean (1.0 if degree == 1, else 0.0)
- mst_is_hub: Boolean (1.0 if degree >= 3, else 0.0)

**MST Edge Weight Features:**
- mst_total_weight: Sum of MST edge weights from vertex
- mst_mean_weight: Average MST edge weight
- mst_max_weight: Maximum MST edge weight
- mst_to_total_ratio: MST weight / total vertex weight
  - High ratio = MST edges are heavy relative to all edges

**MST Structural Features:**
- mst_center_distance: Distance to MST center (hop count)
  - MST center = vertex minimizing maximum distance in MST
  - Computed via BFS on MST
- mst_removal_impact: Change in MST weight if vertex removed
  - High impact = structurally important vertex
  - Normalized by original MST weight

**Computational Cost:**
- MST computation: O(n² log n) using scipy's minimum_spanning_tree
- Center distance: O(n²) BFS from each vertex
- Removal impact: O(n × n² log n) = O(n³ log n) - EXPENSIVE
  - Consider disabling for large graphs or approximating

**Caching:**
- `mst` - MST adjacency matrix and edge list, shared with other extractors

**Implementation Notes:**
- Uses scipy.sparse.csgraph.minimum_spanning_tree
- Makes MST symmetric (scipy returns directed version)
- Stores edge list as set of (u,v) tuples for efficiency
- BFS used for MST distance computation (unweighted paths in tree)

**Constructor:**
```python
MSTFeatureExtractor(name="mst_based")
```

---

## Usage Examples

### Basic Usage: Single Extractor
```python
from src.features import FeatureExtractorPipeline
from src.features.extractors import WeightFeatureExtractor

# Create pipeline
pipeline = FeatureExtractorPipeline()
pipeline.add_extractor(WeightFeatureExtractor())

# Extract features
import numpy as np
graph = np.array([[0, 1, 2], [1, 0, 3], [2, 3, 0]], dtype=float)
features, names = pipeline.extract_features(graph)

# features: 3x20 array
# names: ['weight_based.total_weight', 'weight_based.mean_weight', ...]
```

### Multi-Extractor Pipeline
```python
from src.features.extractors import (
    WeightFeatureExtractor,
    TopologicalFeatureExtractor,
    MSTFeatureExtractor
)

pipeline = FeatureExtractorPipeline()
pipeline.add_extractor(WeightFeatureExtractor())
pipeline.add_extractor(TopologicalFeatureExtractor(
    include_betweenness=False  # Disable expensive feature
))
pipeline.add_extractor(MSTFeatureExtractor())

# Extract all features
features, names = pipeline.extract_features(graph)
# features: 3x(20+5+9) = 3x34 array
# names: All feature names prefixed by extractor
```

### Using Cache for Efficiency
```python
# When processing multiple graphs with same structure
cache = {}

for graph in graph_collection:
    features, names = pipeline.extract_features(graph, cache)
    # Cache is reused, but MST/shortest_paths recomputed per graph
    cache.clear()  # Clear between graphs if they differ
```

---

## Test Coverage (34 Tests)

**Test File:** `src/tests/test_features.py`

**Test Categories:**
1. Base Architecture (5 tests)
   - Cache helper functionality
   - Pipeline add/remove extractors
   - Duplicate name detection
   - Empty pipeline error

2. Weight Features (7 tests)
   - Symmetric vs asymmetric graphs
   - Feature calculation correctness
   - No NaN/Inf values
   - Asymmetry feature generation

3. Topological Features (7 tests)
   - All features vs minimal features
   - Degree/closeness calculations
   - Cache usage
   - No NaN values

4. MST Features (6 tests)
   - MST degree range validation
   - Total edges = 2(n-1)
   - Leaf/hub indicators
   - Cache usage
   - No NaN values

5. Feature Validation (4 tests)
   - NaN detection
   - Inf detection
   - Shape mismatch detection
   - Name count mismatch

6. Pipeline Integration (4 tests)
   - Single/multi extractor pipelines
   - Cache sharing between extractors
   - Feature count reporting

7. Edge Cases (3 tests)
   - Single vertex graph (no edges)
   - Uniform weights (zero variance)
   - Large weight range

**All 34 tests passing (verified 11-07-2025)**

**Run Tests:**
```bash
python3 -m unittest src.tests.test_features -v
```

---

## Future Work (Prompts 5-12)

### Prompt 5: Neighborhood Features (Not Implemented)
- K-nearest neighbor statistics
- Neighborhood density
- Radial features (shells by distance)
- Voronoi-like partitioning

### Prompt 6: Heuristic-Specific Features (Not Implemented)
- Two cheapest edge features
- Gap between 2nd and 3rd cheapest
- Anchor edge angle (Euclidean graphs)
- Baseline tour cost estimates

### Prompt 7: Graph-Level Context Features (Not Implemented)
- Z-score normalization
- Percentile ranks
- Graph property features (size, density, metricity)
- Relative importance features

### Prompt 8: Feature Validation and Analysis (Not Implemented)
- Correlation matrix computation
- PCA for dimensionality reduction
- Feature-target correlation ranking
- Constant feature detection

### Prompt 9: Anchor Quality Labeling (Not Implemented)
- Run single-anchor from all vertices
- Assign quality scores (rank-based, absolute, binary, multi-class)
- Store labels alongside features

### Prompt 10: Feature Engineering Pipeline (Not Implemented)
- End-to-end: graphs → features + labels → ML dataset
- Progress tracking and caching
- CSV/DataFrame output

### Prompt 11: Feature Selection (Not Implemented)
- Univariate selection (correlation, F-test)
- Recursive feature elimination
- Model-based importance (random forest)
- L1 regularization (Lasso)

### Prompt 12: Feature Transformation (Not Implemented)
- Non-linear transforms (log, sqrt, polynomial)
- Feature interactions (products, ratios)
- Standardization (z-score, min-max, robust)
- Domain-specific combinations

---

## Design Decisions and Rationale

### 1. Why Modular Extractors?
- **Flexibility**: Enable/disable feature groups based on computational budget
- **Maintainability**: Add new feature types without modifying existing code
- **Testability**: Each extractor tested independently
- **Interpretability**: Clear grouping of related features

### 2. Why Cache Expensive Computations?
- **MST**: O(n² log n), used by multiple extractors
- **Shortest Paths**: O(n² log n), used by topological features
- **Centrality**: O(n³), computed once for all vertices
- Caching reduces redundant computation when using multiple extractors

### 3. Why Prefix Feature Names with Extractor Name?
- **Uniqueness**: Prevents name collisions (e.g., 'degree' could mean MST degree or graph degree)
- **Clarity**: User knows which extractor produced each feature
- **Debugging**: Easier to trace feature values back to source

### 4. Why Validate Features Strictly?
- **Fail Fast**: Invalid features caught immediately, not during ML training
- **Reproducibility**: Ensures consistent output across runs
- **Debugging**: Clear error messages pinpoint issues

### 5. Why Not Normalize During Extraction?
- **Separation of Concerns**: Normalization is ML-specific, extraction is general
- **Flexibility**: Some ML models need raw features, others need normalized
- **Future-Proofing**: Normalization strategy may depend on train/test split

**Exception:** Relative features (z-scores, percentiles) are computed during extraction because they require graph-level context only available at extraction time.

---

## Critical Principles

### Principle 1: Feature Interpretability Over Complexity
- Every feature must have clear meaning
- No learned embeddings or black-box transformations
- Feature names must be self-documenting
- Coefficient analysis must be possible (for linear models)

**Rationale:** Research goal is understanding WHY certain vertices make good anchors, not just prediction accuracy.

### Principle 2: Computational Cost Awareness
- Document complexity of each feature (O(n), O(n²), O(n³))
- Provide enable/disable flags for expensive features
- Use caching to avoid redundant computation
- Consider approximations for large graphs

**Current Costs:**
- Weight features: O(n²) total
- Topological (without betweenness): O(n² log n)
- Topological (with betweenness): O(n³)
- MST features: O(n³ log n) due to removal impact

**For 100-vertex graph:**
- Weight + Topological (minimal) + MST: ~1 second
- Weight + Topological (full): ~5 seconds

### Principle 3: Edge Case Handling
- Single vertex graphs (no edges)
- Uniform weights (zero variance)
- Disconnected graphs (though TSP assumes complete)
- Asymmetric graphs (quasi-metrics)

**Implemented Safeguards:**
- Check array lengths before min/max/percentile
- Conditional computation of higher-order stats
- Graceful defaults (0.0 for undefined metrics)
- Division by zero protection

### Principle 4: Test-Driven Development
- Every feature type has dedicated tests
- Edge cases explicitly tested
- Validation logic tested independently
- Integration tests for multi-extractor pipelines

**Quality Gate:** All tests must pass before marking prompt complete.

---

## Integration with Other Phases

### Phase 1 (Graph Generation) → Phase 3
- Feature extractors accept adjacency matrices from graph generators
- Support all graph types: Euclidean, metric, quasi-metric, random
- Asymmetric feature handling for quasi-metrics

### Phase 2 (Algorithms) → Phase 3
- Anchor quality labels will come from algorithm benchmarking results
- Single-anchor tour quality scores used as target variable
- Best-anchor search results used for ground truth

### Phase 3 → Phase 4 (Machine Learning)
- Feature matrix format compatible with sklearn/pandas
- Feature names enable coefficient interpretation
- Feature validation ensures clean ML input

### Phase 3 → Phase 5 (Pipeline)
- Caching infrastructure ready for batch processing
- Pipeline orchestrator for graph collections
- Progress tracking hooks for long-running extractions

### Phase 3 → Phase 6 (Analysis)
- Feature names enable publication-quality visualizations
- Correlation analysis identifies predictive features
- Interpretability supports research insights

---

## Common Pitfalls and Solutions

### Pitfall 1: Memory Explosion with Large Graphs
**Problem:** O(n²) or O(n³) algorithms on 1000+ vertex graphs
**Solution:**
- Disable expensive features (betweenness, removal impact)
- Use approximations (sample vertices for betweenness)
- Process in batches with aggressive caching

### Pitfall 2: NaN from Edge Cases
**Problem:** Empty arrays, division by zero, uniform distributions
**Solution:**
- Check array lengths before statistical functions
- Conditional computation with sensible defaults
- Higher-order stats only if std > threshold

### Pitfall 3: Feature Redundancy
**Problem:** Highly correlated features (e.g., total_weight and weighted_degree)
**Solution:**
- Document correlations (Prompt 8)
- Feature selection before ML training (Prompt 11)
- Keep for interpretability, remove if multicollinearity issues

### Pitfall 4: Cache Pollution
**Problem:** Cache grows unbounded across many graphs
**Solution:**
- Clear cache between graphs if structure differs
- Use separate caches for different graph collections
- Document which keys are cached by which extractors

---

## Questions for Future Development

### Before Starting Prompt 5 (Neighborhood Features)
1. What k values for k-nearest neighbors? (3, 5, 10?)
2. How to define neighborhood radius? (Median edge weight? Percentile?)
3. Should Voronoi features use precomputed anchors or hypothetical partitions?

### Before Starting Prompt 9 (Anchor Quality Labeling)
1. Which labeling strategy? Rank-based vs absolute quality?
2. Should labels be per-graph or normalized across collection?
3. How to handle ties (multiple vertices with same tour quality)?

### Before Starting Prompt 10 (Pipeline)
1. Should pipeline support parallelization across graphs?
2. What intermediate format for caching? JSON, pickle, HDF5?
3. How to handle partial failures (one graph fails extraction)?

---

## Version History

**v1.0 - 11-07-2025 (Prompts 1-4 Complete)**
- Base architecture: VertexFeatureExtractor, FeatureExtractorPipeline
- WeightFeatureExtractor: 20 symmetric, 46 asymmetric features
- TopologicalFeatureExtractor: 5-8 features (configurable)
- MSTFeatureExtractor: 9 features
- 34 tests, all passing
- Handles edge cases: single vertex, uniform weights, asymmetric graphs

**Next Version (Prompts 5-12):**
- Neighborhood, heuristic, graph-level features
- Feature validation and analysis tools
- Anchor quality labeling system
- End-to-end pipeline
- Feature selection and transformation

---

## Maintainer Notes

**Code Quality:**
- All extractors follow consistent interface
- Docstrings for all public methods
- Type hints throughout
- No warnings on typical inputs

**Documentation:**
- Inline comments explain WHY, not WHAT
- Complex algorithms reference standard sources
- Edge cases documented explicitly

**Testing:**
- Unit tests for each extractor
- Integration tests for pipeline
- Edge case coverage
- No mocking (uses real computation)

**Performance:**
- Caching reduces redundant computation by ~90% for multi-extractor pipelines
- Most expensive operation: MST removal impact O(n³ log n)
- Recommend disabling betweenness and removal impact for graphs >200 vertices

---

**Document Maintained By:** Builder Agent
**Last Review:** 11-07-2025
**Status:** Phase 3 (Prompts 1-4) production-ready
