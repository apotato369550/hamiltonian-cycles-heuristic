# Phase 2: Algorithm Benchmarking Pipeline Implementation Plan

## Overview

Build a rigorous benchmarking system to evaluate anchor-based TSP heuristics against baseline algorithms. System must provide reproducible, statistically sound performance comparisons across diverse graph types with full tour validation.

**Core Research Question:** Which TSP algorithms (baseline vs anchor-based) perform best on which graph types?

**Success Criteria:**
- Benchmark 5+ algorithms on 100 graphs in <1 hour
- 100% valid Hamiltonian cycles (or graceful failure)
- Statistical comparison framework operational
- Publication-ready visualizations

---

## Build Steps

### Step 1: Core Algorithm Interface & Registry

**Goal:** Establish unified interface for all TSP algorithms with registry system for selection and composition

**Files to create:**
- `src/algorithms/__init__.py` - Package initialization, exports
- `src/algorithms/base.py` - Abstract base classes and data structures
- `src/algorithms/registry.py` - Algorithm registry system

**Key components:**

`base.py`:
- `TourResult` dataclass: tour (List[int]), weight (float), runtime (float), metadata (Dict), success (bool)
- `AlgorithmMetadata` dataclass: name, version, parameters, applicability constraints
- `TSPAlgorithm` abstract base class:
  - `solve(adjacency_matrix, **kwargs) -> TourResult`
  - `get_metadata() -> AlgorithmMetadata`
  - `is_applicable(graph_properties) -> bool`
  - Built-in runtime tracking decorator
  - Exception handling wrapper

`registry.py`:
- `AlgorithmRegistry` singleton class
- `register(name, algorithm_class, tags, constraints)` decorator
- `get_algorithm(name) -> TSPAlgorithm`
- `list_algorithms(tags, applicable_to_graph) -> List[str]`
- Support algorithm families: "nearest_neighbor", "nearest_neighbor_best_start"
- Version tracking for algorithm improvements

**Tests:**
- Algorithm interface contract validation
- Registry registration and retrieval
- Applicability filtering
- Metadata extraction

**Success criteria:**
- Can instantiate algorithm from registry by name
- Runtime automatically tracked
- Invalid tours caught and flagged in TourResult
- Registry filters algorithms by graph applicability

**Integration notes:**
- Follow Phase 1 pattern: dataclasses for data, classes for behavior
- Use decorators for runtime tracking (like @dataclass)
- Registry pattern similar to batch_generator's type dispatch

---

### Step 2: Tour Validation & Quality Metrics

**Goal:** Robust validation of tours and comprehensive quality metric computation

**Files to create:**
- `src/algorithms/validation.py` - Tour validation functions
- `src/algorithms/metrics.py` - Quality metrics computation

**Key components:**

`validation.py`:
- `validate_tour(tour, adjacency_matrix) -> ValidationResult`
  - Check: exactly n vertices, no duplicates
  - Check: forms valid cycle (last connects to first)
  - Check: all edges exist in graph
  - Return: ValidationResult(valid, errors, warnings)
- `validate_tour_constraints(tour, constraints)` - verify anchor edges, other constraints
- `TourValidator` class for batch validation with caching

`metrics.py`:
- `compute_tour_weight(tour, adjacency_matrix) -> float`
- `compute_tour_statistics(tours) -> Dict` - mean, median, std, min, max
- `compute_optimality_gap(heuristic_weight, optimal_weight) -> float`
- `compute_relative_performance(algorithm_weights, baseline_weights) -> Dict`
- `compute_tour_properties(tour, adjacency_matrix) -> Dict` - max edge, smoothness, etc.
- `MetricsCalculator` class with caching for expensive computations

**Tests:**
- Valid tour passes validation
- Invalid tours caught (missing vertex, cycle break, non-existent edge)
- Tour weight computation correct
- Statistical metrics accurate
- Edge cases: single-vertex, two-vertex tours

**Success criteria:**
- Detects all tour invalidity types
- Clear error messages for validation failures
- Metrics computation matches manual calculation
- Handles asymmetric graphs correctly

**Integration notes:**
- Validation called by every algorithm's solve() wrapper
- Metrics computation separate from validation (can compute metrics on invalid tours for debugging)
- Similar pattern to Phase 1's verify_graph_properties()

---

### Step 3: Baseline Algorithm Implementations

**Goal:** Correct, tested implementations of standard TSP algorithms for comparison

**Files to create:**
- `src/algorithms/nearest_neighbor.py` - Nearest neighbor variants
- `src/algorithms/greedy.py` - Greedy edge-picking algorithm
- `src/algorithms/exact.py` - Held-Karp for small graphs

**Key components:**

`nearest_neighbor.py`:
- `NearestNeighborAlgorithm(TSPAlgorithm)`:
  - Parameter: start_vertex (int or "random" or "best")
  - Deterministic tie-breaking (lowest vertex index)
  - If start="best", try all vertices, return best tour
- Register: "nearest_neighbor_random", "nearest_neighbor_best"

`greedy.py`:
- `GreedyEdgeAlgorithm(TSPAlgorithm)`:
  - Sort edges by weight
  - Use union-find for cycle detection
  - Add edge if: no cycle created AND no vertex exceeds degree 2
  - Continue until Hamiltonian cycle formed
- Register: "greedy_edge"

`exact.py`:
- `HeldKarpAlgorithm(TSPAlgorithm)`:
  - Applicability: only graphs with n ≤ 15 (configurable)
  - Dynamic programming: O(n² × 2^n)
  - Timeout protection (default 60s)
  - Cache results to disk (use graph ID as key)
- Register: "held_karp_exact"

**Tests (per algorithm):**
- Produces valid tour on small graphs (n=5, 10, 15)
- Produces valid tour on all graph types (euclidean, metric, random, quasi-metric)
- Deterministic: same seed → same tour
- Handles edge cases: n=3, n=4
- Nearest neighbor best finds optimal on trivial graphs
- Held-Karp finds known optimal solutions

**Success criteria:**
- All algorithms produce valid tours 100% of time
- Greedy never creates invalid intermediate state
- Held-Karp matches known optimal for small test cases
- Runtime reasonable: nearest neighbor <1s for n=100, greedy <5s for n=100

**Integration notes:**
- Christofides deferred to later (complex, may use library)
- Each algorithm self-contained in own file
- All algorithms registered automatically via decorators
- Follow Phase 1 pattern: generator functions + wrapper class

---

### Step 4: Anchor-Based Algorithm Implementations

**Goal:** Implement research algorithms with full parameterization and metadata tracking

**Files to create:**
- `src/algorithms/single_anchor.py` - Single anchor heuristic
- `src/algorithms/best_anchor.py` - Best anchor search
- `src/algorithms/multi_anchor.py` - Multi-anchor heuristic

**Key components:**

`single_anchor.py`:
- `SingleAnchorAlgorithm(TSPAlgorithm)`:
  - Parameters: anchor_vertex (int), edge_selection_strategy
  - Pre-commit two cheapest edges from anchor
  - Build remaining tour greedily (nearest neighbor from partial tour)
  - Metadata: anchor vertex, anchor edges used, anchor edge weights
  - Handle failure: if pre-committed edges prevent valid tour, return failure with metadata
- Register: "single_anchor"

`best_anchor.py`:
- `BestAnchorAlgorithm(TSPAlgorithm)`:
  - Try single_anchor starting from each vertex
  - Track best tour and which anchor produced it
  - Metadata: optimal_anchor, all_anchor_weights (list), search_time
  - Optimization: early stopping if optimal found (if known optimal available)
- Register: "best_anchor_exhaustive"

`multi_anchor.py`:
- `MultiAnchorAlgorithm(TSPAlgorithm)`:
  - Parameters: num_anchors (int), anchor_selection_strategy
  - Anchor selection strategies:
    - "random": random K vertices
    - "distributed": maximize pairwise distances (greedy)
    - "mst_degree": vertices with highest degree in MST
  - Pre-commit edges from each anchor
  - Connect anchors into tour skeleton
  - Fill in remaining vertices
  - Metadata: anchor_vertices, anchor_selection_time, construction_time
- Register: "multi_anchor_random", "multi_anchor_distributed", "multi_anchor_mst"

**Tests (per algorithm):**
- Produces valid tour or graceful failure
- Anchor metadata correctly captured
- Single anchor: anchors with different vertices give different tours
- Best anchor: finds best among all single-anchor options
- Multi anchor: different strategies produce different anchor sets
- Deterministic with seed control

**Success criteria:**
- All anchor algorithms track which vertices used as anchors
- Failure cases handled gracefully (don't crash, return TourResult with success=False)
- Best anchor provably optimal among all single-anchor options
- Multi-anchor strategies produce well-distributed anchors

**Integration notes:**
- Best anchor uses single_anchor internally
- Multi-anchor may use MST computation (import from scipy or implement)
- All anchor metadata captured in TourResult.metadata for Phase 3 analysis
- Reference guide prompts 4, 5 for implementation details

---

### Step 5: Single-Graph Benchmarking Runner

**Goal:** Run all algorithms on one graph with timeout protection, validation, and comparative analysis

**Files to create:**
- `src/algorithms/single_benchmark.py` - Single graph benchmarking

**Key components:**

`single_benchmark.py`:
- `BenchmarkConfig` dataclass:
  - algorithms (List[str]), timeout_per_algorithm (float), validate_tours (bool)
  - trials_per_algorithm (int, for stochastic algorithms)
- `SingleGraphBenchmark` class:
  - `run(graph_instance, config) -> BenchmarkResult`
  - For each algorithm:
    - Check applicability
    - Run with timeout protection (signal.alarm or threading)
    - Validate tour if enabled
    - Compute all metrics
    - Catch and log exceptions
  - Return `BenchmarkResult`: graph_metadata, algorithm_results (List[TourResult]), comparative_stats
- `comparative_stats` includes:
  - Winner (by tour weight)
  - Win margin (percentage)
  - Runtime comparison
  - Algorithm rankings

**Tests:**
- All algorithms run successfully on valid graph
- Timeout protection works (use artificially slow mock algorithm)
- Invalid tours caught and flagged
- Comparative stats correctly identify winner
- Handles partial failure (some algorithms succeed, some fail)

**Success criteria:**
- Can run 5 algorithms on n=50 graph in <10 seconds
- Timeout prevents hanging on expensive algorithms
- Results structure contains all information for analysis
- Clear logging of progress and failures

**Integration notes:**
- Uses GraphInstance from Phase 1
- Uses algorithm registry from Step 1
- Uses validation from Step 2
- Reference guide prompt 6

---

### Step 6: Batch Benchmarking System & Results Storage

**Goal:** High-level orchestration for running experiments across graph collections with resumption support

**Files to create:**
- `src/algorithms/batch_benchmark.py` - Batch benchmarking orchestration
- `src/algorithms/results_storage.py` - Results persistence and retrieval

**Key components:**

`batch_benchmark.py`:
- `BatchBenchmarkConfig` dataclass:
  - graph_collection_path, algorithm_names, filters (graph_type, size_range)
  - trials_per_combination (int), timeout_seconds, output_directory
  - experiment_name, random_seed, resume_from_checkpoint
- `BatchBenchmarker` class:
  - Load graphs from storage (use Phase 1 GraphStorage)
  - Apply filters to select relevant graphs
  - For each graph: run single_benchmark
  - Save results incrementally (after each graph)
  - Checkpoint state for resumption
  - Progress tracking: completed/total, ETA, current win rates
  - Summary statistics updated in real-time
- Output: `ExperimentResults` object + saved JSON/CSV

`results_storage.py`:
- `ResultsStorage` class:
  - Save format: JSON for full results, CSV for tabular summary
  - Directory structure: `results/{experiment_name}/{timestamp}/`
  - Files: `full_results.json`, `summary.csv`, `experiment_config.yaml`, `checkpoint.json`
  - Metadata: git commit hash, timestamp, machine specs, random seeds
- `load_results(experiment_name) -> ExperimentResults`
- `query_results(filters) -> DataFrame` - filter by graph type, algorithm, performance
- Version control: multiple runs don't overwrite, use timestamps

**Tests:**
- Batch run on small collection (5 graphs, 2 algorithms)
- Resumption works after interruption
- Results correctly saved and loaded
- Query functions filter correctly
- Incremental save prevents data loss on crash

**Success criteria:**
- Run 100 graphs × 5 algorithms in <1 hour
- Checkpoint every 10 graphs
- Resume picks up exactly where left off
- Results queryable by multiple dimensions

**Integration notes:**
- Integrate with Phase 1 GraphStorage for loading graphs
- Similar to Phase 1 batch_generator structure
- YAML config format matches Phase 1 conventions
- Reference guide prompts 7, 8

---

### Step 7: Statistical Analysis Tools

**Goal:** Rigorous statistical comparison of algorithm performance

**Files to create:**
- `src/algorithms/statistics.py` - Statistical analysis functions
- `src/algorithms/comparison.py` - Pairwise algorithm comparison

**Key components:**

`statistics.py`:
- `compute_descriptive_stats(results) -> Dict`:
  - Per algorithm: mean/median/std of tour weight, runtime
  - Per graph type: performance breakdown
  - Per graph size: scaling behavior
- `paired_comparison_test(algorithm_a_results, algorithm_b_results) -> TestResult`:
  - Wilcoxon signed-rank test (non-parametric, for paired samples)
  - Effect size: Cohen's d
  - Confidence interval for mean difference
  - Return: statistic, p_value, effect_size, interpretation
- `multi_algorithm_comparison(results) -> ComparisonMatrix`:
  - Pairwise tests for all algorithm pairs
  - Multiple testing correction (Bonferroni)
  - Win/loss/tie matrix
- `analyze_by_factor(results, factor) -> Dict`:
  - Factor: graph_type, size, metricity_score
  - ANOVA or Kruskal-Wallis for multi-group comparison

`comparison.py`:
- `AlgorithmComparator` class:
  - Load results from storage
  - Generate comparison tables (markdown, LaTeX)
  - Create statistical test reports
  - Rank algorithms by multiple criteria
- `generate_comparison_report(results) -> str` - human-readable summary

**Tests:**
- Statistical tests produce correct results on synthetic data
- Effect size calculations accurate
- Multiple testing correction works
- Report generation creates valid markdown

**Success criteria:**
- Can answer: "Is algorithm A significantly better than B?" with p-value and effect size
- Can answer: "Which algorithm is best for graph type X?" with statistical confidence
- Comparison tables ready for paper inclusion

**Integration notes:**
- Use scipy.stats for statistical tests
- Generate outputs matching academic publication standards
- Reference guide prompt 9

---

### Step 8: Visualization & Reporting

**Goal:** Publication-quality visualizations and automated reports

**Files to create:**
- `src/algorithms/visualization.py` - Plotting functions
- `src/algorithms/reporting.py` - Automated report generation

**Key components:**

`visualization.py`:
- `plot_performance_comparison(results) -> Figure`:
  - Box plots: tour weights by algorithm, grouped by graph type
  - Line plots: performance vs. graph size
  - Scatter: runtime vs. quality (Pareto frontier)
  - Bar charts: win rates for pairwise comparisons
- `plot_performance_heatmap(results) -> Figure`:
  - Rows: graph instances, Columns: algorithms, Cells: relative performance
- `plot_tour_comparison(graph, tours) -> Figure`:
  - For Euclidean graphs: overlay all algorithm tours
  - Color-coded by algorithm
  - Legend with tour weights
- `plot_scaling_analysis(results) -> Figure`:
  - Runtime vs. n, quality vs. n
  - Log-log plots for asymptotic analysis
- All plots: 300 DPI, colorblind-friendly palette, clear labels

`reporting.py`:
- `generate_html_report(results, output_path)`:
  - Executive summary: best algorithm by graph type
  - Statistical test results with tables
  - All key visualizations embedded
  - Methodology section
  - Interactive: sortable tables, zoomable plots
- `generate_markdown_report(results) -> str`:
  - For documentation/README
  - Static images + tables

**Tests:**
- Plots generate without errors
- Plots correctly represent data (check specific data points)
- HTML report valid and renders correctly
- Colorblind-friendly palette verification

**Success criteria:**
- Reviewer can understand results from visualizations alone
- All plots publication-ready (high DPI, clear labels, professional appearance)
- HTML report interactive and comprehensive

**Integration notes:**
- Use matplotlib/seaborn for plotting
- Follow Phase 1 visualization.py patterns
- Reference guide prompt 10

---

## Critical Implementation Notes

### Architectural Principles Applied

**Modularity:**
- Each algorithm is self-contained, registered independently
- Benchmarking pipeline reusable: Step 5 (single graph) → Step 6 (batch)
- Can run Phase 2 on Phase 1 graphs without regenerating

**Reproducibility:**
- All algorithms accept random_seed parameter
- YAML configuration controls all experimental parameters
- Git commit hash recorded with results
- Checkpoint system enables exact resumption

**Fail Fast, Fail Clearly:**
- Tour validation catches invalid tours immediately
- Timeout protection prevents hanging
- Graceful failure: return TourResult(success=False) rather than crash
- Clear error messages in validation failures

**Test Everything:**
- Each algorithm tested on multiple graph types
- Validation tested on known-invalid tours
- Statistical tests verified on synthetic data
- End-to-end batch test on small collection

### Pitfalls to Avoid

1. **Asymmetric graph handling:** Tour weight calculation must use correct direction (i→j vs j→i)
2. **Floating point comparison:** Use epsilon tolerance for equality checks
3. **Algorithm applicability:** Don't run Christofides on non-metric graphs
4. **Timeout implementation:** signal.alarm not available on Windows, use threading.Timer
5. **Result versioning:** Don't overwrite old results when re-running experiments
6. **Statistical assumptions:** Check normality before t-tests, use non-parametric alternatives
7. **Memory management:** Don't load all results into memory for large experiments

### Phase 1 Integration Points

- **GraphInstance:** Used as input to all algorithms
- **GraphStorage:** Used to load graph collections for batch benchmarking
- **Verification patterns:** Tour validation follows verify_graph_properties() structure
- **Dataclass patterns:** TourResult, AlgorithmMetadata follow GraphMetadata pattern
- **Batch patterns:** BatchBenchmarker follows BatchGenerator structure
- **Testing patterns:** Comprehensive test coverage like Phase 1's 34 tests

### Key Design Decisions

**Algorithm Interface:**
- Decision: Return dataclass (TourResult) not tuple
- Why: Extensible, self-documenting, type-safe
- Alternative: Return tour, weight separately (rejected: harder to extend)

**Registry Pattern:**
- Decision: Decorator-based registration
- Why: Clean, automatic, follows Python conventions
- Alternative: Manual registration in __init__.py (rejected: error-prone)

**Results Storage:**
- Decision: JSON for full results, CSV for summaries
- Why: JSON human-readable + hierarchical, CSV for pandas/R
- Alternative: SQLite database (deferred: overkill for current scale)

**Statistical Tests:**
- Decision: Non-parametric tests (Wilcoxon) as default
- Why: No normality assumption, robust
- Alternative: t-tests (available but require normality check)

---

## Testing Strategy

### Unit Tests (per component)
- Algorithm interface contract validation
- Tour validation on known invalid tours
- Metrics computation on hand-calculated examples
- Registry registration and retrieval
- Storage save/load round-trips

### Integration Tests
- Single graph benchmark on all graph types
- Batch benchmark on small collection (5 graphs, 2 algorithms)
- Results storage and retrieval
- Statistical analysis on synthetic data

### End-to-End Tests
- Full pipeline: generate graphs → benchmark → analyze → report
- Resumption test: interrupt and resume batch benchmark
- Reproducibility test: same config → same results

### Test Organization
- `src/tests/test_algorithms/` directory
- Files: `test_interface.py`, `test_validation.py`, `test_baselines.py`, `test_anchors.py`, `test_benchmarking.py`, `test_statistics.py`
- Target: 50+ tests covering all components
- Run: `python -m pytest src/tests/test_algorithms/`

---

## Estimated Effort

**Step 1:** Interface & Registry - 0.5 days
**Step 2:** Validation & Metrics - 0.5 days
**Step 3:** Baseline Algorithms - 1.5 days (Held-Karp most complex)
**Step 4:** Anchor Algorithms - 1.5 days (multi-anchor most complex)
**Step 5:** Single Benchmark - 0.5 days
**Step 6:** Batch & Storage - 1.0 days
**Step 7:** Statistics - 1.0 days
**Step 8:** Visualization - 1.0 days

**Total: 7-8 days of focused implementation**

Additional time for:
- Testing and debugging: +3 days
- Documentation: +1 day
- Integration refinement: +1 day

**Phase 2 total: 12-13 days** (within 2-3 week estimate from CLAUDE.md)

---

## Success Validation Checklist

Before marking Phase 2 complete, verify:

- [ ] All algorithms produce valid tours 100% of time (or fail gracefully)
- [ ] Benchmark 100 graphs × 5 algorithms in <1 hour
- [ ] All tours validated as proper Hamiltonian cycles
- [ ] Statistical comparison framework operational
- [ ] Held-Karp finds optimal solutions on small test cases
- [ ] Best anchor provably finds best among all single-anchor options
- [ ] Checkpoint/resume works correctly
- [ ] Results reproducible from config + seed
- [ ] Publication-quality visualizations generated
- [ ] Statistical tests provide clear significance results
- [ ] HTML report comprehensive and professional
- [ ] 50+ tests passing
- [ ] Code follows architectural principles
- [ ] Documentation complete

---

## Next Steps After Phase 2 Completion

With Phase 2 complete, you will have:
1. Performance data for all algorithms on diverse graph types
2. Statistical evidence of which algorithms excel where
3. Rich metadata about anchor usage for each tour

**Phase 3 Preparation:**
- Identify graphs where anchor algorithms beat baselines (high anchor quality)
- Identify graphs where anchor algorithms lose (low anchor quality)
- Question: What graph properties distinguish these cases?
- Answer: Extract vertex features to predict anchor quality

**Phase 3 Input:**
- Graph instances (from Phase 1)
- Anchor quality scores (from Phase 2 benchmarking results)
- Best anchor vertex per graph (from best_anchor metadata)

**Documentation Updates:**
- Add Phase 2 change log to `docs/`
- Update CLAUDE.md with any new architectural principles discovered
- Update success metrics with actual benchmarking performance

---

**Plan Version:** 1.0
**Created:** 03-11-2025
**Author:** Planner Agent (Sonnet 4)
**For Execution By:** Builder Agent (Haiku)
**Estimated Duration:** 2-3 weeks
