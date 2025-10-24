# Metaprompt 2: Algorithm Benchmarking Pipeline

## Context
You have a graph generation system producing diverse TSP instances. Now you need a rigorous benchmarking pipeline to systematically test your anchor-based heuristics against established baselines. This isn't about running algorithms once and eyeballing results - it's about building a reproducible, statistically sound experimental framework.

The pipeline will run multiple algorithms on hundreds of graphs, collect performance metrics, detect which algorithms excel on which graph types, and produce publication-ready comparative analysis.

---

## Prompt 1: Unified Algorithm Interface

Design a clean interface that all TSP algorithms must implement, enabling fair comparison and easy addition of new algorithms.

The interface should require each algorithm to expose:
- A solve() method that takes a graph and returns a Hamiltonian cycle (as a list of vertex indices)
- Metadata: algorithm name, parameter configuration, version
- Optional: progress callbacks for long-running algorithms, early stopping support

The interface should return:
- The tour (list of vertices forming a Hamiltonian cycle)
- Tour weight (total cost)
- Runtime (wall-clock time in seconds)
- Algorithm-specific metadata (e.g., for best-anchor: which vertex was the anchor? For multi-anchor: how many anchors used?)
- Success/failure status (did it find a valid cycle?)

Design considerations:
- Should the interface support warm-starting (providing a partial solution to improve)?
- Should it support time limits (stop after X seconds and return best-so-far)?
- How do you handle stochastic algorithms that need multiple runs?

Think about: If an algorithm fails (throws an exception or returns an invalid tour), how should the interface handle this gracefully for batch benchmarking?

---

## Prompt 2: Algorithm Registry System

Create a registry system that maintains all available algorithms and their configurations.

The registry should:
- Map algorithm names to implementations (e.g., "nearest_neighbor" -> NearestNeighborSolver class)
- Support parameterized algorithms: "single_anchor_v5" might be anchor algorithm with specific edge selection rules
- Enable algorithm families: "best_anchor_all" runs best-anchor search across all vertices, while "best_anchor_sample_20" only tries 20 random vertices
- Tag algorithms with applicability: some algorithms only work on metric graphs (Christofides), others work universally
- Version algorithms: if you improve multi-anchor, old results used "multi_anchor_v1", new results use "multi_anchor_v2"

The registry enables:
- Selecting algorithms for benchmarking via configuration: "run these 5 algorithms on this graph collection"
- Automatic filtering: don't try to run Christofides on non-metric graphs
- Fair comparison: group results by algorithm family or version

Think about: Should the registry support algorithm composition? E.g., "nearest_neighbor + 2opt" as a pipeline?

---

## Prompt 3: Baseline Algorithm Implementations

Standardize implementations of the baseline algorithms you're comparing against. These need to be correct, efficient, and well-tested.

**Nearest Neighbor:**
- Start at a specified vertex (or try all vertices for best-of-all-starts variant)
- Greedily select nearest unvisited neighbor at each step
- Return to start after visiting all vertices
- Handle ties: if multiple neighbors have equal distance, use consistent tie-breaking (e.g., lowest vertex index)

**Greedy Edge-Picking:**
- Sort all edges by weight (ascending)
- Iteratively add cheapest edge that doesn't create a cycle or give any vertex degree > 2
- Continue until you have a Hamiltonian cycle
- This requires tracking partial tour structure - use union-find for cycle detection

**Christofides Algorithm:**
- Only applicable to metric graphs - throw an error if applied to non-metric
- Steps: 1) Find MST, 2) Find minimum-weight perfect matching on odd-degree vertices, 3) Combine into Eulerian graph, 4) Find Eulerian tour, 5) Convert to Hamiltonian by skipping repeated vertices
- This is complex - consider using existing library implementations if available

For each baseline, include:
- Parameter variations: nearest neighbor from random start vs. best start
- Input validation: check that the graph is appropriate for the algorithm
- Output validation: verify the returned tour is actually a valid Hamiltonian cycle

Think about: Should you implement approximate versions of expensive algorithms? E.g., approximate minimum matching instead of exact matching for Christofides?

---

## Prompt 4: Anchor-Based Algorithm Implementations

Standardize your novel anchor-based heuristics. These are your research contributions, so they need especially careful implementation.

**Single-Anchor Heuristic:**
- Accept parameters: graph, anchor vertex, edge selection strategy
- Pre-commit the two cheapest edges from the anchor vertex
- Build the rest of the tour greedily (specify exact greedy rule: nearest-neighbor-like growth? cheapest edge that maintains cycle properties?)
- Handle edge cases: what if the two cheapest anchor edges don't lead to a valid Hamiltonian cycle?

**Best-Anchor Search:**
- Try single-anchor heuristic starting from every vertex
- Track the best tour found across all starting vertices
- Return the best tour along with metadata about which vertex was optimal
- This is O(n) times slower than single-anchor - document runtime expectations

**Multi-Anchor Heuristic:**
- Accept parameters: number of anchors, anchor selection strategy
- Select K well-distributed anchor vertices (how? random? MST-based? geometric spacing?)
- Pre-commit cheap edges from each anchor
- Connect anchors into a tour (how? nearest-neighbor between anchors? Christofides on anchor subset?)

**Improved Multi-Anchor:**
- Use MST to identify structurally important vertices (high MST degree)
- Create regions around each anchor (Voronoi-like partitioning?)
- Build local paths within each region
- Connect regions efficiently

For all anchor variants:
- Document the exact greedy rules and tie-breaking procedures
- Include assertions that verify tour validity
- Track detailed metadata: anchor vertices used, edge weights of anchor edges, whether the algorithm encountered any constraints

Think about: Should anchor selection itself be parameterizable? What if you want to try "anchor at vertex with highest degree" vs "anchor at vertex with lowest total edge weight"?

---

## Prompt 5: Tour Validation and Quality Metrics

Build robust validation and metrics computation for algorithm outputs.

**Tour Validation:**
- Check that the tour visits exactly n vertices (no duplicates, no missing vertices)
- Verify it forms a valid cycle (last vertex connects back to first)
- Confirm all edges in the tour exist in the original graph
- For algorithms that should preserve specific properties (e.g., anchor edges), verify those constraints

**Quality Metrics:**
- Tour weight: sum of all edge weights in the tour
- Tour weight statistics across multiple runs: mean, median, std dev, min, max
- For small graphs with known optimal solutions: optimality gap = (heuristic_weight - optimal_weight) / optimal_weight
- Approximation ratio: heuristic_weight / optimal_weight

**Comparative Metrics:**
- Relative performance: how much better/worse than nearest neighbor?
- Win rate: across N instances, how often does algorithm A beat algorithm B?
- Rank statistics: for each graph, rank all algorithms by tour quality - what's the average rank of each algorithm?

Think about: Should you compute additional tour properties? E.g., tour "smoothness" (how much edge weights vary along the tour), maximum edge weight used, number of edges above the median weight?

---

## Prompt 6: Single-Graph Benchmarking Runner

Create a function that takes one graph and runs all specified algorithms on it, collecting detailed results.

The runner should:
- Accept: a graph instance, a list of algorithm names/configurations, optional timeout per algorithm
- For each algorithm:
  - Check if the algorithm is applicable to this graph type
  - Run the algorithm with timeout protection
  - Validate the returned tour
  - Compute all quality metrics
  - Record runtime
  - Catch and log any errors/exceptions
- Return a structured result object containing:
  - Graph metadata (ID, type, size, properties)
  - For each algorithm: tour, weight, runtime, metadata, success status
  - Comparative statistics: which algorithm won, by how much

Include logging:
- Progress updates: "Running nearest_neighbor on euclidean_50_12345..."
- Warnings: "Algorithm X failed on this graph: timeout"
- Summary: "Completed 5/5 algorithms in 2.3 seconds"

Think about: Should you run algorithms in random order to avoid bias from CPU warming or cache effects? Should you run each algorithm multiple times and average?

---

## Prompt 7: Batch Benchmarking System

Build a high-level system that runs benchmarking campaigns across entire graph collections.

The system should:
- Accept a configuration file specifying:
  - Path to graph collection (directory of saved graphs)
  - Algorithms to benchmark
  - Filters: only run on graphs matching criteria (e.g., only metric graphs, only size 50-100)
  - Number of trials per algorithm-graph combination (for stochastic algorithms)
  - Timeout limits
- Load graphs from disk
- Apply filters to select relevant graphs
- For each graph, run single-graph benchmarking
- Save results incrementally (don't lose everything if the run crashes halfway)
- Produce a comprehensive results database (JSON, CSV, or SQLite)

Support resumption: if the run crashes or is interrupted, be able to resume from where it left off without re-running completed experiments.

Include progress tracking:
- Overall progress: "Completed 150/500 graph-algorithm combinations"
- Estimated time remaining based on average runtime so far
- Summary statistics updated in real-time: "Current win rate: nearest_neighbor 45%, best_anchor 55%"

Think about: Should you parallelize across graphs? Across algorithms? What's the optimal parallelization strategy to maximize throughput while maintaining result quality?

---

## Prompt 8: Results Storage and Retrieval

Design a system for storing benchmarking results that supports efficient querying and analysis.

**Storage format options:**
- JSON: human-readable, supports nested structures, easy to inspect
- CSV: flat structure, easy to load into pandas/R for analysis
- SQLite: relational database, supports complex queries
- Combination: raw results as JSON, aggregated results as CSV

The storage should capture:
- Full graph metadata
- For each algorithm run: complete results including tour, weight, runtime, all metrics
- Experimental metadata: timestamp, git commit hash, machine specs, random seeds
- Relationships: link results back to source graph files

**Retrieval API:**
- Query by graph type: "give me all results on Euclidean graphs"
- Query by algorithm: "give me all best_anchor results"
- Query by performance: "give me graphs where algorithm X beat algorithm Y by >10%"
- Aggregate queries: "average tour weight by algorithm and graph type"

Support versioning: if you re-run experiments with improved algorithms, don't overwrite old results - keep both versions with timestamps.

Think about: How do you handle the volume? If you run 10 algorithms on 1000 graphs, that's 10,000 result records. At what scale do you need a real database instead of files?

---

## Prompt 9: Statistical Analysis Tools

Build tools for statistically rigorous comparison of algorithm performance.

**Descriptive statistics:**
- For each algorithm: mean/median/std dev of tour weights, runtime, optimality gaps
- Win/loss/tie counts for pairwise algorithm comparisons
- Performance by graph type and size: does algorithm A beat B on small graphs but lose on large graphs?

**Inferential statistics:**
- Paired t-tests or Wilcoxon signed-rank tests: is the performance difference between algorithms A and B statistically significant?
- Effect size measures: not just "is it significant?" but "how big is the difference?"
- Confidence intervals for mean performance differences

**Multi-factor analysis:**
- Which factors predict algorithm performance? Graph type? Size? Metricity score?
- Interaction effects: does algorithm A's advantage over B depend on graph type?

Output formats:
- Summary tables: markdown or LaTeX tables ready for papers
- Statistical test reports: "Algorithm A beats B with p<0.001, effect size d=0.8"
- Comparison matrices: heatmap showing pairwise win rates

Think about: What assumptions are your statistical tests making? Are algorithm performances normally distributed? Do you need non-parametric tests?

---

## Prompt 10: Visualization and Reporting

Create visualization tools to make benchmark results interpretable and publication-ready.

**Performance comparison plots:**
- Box plots: distribution of tour weights for each algorithm, grouped by graph type
- Line plots: performance vs. graph size, one line per algorithm
- Scatter plots: runtime vs. solution quality (Pareto frontier analysis)
- Bar charts: win rates for pairwise comparisons

**Detailed analysis plots:**
- Heatmaps: rows are graph instances, columns are algorithms, cells show relative performance
- Rank plots: for each graph, show algorithm rankings
- Improvement plots: show distribution of improvement when using algorithm A vs. baseline B

**Case study visualizations:**
For interesting individual graphs:
- Overlay all algorithm tours on the graph structure (different colors per algorithm)
- Show edge weight heatmap with tours highlighted
- Compare tours side-by-side: best tour vs. worst tour

**Summary reports:**
Generate automated HTML or markdown reports containing:
- Executive summary: best algorithm by graph type
- Statistical test results
- All key plots
- Methodology description
- Raw data tables for reproducibility

Think about: What makes a visualization useful for research? Clarity? Information density? What plots would convince a skeptical reviewer that your algorithm actually works better?

---

## Prompt 11: Experiment Configuration Management

Design a system for specifying, versioning, and documenting experimental configurations.

Use configuration files (YAML recommended) with structure like:
```
experiment_name: "baseline_comparison_v1"
graph_collection: "data/graphs/baseline_set"
algorithms:
  - name: "nearest_neighbor"
    params: {start_vertex: "best"}
  - name: "single_anchor"
    params: {anchor_selection: "lowest_weight"}
filters:
  graph_types: ["euclidean", "metric", "random"]
  size_range: [20, 100]
  metricity_threshold: 0.8
trials_per_combination: 5
timeout_seconds: 60
output_directory: "results/baseline_v1"
random_seed: 42
```

The configuration system should:
- Validate configurations before running experiments (catch typos, invalid parameters)
- Support configuration inheritance: base configuration + experiment-specific overrides
- Auto-generate experiment documentation from configs
- Version configurations alongside results

Include a command-line interface:
```
python run_benchmark.py --config experiments/baseline_v1.yaml
```

Think about: Should configurations be executable code (Python) or declarative data (YAML)? Trade-offs: flexibility vs. reproducibility?

---

## Prompt 12: Optimal Solution Computation for Small Graphs

For small graphs (≤15 vertices), compute exact optimal solutions to measure optimality gaps.

Implement or integrate:
- Held-Karp dynamic programming algorithm: O(n² * 2ⁿ) time, O(n * 2ⁿ) space
- Practical for n ≤ 20 with careful implementation
- For n > 20, use integer programming solvers (e.g., OR-Tools, Gurobi)

The optimal solver should:
- Accept a graph
- Return the optimal tour and its weight
- Report runtime (optimal solving can take minutes for n=18)
- Cache results: once you've solved a graph optimally, save the result

Use optimal solutions to:
- Validate algorithms: any algorithm returning worse than optimal is working correctly (no accidental "super-optimal" results suggesting bugs)
- Compute precise optimality gaps for heuristics
- Identify hard instances: graphs where all heuristics perform poorly relative to optimal

Think about: Should you pre-compute optimal solutions for your graph collection and save them, or compute on-demand? Pre-computation is slow but makes benchmarking faster.

---

## Prompt 13: Performance Profiling and Optimization

Build tools to understand where your algorithms spend time and identify bottlenecks.

**Profiling tools:**
- Runtime breakdown: how much time in graph loading, algorithm execution, validation, metric computation, storage?
- Algorithm-specific profiling: for anchor algorithms, time spent in anchor selection vs. tour construction
- Memory profiling: which operations allocate the most memory?

**Optimization opportunities to investigate:**
- Graph representation: adjacency matrix vs. adjacency list vs. edge list - which is fastest for your algorithms?
- Numpy/numba acceleration: can critical loops be compiled?
- Caching: are you recomputing the same things (MST, edge lists) multiple times?
- Parallelization: which operations are embarrassingly parallel?

Document performance characteristics:
- Asymptotic complexity: actual runtime vs. graph size for each algorithm
- Memory usage: actual memory vs. graph size
- Scaling limits: at what graph size does each algorithm become impractical?

Think about: Is optimizing premature? Should you profile first or build the full pipeline first?

---

## Success Criteria

You've succeeded when:
- You can run all algorithms on 500 diverse graphs in under an hour
- Every algorithm produces valid tours 100% of the time (or fails gracefully)
- You have publication-ready comparison plots showing clear performance differences
- Statistical tests confirm whether differences are significant
- A colleague can reproduce your exact results using your configuration files
- You can answer "which algorithm is best for X graph type?" with statistical confidence

## What NOT to Do

- Don't implement algorithms from scratch if good libraries exist (e.g., Christofides)
- Don't run experiments without proper statistical design (need enough samples for significance)
- Don't trust results without validation (verify tours are actually valid)
- Don't optimize code before you have a performance problem
- Don't lose results - save incrementally, version everything

## Next Steps After Completion

With robust benchmarking complete, you'll have identified:
- Which graph types favor which algorithms
- When your anchor-based heuristics beat baselines
- Which graphs expose weaknesses in each approach

This data becomes the foundation for feature engineering - understanding WHAT properties make certain vertices good anchors.
