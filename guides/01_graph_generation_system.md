# Metaprompt 1: Graph Generation System Architecture

## Context
You're building the foundation of a TSP research pipeline. The graph generation system must produce diverse, verifiable graph instances across multiple dimensions: symmetric/asymmetric, metric/non-metric, various sizes, and different structural properties. Think of this as creating your "test universe" - each graph type stresses algorithms differently and reveals different insights.

## What You're Building
A modular graph generation system that can produce controlled, reproducible graph instances with verified properties. This is NOT about writing code quickly - it's about building a clean, testable foundation that will support months of research.

---

## Prompt 1: Core Graph Data Structure

Design a clean, minimal data structure for representing TSP graph instances. This should:

- Store the adjacency matrix (edge weights between all vertex pairs)
- Include metadata: graph type, size, generation parameters, random seed
- Track verified properties: is_metric (boolean), is_symmetric (boolean), weight_range (min/max), density
- Support serialization to/from JSON for reproducibility
- Include a unique identifier (hash or UUID) for each graph instance

The structure should be simple enough to inspect manually for small graphs but efficient enough for graphs with 500+ vertices.

Think about: How will you verify the stored properties match the actual graph? Should verification happen at construction time or separately?

---

## Prompt 2: Euclidean Graph Generator

Implement a generator for Euclidean graphs - points in 2D or 3D space where edge weights equal geometric distances.

The generator should:
- Accept parameters: number of vertices, dimensionality (2D or 3D), coordinate bounds, random seed
- Generate random point coordinates within specified bounds
- Compute pairwise Euclidean distances as edge weights
- Handle the coordinate scaling problem: ensure weight ranges span desired values (e.g., if you want weights from 1-100, scale coordinates appropriately)
- Optionally support non-random point distributions: clustered points, grid-based, radially distributed

Include verification logic that confirms:
- The graph is symmetric (distance from A to B equals distance from B to A)
- The graph is metric (triangle inequality holds for all vertex triplets)
- Weight ranges match expectations

Think about edge cases: What if two points are identical? How do you handle floating-point precision when checking metricity?

---

## Prompt 3: Metric Graph Generator (Non-Euclidean)

Create a generator for metric graphs that aren't necessarily embeddable in Euclidean space. This is trickier than Euclidean generation.

Strategy approach 1 - MST-based construction:
- Start with a minimum spanning tree connecting all vertices
- Assign initial edge weights to tree edges from a controlled distribution
- For non-tree edges, compute weight as the path distance through the tree (this guarantees triangle inequality)
- Optionally perturb non-tree edge weights slightly while maintaining metricity

Strategy approach 2 - Distance matrix completion:
- Generate a partial distance matrix with random values
- Use mathematical completion techniques to fill remaining entries while enforcing triangle inequality
- This is computationally expensive but produces more "random-looking" metric graphs

The generator should:
- Accept parameters: number of vertices, weight distribution parameters, metric strictness (how tightly the triangle inequality is satisfied), random seed
- Verify metricity after generation (check all vertex triplets)
- Handle symmetric vs asymmetric cases (for quasi-metric graphs, allow directional cost differences while maintaining triangle inequality)

Think about: How do you efficiently verify metricity for 100+ vertex graphs? (Hint: this is O(n³) - is there a faster approximate check?)

---

## Prompt 4: Random Graph Generator

Implement the simplest generator - random edge weights with no structural constraints. These serve as baseline "chaotic" graphs.

The generator should:
- Accept parameters: number of vertices, weight distribution (uniform, normal, exponential, etc.), weight range bounds, symmetric vs asymmetric, random seed
- Draw each edge weight independently from the specified distribution
- For symmetric graphs, ensure weight(A,B) = weight(B,A)
- For asymmetric graphs, draw forward and backward weights independently

Include analysis logic that computes:
- What percentage of vertex triplets satisfy the triangle inequality (metricity score)
- Weight distribution statistics: mean, std dev, skewness, kurtosis
- Density metrics if supporting sparse graphs

The key insight: random graphs are almost always non-metric unless you have very few vertices. Document this in the output.

Think about: Should you support different distributions for different edge types? For example, local edges (nearby vertices) drawn from one distribution, distant edges from another?

---

## Prompt 5: Graph Property Verification System

Build a standalone verification module that validates graph properties independently of generation.

This module should:
- Take any graph instance and verify its claimed properties
- Check symmetry: ensure weight(i,j) = weight(j,i) for all i,j
- Check metricity: verify triangle inequality for all triplets (or a large random sample for huge graphs)
- Compute actual weight statistics and compare to metadata
- For Euclidean graphs with stored coordinates, verify that computed distances match edge weights (within floating-point tolerance)
- Flag any inconsistencies between metadata claims and actual properties

Design this for two use cases:
1. Automated testing during generation (fast, sample-based checking)
2. Full verification for research publication (exhaustive, slow, 100% coverage)

Think about: How do you report verification failures? Should you throw exceptions, return error objects, or log warnings?

---

## Prompt 6: Graph Instance Storage and Retrieval

Design a system for saving generated graphs to disk and loading them later.

The storage system should:
- Save graphs as JSON files with human-readable structure (for small graphs you want to inspect manually)
- Include complete metadata: generation parameters, timestamp, code version/git commit, verified properties
- Use consistent naming: graphtype_size_seed_id.json (e.g., euclidean_50_12345_abc123.json)
- Support batch operations: save all graphs from an experimental run with a manifest file listing the batch
- Include the random seed so you can regenerate if needed

The retrieval system should:
- Load graphs by ID, by properties (give me all metric graphs with 50-100 vertices), or by batch
- Verify integrity on load (check that JSON structure is valid)
- Optionally re-verify properties on load (to catch corruption or format changes)

Think about: Should you compress large adjacency matrices? How do you handle backwards compatibility if you change the graph format later?

---

## Prompt 7: Batch Generation Pipeline

Create a high-level batch generation system that produces diverse graph collections for experiments.

The pipeline should:
- Accept a configuration file (YAML or JSON) specifying:
  - Graph types to generate (Euclidean, metric, random)
  - Sizes to generate (e.g., [20, 50, 100, 200])
  - Number of instances per type/size combination
  - Random seed range
  - Any type-specific parameters
- Generate all specified graphs with progress tracking
- Save each graph with verification
- Produce a summary report: how many graphs generated, how many passed verification, property distributions
- Handle failures gracefully: if one graph generation fails, log it and continue

Support reproducibility: running the same configuration file with the same seed range should produce identical graphs.

Think about: Should you parallelize generation for speed? How do you ensure random seeds don't collide across parallel workers?

---

## Prompt 8: Graph Visualization Utilities

Build simple visualization tools for inspecting generated graphs.

For small graphs (≤20 vertices):
- Produce a graph layout visualization showing vertices and edges
- Color-code edges by weight (e.g., blue for cheap, red for expensive)
- Highlight structural features: MST edges in bold, heaviest edges in dashed lines
- For Euclidean graphs, use actual coordinates for layout; for others, use force-directed layout

For all graphs:
- Generate weight distribution histograms
- Produce heatmaps of the adjacency matrix
- Show summary statistics: mean/median edge weight, weight range, metricity score

Output formats: save as PNG/SVG for inclusion in reports, or display interactively if using Jupyter notebooks.

Think about: How do you make visualizations useful for debugging? If a graph fails verification, what visualization would help you understand why?

---

## Prompt 9: Test Suite for Graph Generators

Design a comprehensive test suite to ensure all generators work correctly.

Tests should cover:

**Property tests for each generator type:**
- Euclidean graphs are always symmetric and metric
- Metric graphs always satisfy triangle inequality
- Random symmetric graphs have matching forward/backward weights
- Graph size matches requested vertex count

**Edge case tests:**
- Very small graphs (3-4 vertices) - manually verify correctness
- Degenerate cases: all edges equal weight, weights from extremely narrow range
- Extreme weight ranges: very small (0.001-0.002) vs very large (1000000+)
- Floating-point precision: ensure no precision errors break metricity

**Consistency tests:**
- Generate the same graph twice with the same seed - should be identical
- Generate with different seeds - should be different
- Save and load a graph - should be identical after round-trip

**Performance benchmarks:**
- How long to generate graphs of different sizes?
- Does verification time scale as expected (O(n³) for metricity)?

Think about: Should tests be deterministic (fixed seeds) or randomized (random seeds but checking properties)? Both serve different purposes.

---

## Prompt 10: Graph Collection Analysis Tools

Create analysis tools for understanding the diversity of your generated graph collection.

These tools should:
- Scan a directory of saved graphs and produce summary statistics
- Compute coverage metrics: how many graphs of each type/size combination?
- Analyze property distributions: what percentage are metric? What's the distribution of metricity scores for random graphs?
- Identify outliers: graphs with unusual weight distributions or unexpected properties
- Check for duplicates or near-duplicates (graphs that are suspiciously similar despite different seeds)

Produce visualizations:
- Scatter plots: graph size vs metricity score, colored by graph type
- Distribution plots: weight range distributions across all graphs
- Coverage heatmap: rows are graph types, columns are sizes, cells show instance counts

This helps you verify you've generated a balanced, diverse dataset before running expensive benchmarking.

Think about: What does "diversity" mean for a graph collection? Is 100 nearly-identical Euclidean graphs diverse? How do you measure it?

---

## Success Criteria

You've succeeded when:
- You can generate 100 graphs across 5 types and 4 sizes in under a minute
- Every generated graph passes verification for its claimed properties
- You can save and reload graphs with perfect fidelity
- The test suite catches intentional bugs you introduce
- A colleague could generate identical graphs using just your configuration file and seeds
- You can visualize any small graph and immediately understand its structure

## What NOT to Do

- Don't optimize prematurely - clean code first, speed later
- Don't skip verification - you'll waste weeks debugging with corrupted graphs
- Don't hardcode parameters - everything should be configurable
- Don't mix generation and analysis logic - keep modules separate
- Don't generate graphs without seeds - reproducibility is sacred

## Next Steps After Completion

Once graph generation is solid, you'll build the algorithm benchmarking system on top of this foundation. The graphs you generate here will be reused across hundreds of algorithm runs, so invest in quality now.
