# Metaprompt 3: Feature Engineering System

## Context
You've generated diverse graphs and benchmarked algorithms. Now comes the machine learning bridge: extracting structural features from graphs and vertices that predict anchor quality. This is the heart of your research contribution - understanding WHAT makes a vertex a good anchor.

This isn't about throwing every possible feature at a model. It's about thoughtful feature design informed by graph theory, algorithm behavior, and TSP intuition. You're building the language to describe "good anchor vertices" mathematically.

---

## Prompt 1: Vertex Feature Extraction Architecture

Design a clean, extensible architecture for computing vertex-level features.

The system should:
- Accept a graph and return a feature matrix: rows are vertices, columns are features
- Support modular feature extractors: each feature type (weight-based, centrality, structural) is its own module
- Enable feature selection: easily enable/disable feature groups for experimentation
- Cache expensive computations: if multiple features need the MST, compute it once
- Handle feature naming: each column gets a descriptive name like "total_edge_weight", "mst_degree", "betweenness_centrality"

Architecture pattern to consider:
- Base class: VertexFeatureExtractor with extract() method
- Derived classes: WeightFeatureExtractor, CentralityFeatureExtractor, etc.
- Orchestrator: FeatureExtractorPipeline that runs all enabled extractors

Include validation:
- Check that all vertices get the same number of features
- Verify no NaN or infinite values unless expected
- Confirm feature values are in reasonable ranges

Think about: Should features be normalized during extraction or later during ML? Should you support feature transformations (log, square, standardization)?

---

## Prompt 2: Weight-Based Vertex Features

Implement extractors for features derived from edge weights incident to each vertex.

**Basic statistics:**
- Total weight: sum of all edges from this vertex
- Mean weight: average distance to other vertices
- Median weight: middle value of edge weights
- Standard deviation: spread of edge weights
- Variance: squared spread

**Distribution features:**
- Min edge weight: distance to nearest neighbor
- Max edge weight: distance to farthest vertex
- Min/max ratio: how concentrated are nearby neighbors?
- Quantiles: 25th, 50th, 75th percentile of edge weights
- Skewness: is distribution of weights symmetric or skewed?
- Kurtosis: how heavy are the tails of the distribution?

**Relative features:**
- Rank of cheapest edge: how does this vertex's cheapest edge compare to all cheapest edges in the graph?
- Proportion of edges below graph median: how "central" is this vertex weight-wise?
- Distance to graph centroid: define centroid as vertex minimizing sum of distances to all others

For asymmetric graphs, compute separate features for:
- Outgoing edges (edges leaving this vertex)
- Incoming edges (edges arriving at this vertex)
- Asymmetry metrics: difference between outgoing and incoming statistics

Think about: Are raw values useful or should you normalize by graph statistics? E.g., "this vertex's mean edge weight is 1.5 standard deviations below the graph mean."

---

## Prompt 3: Topological Vertex Features

Implement features based on graph topology and centrality measures.

**Degree-based features:**
- Degree: number of neighbors (constant in complete graphs, interesting in sparse graphs)
- Weighted degree: sum of edge weights (same as total weight, but conceptually different)

**Centrality measures:**
- Closeness centrality: inverse of average shortest path distance to all other vertices
  - Captures "how central" a vertex is in the graph
  - High closeness = close to everything, potentially bad for anchoring
- Betweenness centrality: how many shortest paths pass through this vertex?
  - High betweenness = structural importance as a bridge
  - Expensive to compute: O(n³) naive, O(n²log n + nm) with Dijkstra for each vertex
- Eigenvector centrality: vertex importance based on connection to other important vertices
  - Power iteration method for computation

**Clustering features:**
- Clustering coefficient: how interconnected are this vertex's neighbors?
  - Local clustering: proportion of possible edges between neighbors that actually exist
  - High clustering = vertex is in a tight community

**Distance-based features:**
- Eccentricity: maximum shortest path distance from this vertex
  - Periphery vertices have high eccentricity
- Average shortest path length: mean distance to all other vertices (related to closeness)

Think about: Some centrality measures are expensive - should you approximate them for large graphs? Should you compute on the full graph or on a weighted graph derived from edge weights?

---

## Prompt 4: MST-Based Vertex Features

Implement features derived from the minimum spanning tree of the graph. The MST reveals structural importance.

**Basic MST features:**
- MST degree: how many edges incident to this vertex are in the MST?
  - High MST degree = structurally important "hub" vertex
  - This is cheap to compute: O(n²) for MST, O(n) for degree counting
- Is this vertex an MST leaf? (boolean feature: MST degree = 1)
- Is this vertex an MST hub? (boolean: MST degree ≥ k for some threshold k)

**MST edge weight features:**
- Total weight of MST edges incident to this vertex
- Mean weight of MST edges from this vertex
- Ratio of MST edge weights to all edge weights from this vertex

**MST path features:**
- Distance to MST center: define center as vertex minimizing max distance in MST
- Depth in MST tree (if rooted at a specific vertex)
- Number of MST leaves reachable from this vertex

**MST-based importance:**
- If this vertex were removed, how many MST components would result?
- Total weight of MST vs. total weight of MST with this vertex removed (measures structural importance)

Think about: Should you compute MST on the original graph or on a transformed version (e.g., inverted weights to prioritize long edges)? Different MST definitions might reveal different structural aspects.

---

## Prompt 5: Neighborhood and Regional Features

Implement features capturing local neighborhood structure around each vertex.

**K-nearest neighbors features:**
- For k=1,2,3,5, compute statistics of k-nearest neighbors
- Mean weight of k-nearest neighbors
- Variance of k-nearest neighbor weights
- Are the k-nearest neighbors clustered together or spread out?

**Neighborhood density:**
- Define neighborhood as vertices within distance d (e.g., within median edge weight)
- How many vertices are in the neighborhood?
- What's the total weight of edges within the neighborhood?
- Is the neighborhood tightly connected (many intra-neighborhood edges)?

**Radial features:**
- Divide vertices into shells by distance from this vertex
- Features: number of vertices in each shell, mean weight within each shell
- Captures whether the graph is "uniform" or "layered" around this vertex

**Voronoi-like features:**
- If you partition the graph by nearest anchor vertex (Voronoi diagram concept)
- How many vertices would be closest to this vertex?
- What's the total weight within this region?

Think about: Do neighborhood features help distinguish between locally central vertices (good neighbors) and globally central vertices (potentially bad anchors)? How do you choose neighborhood size parameters?

---

## Prompt 6: Heuristic-Specific Features

Implement features directly inspired by how anchor-based heuristics work.

**Anchor edge features:**
- Weight of two cheapest edges from this vertex (these become the anchor edges)
- Sum and product of two cheapest edge weights
- Gap: difference between 2nd cheapest and 3rd cheapest edge
  - Large gap = clear anchor choice, small gap = ambiguous
- Ratio: (2nd cheapest) / (cheapest edge)

**Tour construction features:**
- Estimate of "remaining tour cost" if we start from this vertex and use anchor edges
- Prediction of tour quality based on local greedy simulation (fast approximate tour construction)

**Constraint features:**
- How constraining are the anchor edges?
- Do the anchor edges "point" in opposite directions (good spread) or same direction (potential dead-end)?
- Angle between anchor edges in Euclidean space (only for Euclidean graphs)

**Baseline comparison features:**
- Nearest neighbor tour cost starting from this vertex (without anchor constraint)
- Compare to anchor tour cost: is anchor helping or hurting?

Think about: Are these features "cheating"? You're predicting anchor quality using features derived from anchor behavior. But if they're cheaper to compute than full anchor search, they're still useful.

---

## Prompt 7: Graph-Level Context Features

Some vertex features depend on global graph properties. Implement features that contextualize vertices within their graph.

**Normalization features:**
- For each vertex feature (e.g., total edge weight), compute:
  - Z-score: (value - graph mean) / graph std dev
  - Percentile rank: what percentile is this vertex in the graph's distribution?
  - Distance from graph median

**Graph property features:**
These are constant across all vertices in a graph but help ML models distinguish graph types:
- Graph size (number of vertices)
- Graph density (if sparse)
- Metricity score (percentage of triplets satisfying triangle inequality)
- Weight distribution statistics: mean, std dev, skewness, kurtosis of all edge weights
- Graph diameter (longest shortest path)

**Relative importance features:**
- Ratio of this vertex's MST degree to max MST degree in graph
- Ratio of this vertex's total weight to graph total weight
- Relative centrality: this vertex's closeness centrality / max closeness in graph

Think about: Should graph-level features be included explicitly or should normalization handle it? If your training set has balanced graph types, normalization might be sufficient. If not, explicit graph-level features help.

---

## Prompt 8: Feature Validation and Analysis

Build tools to validate extracted features and understand their relationships.

**Sanity checks:**
- Range validation: are feature values in expected ranges? (e.g., betweenness centrality should be [0, 1])
- Correlation checks: are some features perfectly correlated? (e.g., total weight and weighted degree in complete graphs)
- Constant features: identify features with zero variance (useless for ML)

**Exploratory data analysis:**
- Compute correlation matrix for all features
- Identify highly correlated feature pairs (>0.95 correlation)
- Cluster features by similarity
- Perform PCA: which principal components explain most variance?

**Feature-target correlation:**
- Correlate each feature with anchor quality score
- Rank features by predictive power (univariate correlation)
- Identify surprising correlations (features you didn't expect to matter)
- Create scatter plots: feature value vs. anchor quality

**Feature distributions:**
- Histograms for each feature across all vertices in dataset
- Check for skewness, outliers, multi-modal distributions
- Identify features needing transformation (log, square root, etc.)

Think about: Should you remove highly correlated features before ML training? Some models (linear regression) are sensitive to multicollinearity, others (random forests) aren't.

---

## Prompt 9: Anchor Quality Labeling System

Design a system for assigning "anchor quality scores" to vertices based on algorithm performance. This is your ML target variable.

**Labeling strategies:**

**Strategy 1 - Absolute quality:**
- Run single-anchor from each vertex, record tour weight
- Score = 1 / tour_weight (lower weight = higher score)
- Or score = max_weight_in_graph - tour_weight

**Strategy 2 - Rank-based quality:**
- Run single-anchor from all vertices
- Rank vertices by tour quality (best = rank 1)
- Score = percentile rank (0-100 scale)
- Normalizes across graphs of different scales

**Strategy 3 - Binary classification:**
- Label top k% of vertices as "good anchors" (positive class)
- Label rest as "bad anchors" (negative class)
- Simplifies ML problem to classification

**Strategy 4 - Relative to optimal:**
- For small graphs with known optimal tour weight
- Score = optimal_weight / anchor_tour_weight
- Score of 1.0 = perfect, score of 1.5 = 50% worse than optimal

**Strategy 5 - Multi-class:**
- Excellent anchors (top 10%)
- Good anchors (10-30%)
- Mediocre anchors (30-70%)
- Poor anchors (70-100%)

The labeling system should:
- Be deterministic: same graph + same vertex = same score
- Handle ties: if multiple vertices produce identical tours
- Store metadata: which algorithm was used for labeling (single-anchor? multi-anchor?)

Think about: Which strategy best captures your research question? If you care about finding THE best anchor, rank-based or binary might be best. If you care about avoiding terrible anchors, maybe multi-class.

---

## Prompt 10: Feature Engineering Pipeline

Create an end-to-end pipeline that takes a graph collection and produces an ML-ready dataset.

The pipeline should:
1. Load a collection of graphs from disk
2. For each graph:
   a. Run best-anchor search to identify optimal vertex (expensive!)
   b. Run single-anchor from every vertex to get anchor quality labels
   c. Extract all features for all vertices
   d. Combine features and labels into a dataset row for each vertex
3. Aggregate all rows into a single dataset (DataFrame or CSV)
4. Add metadata columns: graph_id, vertex_id, graph_type, graph_size
5. Perform feature validation and cleaning
6. Save as ML-ready format

Include progress tracking:
- "Processing graph 15/100..."
- "Extracted features for 750/5000 vertices..."
- Estimated time remaining

Support caching:
- If a graph has already been processed, skip it (unless explicitly re-running)
- Store intermediate results: features and labels separately for easier debugging

Include summary statistics:
- Feature coverage: any missing values?
- Label distribution: balanced or skewed?
- Dataset size: how many training examples?

Think about: This pipeline is expensive (best-anchor search is O(n²)). Should you parallelize? Should you process a subset of graphs first to validate the pipeline before running on full collection?

---

## Prompt 11: Feature Selection Utilities

Build tools to identify which features are most predictive, enabling dimensionality reduction.

**Univariate feature selection:**
- Compute correlation between each feature and target (anchor quality)
- Rank features by absolute correlation
- Use statistical tests: F-test, mutual information

**Recursive feature elimination:**
- Train a simple model with all features
- Iteratively remove least important features
- Measure performance degradation

**Model-based feature importance:**
- Train tree-based models (random forest)
- Extract feature importance scores
- Identify top-k most important features

**Regularization-based selection:**
- Train linear models with L1 regularization (Lasso)
- L1 drives unimportant feature coefficients to zero
- Selected features are those with non-zero coefficients

**Greedy forward selection:**
- Start with zero features
- Iteratively add feature that most improves model performance
- Stop when performance plateaus

Output:
- Ranked feature lists
- Plots: feature importance bar charts
- Selected feature sets: "top 10 features by importance"

Think about: Should you select features globally or per graph type? Maybe different features matter for Euclidean vs. random graphs?

---

## Prompt 12: Feature Transformation and Engineering

Beyond raw features, create derived features through transformations and combinations.

**Non-linear transformations:**
- Log transform: log(total_weight) for skewed features
- Square root: sqrt(betweenness_centrality)
- Polynomial: total_weight², total_weight³
- Inverse: 1/mean_edge_weight

**Feature interactions:**
- Products: mst_degree × total_weight
- Ratios: min_edge_weight / mean_edge_weight
- Differences: max_weight - min_weight

**Binning and discretization:**
- Convert continuous features to categorical: "low", "medium", "high" MST degree
- One-hot encode: useful for tree models

**Domain-specific combinations:**
- "Anchor favorability" = (min1 + min2) / mean_weight (anchor edge quality relative to average)
- "Centrality score" = weighted combination of multiple centrality metrics
- "Hub score" = MST degree × betweenness centrality

**Standardization:**
- Z-score normalization: (x - mean) / std
- Min-max scaling: (x - min) / (max - min)
- Robust scaling: use median and IQR instead of mean and std

The transformation system should:
- Be configurable: specify which transformations to apply
- Handle edge cases: log of zero, division by zero
- Maintain interpretability: document what each derived feature represents

Think about: Should transformations be part of feature extraction or part of ML preprocessing? Separation of concerns vs. convenience?

---

## Success Criteria

You've succeeded when:
- You can extract 20-50 meaningful features for any vertex in any graph type
- Feature extraction for a 100-vertex graph completes in under 10 seconds
- You've identified 5-10 features that clearly correlate with anchor quality (|r| > 0.3)
- The ML-ready dataset is clean: no missing values, reasonable distributions, clear documentation
- Feature names are self-explanatory: a colleague can understand what "mst_degree_normalized" means

## What NOT to Do

- Don't extract thousands of redundant features hoping ML will figure it out
- Don't skip validation: bad features = bad models
- Don't ignore computational cost: betweenness centrality is O(n³), use it wisely
- Don't forget about feature interpretability: you need to explain WHY your model works
- Don't assume features that work on one graph type work on all types

## Next Steps After Completion

With features extracted and labeled, you'll move to machine learning: training models to predict anchor quality from features. The quality of your ML models depends entirely on the quality of features you've engineered here. Invest time in thoughtful feature design, not just throwing everything at the wall.
