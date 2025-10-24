# Metaprompt 5: Pipeline Integration and Workflow Management

## Context
You've built the components: graph generation, algorithm benchmarking, feature engineering, and ML models. Now comes the hard part: integrating everything into a cohesive, reproducible research pipeline. This isn't just about making things work - it's about building a system that you (and others) can use reliably for months of experimentation.

Think of this as the "operating system" for your TSP research. Good integration means experiments are reproducible, results are trustworthy, and you can iterate quickly on research questions.

---

## Prompt 1: Pipeline Architecture Design

Design the overall architecture connecting all components into a unified workflow.

**High-level pipeline stages:**

Stage 1: Data Generation
- Input: generation configuration (graph types, sizes, counts)
- Output: collection of verified graph instances (saved to disk)

Stage 2: Algorithm Benchmarking
- Input: graph collection, algorithm configurations
- Output: performance database (tours, weights, runtimes for each graph-algorithm pair)

Stage 3: Feature Engineering
- Input: graph collection, benchmarking results
- Output: ML dataset (features + anchor quality labels for each vertex)

Stage 4: Model Training
- Input: ML dataset, model configurations
- Output: trained models (saved model files + performance reports)

Stage 5: Model Evaluation
- Input: trained models, test graphs
- Output: generalization performance metrics, comparative analysis

**Design principles:**
- Modularity: each stage can run independently
- Idempotency: running a stage twice produces same results
- Resumability: if a stage crashes, can resume without starting over
- Observability: clear logging at each stage
- Configurability: everything controlled via config files

**Data flow:**
- Use filesystem as data layer: each stage reads/writes files
- Clear directory structure: data/graphs/, results/benchmarks/, results/features/, models/
- Manifest files: each stage produces a manifest listing what it created

Think about: Should stages be Python scripts, command-line tools, or Jupyter notebooks? Scripts are reproducible, notebooks are explorable. Maybe both?

---

## Prompt 2: Configuration Management System

Build a comprehensive configuration system for specifying experiments.

**Configuration file structure (YAML recommended):**

```yaml
experiment:
  name: "baseline_comparison_v1"
  description: "Compare anchor heuristics to nearest neighbor"
  random_seed: 42

graph_generation:
  enabled: true
  types:
    - type: "euclidean"
      sizes: [20, 50, 100]
      instances_per_size: 20
      dimension: 2
    - type: "metric"
      sizes: [50, 100]
      instances_per_size: 10

benchmarking:
  enabled: true
  algorithms:
    - name: "nearest_neighbor"
      params: {strategy: "best_start"}
    - name: "single_anchor"
      params: {selection: "lowest_weight"}
    - name: "best_anchor"
  timeout_seconds: 300

feature_engineering:
  enabled: true
  feature_groups:
    - "weight_based"
    - "mst_based"
    - "centrality"
  labeling_strategy: "rank_based"

model_training:
  enabled: true
  models:
    - type: "linear_regression"
      regularization: "ridge"
      cv_folds: 5
    - type: "random_forest"
      n_estimators: 100
  test_split: 0.2
```

**Configuration system features:**
- Validation: check for invalid parameters before running experiment
- Inheritance: base config + experiment-specific overrides
- Templates: common configurations as reusable templates
- Documentation: auto-generate docs from config schema

**Command-line interface:**
```
python run_experiment.py --config experiments/baseline_v1.yaml
python run_experiment.py --config experiments/baseline_v1.yaml --stage feature_engineering
python run_experiment.py --config experiments/baseline_v1.yaml --dry-run
```

Think about: Should you support dynamic configs (Python code) or static configs (YAML)? Static is more reproducible but less flexible.

---

## Prompt 3: Experiment Tracking and Metadata

Build a system to track experiments, their configurations, and their results.

**Experiment metadata:**
- Unique experiment ID (UUID or hash)
- Timestamp: when experiment started/completed
- Configuration: full config file used
- Code version: git commit hash
- Environment: Python version, package versions, OS
- Status: running, completed, failed
- Results summary: key metrics, output locations

**Experiment registry:**
- Database (SQLite) or JSON file tracking all experiments
- Enables queries: "show me all experiments on metric graphs"
- Enables comparisons: "how do results compare between v1 and v2?"

**Logging system:**
- Structured logs: JSON lines format for easy parsing
- Log levels: DEBUG, INFO, WARNING, ERROR
- Per-stage logs: separate log files for generation, benchmarking, etc.
- Progress tracking: "Stage 2/5: 45% complete"

**Result organization:**
```
experiments/
  baseline_v1_20250124_abc123/
    config.yaml
    metadata.json
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

Think about: How do you handle experiment variants? If you change one parameter, is it a new experiment or a modification?

---

## Prompt 4: Reproducibility Infrastructure

Ensure all experiments are perfectly reproducible by anyone with the code and config.

**Random seed management:**
- Set seeds at every stage: graph generation, algorithm randomness, train/test splits, model initialization
- Propagate seed from experiment config to all components
- Document: "To reproduce, use config X with seed Y"

**Environment management:**
- requirements.txt or environment.yml listing all dependencies with exact versions
- Virtual environment or conda environment
- Docker container for ultimate reproducibility

**Code versioning:**
- Tag releases: v1.0, v2.0
- Record git commit hash with every experiment
- If someone has your code at commit X and config Y, they should get identical results

**Data versioning:**
- Generated graphs are reproducible from seeds
- But also save generated graphs to guarantee reproducibility if generation code changes
- Use data versioning tools (DVC, git-lfs) for large datasets

**Reproducibility checklist:**
- [ ] Run experiment twice with same config - identical results?
- [ ] Run on different machine - identical results?
- [ ] Run 6 months later - identical results?
- [ ] Another researcher runs it - identical results?

Think about: How do you handle non-deterministic algorithms? Force determinism or run multiple trials and report distributions?

---

## Prompt 5: Automated Testing and Validation

Build comprehensive tests to catch bugs before they corrupt research results.

**Unit tests:**
- Graph generation: verify properties (metricity, symmetry)
- Algorithm implementations: test on small known graphs
- Feature extraction: verify calculations on toy examples
- Model training: smoke tests that training completes

**Integration tests:**
- End-to-end pipeline: run minimal experiment (2 graphs, 2 algorithms) and verify completion
- Stage interfaces: verify stage outputs have expected format for next stage's inputs

**Validation tests:**
- Tour validation: every algorithm output is a valid Hamiltonian cycle
- Feature validation: no NaN, no infinite values, reasonable ranges
- Model validation: predictions are in expected range

**Regression tests:**
- Save "gold standard" results from known-good runs
- Re-run and compare: if results differ, investigate why
- Catches unintended changes in algorithm behavior

**Property-based tests:**
- Use hypothesis or similar: generate random graphs, verify invariants
- E.g., "for any graph, best-anchor tour ≤ any single-anchor tour"

**Continuous testing:**
- Run test suite on every code commit
- Automated with GitHub Actions, Travis CI, or similar
- Prevents bugs from sneaking in

Think about: How much testing is enough? You can't test everything. Focus on components where bugs would be catastrophic (graph generation, tour validation).

---

## Prompt 6: Performance Monitoring and Profiling

Build tools to understand where time is spent and optimize bottlenecks.

**Runtime profiling:**
- Instrument each pipeline stage with timing
- Report: "Graph generation: 2m, Benchmarking: 45m, Features: 10m, Training: 3m"
- Identify bottlenecks: where does 80% of time go?

**Per-graph profiling:**
- Track runtime vs. graph size
- Identify scaling behavior: is it O(n²) as expected? O(n³)? Worse?
- Plot: runtime vs. n on log-log scale to verify complexity

**Algorithm profiling:**
- Which algorithms are slowest?
- For slow algorithms, profile internally: which steps are bottlenecks?
- Use cProfile or line_profiler for detailed Python profiling

**Memory profiling:**
- Track memory usage per stage
- Identify memory leaks or excessive allocation
- Use memory_profiler or tracemalloc

**Optimization opportunities:**
- Parallelization: which operations can run in parallel?
- Caching: which computations are repeated unnecessarily?
- Algorithmic improvements: can you reduce complexity?
- Implementation: can you use numpy/numba for critical loops?

**Performance regression tracking:**
- Monitor key metrics over time: "graph generation time per graph"
- Alert if performance degrades significantly in new code

Think about: Premature optimization is the root of all evil. Profile first, optimize second. Don't optimize until you've identified actual bottlenecks.

---

## Prompt 7: Parallel Execution and Scaling

Design parallelization strategy to maximize throughput on multi-core machines.

**Embarrassingly parallel operations:**
- Graph generation: each graph is independent
- Benchmarking: each graph-algorithm combination is independent
- Feature extraction: each graph is independent (within-graph computations may not be parallelizable)

**Parallelization approaches:**

**Option 1: Multiprocessing**
- Python multiprocessing module
- Spawn worker processes for parallel tasks
- Pro: true parallelism (bypasses GIL), simple for independent tasks
- Con: overhead of process creation, serialization costs

**Option 2: Joblib**
- High-level parallel computing library
- Parallel(n_jobs=-1)(delayed(func)(arg) for arg in args)
- Pro: easy to use, caches results
- Con: same limitations as multiprocessing

**Option 3: Dask**
- Distributed computing framework
- Scales from single machine to clusters
- Pro: handles large-scale data, sophisticated scheduling
- Con: more complex, overhead for small problems

**Parallelization strategy per stage:**
- Graph generation: parallelize over graphs
- Benchmarking: parallelize over (graph, algorithm) pairs
- Feature extraction: parallelize over graphs
- Model training: some algorithms parallelize internally (random forest), others don't (single tree)

**Resource management:**
- Don't oversubscribe: n_workers ≤ n_cpu_cores
- Memory limits: if each graph uses 1GB, limit parallelism to avoid OOM
- Shared resources: MST computation for multiple feature extractors - compute once per graph

Think about: What's your target hardware? Laptop (4 cores), workstation (16 cores), cluster (hundreds of cores)? Design for your use case.

---

## Prompt 8: Error Handling and Fault Tolerance

Design robust error handling so individual failures don't crash entire experiments.

**Error categories:**

**Recoverable errors:**
- Algorithm timeout on one graph: log warning, skip, continue
- Feature extraction failure: log error, skip graph, continue
- Invalid graph generation: retry or skip

**Fatal errors:**
- Configuration file invalid: fail immediately
- Output directory not writable: fail immediately
- Required dependency missing: fail immediately

**Error handling patterns:**

**Try-continue pattern:**
```python
for graph in graphs:
    try:
        result = benchmark_graph(graph)
        results.append(result)
    except Exception as e:
        logger.error(f"Failed on graph {graph.id}: {e}")
        # continue with next graph
```

**Retry with backoff:**
- If operation fails, retry a few times with exponential backoff
- Useful for transient failures (network, disk I/O)

**Graceful degradation:**
- If expensive feature (betweenness centrality) times out, skip it
- Model trains with fewer features rather than failing completely

**Checkpointing:**
- Save intermediate results periodically
- If pipeline crashes, resume from last checkpoint
- E.g., save benchmarking results after every 10 graphs

**Error reporting:**
- Summary at end: "Completed 95/100 graphs, 5 failures"
- Error log with details for manual inspection
- Automatic alerting for critical failures (email, Slack)

Think about: How do you balance robustness vs. correctness? If 5% of graphs fail, is that acceptable or does it bias results?

---

## Prompt 9: Results Analysis and Reporting

Build automated analysis and report generation to interpret experimental results.

**Statistical analysis:**
- Compute aggregate metrics across all graphs
- Perform statistical tests: algorithm A vs. B
- Generate comparison tables: markdown or LaTeX

**Visualization generation:**
- Performance comparison plots (box plots, line plots)
- Feature correlation heatmaps
- Model performance plots (actual vs. predicted)
- All plots saved as PNG/PDF for inclusion in papers

**Automated report generation:**
- HTML or markdown report with:
  - Experiment configuration
  - Summary statistics
  - Key findings ("Algorithm X outperforms Y by 15% on average")
  - All plots
  - Statistical test results
  - Raw data tables
- Report generated automatically at end of experiment

**Comparison reports:**
- Compare multiple experiments side-by-side
- "How did results change from v1 to v2?"
- Highlight significant differences

**Insight extraction:**
- Identify patterns: "Best-anchor wins on metric graphs but loses on random graphs"
- Flag anomalies: graphs where all algorithms perform poorly (hard instances)
- Suggest next steps: "Model performs poorly on large graphs - collect more training data?"

Think about: Should reports be generated automatically or manually triggered? Auto-generation ensures you never forget, but might create clutter.

---

## Prompt 10: Interactive Exploration Tools

Build tools for interactive exploration of results during and after experiments.

**Jupyter notebooks:**
- Load experiment results into pandas DataFrames
- Interactive plotting with matplotlib/seaborn/plotly
- Ad-hoc analysis: "Show me all graphs where nearest-neighbor beat best-anchor"
- Quick hypothesis testing without re-running full pipeline

**Command-line query tool:**
```bash
python query_results.py --experiment baseline_v1 --metric tour_weight --group-by graph_type
python query_results.py --experiment baseline_v1 --compare nearest_neighbor best_anchor
```

**Web dashboard (optional, advanced):**
- Real-time monitoring of running experiments
- Interactive plots (zoom, filter, drill-down)
- Result browser: click on graph to see detailed analysis
- Tools: Streamlit, Dash, or custom Flask app

**Result database queries:**
- If using SQLite for results storage
- SQL queries for complex analysis
- Join tables: graphs × algorithms × features

**Case study explorer:**
- Select a specific graph
- Visualize graph structure
- Show all algorithm tours overlaid
- Display feature values and predictions
- Compare predicted vs. actual best anchor

Think about: How much time should you invest in exploration tools? They're invaluable for understanding results but can be a time sink. Start simple, expand as needed.

---

## Prompt 11: Documentation System

Create comprehensive documentation for the entire pipeline.

**Code documentation:**
- Docstrings for every public function/class
- Module-level docstrings explaining purpose
- Examples in docstrings
- Type hints for function signatures

**Architecture documentation:**
- README.md with high-level overview
- Architecture diagram showing component relationships
- Data flow diagrams
- API documentation for key interfaces

**User documentation:**
- Getting started guide: setup, running first experiment
- Configuration guide: all config options explained
- Troubleshooting guide: common errors and solutions
- FAQ: frequently asked questions

**Research documentation:**
- Methodology notes: why you made certain design choices
- Algorithm descriptions: detailed pseudocode for each algorithm
- Feature descriptions: what each feature measures and why it matters
- Results interpretation guide: how to understand outputs

**Example-driven documentation:**
- Tutorial: "Running your first experiment"
- Walkthrough: "Analyzing experimental results"
- Case study: "Investigating why algorithm X beats Y on metric graphs"

**Documentation formats:**
- Markdown for text documentation
- Sphinx for auto-generated API docs from docstrings
- Jupyter notebooks for interactive tutorials
- Comments in config files explaining options

Think about: Write documentation as you build, not after. Future you will thank present you.

---

## Prompt 12: Version Control and Collaboration

Set up version control and collaboration infrastructure.

**Git repository structure:**
```
tsp-research/
  .gitignore        (ignore data/, results/, models/)
  README.md
  requirements.txt
  setup.py
  src/
    graph_gen/
    algorithms/
    features/
    ml/
    pipeline/
  tests/
  experiments/
    configs/        (version controlled)
    results/        (not version controlled - too large)
  docs/
  notebooks/
```

**What to version control:**
- All source code
- Configuration files
- Test files
- Documentation
- Small example datasets for testing
- Requirements/environment specs

**What NOT to version control:**
- Large generated datasets (use git-lfs or external storage)
- Experimental results (store separately, link in commit messages)
- Temporary files, caches
- Virtual environments

**Commit practices:**
- Descriptive commit messages: "Add MST-based feature extractors" not "update code"
- Logical commits: one feature per commit
- Tag significant versions: v1.0-baseline, v2.0-improved-features
- Link commits to experiments: "Results in experiments/baseline_v1 generated with commit abc123"

**Collaboration features:**
- Code review: use pull requests for major changes
- Issue tracking: bugs, feature requests, research questions
- Project board: track progress on research goals
- Branch strategy: main for stable, feature branches for development

Think about: If collaborating with others, establish conventions early. Code style, branch naming, commit message format, etc.

---

## Success Criteria

You've succeeded when:
- You can run a complete experiment from scratch with a single command
- A colleague can reproduce your results using just config files and code
- Pipeline logs clearly show progress and any errors
- Experiment takes <1 hour for 100 graphs with 5 algorithms
- Automated reports generated at experiment completion provide key insights
- You can compare results across 10 different experiments easily

## What NOT to Do

- Don't build a complex workflow system (Airflow, Luigi) unless truly needed
- Don't over-engineer: simple scripts beat complex frameworks for research
- Don't ignore reproducibility for speed: trustworthy results matter more than fast results
- Don't version control large data files (they bloat git repos)
- Don't skip documentation thinking you'll remember how it works

## Next Steps After Completion

With solid pipeline integration:
1. Run systematic experiments varying one parameter at a time
2. Build a research workflow: hypothesis → experiment → analysis → insight → next hypothesis
3. Prepare for publication: reproducible results, clear documentation, shareable code
4. Extend pipeline for new research directions: new graph types, new algorithms, new features
5. Open-source the pipeline: others can build on your work

Good infrastructure is invisible when it works but catastrophic when it fails. Invest time here - it's the foundation of trustworthy research.
