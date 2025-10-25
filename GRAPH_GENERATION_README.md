# Graph Generation System

A comprehensive, modular system for generating, verifying, storing, and analyzing TSP (Traveling Salesman Problem) graph instances. This system was built following the architecture outlined in the Graph Generation System guide to support TSP research with diverse, reproducible graph collections.

## Features

### 🎯 Core Capabilities

- **Multiple Graph Types**
  - Euclidean graphs (2D/3D point-based with true geometric distances)
  - Metric graphs (non-Euclidean but satisfying triangle inequality)
  - Quasi-metric graphs (asymmetric metric graphs)
  - Random graphs (baseline chaotic instances)

- **Robust Verification**
  - Property verification (symmetry, metricity, weight statistics)
  - Fast sampling mode for large graphs
  - Exhaustive verification for research publication
  - Automatic verification during generation

- **Flexible Storage**
  - JSON-based persistence with human-readable format
  - Batch operations with manifest files
  - Query system for finding graphs by properties
  - Complete reproducibility with random seeds

- **Batch Generation Pipeline**
  - YAML/JSON configuration files
  - Progress tracking and error handling
  - Automatic verification and reporting
  - Parallel-ready architecture

- **Visualization Tools**
  - Graph layout visualization (force-directed or coordinate-based)
  - Weight distribution histograms
  - Adjacency matrix heatmaps
  - Summary statistics panels

- **Collection Analysis**
  - Coverage metrics across type/size combinations
  - Diversity analysis
  - Outlier detection
  - Comprehensive reporting and visualization

## Installation

```bash
# Clone the repository
cd hamiltonian-cycles-heuristic

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Generate a Single Graph

```python
from src.graph_generation import generate_euclidean_graph, create_graph_instance

# Generate a Euclidean graph
matrix, coords = generate_euclidean_graph(
    num_vertices=20,
    dimensions=2,
    weight_range=(1.0, 100.0),
    random_seed=42
)

# Create graph instance with verification
graph = create_graph_instance(
    adjacency_matrix=matrix,
    graph_type='euclidean',
    generation_params={'dimensions': 2},
    random_seed=42,
    coordinates=coords,
    verify=True
)

# Print summary
print(graph.summary())
```

### Batch Generation

```python
from src.graph_generation import generate_batch_from_config

# Generate from configuration file
report = generate_batch_from_config('config/example_batch_config.yaml')

print(f"Generated {report['total_generated']} graphs")
```

### Analyze a Collection

```python
from src.graph_generation import analyze_collection

# Analyze all graphs in a batch
analysis = analyze_collection(
    batch_name='my_batch',
    create_visualizations=True
)

print(f"Total graphs: {analysis['summary']['total_graphs']}")
print(f"Diversity score: {analysis['diversity_metrics']['diversity_score']:.2f}")
```

## Project Structure

```
hamiltonian-cycles-heuristic/
├── src/
│   ├── graph_generation/
│   │   ├── __init__.py                 # Package exports
│   │   ├── graph_instance.py           # Core data structures
│   │   ├── euclidean_generator.py      # Euclidean graph generation
│   │   ├── metric_generator.py         # Metric graph generation
│   │   ├── random_generator.py         # Random graph generation
│   │   ├── verification.py             # Property verification
│   │   ├── storage.py                  # Graph persistence
│   │   ├── batch_generator.py          # Batch generation pipeline
│   │   ├── visualization.py            # Visualization utilities
│   │   └── collection_analysis.py      # Collection analysis
│   ├── tests/
│   │   └── test_graph_generators.py    # Comprehensive test suite
│   └── main.py                         # Demo script
├── config/
│   └── example_batch_config.yaml       # Example configuration
├── data/
│   └── graphs/                         # Generated graphs storage
├── guides/
│   └── 01_graph_generation_system.md   # System architecture guide
├── requirements.txt                    # Python dependencies
└── GRAPH_GENERATION_README.md          # This file
```

## Graph Types

### Euclidean Graphs

Vertices are points in 2D or 3D space, edge weights are geometric distances.

```python
from src.graph_generation import generate_euclidean_graph

# Uniform distribution
matrix, coords = generate_euclidean_graph(
    num_vertices=50,
    dimensions=2,
    distribution='uniform',
    random_seed=42
)

# Clustered distribution
matrix, coords = generate_euclidean_graph(
    num_vertices=50,
    distribution='clustered',
    distribution_params={'num_clusters': 3, 'cluster_std': 5.0},
    random_seed=42
)

# Grid distribution
matrix, coords = generate_euclidean_graph(
    num_vertices=49,
    distribution='grid',
    random_seed=42
)
```

**Properties:** Always symmetric and metric

### Metric Graphs

Non-Euclidean graphs that satisfy the triangle inequality.

```python
from src.graph_generation import generate_metric_graph

# MST-based strategy (faster)
matrix = generate_metric_graph(
    num_vertices=50,
    weight_range=(1.0, 100.0),
    strategy='mst',
    metric_strictness=1.0,
    random_seed=42
)

# Completion-based strategy (more random-looking)
matrix = generate_metric_graph(
    num_vertices=50,
    strategy='completion',
    random_seed=42
)
```

**Properties:** Symmetric (by default), metric

### Quasi-Metric Graphs

Asymmetric graphs that still satisfy triangle inequality.

```python
from src.graph_generation import generate_quasi_metric_graph

matrix = generate_quasi_metric_graph(
    num_vertices=50,
    asymmetry_factor=0.2,  # Degree of asymmetry
    random_seed=42
)
```

**Properties:** Asymmetric, metric

### Random Graphs

Fully random edge weights without structural constraints.

```python
from src.graph_generation import generate_random_graph

# Uniform distribution
matrix = generate_random_graph(
    num_vertices=50,
    distribution='uniform',
    weight_range=(1.0, 100.0),
    random_seed=42
)

# Normal distribution
matrix = generate_random_graph(
    num_vertices=50,
    distribution='normal',
    distribution_params={'mean_ratio': 0.5, 'std_ratio': 0.2},
    random_seed=42
)

# Bimodal distribution
matrix = generate_random_graph(
    num_vertices=50,
    distribution='bimodal',
    random_seed=42
)
```

**Properties:** Almost always non-metric

## Batch Configuration

Create a YAML configuration file for batch generation:

```yaml
batch_name: my_experiment
output_directory: data/graphs
verification_mode: fast  # 'fast', 'full', or 'none'
continue_on_error: true

graphs:
  - type: euclidean
    sizes: [20, 50, 100]
    instances_per_size: 10
    seed_start: 1000
    parameters:
      dimensions: 2
      weight_range: [1.0, 100.0]
      distribution: uniform

  - type: metric
    sizes: [20, 50, 100]
    instances_per_size: 10
    seed_start: 2000
    parameters:
      weight_range: [1.0, 100.0]
      strategy: mst

  - type: random
    sizes: [20, 50, 100]
    instances_per_size: 10
    seed_start: 3000
    parameters:
      distribution: uniform
      is_symmetric: true
```

Then generate:

```python
from src.graph_generation import generate_batch_from_config

report = generate_batch_from_config('config/my_experiment.yaml')
```

## Verification

The system provides comprehensive property verification:

```python
from src.graph_generation import GraphVerifier, print_verification_report

verifier = GraphVerifier(fast_mode=False)  # Use exhaustive checking
results = verifier.verify_all(
    adjacency_matrix=matrix,
    coordinates=coords  # Optional
)

print_verification_report(results)
```

**Verified Properties:**
- Symmetry: weight(i,j) == weight(j,i)
- Metricity: Triangle inequality for all vertex triplets
- Weight statistics: Valid ranges, no NaN/Inf values
- Euclidean distances: Computed distances match weights (if coordinates provided)

## Storage and Retrieval

```python
from src.graph_generation import GraphStorage

storage = GraphStorage(base_directory='data/graphs')

# Save a graph
filepath = storage.save_graph(graph, subdirectory='experiment_1')

# Load a graph
graph = storage.load_graph(filepath, verify=True)

# Save a batch
manifest = storage.save_batch(graphs, batch_name='my_batch')

# Load a batch
graphs = storage.load_batch('my_batch')

# Query graphs
euclidean_graphs = storage.find_graphs(graph_type='euclidean')
medium_graphs = storage.find_graphs(size_range=(20, 50))
metric_graphs = storage.find_graphs(is_metric=True)

# Get storage stats
stats = storage.get_storage_stats()
```

## Visualization

```python
from src.graph_generation import visualize_graph

# Create all visualizations
viz_files = visualize_graph(
    graph,
    output_dir='visualizations',
    create_all=True
)

# Or use the visualizer directly
from src.graph_generation import GraphVisualizer

visualizer = GraphVisualizer(output_dir='visualizations')

# Individual visualizations
visualizer.visualize_graph_layout(graph, save_as='layout.png')
visualizer.visualize_weight_distribution(graph, save_as='weights.png')
visualizer.visualize_adjacency_heatmap(graph, save_as='heatmap.png')
visualizer.visualize_summary_stats(graph, save_as='summary.png')
```

## Collection Analysis

```python
from src.graph_generation import CollectionAnalyzer, GraphStorage

storage = GraphStorage()
analyzer = CollectionAnalyzer(storage)

# Load and analyze
graphs = storage.load_batch('my_batch')
analysis = analyzer.analyze_collection(graphs=graphs)

# View results
print(f"Total graphs: {analysis['summary']['total_graphs']}")
print(f"Coverage: {analysis['coverage']}")
print(f"Diversity score: {analysis['diversity_metrics']['diversity_score']}")

# Create visualizations
viz_file = analyzer.visualize_collection(graphs, output_file='analysis.png')
report_file = analyzer.export_analysis_report(graphs, output_file='report.json')
```

## Running Tests

```bash
# Run all tests
cd src
python tests/test_graph_generators.py

# Or with pytest
pytest tests/test_graph_generators.py -v

# With coverage
pytest tests/test_graph_generators.py --cov=graph_generation
```

## Demo

Run the complete demonstration:

```bash
cd src
python main.py
```

This will demonstrate:
1. Single graph generation
2. Multiple graph types
3. Batch generation
4. Collection analysis
5. Storage and querying

## Performance

**Generation Speed** (approximate, on modern hardware):
- Euclidean: ~0.1s for 100 vertices, ~0.5s for 500 vertices
- Metric: ~0.2s for 100 vertices, ~1s for 500 vertices
- Random: ~0.05s for 100 vertices, ~0.2s for 500 vertices

**Verification Speed:**
- Fast mode (sampling): ~0.1s for 100 vertices
- Full mode (exhaustive): ~1s for 100 vertices, ~20s for 500 vertices

## Success Criteria

According to the guide, the system succeeds when:

✅ Can generate 100 graphs across 5 types and 4 sizes in under a minute
✅ Every generated graph passes verification for its claimed properties
✅ Can save and reload graphs with perfect fidelity
✅ Test suite catches intentional bugs
✅ A colleague can generate identical graphs using config file and seeds
✅ Can visualize any small graph and understand its structure immediately

## Future Extensions

This graph generation system serves as the foundation for:
- Algorithm benchmarking pipeline (Guide 02)
- Feature engineering system (Guide 03)
- Machine learning component (Guide 04)
- Pipeline integration (Guide 05)
- Analysis and visualization (Guide 06)

## Contributing

When adding new graph types:
1. Create generator in appropriate module
2. Add verification logic if needed
3. Add tests to test suite
4. Update documentation
5. Add example to configuration

## License

See LICENSE.md in the root directory.

## Citation

If you use this graph generation system in your research, please cite:

```bibtex
@software{tsp_graph_generation,
  title = {TSP Graph Generation System},
  year = {2024},
  author = {Your Name},
  note = {Part of Hamiltonian Cycles Heuristic Research Pipeline}
}
```

## Contact

For questions, issues, or contributions, please open an issue on the repository.

---

**Built with clean code principles for months of research.**
