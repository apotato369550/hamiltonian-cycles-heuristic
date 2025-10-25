# Quick Start Guide - Graph Generation System

## Installation

```bash
pip install -r requirements.txt
```

## 30-Second Demo

```bash
cd src
python main.py
```

This runs 5 complete demonstrations of the system.

## Generate Your First Graph

```python
from src.graph_generation import generate_euclidean_graph, create_graph_instance

# Generate
matrix, coords = generate_euclidean_graph(num_vertices=20, random_seed=42)

# Create instance with verification
graph = create_graph_instance(matrix, 'euclidean', {}, 42, coords, verify=True)

# View summary
print(graph.summary())
```

## Generate Multiple Graph Types

```python
from src.graph_generation import (
    generate_euclidean_graph,
    generate_metric_graph,
    generate_random_graph,
    create_graph_instance
)

# Euclidean (always metric and symmetric)
matrix1, coords = generate_euclidean_graph(20, random_seed=1)
graph1 = create_graph_instance(matrix1, 'euclidean', {}, 1, coords)

# Metric (non-Euclidean but satisfies triangle inequality)
matrix2 = generate_metric_graph(20, random_seed=2)
graph2 = create_graph_instance(matrix2, 'metric', {}, 2)

# Random (typically non-metric)
matrix3 = generate_random_graph(20, random_seed=3)
graph3 = create_graph_instance(matrix3, 'random', {}, 3)

print(f"Graph 1 is metric: {graph1.properties.is_metric}")
print(f"Graph 2 is metric: {graph2.properties.is_metric}")
print(f"Graph 3 is metric: {graph3.properties.is_metric}")
print(f"Graph 3 metricity: {graph3.properties.metricity_score:.1%}")
```

## Save and Load

```python
from src.graph_generation import GraphStorage

storage = GraphStorage()

# Save
filepath = storage.save_graph(graph, subdirectory='my_graphs')
print(f"Saved to: {filepath}")

# Load
loaded = storage.load_graph(filepath)
print(f"Loaded: {loaded.id}")
```

## Batch Generation

### 1. Create config file: `my_config.yaml`

```yaml
batch_name: my_first_batch
output_directory: data/graphs
verification_mode: fast

graphs:
  - type: euclidean
    sizes: [20, 50]
    instances_per_size: 5
    seed_start: 1000
    parameters:
      dimensions: 2
      weight_range: [1.0, 100.0]
```

### 2. Generate

```python
from src.graph_generation import generate_batch_from_config

report = generate_batch_from_config('my_config.yaml')
print(f"Generated {report['total_generated']} graphs in {report['duration_seconds']:.1f}s")
```

## Analyze a Collection

```python
from src.graph_generation import analyze_collection

analysis = analyze_collection(batch_name='my_first_batch')

print(f"Total: {analysis['summary']['total_graphs']}")
print(f"Types: {list(analysis['coverage']['by_type_and_size'].keys())}")
print(f"Diversity: {analysis['diversity_metrics']['diversity_score']:.2f}")
```

## Visualize a Graph

```python
from src.graph_generation import visualize_graph

# Creates layout, weights, heatmap, and summary visualizations
files = visualize_graph(graph, output_dir='visualizations')

for f in files:
    print(f"Created: {f}")
```

## Run Tests

```bash
cd src
python tests/test_graph_generators.py
```

## Common Patterns

### Find Specific Graphs

```python
storage = GraphStorage()

# All Euclidean graphs
euclidean = storage.find_graphs(graph_type='euclidean')

# Graphs with 20-50 vertices
medium = storage.find_graphs(size_range=(20, 50))

# Only metric graphs
metric = storage.find_graphs(is_metric=True)

# Custom filter
large_metric = storage.find_graphs(
    custom_filter=lambda g: g.metadata.size > 100 and g.properties.is_metric
)
```

### Verify Properties

```python
from src.graph_generation import GraphVerifier, print_verification_report

verifier = GraphVerifier(fast_mode=False)  # Exhaustive verification
results = verifier.verify_all(graph.adjacency_matrix, graph.coordinates)

print_verification_report(results)
```

### Custom Distribution

```python
# Clustered Euclidean
matrix, coords = generate_euclidean_graph(
    num_vertices=30,
    distribution='clustered',
    distribution_params={
        'num_clusters': 4,
        'cluster_std': 8.0
    },
    random_seed=42
)

# Bimodal random
matrix = generate_random_graph(
    num_vertices=30,
    distribution='bimodal',
    distribution_params={
        'mode1_ratio': 0.3,
        'mode2_ratio': 0.7
    },
    random_seed=42
)
```

## File Locations

- **Generated graphs:** `data/graphs/`
- **Visualizations:** `visualizations/`
- **Config files:** `config/`
- **Source code:** `src/graph_generation/`
- **Tests:** `src/tests/`

## Common Issues

### ModuleNotFoundError
```bash
# Make sure you're in the src directory
cd src
python your_script.py

# Or add to path
export PYTHONPATH="${PYTHONPATH}:/path/to/hamiltonian-cycles-heuristic/src"
```

### Missing Dependencies
```bash
pip install numpy scipy matplotlib pyyaml
```

### Visualization Not Working
Matplotlib may require additional setup on some systems. Visualizations are optional - all core functionality works without them.

## Next Steps

1. Read [GRAPH_GENERATION_README.md](GRAPH_GENERATION_README.md) for detailed documentation
2. Explore [src/main.py](src/main.py) for complete examples
3. Customize [config/example_batch_config.yaml](config/example_batch_config.yaml) for your needs
4. Run the test suite to verify installation
5. Start generating graphs for your TSP research!

## Support

- **Full Documentation:** [GRAPH_GENERATION_README.md](GRAPH_GENERATION_README.md)
- **Implementation Details:** [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)
- **Architecture Guide:** [guides/01_graph_generation_system.md](guides/01_graph_generation_system.md)

---

**Ready to build the TSP research pipeline!** 🚀
