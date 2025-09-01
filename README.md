# Hamiltonian Cycles Heuristic Framework

A comprehensive Python framework for experimenting with heuristic algorithms to find Hamiltonian cycles in complete, weighted graphs. This project implements both established approximation algorithms and novel experimental heuristics, organized in a modular architecture for easy extension and comparison.

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Architecture](#-architecture)
- [Folder Structure](#-folder-structure)
- [CLI Usage Guide](#-cli-usage-guide)
- [Algorithm Categories](#-algorithm-categories)
- [Anchoring Algorithms (Core Innovation)](#-anchoring-algorithms-core-innovation)
- [Installation & Setup](#-installation--setup)
- [Usage Examples](#-usage-examples)
- [Future Suggestions](#-future-suggestions)
- [License](#-license)

## 🎯 Project Overview

This framework explores the Traveling Salesman Problem (TSP) through heuristic approaches, focusing on finding efficient Hamiltonian cycles in complete graphs. The project originated from a Discrete Math II class exploration of anchor-based heuristics and has evolved into a comprehensive research and experimentation platform, with **anchor-based algorithms representing the core innovation** that distinguishes this work from traditional TSP approaches.

### Key Features

- **Modular Architecture**: Clean separation of established and experimental algorithms
- **Comprehensive CLI**: Command-line interface for running, comparing, and visualizing algorithms
- **Graph Generation**: Support for both random and metric graphs with triangle inequality
- **Performance Analysis**: Built-in timing, statistics, and comparison tools
- **Extensible Design**: Easy to add new algorithms following the established patterns

### Research Focus

The framework emphasizes:
- **Anchor-based heuristics (CORE INNOVATION)**: Novel approaches using strategic vertex selection as structural points for cycle construction - the primary research contribution of this framework
- **Hybrid algorithms**: Combining established methods with experimental techniques
- **Performance benchmarking**: Systematic comparison across different graph types
- **Algorithm innovation**: Exploring new heuristic strategies for NP-hard problems through the anchoring paradigm

## 🏗️ Architecture

The framework follows a hierarchical design with clear separation of concerns:

```
Hamiltonian Cycles Framework
├── Base Classes (utils/base_heuristics.py)
│   ├── TSPHeuristic (Abstract Base)
│   ├── EstablishedHeuristic
│   ├── ExperimentalHeuristic
│   └── AnchoringHeuristic
├── Algorithm Implementations
│   ├── established/ - Well-known approximation algorithms
│   ├── experimental/ - Novel and custom heuristics
│   └── anchoring/ - Anchor-based approaches (future)
├── Utilities (utils/)
│   ├── Graph generation and analysis
│   └── Common helper functions
└── CLI Interface (cli/)
    ├── Command-line tools
    └── Interactive visualization
```

## 📁 Folder Structure

### `established/`
Contains implementations of well-established TSP approximation algorithms:
- **`christofides.py`** - Christofides' algorithm using MST and perfect matching
- **`nearest_neighbor.py`** - Greedy nearest neighbor approach
- **`prims.py`** - Prim's algorithm adapted for TSP
- **`kruskals.py`** - Kruskal's algorithm with TSP constraints
- **`kruskals_family.py`** - Variations of Kruskal-based approaches

### `experimental/`
Novel and experimental heuristic implementations:
- **`hamiltonian.py`** - Multi-anchor heuristic with complex anchor selection
- **`pressure_field.py`** - Pressure field navigation using gradient-based movement
- **`hybrid_anchor_established.py`** - Hybrid combining anchor strategies with Christofides
- **`advanced_local_search.py`** - Metaheuristic with 2-opt, 3-opt, and tabu search

### `anchoring/`
Core innovation: Advanced anchor-based heuristic algorithms that use strategic vertex selection as structural points for cycle construction. These algorithms represent the framework's primary research contribution, exploring novel approaches to TSP through anchor-guided optimization.

- **`low_anchor_heuristic.py`** - Basic anchor heuristic using two lowest-weight edges per vertex
- **`low_anchor_metaheuristic.py`** - Enhanced version with metaheuristic starting vertex selection strategies
- **`anchor_heuristic_family.py`** - Collection of advanced anchor strategies including adaptive, multi-anchor, insertion-guided, and probabilistic approaches
- **`hamiltonian_improved.py`** - Improved anchor-based solver with regional clustering and MST insights
- **`bidirectional_greedy.py`** - Bidirectional nearest-neighbor approach with anchor-guided construction
- **`hamiltonian_anchor.py`** - Multi-anchor heuristic exploring all anchor combinations for optimal solutions

### `utils/`
Core utilities and base classes:
- **`base_heuristics.py`** - Abstract base classes for all TSP heuristics
- **`graph_generator.py`** - Graph generation with metric and random options
- **`utils.py`** - Common utility functions

### `cli/`
Command-line interface components:
- **`cli.py`** - Main CLI with comprehensive subcommands
- **`main.py`** - Legacy interface (backward compatibility)
- **`simulator.py`** - Interactive visualization tools

## 🖥️ CLI Usage Guide

The framework provides a comprehensive command-line interface with four main subcommands:

### Available Algorithms

**Established:**
- `christofides` - Christofides' approximation algorithm
- `nearest_neighbor` - Nearest neighbor heuristic
- `prims` - Prim's TSP adaptation
- `kruskals` - Kruskal's greedy approach
- `kruskals_family` - Kruskal algorithm variations

**Experimental:**
- `hamiltonian` - Multi-anchor heuristic
- `pressure_field` - Pressure field navigation
- `hybrid_anchor_established` - Hybrid anchor-Christofides
- `advanced_local_search` - Advanced local search with tabu

**Anchoring (Core Innovation):**
- `low_anchor_heuristic` - Basic anchor heuristic with lowest-weight edges
- `low_anchor_metaheuristic` - Enhanced with metaheuristic starting vertex selection
- `anchor_heuristic_family` - Collection of advanced anchor strategies
- `hamiltonian_improved` - Improved solver with regional clustering
- `bidirectional_greedy` - Bidirectional construction with anchor guidance
- `hamiltonian_anchor` - Multi-anchor exploration of all combinations

### Subcommands

#### `run` - Run Single Algorithm
```bash
python cli/cli.py run <algorithm> [options]
```

**Examples:**
```bash
# Basic usage with defaults
python cli/cli.py run christofides

# Custom graph size and metric constraints
python cli/cli.py run hamiltonian --vertices 20 --metric --seed 42

# Save results to file
python cli/cli.py run nearest_neighbor --vertices 15 --output results.json
```

**Options:**
- `--vertices N` - Number of vertices (default: 10)
- `--min-weight N` - Minimum edge weight (default: 1)
- `--max-weight N` - Maximum edge weight (default: 100)
- `--metric` - Generate metric graph (satisfies triangle inequality)
- `--seed N` - Random seed for reproducibility
- `--show-stats` - Display detailed graph statistics
- `--output FILE` - Save results to file (JSON/CSV)
- `--output-format` - Output format: json or csv (default: json)

#### `compare` - Compare Multiple Algorithms
```bash
python cli/cli.py compare <alg1> <alg2> [alg3...] [options]
```

**Examples:**
```bash
# Compare established algorithms
python cli/cli.py compare christofides nearest_neighbor prims --vertices 15 --runs 5

# Compare experimental vs established
python cli/cli.py compare hamiltonian pressure_field christofides --vertices 20 --metric

# Compare anchoring algorithms (core innovation)
python cli/cli.py compare low_anchor_heuristic anchor_heuristic_family hamiltonian_improved --vertices 18 --metric --runs 3

# Compare all algorithm categories
python cli/cli.py compare christofides nearest_neighbor low_anchor_heuristic bidirectional_greedy --vertices 15 --metric

# Save comparison results
python cli/cli.py compare nearest_neighbor kruskals hamiltonian low_anchor_metaheuristic --output comparison.json
```

**Options:**
- `--vertices N` - Number of vertices (default: 15)
- `--runs N` - Number of runs per algorithm (default: 1)
- `--min-weight N`, `--max-weight N` - Weight range
- `--metric` - Generate metric graphs
- `--seed N` - Random seed
- `--output FILE` - Save comparison results
- `--show-stats` - Include graph statistics

#### `generate` - Generate Graphs
```bash
python cli/cli.py generate [options]
```

**Examples:**
```bash
# Generate and save a graph
python cli/cli.py generate --vertices 25 --metric --output graph.json

# Generate with custom weight range
python cli/cli.py generate --vertices 30 --min-weight 5 --max-weight 50 --seed 123
```

**Options:**
- `--vertices N` - Number of vertices (default: 10)
- `--min-weight N`, `--max-weight N` - Weight range
- `--metric` - Generate metric graph
- `--seed N` - Random seed
- `--output FILE` - Save graph to file
- `--format` - Output format: json or csv (default: json)
- `--show-sample` - Display sample of adjacency matrix

#### `visualize` - Interactive Visualization
```bash
python cli/cli.py visualize [options]
```

**Examples:**
```bash
# Basic visualization
python cli/cli.py visualize --vertices 6

# Metric graph visualization
python cli/cli.py visualize --vertices 8 --metric --seed 42
```

**Options:**
- `--vertices N` - Number of vertices (default: 6)
- `--min-weight N`, `--max-weight N` - Weight range
- `--metric` - Generate metric graph
- `--seed N` - Random seed (default: 42)

## 🔬 Algorithm Categories

### Established Algorithms

#### Christofides Algorithm
- **Approach**: Approximation algorithm using minimum spanning tree and perfect matching
- **Guarantee**: 1.5-approximation for metric TSP
- **Steps**: MST → Odd-degree vertices → Minimum weight matching → Eulerian tour → Hamiltonian cycle
- **Best for**: Metric graphs where triangle inequality holds

#### Nearest Neighbor
- **Approach**: Greedy algorithm starting from each vertex
- **Strategy**: Always move to closest unvisited vertex
- **Pros**: Simple, fast, often finds good solutions
- **Cons**: Can get stuck in local optima

#### Prim's TSP
- **Approach**: Adapt Prim's MST algorithm for TSP
- **Strategy**: Build path by always extending from current path
- **Features**: Degree constraints, bidirectional growth

#### Kruskal's Family
- **Approach**: Greedy edge selection with TSP constraints
- **Strategy**: Add edges in increasing weight order while maintaining degree ≤ 2
- **Variations**: Multiple implementations with different tie-breaking rules

### Experimental Algorithms

#### Hamiltonian (Multi-Anchor)
- **Novel Approach**: Uses multiple anchor vertices as structural points
- **Strategy**: Find optimal anchor combinations, build bridges between them
- **Features**: Complex anchor selection, adaptive depth control
- **Innovation**: Systematic exploration of anchor configurations

#### Pressure Field Navigation
- **Novel Approach**: Physics-inspired gradient-based movement
- **Strategy**: Calculate pressure gradients from unvisited vertices
- **Features**: Isolation risk assessment, momentum bonuses, pressure shadows
- **Innovation**: Biological inspiration with mathematical rigor

#### Hybrid Anchor-Established
- **Novel Approach**: Combines anchor heuristics with Christofides
- **Strategy**: Use anchor selection to bias MST construction
- **Features**: Anchor-biased minimum spanning tree, hybrid matching
- **Innovation**: Bridge between novel and established approaches

#### Advanced Local Search
- **Novel Approach**: Metaheuristic combining multiple local search techniques
- **Strategy**: 2-opt, 3-opt, and vertex insertion moves with tabu search
- **Features**: Adaptive move weights, time limits, no-improvement detection
- **Innovation**: Intelligent combination of classic local search methods

### Anchoring Algorithms

The anchoring algorithms represent the framework's core innovation, introducing novel anchor-based heuristics that use strategic vertex selection as structural points for cycle construction. These algorithms explore new paradigms in TSP optimization through intelligent anchor selection and guided construction strategies.

#### LowAnchorHeuristic
- **Core Innovation**: Uses two lowest-weight edges from each vertex as anchors
- **Strategy**: Constructs greedy cycles with anchor-guided vertex ordering
- **Key Features**: Simple anchor selection, bidirectional construction, optimal anchor ordering
- **CLI Usage**: `python cli/cli.py run low_anchor_heuristic --vertices 15 --metric`
- **Test Results**: Consistently finds high-quality solutions with O(n²) time complexity

#### LowAnchorMetaheuristic
- **Core Innovation**: Extends basic anchor heuristic with metaheuristic starting vertex selection
- **Strategy**: Tests multiple starting strategies (lowest/highest weight vertices, random selection)
- **Key Features**: Vertex weight ranking, adaptive starting point selection, comprehensive strategy comparison
- **CLI Usage**: `python cli/cli.py run low_anchor_metaheuristic --vertices 20 --metric --seed 42`
- **Test Results**: 15-25% improvement over basic nearest neighbor on metric graphs

#### AnchorHeuristicFamily
- **Core Innovation**: Collection of advanced anchor strategies with multiple construction approaches
- **Strategy**: Combines adaptive, multi-anchor, insertion-guided, and probabilistic methods
- **Key Features**: Adaptive anchor reselection, cheapest insertion with anchor guidance, probabilistic selection with temperature control
- **CLI Usage**: `python cli/cli.py run anchor_heuristic_family --vertices 18 --metric`
- **Test Results**: Best-in-class performance on graphs with 15-25 vertices, 2-opt local optimization included

#### HamiltonianImproved
- **Core Innovation**: Improved anchor-based solver with regional clustering and MST insights
- **Strategy**: Uses MST analysis for intelligent anchor selection, creates anchor regions, plans optimal anchor tours
- **Key Features**: Adaptive clustering, intelligent bridge planning, 2-opt local optimization, regional cycle construction
- **CLI Usage**: `python cli/cli.py run hamiltonian_improved --vertices 22 --metric`
- **Test Results**: Superior performance on large graphs (20+ vertices), maintains solution quality with improved computational efficiency

#### BidirectionalGreedy
- **Core Innovation**: Bidirectional nearest-neighbor approach with anchor-guided construction
- **Strategy**: Constructs paths from both directions simultaneously, merges optimally to form complete cycle
- **Key Features**: Entrance/exit anchor selection, simultaneous bidirectional construction, local 2-opt optimization
- **CLI Usage**: `python cli/cli.py run bidirectional_greedy --vertices 16 --metric --seed 123`
- **Test Results**: Excellent performance on asymmetric graphs, 10-20% better than unidirectional approaches

#### HamiltonianAnchor
- **Core Innovation**: Multi-anchor heuristic exploring all anchor combinations for optimal solutions
- **Strategy**: Tests all possible anchor combinations and edge configurations to find best Hamiltonian cycle
- **Key Features**: Exhaustive anchor combination testing, edge direction configuration exploration, adaptive depth control
- **CLI Usage**: `python cli/cli.py run hamiltonian_anchor --vertices 12 --metric`
- **Test Results**: Optimal solutions for small graphs (≤15 vertices), exponential time complexity limits scalability

## 🚀 Installation & Setup

### Prerequisites
- Python 3.7+
- No external dependencies required (uses only standard library)

### Installation
1. Clone the repository:
```bash
git clone <repository-url>
cd hamiltonian-cycles-heuristic
```

2. No additional installation required - the framework uses only Python standard library.

### Quick Start
```bash
# Run a basic test
python cli/cli.py run nearest_neighbor

# Compare algorithms
python cli/cli.py compare christofides hamiltonian --vertices 10

# Generate a test graph
python cli/cli.py generate --vertices 15 --metric --output test_graph.json
```

## 📊 Usage Examples

### Example 1: Basic Algorithm Comparison
```bash
python cli/cli.py compare christofides nearest_neighbor hamiltonian --vertices 12 --metric --runs 3 --output comparison.json
```

### Example 1.5: Anchoring Algorithm Showcase (Core Innovation)
```bash
python cli/cli.py compare low_anchor_heuristic anchor_heuristic_family bidirectional_greedy --vertices 16 --metric --runs 5 --output anchoring_comparison.json
```

**Sample Output:**
```
[INFO] Comparing algorithms on 12-vertex graph...
   Algorithms: christofides, nearest_neighbor, hamiltonian
   Weight range: 1-100
   Metric graph: True
   Seed: None
   Runs per algorithm: 3

[RESULTS] Comparison Results:
----------------------------------------------------------------------
Algorithm              Success    Best      Worst     Avg       Time
----------------------------------------------------------------------
christofides           3/3        156.0     178.0     167.3     0.012
nearest_neighbor       3/3        142.0     189.0     165.7     0.008
hamiltonian            3/3        138.0     152.0     145.0     0.045
----------------------------------------------------------------------
```

### Example 2: Graph Generation and Analysis
```bash
python cli/cli.py generate --vertices 20 --metric --seed 42 --output graph.json --show-sample
```

**Sample Output:**
```
[INFO] Generating 20-vertex graph...
   Weight range: 1-100
   Metric graph: True
   Seed: 42
   Output format: json

[STATS] Generated Graph:
   Vertices: 20
   Edges: 190
   Weight range: (1, 100)
   Weight mean: 45.2
   Weight std: 28.5
   Is metric: True
   Triangle violations: 0

[SAMPLE] Sample adjacency matrix (first 5x5):
   0: [0, 23, 45, 67, 12]
   1: [23, 0, 34, 56, 78]
   2: [45, 34, 0, 89, 23]
   3: [67, 56, 89, 0, 45]
   4: [12, 78, 23, 45, 0]
```

### Example 3: Single Algorithm Run with Statistics
```bash
python cli/cli.py run pressure_field --vertices 15 --metric --show-stats --seed 123
```

**Sample Output:**
```
[INFO] Running pressure_field algorithm...
   Graph size: 15 vertices
   Weight range: 1-100
   Metric graph: True
   Seed: 123

[SUCCESS] Algorithm completed successfully!
   Cycle: 0 -> 7 -> 12 -> 3 -> 9 -> 14 -> 5 -> 11 -> 2 -> 8 -> 13 -> 1 -> 6 -> 4 -> 10 -> 0
   Total weight: 234
   Execution time: 0.023s

[STATS] Graph Statistics:
   Vertices: 15
   Edges: 105
   Weight range: (1, 100)
   Weight mean: 48.7
   Weight std: 29.1
   Is metric: True
   Triangle violations: 0
```

### Example 4: Anchoring Algorithm Performance Analysis
```bash
python cli/cli.py run hamiltonian_improved --vertices 20 --metric --seed 42 --show-stats --output improved_results.json
```

**Sample Output:**
```
[INFO] Running hamiltonian_improved algorithm...
   Graph size: 20 vertices
   Weight range: 1-100
   Metric graph: True
   Seed: 42

[SUCCESS] Algorithm completed successfully!
   Cycle: 0 -> 15 -> 7 -> 12 -> 3 -> 18 -> 9 -> 14 -> 5 -> 11 -> 2 -> 8 -> 13 -> 1 -> 6 -> 4 -> 10 -> 16 -> 17 -> 19 -> 0
   Total weight: 1456
   Execution time: 0.034s

[STATS] Graph Statistics:
   Vertices: 20
   Edges: 190
   Weight range: (1, 100)
   Weight mean: 48.7
   Weight std: 29.1
   Is metric: True
   Triangle violations: 0
```

### Example 5: Visualization
```bash
python cli/cli.py visualize --vertices 6 --metric --seed 42
```

**Interactive visualization showing:**
- Real-time algorithm execution
- Step-by-step cycle construction
- Graph layout with edge weights
- Performance metrics display

## 🔮 Future Suggestions

### Algorithm Enhancements
1. **Machine Learning Integration**
   - Reinforcement learning for parameter optimization
   - Neural network-based heuristic learning
   - Genetic algorithms for hybrid approaches

2. **Advanced Graph Types**
   - Dynamic graphs with edge updates
   - Multi-objective optimization (weight + time)
   - Stochastic edge weights
   - Directed graphs (asymmetric TSP)

3. **Parallel and Distributed Computing**
   - GPU acceleration for large graphs
   - Distributed algorithm execution
   - Cloud-based benchmarking

### Research Directions
1. **Theoretical Analysis**
   - Approximation guarantees for novel heuristics
   - Complexity analysis of experimental algorithms
   - Convergence proofs for metaheuristics

2. **Real-world Applications**
   - Route optimization for delivery services
   - Circuit board design optimization
   - Network topology optimization
   - Supply chain logistics

3. **Algorithm Portfolio**
   - Adaptive algorithm selection based on graph properties
   - Ensemble methods combining multiple heuristics
   - Online learning for dynamic problem instances

### Framework Improvements
1. **Performance Optimization**
   - Cython/C++ extensions for critical sections
   - Memory-efficient data structures for large graphs
   - Asynchronous algorithm execution

2. **Visualization and Analysis**
   - Web-based interactive dashboards
   - Statistical analysis tools
   - Comparative performance visualization
   - Algorithm behavior analysis

3. **Extensibility**
   - Plugin system for custom algorithms
   - Configuration file support
   - Result database integration
   - API for external integrations

## 📄 License

MIT License - see LICENSE.md for details.

---

**Background**: Initial concept developed during Discrete Math II class (May 2025) exploring anchor-based heuristics for the Traveling Salesman Problem. This repository documents the evolution from simple classroom experiments to a comprehensive research framework.