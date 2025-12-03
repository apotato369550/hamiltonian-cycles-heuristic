# End-to-End Integration Plan: Complete Research Pipeline
**Date**: December 3, 2025
**Purpose**: Integrate all phases (graph generation → algorithms → features → ML → analysis) into a reproducible research workflow
**Target**: Single-command execution with comprehensive analysis, visualization, and reporting

---

## Executive Summary

### What We Have
- ✅ **Phase 1-4**: Complete implementations (graph generation, algorithms, features, ML)
- ✅ **Phase 5 (67%)**: Pipeline infrastructure (orchestration, config, tracking, reproducibility)
- ✅ **375 tests passing** for Phases 1-4 and Phase 5 Prompts 1-4

### What We Need
1. **Stage Factory Functions** - Connect phase components to pipeline orchestrator
2. **Benchmark Storage Layer** - Persistent algorithm results storage
3. **Unified Configuration** - Single YAML for complete experiments
4. **Analysis & Reporting** - Automated result analysis and visualization
5. **End-to-End Examples** - Jupyter notebook + Python script demonstrating workflow

### Deliverables
1. **`src/pipeline/stages.py`** - Stage factory implementations
2. **`src/algorithms/storage.py`** - Benchmark results storage
3. **`config/complete_experiment_template.yaml`** - Unified experiment config
4. **`notebooks/01_end_to_end_workflow.ipynb`** - Interactive research workflow
5. **`experiments/run_experiment.py`** - Command-line experiment runner
6. **`src/pipeline/analysis.py`** - Results analysis and reporting tools
7. **`src/pipeline/visualization.py`** - Publication-quality visualizations
8. **`src/pipeline/test_results_summary.py`** - **NEW**: Automated test results summarizer (success/failure rates, observations, interpretations)

---

## Architecture Overview

### Data Flow
```
[1. Graph Generation]
    ↓ (GraphStorage → data/graphs/)
[2. Algorithm Benchmarking]
    ↓ (BenchmarkStorage → results/benchmarks/)
[3. Feature Extraction]
    ↓ (FeatureDatasetPipeline → data/features/)
[4. ML Training & Prediction]
    ↓ (Model artifacts → models/)
[5. Analysis & Reporting]
    ↓ (Reports, plots → experiments/{exp_id}/reports/)
```

### Integration Points
- **Stage 1→2**: Load graphs from `GraphStorage`
- **Stage 2→3**: Pass benchmark results to feature labeling
- **Stage 3→4**: Features DataFrame → ML training
- **Stage 4→2**: ML predictions → algorithm execution with predicted anchors
- **All→5**: Load saved artifacts for analysis

---

## Part 1: Stage Factory Functions

### File: `src/pipeline/stages.py`

#### Purpose
Bridge between pipeline orchestrator and phase-specific implementations.

#### Implementation Template

```python
"""
Pipeline stage factory functions.

Each factory creates a PipelineStage that wraps phase-specific logic
into the unified pipeline interface.
"""

from pathlib import Path
from typing import Dict, Any, List
import numpy as np

from .orchestrator import PipelineStage, StageResult
from .reproducibility import ReproducibilityManager
from graph_generation import BatchGenerator, GraphStorage
from algorithms import AlgorithmRegistry
from features import (
    FeatureExtractorPipeline,
    WeightBasedExtractor,
    TopologicalExtractor,
    MSTBasedExtractor,
    AnchorQualityLabeler
)
from ml import DatasetPreparator, LinearRegressionModel


def create_graph_generation_stage(
    config: Dict[str, Any],
    repro_manager: ReproducibilityManager,
    output_dir: Path
) -> PipelineStage:
    """
    Create graph generation pipeline stage.

    Config structure:
        graph_generation:
            batch_name: "exp_batch_001"
            types:
                - type: "euclidean"
                  sizes: [20, 50, 100]
                  instances_per_size: 10
                - type: "metric"
                  sizes: [50, 100]
                  instances_per_size: 5

    Outputs:
        - graphs: List of GraphInstance objects
        - graph_paths: List of file paths to saved graphs
        - batch_manifest: Path to batch manifest file
    """
    def execute(inputs: Dict[str, Any]) -> StageResult:
        # Extract config
        gen_config = config.get('graph_generation', {})
        batch_name = gen_config.get('batch_name', 'default_batch')

        # Set seed for reproducibility
        seed = repro_manager.propagate_seed('graph_generation')

        # Initialize storage
        storage = GraphStorage(str(output_dir / 'graphs'))

        # Generate graphs per type
        all_graphs = []
        all_paths = []

        for graph_spec in gen_config.get('types', []):
            graph_type = graph_spec['type']

            for size in graph_spec['sizes']:
                for instance in range(graph_spec['instances_per_size']):
                    # Create generator with seeded RNG
                    if graph_type == 'euclidean':
                        from graph_generation import EuclideanGraphGenerator
                        generator = EuclideanGraphGenerator(
                            seed=seed + instance,
                            dimension=graph_spec.get('dimension', 2)
                        )
                    elif graph_type == 'metric':
                        from graph_generation import MetricGraphGenerator
                        generator = MetricGraphGenerator(
                            seed=seed + instance,
                            strategy=graph_spec.get('strategy', 'completion')
                        )
                    # ... handle other types

                    # Generate graph
                    graph = generator.generate(
                        n_vertices=size,
                        weight_range=tuple(graph_spec.get('weight_range', [1.0, 100.0]))
                    )

                    # Save graph
                    path = storage.save_graph(graph, batch_name=batch_name)
                    all_graphs.append(graph)
                    all_paths.append(str(path))

        # Save batch manifest
        manifest_path = storage.save_batch_manifest(batch_name, all_paths)

        return StageResult(
            success=True,
            outputs={
                'graphs': all_graphs,
                'graph_paths': all_paths,
                'batch_manifest': str(manifest_path),
                'num_graphs': len(all_graphs)
            },
            metadata={
                'batch_name': batch_name,
                'graph_types': [spec['type'] for spec in gen_config['types']],
                'total_graphs': len(all_graphs)
            }
        )

    return PipelineStage(
        name='graph_generation',
        execute_fn=execute,
        required_inputs=[],  # No upstream dependencies
        expected_outputs=['graphs', 'graph_paths', 'batch_manifest']
    )


def create_benchmarking_stage(
    config: Dict[str, Any],
    repro_manager: ReproducibilityManager,
    output_dir: Path
) -> PipelineStage:
    """
    Create algorithm benchmarking pipeline stage.

    Config structure:
        benchmarking:
            algorithms:
                - name: "nearest_neighbor"
                  params: {}
                - name: "single_anchor"
                  params: {bidirectional: true}
            exhaustive_anchors: true  # Test all anchors for labeling
            timeout_seconds: 300

    Inputs (from graph_generation):
        - graphs: List[GraphInstance]
        - graph_paths: List[str]

    Outputs:
        - benchmark_results: List[BenchmarkResult]
        - results_db_path: Path to saved results database
    """
    def execute(inputs: Dict[str, Any]) -> StageResult:
        from algorithms.storage import BenchmarkStorage  # NEW MODULE

        # Extract inputs
        graphs = inputs.get('graphs', [])
        if not graphs and 'graph_paths' in inputs:
            # Load graphs from paths
            storage = GraphStorage(str(output_dir / 'graphs'))
            graphs = [storage.load_graph(Path(p)) for p in inputs['graph_paths']]

        # Extract config
        bench_config = config.get('benchmarking', {})
        algo_specs = bench_config.get('algorithms', [])
        exhaustive_anchors = bench_config.get('exhaustive_anchors', False)
        timeout = bench_config.get('timeout_seconds', 300)

        # Initialize storage
        bench_storage = BenchmarkStorage(str(output_dir / 'benchmarks'))

        # Set seed for reproducibility
        seed = repro_manager.propagate_seed('benchmarking')

        all_results = []

        # Benchmark each algorithm on each graph
        for graph_idx, graph in enumerate(graphs):
            graph_id = f"graph_{graph_idx:04d}"

            for algo_spec in algo_specs:
                algo_name = algo_spec['name']
                algo_params = algo_spec.get('params', {})

                # Get algorithm from registry
                algo = AlgorithmRegistry.get_algorithm(
                    algo_name,
                    random_seed=seed,
                    **algo_params
                )

                if exhaustive_anchors:
                    # Test all possible anchors for labeling
                    for anchor_vertex in range(graph.n_vertices):
                        result = algo.solve(
                            graph.adjacency_matrix,
                            anchor_vertex=anchor_vertex if 'anchor' in algo_name else None
                        )

                        bench_result = {
                            'graph_id': graph_id,
                            'graph_type': graph.metadata.get('type', 'unknown'),
                            'graph_size': graph.n_vertices,
                            'algorithm': algo_name,
                            'anchor_vertex': anchor_vertex if 'anchor' in algo_name else None,
                            'tour_weight': result.weight,
                            'runtime': result.metadata.get('runtime', 0.0),
                            'tour': result.tour
                        }
                        all_results.append(bench_result)
                        bench_storage.save_result(bench_result)
                else:
                    # Single run per algorithm
                    result = algo.solve(graph.adjacency_matrix)

                    bench_result = {
                        'graph_id': graph_id,
                        'graph_type': graph.metadata.get('type', 'unknown'),
                        'graph_size': graph.n_vertices,
                        'algorithm': algo_name,
                        'tour_weight': result.weight,
                        'runtime': result.metadata.get('runtime', 0.0),
                        'tour': result.tour
                    }
                    all_results.append(bench_result)
                    bench_storage.save_result(bench_result)

        # Save complete results database
        results_db_path = bench_storage.save_database()

        return StageResult(
            success=True,
            outputs={
                'benchmark_results': all_results,
                'results_db_path': str(results_db_path),
                'num_results': len(all_results)
            },
            metadata={
                'algorithms': [spec['name'] for spec in algo_specs],
                'num_graphs': len(graphs),
                'exhaustive_anchors': exhaustive_anchors
            }
        )

    return PipelineStage(
        name='benchmarking',
        execute_fn=execute,
        required_inputs=['graphs'],
        expected_outputs=['benchmark_results', 'results_db_path']
    )


def create_feature_extraction_stage(
    config: Dict[str, Any],
    repro_manager: ReproducibilityManager,
    output_dir: Path
) -> PipelineStage:
    """
    Create feature extraction pipeline stage.

    Config structure:
        feature_extraction:
            extractors:
                - weight_based
                - topological
                - mst_based
            labeling_strategy: "rank_based"
            output_format: "csv"  # or "pickle"

    Inputs:
        - graphs: List[GraphInstance]
        - benchmark_results: List[BenchmarkResult] (for labeling)

    Outputs:
        - feature_dataset_path: Path to saved features
        - feature_names: List of feature names
        - num_features: int
    """
    def execute(inputs: Dict[str, Any]) -> StageResult:
        # Extract inputs
        graphs = inputs['graphs']
        benchmark_results = inputs.get('benchmark_results', [])

        # Extract config
        feat_config = config.get('feature_extraction', {})
        extractor_names = feat_config.get('extractors', ['weight_based'])
        labeling_strategy = feat_config.get('labeling_strategy', 'rank_based')
        output_format = feat_config.get('output_format', 'csv')

        # Build feature extractor pipeline
        pipeline = FeatureExtractorPipeline()

        if 'weight_based' in extractor_names:
            pipeline.add_extractor(WeightBasedExtractor())
        if 'topological' in extractor_names:
            pipeline.add_extractor(TopologicalExtractor())
        if 'mst_based' in extractor_names:
            pipeline.add_extractor(MSTBasedExtractor())
        # ... add others as specified

        # Extract features for all graphs
        all_features = []
        all_labels = []
        all_metadata = []

        for graph_idx, graph in enumerate(graphs):
            graph_id = f"graph_{graph_idx:04d}"

            # Extract features
            features, feature_names = pipeline.extract_features(graph.adjacency_matrix)

            # Get labels from benchmark results
            graph_bench_results = [
                r for r in benchmark_results
                if r['graph_id'] == graph_id and 'anchor' in r['algorithm']
            ]

            if graph_bench_results:
                anchor_weights = [r['tour_weight'] for r in graph_bench_results]
                labeler = AnchorQualityLabeler(strategy=labeling_strategy)
                labels = labeler.label_from_weights(anchor_weights)
            else:
                labels = np.zeros(graph.n_vertices)  # No labels available

            # Store per-vertex features
            for vertex_idx in range(graph.n_vertices):
                all_features.append(features[vertex_idx])
                all_labels.append(labels[vertex_idx])
                all_metadata.append({
                    'graph_id': graph_id,
                    'vertex_id': vertex_idx,
                    'graph_type': graph.metadata.get('type', 'unknown'),
                    'graph_size': graph.n_vertices
                })

        # Save features
        features_array = np.array(all_features)
        labels_array = np.array(all_labels)

        output_path = output_dir / 'features' / 'feature_dataset'
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_format == 'csv':
            import pandas as pd
            df = pd.DataFrame(features_array, columns=feature_names)
            df['label'] = labels_array
            for key in all_metadata[0].keys():
                df[key] = [m[key] for m in all_metadata]
            df.to_csv(f"{output_path}.csv", index=False)
            final_path = f"{output_path}.csv"
        else:  # pickle
            import pickle
            data = {
                'features': features_array,
                'labels': labels_array,
                'feature_names': feature_names,
                'metadata': all_metadata
            }
            with open(f"{output_path}.pkl", 'wb') as f:
                pickle.dump(data, f)
            final_path = f"{output_path}.pkl"

        return StageResult(
            success=True,
            outputs={
                'feature_dataset_path': final_path,
                'feature_names': feature_names,
                'num_features': len(feature_names),
                'num_vertices': len(all_features)
            },
            metadata={
                'extractors': extractor_names,
                'labeling_strategy': labeling_strategy,
                'output_format': output_format
            }
        )

    return PipelineStage(
        name='feature_extraction',
        execute_fn=execute,
        required_inputs=['graphs', 'benchmark_results'],
        expected_outputs=['feature_dataset_path', 'feature_names']
    )


def create_training_stage(
    config: Dict[str, Any],
    repro_manager: ReproducibilityManager,
    output_dir: Path
) -> PipelineStage:
    """
    Create ML training pipeline stage.

    Config structure:
        training:
            models:
                - type: "linear_ridge"
                  alpha: 1.0
                - type: "random_forest"
                  n_estimators: 100
            test_split: 0.2
            cv_folds: 5

    Inputs:
        - feature_dataset_path: Path to feature dataset

    Outputs:
        - trained_models: List of model info dicts
        - model_paths: List of paths to saved models
        - evaluation_results: Performance metrics
    """
    def execute(inputs: Dict[str, Any]) -> StageResult:
        import pickle
        import pandas as pd
        from ml import (
            DatasetPreparator, MLProblemType,
            LinearRegressionModel, ModelType,
            TreeBasedModel,
            ModelEvaluator, RegressionMetric
        )

        # Load features
        feature_path = inputs['feature_dataset_path']
        if feature_path.endswith('.csv'):
            df = pd.read_csv(feature_path)
            X = df[inputs['feature_names']].values
            y = df['label'].values
        else:  # pickle
            with open(feature_path, 'rb') as f:
                data = pickle.load(f)
            X = data['features']
            y = data['labels']

        # Extract config
        train_config = config.get('training', {})
        model_specs = train_config.get('models', [])
        test_split = train_config.get('test_split', 0.2)

        # Set seed for reproducibility
        seed = repro_manager.propagate_seed('training')

        # Prepare dataset
        prep = DatasetPreparator(problem_type=MLProblemType.REGRESSION)
        X_clean, y_clean, metadata = prep.prepare(X, y)

        # Split data
        from ml.dataset import DatasetSplitter, SplitStrategy
        splitter = DatasetSplitter(strategy=SplitStrategy.RANDOM_SPLIT)
        splits = splitter.split(X_clean, y_clean, test_size=test_split, random_state=seed)
        X_train, X_test, y_train, y_test = splits['X_train'], splits['X_test'], splits['y_train'], splits['y_test']

        # Train models
        trained_models = []
        model_paths = []
        evaluation_results = []

        for model_spec in model_specs:
            model_type_str = model_spec['type']

            # Create model
            if 'linear' in model_type_str:
                if 'ridge' in model_type_str:
                    model_type = ModelType.LINEAR_RIDGE
                elif 'lasso' in model_type_str:
                    model_type = ModelType.LINEAR_LASSO
                else:
                    model_type = ModelType.LINEAR_OLS

                model = LinearRegressionModel(
                    model_type=model_type,
                    **{k: v for k, v in model_spec.items() if k != 'type'}
                )
            elif 'forest' in model_type_str:
                model = TreeBasedModel(
                    model_type=ModelType.RANDOM_FOREST,
                    **{k: v for k, v in model_spec.items() if k != 'type'}
                )
            # ... handle other model types

            # Train
            model.fit(X_train, y_train)

            # Evaluate
            evaluator = ModelEvaluator()
            y_pred = model.predict(X_test)
            metrics = evaluator.evaluate_single(y_test, y_pred, metric_type=RegressionMetric.R2)

            # Save model
            model_path = output_dir / 'models' / f"{model_type_str}_seed{seed}.pkl"
            model_path.parent.mkdir(parents=True, exist_ok=True)
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)

            trained_models.append({
                'model_type': model_type_str,
                'performance': metrics,
                'path': str(model_path)
            })
            model_paths.append(str(model_path))
            evaluation_results.append(metrics)

        return StageResult(
            success=True,
            outputs={
                'trained_models': trained_models,
                'model_paths': model_paths,
                'evaluation_results': evaluation_results,
                'best_model_path': trained_models[0]['path']  # TODO: select best by metric
            },
            metadata={
                'num_train': len(X_train),
                'num_test': len(X_test),
                'num_models': len(trained_models)
            }
        )

    return PipelineStage(
        name='training',
        execute_fn=execute,
        required_inputs=['feature_dataset_path', 'feature_names'],
        expected_outputs=['trained_models', 'model_paths', 'best_model_path']
    )


def create_evaluation_stage(
    config: Dict[str, Any],
    repro_manager: ReproducibilityManager,
    output_dir: Path
) -> PipelineStage:
    """
    Create model evaluation pipeline stage.

    Tests: Can ML-predicted anchors produce competitive tours?

    Inputs:
        - graphs: Test graphs
        - trained_models: Trained model info
        - model_paths: Paths to saved models

    Outputs:
        - evaluation_report: Dict with comparative results
        - report_path: Path to saved report
    """
    def execute(inputs: Dict[str, Any]) -> StageResult:
        import pickle
        import pandas as pd

        # Load best model
        model_path = inputs['best_model_path']
        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        # Get test graphs
        graphs = inputs['graphs']

        # Extract features and predict best anchors
        pipeline = FeatureExtractorPipeline()
        # ... configure extractors from config ...

        results = []

        for graph in graphs:
            # Extract features
            features, _ = pipeline.extract_features(graph.adjacency_matrix)

            # Predict best anchor
            predictions = model.predict(features)
            predicted_anchor = np.argmax(predictions)

            # Run algorithm with predicted anchor
            algo = AlgorithmRegistry.get_algorithm('single_anchor')
            predicted_result = algo.solve(graph.adjacency_matrix, anchor_vertex=predicted_anchor)

            # Compare to baselines
            nn_algo = AlgorithmRegistry.get_algorithm('nearest_neighbor')
            nn_result = nn_algo.solve(graph.adjacency_matrix)

            random_anchor = np.random.randint(0, graph.n_vertices)
            random_result = algo.solve(graph.adjacency_matrix, anchor_vertex=random_anchor)

            # Record results
            results.append({
                'graph_id': graph.metadata.get('id', 'unknown'),
                'graph_type': graph.metadata.get('type', 'unknown'),
                'graph_size': graph.n_vertices,
                'predicted_anchor_tour': predicted_result.weight,
                'random_anchor_tour': random_result.weight,
                'nearest_neighbor_tour': nn_result.weight,
                'improvement_vs_random': (random_result.weight - predicted_result.weight) / random_result.weight,
                'improvement_vs_nn': (nn_result.weight - predicted_result.weight) / nn_result.weight
            })

        # Generate report
        df = pd.DataFrame(results)
        report_path = output_dir / 'reports' / 'evaluation_report.csv'
        report_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(report_path, index=False)

        # Compute summary statistics
        summary = {
            'mean_improvement_vs_random': df['improvement_vs_random'].mean(),
            'mean_improvement_vs_nn': df['improvement_vs_nn'].mean(),
            'win_rate_vs_random': (df['improvement_vs_random'] > 0).mean(),
            'win_rate_vs_nn': (df['improvement_vs_nn'] > 0).mean(),
            'num_graphs_tested': len(graphs)
        }

        return StageResult(
            success=True,
            outputs={
                'evaluation_report': summary,
                'report_path': str(report_path),
                'detailed_results': results
            },
            metadata=summary
        )

    return PipelineStage(
        name='evaluation',
        execute_fn=execute,
        required_inputs=['graphs', 'best_model_path'],
        expected_outputs=['evaluation_report', 'report_path']
    )
```

#### Key Design Decisions
- **Inputs/Outputs**: Stages communicate via dictionaries (outputs of stage N → inputs of stage N+1)
- **Storage**: Each stage saves artifacts to disk (graphs, benchmarks, features, models)
- **Reproducibility**: `repro_manager.propagate_seed()` ensures deterministic execution
- **Error Handling**: Stages return `StageResult` with success/failure status
- **Config-Driven**: All parameters controlled by experiment config

---

## Part 2: Benchmark Storage Layer

### File: `src/algorithms/storage.py`

#### Purpose
Persistent storage for algorithm benchmark results (similar to `GraphStorage`).

#### Implementation Template

```python
"""
Benchmark results storage system.

Stores algorithm performance results for later analysis and label generation.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import sqlite3


class BenchmarkStorage:
    """
    Store and retrieve benchmark results.

    Storage format: SQLite database for efficient querying

    Schema:
        benchmarks (
            id INTEGER PRIMARY KEY,
            graph_id TEXT,
            graph_type TEXT,
            graph_size INTEGER,
            algorithm TEXT,
            anchor_vertex INTEGER,
            tour_weight REAL,
            runtime REAL,
            tour TEXT,  -- JSON serialized
            metadata TEXT  -- JSON serialized
        )
    """

    def __init__(self, storage_dir: str):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.storage_dir / 'benchmarks.db'
        self._init_database()

    def _init_database(self):
        """Initialize database schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS benchmarks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                graph_id TEXT NOT NULL,
                graph_type TEXT,
                graph_size INTEGER,
                algorithm TEXT NOT NULL,
                anchor_vertex INTEGER,
                tour_weight REAL NOT NULL,
                runtime REAL,
                tour TEXT,
                metadata TEXT
            )
        """)

        # Create indices for common queries
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_graph_id ON benchmarks(graph_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_algorithm ON benchmarks(algorithm)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_graph_type ON benchmarks(graph_type)")

        conn.commit()
        conn.close()

    def save_result(self, result: Dict[str, Any]) -> int:
        """Save single benchmark result, return result ID."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO benchmarks
            (graph_id, graph_type, graph_size, algorithm, anchor_vertex,
             tour_weight, runtime, tour, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            result['graph_id'],
            result.get('graph_type'),
            result.get('graph_size'),
            result['algorithm'],
            result.get('anchor_vertex'),
            result['tour_weight'],
            result.get('runtime', 0.0),
            json.dumps(result.get('tour', [])),
            json.dumps(result.get('metadata', {}))
        ))

        result_id = cursor.lastrowid
        conn.commit()
        conn.close()

        return result_id

    def load_results(
        self,
        graph_id: Optional[str] = None,
        algorithm: Optional[str] = None,
        graph_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Load benchmark results with optional filtering."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        query = "SELECT * FROM benchmarks WHERE 1=1"
        params = []

        if graph_id:
            query += " AND graph_id = ?"
            params.append(graph_id)
        if algorithm:
            query += " AND algorithm = ?"
            params.append(algorithm)
        if graph_type:
            query += " AND graph_type = ?"
            params.append(graph_type)

        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()

        results = []
        for row in rows:
            results.append({
                'id': row[0],
                'graph_id': row[1],
                'graph_type': row[2],
                'graph_size': row[3],
                'algorithm': row[4],
                'anchor_vertex': row[5],
                'tour_weight': row[6],
                'runtime': row[7],
                'tour': json.loads(row[8]) if row[8] else [],
                'metadata': json.loads(row[9]) if row[9] else {}
            })

        return results

    def get_anchor_weights(self, graph_id: str) -> List[float]:
        """Get tour weights for all anchors on a specific graph."""
        results = self.load_results(graph_id=graph_id)
        anchor_results = [r for r in results if 'anchor' in r['algorithm']]
        anchor_results.sort(key=lambda r: r.get('anchor_vertex', 0))
        return [r['tour_weight'] for r in anchor_results]

    def save_database(self) -> Path:
        """Return path to database (already saved incrementally)."""
        return self.db_path
```

#### Key Design Decisions
- **SQLite**: Efficient querying, no external dependencies
- **Incremental Saving**: Results saved one-by-one (no need to batch)
- **Indexed Queries**: Fast lookups by graph_id, algorithm, graph_type
- **JSON Serialization**: Complex data (tours, metadata) stored as JSON

---

## Part 3: Unified Configuration

### File: `config/complete_experiment_template.yaml`

#### Purpose
Single configuration file for entire end-to-end experiment.

```yaml
# Complete TSP Research Experiment Configuration
experiment:
  name: "baseline_comparison_v1"
  description: "Compare anchor-based heuristics to nearest neighbor baseline"
  random_seed: 42
  output_dir: "experiments/baseline_v1"

# Phase 1: Graph Generation
graph_generation:
  enabled: true
  batch_name: "baseline_graphs"
  types:
    - type: "euclidean"
      sizes: [20, 50, 100]
      instances_per_size: 20
      dimension: 2
      weight_range: [1.0, 100.0]

    - type: "metric"
      sizes: [50, 100]
      instances_per_size: 10
      strategy: "completion"
      weight_range: [10.0, 50.0]

    - type: "random"
      sizes: [20, 50]
      instances_per_size: 5
      weight_range: [1.0, 100.0]

# Phase 2: Algorithm Benchmarking
benchmarking:
  enabled: true
  algorithms:
    # Baselines
    - name: "nearest_neighbor"
      params:
        strategy: "best_start"

    - name: "greedy_edge"
      params: {}

    # Anchor-based
    - name: "single_anchor"
      params:
        bidirectional: true

    - name: "best_anchor"
      params: {}

  exhaustive_anchors: true  # Test all anchors for each graph (for labeling)
  timeout_seconds: 300
  storage_format: "sqlite"

# Phase 3: Feature Extraction
feature_extraction:
  enabled: true
  extractors:
    - weight_based
    - topological
    - mst_based
    - neighborhood
    - heuristic
    - graph_context

  labeling_strategy: "rank_based"  # Options: rank_based, absolute, binary, multiclass, relative
  labeling_params:
    percentile_top: 20    # Top 20% are "good" anchors
    percentile_bottom: 20 # Bottom 20% are "bad" anchors

  output_format: "csv"  # or "pickle"

# Phase 4: Machine Learning
training:
  enabled: true
  models:
    - type: "linear_ridge"
      alpha: 1.0

    - type: "linear_lasso"
      alpha: 0.1

    - type: "random_forest"
      n_estimators: 100
      max_depth: 10
      random_state: 42

  problem_type: "regression"  # Options: regression, classification, ranking
  test_split: 0.2
  stratify_by: "graph_type"

  cross_validation:
    enabled: true
    n_folds: 5
    strategy: "stratified"  # Options: kfold, stratified, group

# Phase 5: Evaluation
evaluation:
  enabled: true
  test_algorithms:
    - name: "single_anchor"
      anchor_source: "ml_prediction"

    - name: "single_anchor"
      anchor_source: "random"

    - name: "nearest_neighbor"
      anchor_source: null

  metrics:
    - tour_weight
    - runtime
    - improvement_vs_baseline

  generate_visualizations: true

# Phase 6: Analysis and Reporting
analysis:
  enabled: true
  statistical_tests:
    - paired_t_test
    - wilcoxon

  visualizations:
    - algorithm_comparison_boxplot
    - feature_importance_barplot
    - predicted_vs_actual_scatter
    - performance_by_graph_type

  report_format: "html"  # Options: html, markdown, pdf
  publication_quality: true  # Use high-DPI, colorblind-friendly palettes

# Pipeline Configuration
pipeline:
  parallel_execution: true
  max_workers: 4
  checkpoint_frequency: "per_stage"  # Save after each stage completes
  error_handling: "continue"  # Options: fail_fast, continue, retry
  logging_level: "INFO"
```

---

## Part 4: Analysis and Reporting Tools

### File: `src/pipeline/analysis.py`

#### Purpose
Load experiment results and generate statistical analysis.

#### Key Functions

```python
"""
Results analysis tools for TSP experiments.
"""

import pandas as pd
from pathlib import Path
from typing import Dict, Any, List
from scipy import stats


class ExperimentAnalyzer:
    """Analyze results from completed experiments."""

    def __init__(self, experiment_dir: Path):
        self.experiment_dir = Path(experiment_dir)

    def load_benchmark_results(self) -> pd.DataFrame:
        """Load benchmark results from SQLite database."""
        # Load from BenchmarkStorage
        pass

    def load_evaluation_results(self) -> pd.DataFrame:
        """Load evaluation results from CSV."""
        # Load from evaluation stage output
        pass

    def compare_algorithms(
        self,
        algorithms: List[str],
        groupby: str = None
    ) -> pd.DataFrame:
        """
        Compare algorithm performance.

        Returns:
            DataFrame with mean, std, median for each algorithm
            (optionally grouped by graph_type or graph_size)
        """
        pass

    def compute_statistical_significance(
        self,
        algorithm_a: str,
        algorithm_b: str,
        test: str = "paired_t_test"
    ) -> Dict[str, float]:
        """
        Test if algorithm A significantly outperforms B.

        Returns:
            {
                'test_statistic': float,
                'p_value': float,
                'effect_size': float (Cohen's d),
                'significant': bool (p < 0.05)
            }
        """
        pass

    def analyze_feature_importance(self, model_path: Path) -> pd.DataFrame:
        """
        Extract feature importance from trained model.

        Returns:
            DataFrame with feature names and importance scores
        """
        pass

    def compute_ml_improvement(self) -> Dict[str, float]:
        """
        Compute practical improvement from ML predictions.

        Returns:
            {
                'mean_improvement_vs_random': float,
                'win_rate_vs_random': float,
                'mean_improvement_vs_baseline': float,
                'computational_speedup': float
            }
        """
        pass

    def generate_summary_report(self) -> str:
        """
        Generate markdown summary report.

        Returns:
            Markdown string with:
            - Experiment configuration
            - Algorithm comparison table
            - Statistical test results
            - ML performance summary
            - Key findings
        """
        pass
```

---

### File: `src/pipeline/visualization.py`

#### Purpose
Publication-quality visualizations of experiment results.

#### Key Functions

```python
"""
Visualization tools for TSP experiments.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path


class ExperimentVisualizer:
    """Create publication-quality visualizations."""

    def __init__(self, style: str = "publication"):
        if style == "publication":
            # High DPI, colorblind-friendly palette
            plt.rcParams['figure.dpi'] = 300
            sns.set_palette("colorblind")
            sns.set_style("whitegrid")

    def plot_algorithm_comparison(
        self,
        results_df: pd.DataFrame,
        metric: str = "tour_weight",
        output_path: Path = None
    ):
        """
        Box plot comparing algorithm performance.

        Shows distribution of tour weights for each algorithm.
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(data=results_df, x='algorithm', y=metric, ax=ax)
        ax.set_xlabel("Algorithm")
        ax.set_ylabel(f"{metric.replace('_', ' ').title()}")
        ax.set_title("Algorithm Performance Comparison")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()

    def plot_feature_importance(
        self,
        importance_df: pd.DataFrame,
        top_k: int = 15,
        output_path: Path = None
    ):
        """
        Horizontal bar chart of top-k feature importances.
        """
        top_features = importance_df.nlargest(top_k, 'importance')

        fig, ax = plt.subplots(figsize=(8, 10))
        ax.barh(top_features['feature_name'], top_features['importance'])
        ax.set_xlabel("Importance")
        ax.set_ylabel("Feature")
        ax.set_title(f"Top {top_k} Feature Importances")
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()

    def plot_predicted_vs_actual(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
        output_path: Path = None
    ):
        """
        Scatter plot: predicted vs actual anchor quality.

        Includes diagonal line for perfect prediction.
        """
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(actuals, predictions, alpha=0.5)

        # Diagonal line
        min_val = min(actuals.min(), predictions.min())
        max_val = max(actuals.max(), predictions.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')

        ax.set_xlabel("Actual Anchor Quality")
        ax.set_ylabel("Predicted Anchor Quality")
        ax.set_title("Model Prediction Accuracy")
        ax.legend()
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()

    def plot_performance_by_graph_type(
        self,
        results_df: pd.DataFrame,
        algorithms: List[str],
        output_path: Path = None
    ):
        """
        Line plot: algorithm performance across graph types.
        """
        grouped = results_df.groupby(['graph_type', 'algorithm'])['tour_weight'].mean().reset_index()

        fig, ax = plt.subplots(figsize=(10, 6))
        for algo in algorithms:
            algo_data = grouped[grouped['algorithm'] == algo]
            ax.plot(algo_data['graph_type'], algo_data['tour_weight'], marker='o', label=algo)

        ax.set_xlabel("Graph Type")
        ax.set_ylabel("Mean Tour Weight")
        ax.set_title("Performance by Graph Type")
        ax.legend()
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()

    def create_summary_figure(
        self,
        experiment_dir: Path,
        output_path: Path = None
    ):
        """
        Multi-panel figure with:
        - Algorithm comparison (top-left)
        - Feature importance (top-right)
        - Predicted vs actual (bottom-left)
        - Performance by type (bottom-right)
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Load data and create subplots
        # ... implementation ...

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
```

---

## Part 4.5: Test Results Summary Module

### File: `src/pipeline/test_results_summary.py`

#### Purpose
Automated analysis and summarization of benchmark test results across all algorithm-graph combinations. Provides statistical summaries, success/failure tracking, and initial observations for research interpretation.

#### Implementation Template

```python
"""
Test results summary and analysis module.

Analyzes benchmark results to provide:
- Success/failure rates per algorithm and graph type
- Statistical summaries of performance metrics
- Anomaly detection and outlier identification
- Initial observations and interpretations
- Data quality assessment
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum

from algorithms.storage import BenchmarkStorage


class TestStatus(Enum):
    """Test execution status."""
    SUCCESS = "success"
    FAILURE = "failure"
    TIMEOUT = "timeout"
    ERROR = "error"


@dataclass
class TestSummary:
    """Summary statistics for a test category."""
    total_tests: int
    successes: int
    failures: int
    success_rate: float
    mean_tour_weight: float
    std_tour_weight: float
    median_tour_weight: float
    mean_runtime: float
    outlier_count: int


@dataclass
class Observation:
    """Single observation or interpretation."""
    category: str  # "performance", "reliability", "anomaly", "pattern"
    severity: str  # "info", "warning", "critical"
    description: str
    affected_items: List[str]  # Graph IDs, algorithm names, etc.
    supporting_data: Dict[str, Any]


class TestResultsSummarizer:
    """
    Comprehensive test results analysis and summarization.

    Analyzes benchmark results to extract insights about:
    - Algorithm performance across graph types
    - Success/failure patterns
    - Data quality issues
    - Performance anomalies
    - Statistical patterns
    """

    def __init__(self, benchmark_storage: BenchmarkStorage):
        """
        Initialize summarizer with benchmark storage.

        Args:
            benchmark_storage: BenchmarkStorage instance containing results
        """
        self.storage = benchmark_storage
        self.results_df = None
        self.observations = []

    def load_results(self) -> pd.DataFrame:
        """Load all benchmark results into DataFrame."""
        results = self.storage.load_results()
        self.results_df = pd.DataFrame(results)
        return self.results_df

    def compute_success_rates(self) -> Dict[str, TestSummary]:
        """
        Compute success/failure rates by algorithm and graph type.

        Returns:
            Dict mapping (algorithm, graph_type) → TestSummary
        """
        if self.results_df is None:
            self.load_results()

        summaries = {}

        # Overall summary
        summaries['overall'] = self._compute_summary(self.results_df)

        # Per algorithm
        for algorithm in self.results_df['algorithm'].unique():
            algo_df = self.results_df[self.results_df['algorithm'] == algorithm]
            summaries[f"algorithm:{algorithm}"] = self._compute_summary(algo_df)

        # Per graph type
        for graph_type in self.results_df['graph_type'].unique():
            type_df = self.results_df[self.results_df['graph_type'] == graph_type]
            summaries[f"graph_type:{graph_type}"] = self._compute_summary(type_df)

        # Per algorithm-graph_type combination
        for algorithm in self.results_df['algorithm'].unique():
            for graph_type in self.results_df['graph_type'].unique():
                combo_df = self.results_df[
                    (self.results_df['algorithm'] == algorithm) &
                    (self.results_df['graph_type'] == graph_type)
                ]
                if len(combo_df) > 0:
                    key = f"{algorithm}@{graph_type}"
                    summaries[key] = self._compute_summary(combo_df)

        return summaries

    def _compute_summary(self, df: pd.DataFrame) -> TestSummary:
        """Compute summary statistics for a subset of results."""
        # Detect failures (NaN tour weights, extreme values, etc.)
        valid_mask = df['tour_weight'].notna() & (df['tour_weight'] > 0)

        total = len(df)
        successes = valid_mask.sum()
        failures = total - successes

        # Outlier detection (values > 3 standard deviations)
        if successes > 0:
            weights = df.loc[valid_mask, 'tour_weight']
            mean = weights.mean()
            std = weights.std()
            outliers = ((weights - mean).abs() > 3 * std).sum()
        else:
            weights = pd.Series([])
            outliers = 0

        return TestSummary(
            total_tests=total,
            successes=successes,
            failures=failures,
            success_rate=successes / total if total > 0 else 0.0,
            mean_tour_weight=weights.mean() if successes > 0 else np.nan,
            std_tour_weight=weights.std() if successes > 0 else np.nan,
            median_tour_weight=weights.median() if successes > 0 else np.nan,
            mean_runtime=df['runtime'].mean() if 'runtime' in df else np.nan,
            outlier_count=outliers
        )

    def identify_patterns(self) -> List[Observation]:
        """
        Identify interesting patterns in the test results.

        Looks for:
        - Algorithms that consistently beat others
        - Graph types that are particularly hard/easy
        - Anomalous results
        - Performance degradation patterns
        """
        observations = []

        if self.results_df is None:
            self.load_results()

        # Pattern 1: Algorithm dominance
        algo_performance = self.results_df.groupby('algorithm')['tour_weight'].mean()
        best_algo = algo_performance.idxmin()
        worst_algo = algo_performance.idxmax()

        if algo_performance[best_algo] < algo_performance[worst_algo] * 0.9:
            observations.append(Observation(
                category="performance",
                severity="info",
                description=f"Algorithm '{best_algo}' consistently outperforms others",
                affected_items=[best_algo],
                supporting_data={
                    'mean_weight': algo_performance[best_algo],
                    'improvement_over_worst': (algo_performance[worst_algo] - algo_performance[best_algo]) / algo_performance[worst_algo]
                }
            ))

        # Pattern 2: Graph type difficulty
        type_performance = self.results_df.groupby('graph_type')['tour_weight'].mean()
        easiest_type = type_performance.idxmin()
        hardest_type = type_performance.idxmax()

        observations.append(Observation(
            category="pattern",
            severity="info",
            description=f"Graph type '{hardest_type}' produces tours {type_performance[hardest_type]/type_performance[easiest_type]:.2f}× longer than '{easiest_type}'",
            affected_items=[hardest_type, easiest_type],
            supporting_data={
                'hardest': hardest_type,
                'easiest': easiest_type,
                'difficulty_ratio': type_performance[hardest_type] / type_performance[easiest_type]
            }
        ))

        # Pattern 3: Failures detection
        failed_mask = self.results_df['tour_weight'].isna() | (self.results_df['tour_weight'] <= 0)
        if failed_mask.any():
            failed_algos = self.results_df.loc[failed_mask, 'algorithm'].unique()
            observations.append(Observation(
                category="reliability",
                severity="warning",
                description=f"Found {failed_mask.sum()} failed tests across {len(failed_algos)} algorithms",
                affected_items=list(failed_algos),
                supporting_data={
                    'failure_count': int(failed_mask.sum()),
                    'failure_rate': float(failed_mask.mean())
                }
            ))

        # Pattern 4: Performance scaling with graph size
        size_corr = self.results_df[['graph_size', 'tour_weight']].corr().iloc[0, 1]
        if size_corr > 0.8:
            observations.append(Observation(
                category="pattern",
                severity="info",
                description=f"Strong correlation (r={size_corr:.3f}) between graph size and tour weight",
                affected_items=[],
                supporting_data={'correlation': size_corr}
            ))

        # Pattern 5: Outlier identification
        summaries = self.compute_success_rates()
        high_outlier_categories = [
            (key, summary.outlier_count)
            for key, summary in summaries.items()
            if summary.outlier_count > 0
        ]

        if high_outlier_categories:
            observations.append(Observation(
                category="anomaly",
                severity="warning",
                description=f"Detected statistical outliers in {len(high_outlier_categories)} test categories",
                affected_items=[key for key, _ in high_outlier_categories],
                supporting_data={
                    'categories': dict(high_outlier_categories)
                }
            ))

        # Pattern 6: Algorithm-graph type interactions
        interaction_matrix = self.results_df.pivot_table(
            values='tour_weight',
            index='algorithm',
            columns='graph_type',
            aggfunc='mean'
        )

        # Find best algorithm for each graph type
        best_per_type = {}
        for graph_type in interaction_matrix.columns:
            best_algo = interaction_matrix[graph_type].idxmin()
            best_per_type[graph_type] = best_algo

        if len(set(best_per_type.values())) > 1:
            observations.append(Observation(
                category="pattern",
                severity="info",
                description="Different graph types favor different algorithms",
                affected_items=list(best_per_type.keys()),
                supporting_data={
                    'best_per_type': best_per_type
                }
            ))

        self.observations = observations
        return observations

    def generate_summary_report(self, output_path: Optional[Path] = None) -> str:
        """
        Generate comprehensive summary report in Markdown format.

        Args:
            output_path: Optional path to save report

        Returns:
            Markdown-formatted report string
        """
        if self.results_df is None:
            self.load_results()

        summaries = self.compute_success_rates()
        observations = self.identify_patterns() if not self.observations else self.observations

        report_lines = []

        # Header
        report_lines.append("# Test Results Summary Report")
        report_lines.append("")
        report_lines.append(f"**Generated**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"**Total Tests**: {len(self.results_df)}")
        report_lines.append("")

        # Overall statistics
        report_lines.append("## Overall Statistics")
        report_lines.append("")
        overall = summaries['overall']
        report_lines.append(f"- **Total Tests**: {overall.total_tests}")
        report_lines.append(f"- **Successes**: {overall.successes} ({overall.success_rate:.1%})")
        report_lines.append(f"- **Failures**: {overall.failures}")
        report_lines.append(f"- **Mean Tour Weight**: {overall.mean_tour_weight:.2f} ± {overall.std_tour_weight:.2f}")
        report_lines.append(f"- **Median Tour Weight**: {overall.median_tour_weight:.2f}")
        report_lines.append(f"- **Mean Runtime**: {overall.mean_runtime:.3f}s")
        report_lines.append(f"- **Outliers Detected**: {overall.outlier_count}")
        report_lines.append("")

        # Per-algorithm summary
        report_lines.append("## Performance by Algorithm")
        report_lines.append("")
        report_lines.append("| Algorithm | Tests | Success Rate | Mean Weight | Std Dev | Median Weight | Outliers |")
        report_lines.append("|-----------|-------|--------------|-------------|---------|---------------|----------|")

        for key, summary in summaries.items():
            if key.startswith('algorithm:'):
                algo_name = key.split(':')[1]
                report_lines.append(
                    f"| {algo_name} | {summary.total_tests} | {summary.success_rate:.1%} | "
                    f"{summary.mean_tour_weight:.2f} | {summary.std_tour_weight:.2f} | "
                    f"{summary.median_tour_weight:.2f} | {summary.outlier_count} |"
                )
        report_lines.append("")

        # Per-graph-type summary
        report_lines.append("## Performance by Graph Type")
        report_lines.append("")
        report_lines.append("| Graph Type | Tests | Success Rate | Mean Weight | Std Dev | Median Weight | Outliers |")
        report_lines.append("|------------|-------|--------------|-------------|---------|---------------|----------|")

        for key, summary in summaries.items():
            if key.startswith('graph_type:'):
                type_name = key.split(':')[1]
                report_lines.append(
                    f"| {type_name} | {summary.total_tests} | {summary.success_rate:.1%} | "
                    f"{summary.mean_tour_weight:.2f} | {summary.std_tour_weight:.2f} | "
                    f"{summary.median_tour_weight:.2f} | {summary.outlier_count} |"
                )
        report_lines.append("")

        # Algorithm-GraphType interaction matrix
        report_lines.append("## Algorithm × Graph Type Performance Matrix")
        report_lines.append("")
        report_lines.append("Mean tour weights for each combination:")
        report_lines.append("")

        interaction_matrix = self.results_df.pivot_table(
            values='tour_weight',
            index='algorithm',
            columns='graph_type',
            aggfunc='mean'
        )

        # Format as markdown table
        header = "| Algorithm | " + " | ".join(interaction_matrix.columns) + " |"
        separator = "|" + "|".join(["---"] * (len(interaction_matrix.columns) + 1)) + "|"
        report_lines.append(header)
        report_lines.append(separator)

        for algo in interaction_matrix.index:
            row = f"| {algo} |"
            for graph_type in interaction_matrix.columns:
                value = interaction_matrix.loc[algo, graph_type]
                row += f" {value:.2f} |"
            report_lines.append(row)
        report_lines.append("")

        # Observations and interpretations
        report_lines.append("## Observations and Interpretations")
        report_lines.append("")

        # Group by category
        obs_by_category = defaultdict(list)
        for obs in observations:
            obs_by_category[obs.category].append(obs)

        for category in ['performance', 'reliability', 'pattern', 'anomaly']:
            if category in obs_by_category:
                report_lines.append(f"### {category.title()}")
                report_lines.append("")

                for obs in obs_by_category[category]:
                    severity_icon = {"info": "ℹ️", "warning": "⚠️", "critical": "❌"}
                    report_lines.append(f"{severity_icon.get(obs.severity, '•')} **{obs.description}**")

                    if obs.affected_items:
                        report_lines.append(f"  - Affected: {', '.join(obs.affected_items)}")

                    if obs.supporting_data:
                        report_lines.append(f"  - Data: {obs.supporting_data}")

                    report_lines.append("")

        # Data quality assessment
        report_lines.append("## Data Quality Assessment")
        report_lines.append("")

        missing_weights = self.results_df['tour_weight'].isna().sum()
        missing_runtimes = self.results_df['runtime'].isna().sum() if 'runtime' in self.results_df else 0

        report_lines.append(f"- **Missing tour weights**: {missing_weights} ({missing_weights/len(self.results_df):.1%})")
        report_lines.append(f"- **Missing runtimes**: {missing_runtimes} ({missing_runtimes/len(self.results_df):.1%})")

        # Check for duplicate tests
        duplicates = self.results_df.duplicated(subset=['graph_id', 'algorithm', 'anchor_vertex']).sum()
        report_lines.append(f"- **Duplicate tests**: {duplicates}")

        report_lines.append("")

        # Key findings summary
        report_lines.append("## Key Findings")
        report_lines.append("")

        # Best algorithm
        algo_perf = self.results_df.groupby('algorithm')['tour_weight'].mean()
        best_algo = algo_perf.idxmin()
        report_lines.append(f"1. **Best performing algorithm**: {best_algo} (mean weight: {algo_perf[best_algo]:.2f})")

        # Hardest graph type
        type_perf = self.results_df.groupby('graph_type')['tour_weight'].mean()
        hardest_type = type_perf.idxmax()
        report_lines.append(f"2. **Most challenging graph type**: {hardest_type} (mean weight: {type_perf[hardest_type]:.2f})")

        # Reliability
        success_rate = summaries['overall'].success_rate
        report_lines.append(f"3. **Overall reliability**: {success_rate:.1%} success rate")

        # Variance
        report_lines.append(f"4. **Result consistency**: Std dev = {summaries['overall'].std_tour_weight:.2f} (CV = {summaries['overall'].std_tour_weight/summaries['overall'].mean_tour_weight:.1%})")

        report_lines.append("")

        # Compile report
        report_text = "\n".join(report_lines)

        # Save if path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(report_text)

        return report_text

    def export_data_summary(self, output_path: Path):
        """
        Export structured data summary as JSON for programmatic access.

        Args:
            output_path: Path to save JSON file
        """
        import json

        summaries = self.compute_success_rates()
        observations = self.identify_patterns() if not self.observations else self.observations

        # Convert to serializable format
        summary_dict = {}
        for key, summary in summaries.items():
            summary_dict[key] = {
                'total_tests': summary.total_tests,
                'successes': summary.successes,
                'failures': summary.failures,
                'success_rate': summary.success_rate,
                'mean_tour_weight': float(summary.mean_tour_weight) if not np.isnan(summary.mean_tour_weight) else None,
                'std_tour_weight': float(summary.std_tour_weight) if not np.isnan(summary.std_tour_weight) else None,
                'median_tour_weight': float(summary.median_tour_weight) if not np.isnan(summary.median_tour_weight) else None,
                'mean_runtime': float(summary.mean_runtime) if not np.isnan(summary.mean_runtime) else None,
                'outlier_count': summary.outlier_count
            }

        observations_dict = [
            {
                'category': obs.category,
                'severity': obs.severity,
                'description': obs.description,
                'affected_items': obs.affected_items,
                'supporting_data': obs.supporting_data
            }
            for obs in observations
        ]

        output_data = {
            'generated_at': pd.Timestamp.now().isoformat(),
            'total_tests': len(self.results_df),
            'summaries': summary_dict,
            'observations': observations_dict
        }

        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)


# Convenience function for quick usage
def summarize_test_results(
    benchmark_storage: BenchmarkStorage,
    output_dir: Path
) -> Tuple[str, List[Observation]]:
    """
    Quick function to generate test summary report.

    Args:
        benchmark_storage: BenchmarkStorage with results
        output_dir: Directory to save reports

    Returns:
        (report_text, observations)
    """
    summarizer = TestResultsSummarizer(benchmark_storage)

    # Generate markdown report
    report_path = output_dir / 'test_results_summary.md'
    report_text = summarizer.generate_summary_report(report_path)

    # Export JSON data
    json_path = output_dir / 'test_results_summary.json'
    summarizer.export_data_summary(json_path)

    return report_text, summarizer.observations
```

#### Key Features

1. **Success/Failure Tracking**: Monitors test execution status and computes reliability metrics
2. **Statistical Summaries**: Mean, median, std dev for tour weights and runtimes
3. **Pattern Detection**: Identifies algorithm dominance, graph type difficulty, performance scaling
4. **Anomaly Detection**: Finds statistical outliers (>3σ from mean)
5. **Interaction Analysis**: Algorithm × graph type performance matrix
6. **Data Quality**: Checks for missing data, duplicates, invalid values
7. **Interpretations**: Generates human-readable observations with severity levels
8. **Multi-Format Export**: Markdown reports + JSON data for programmatic access

#### Usage in Pipeline

Add to evaluation stage in `experiments/run_experiment.py`:

```python
# After benchmarking stage completes
if config.get('analysis.generate_test_summary', True):
    from pipeline.test_results_summary import summarize_test_results
    from algorithms.storage import BenchmarkStorage

    bench_storage = BenchmarkStorage(str(output_dir / 'benchmarks'))
    report_text, observations = summarize_test_results(
        bench_storage,
        output_dir / 'reports'
    )

    print("\n=== Test Results Summary ===")
    print(report_text)

    # Print critical observations
    critical_obs = [obs for obs in observations if obs.severity == 'critical']
    if critical_obs:
        print("\n⚠️ CRITICAL ISSUES DETECTED:")
        for obs in critical_obs:
            print(f"  - {obs.description}")
```

#### Integration with Notebook

Add section in `notebooks/01_end_to_end_workflow.ipynb`:

```python
## Test Results Summary

from pipeline.test_results_summary import TestResultsSummarizer

# Initialize summarizer
summarizer = TestResultsSummarizer(bench_storage)

# Load and analyze results
summarizer.load_results()
summaries = summarizer.compute_success_rates()

# Print overall statistics
print("Overall Test Statistics:")
print(f"  Total: {summaries['overall'].total_tests}")
print(f"  Success Rate: {summaries['overall'].success_rate:.1%}")
print(f"  Mean Weight: {summaries['overall'].mean_tour_weight:.2f}")

# Identify patterns
observations = summarizer.identify_patterns()
print(f"\nFound {len(observations)} observations:")
for obs in observations:
    print(f"  [{obs.severity.upper()}] {obs.description}")

# Generate full report
report = summarizer.generate_summary_report(
    output_path=output_dir / 'reports' / 'test_summary.md'
)
print(f"\nFull report saved to: {output_dir / 'reports' / 'test_summary.md'}")
```

---

## Part 5: End-to-End Jupyter Notebook

### File: `notebooks/01_end_to_end_workflow.ipynb`

#### Purpose
Interactive demonstration of complete research pipeline.

#### Notebook Structure

```markdown
# TSP Anchor-Based Heuristic Research: End-to-End Workflow

## Setup

### Import Dependencies

```python
import sys
sys.path.append('../src')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pipeline.orchestrator import PipelineOrchestrator
from pipeline.config import ExperimentConfig
from pipeline.tracking import ExperimentTracker
from pipeline.reproducibility import ReproducibilityManager
from pipeline.stages import (
    create_graph_generation_stage,
    create_benchmarking_stage,
    create_feature_extraction_stage,
    create_training_stage,
    create_evaluation_stage
)
from pipeline.analysis import ExperimentAnalyzer
from pipeline.visualization import ExperimentVisualizer
```

### Load Configuration

```python
# Load experiment config
config = ExperimentConfig.from_yaml('../config/complete_experiment_template.yaml')
print(f"Experiment: {config.get('experiment.name')}")
print(f"Random seed: {config.get('experiment.random_seed')}")
```

---

## Part 1: Graph Generation

```python
# Create reproducibility manager
repro_manager = ReproducibilityManager(
    master_seed=config.get('experiment.random_seed'),
    git_tracking=True
)

# Create output directory
output_dir = Path(config.get('experiment.output_dir'))
output_dir.mkdir(parents=True, exist_ok=True)

# Create graph generation stage
graph_gen_stage = create_graph_generation_stage(
    config=config.to_dict(),
    repro_manager=repro_manager,
    output_dir=output_dir
)

# Execute
result = graph_gen_stage.execute({})
print(f"Generated {result.outputs['num_graphs']} graphs")
print(f"Batch manifest: {result.outputs['batch_manifest']}")
```

### Visualize Sample Graphs

```python
from graph_generation import GraphStorage, GraphVisualizer

storage = GraphStorage(str(output_dir / 'graphs'))
graphs = result.outputs['graphs'][:3]  # First 3 graphs

visualizer = GraphVisualizer()
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for idx, (graph, ax) in enumerate(zip(graphs, axes)):
    visualizer.visualize(graph, ax=ax)
    ax.set_title(f"Graph {idx+1} ({graph.metadata['type']}, n={graph.n_vertices})")

plt.tight_layout()
plt.show()
```

---

## Part 2: Algorithm Benchmarking

```python
# Create benchmarking stage
benchmark_stage = create_benchmarking_stage(
    config=config.to_dict(),
    repro_manager=repro_manager,
    output_dir=output_dir
)

# Execute
benchmark_result = benchmark_stage.execute(result.outputs)
print(f"Ran {benchmark_result.outputs['num_results']} algorithm benchmarks")
print(f"Results saved to: {benchmark_result.outputs['results_db_path']}")
```

### Analyze Benchmark Results

```python
from algorithms.storage import BenchmarkStorage

bench_storage = BenchmarkStorage(str(output_dir / 'benchmarks'))
all_results = bench_storage.load_results()

df = pd.DataFrame(all_results)
print(df.groupby('algorithm')['tour_weight'].describe())
```

### Compare Algorithms

```python
import seaborn as sns

fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(data=df, x='algorithm', y='tour_weight', ax=ax)
ax.set_xlabel("Algorithm")
ax.set_ylabel("Tour Weight")
ax.set_title("Algorithm Performance Comparison")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
```

---

## Part 3: Feature Extraction

```python
# Create feature extraction stage
feature_stage = create_feature_extraction_stage(
    config=config.to_dict(),
    repro_manager=repro_manager,
    output_dir=output_dir
)

# Execute
feature_result = feature_stage.execute({
    'graphs': result.outputs['graphs'],
    'benchmark_results': benchmark_result.outputs['benchmark_results']
})

print(f"Extracted {feature_result.outputs['num_features']} features")
print(f"For {feature_result.outputs['num_vertices']} vertices")
print(f"Dataset saved to: {feature_result.outputs['feature_dataset_path']}")
```

### Explore Features

```python
# Load feature dataset
feature_df = pd.read_csv(feature_result.outputs['feature_dataset_path'])
print(feature_df.head())

# Feature distributions
feature_cols = feature_result.outputs['feature_names'][:10]  # First 10 features
feature_df[feature_cols].hist(figsize=(15, 10), bins=30)
plt.tight_layout()
plt.show()
```

### Feature Correlations

```python
from features.analysis import FeatureAnalyzer

analyzer = FeatureAnalyzer()
correlation_matrix = analyzer.compute_feature_correlations(
    feature_df[feature_cols].values
)

fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=ax,
            xticklabels=feature_cols, yticklabels=feature_cols)
ax.set_title("Feature Correlation Heatmap")
plt.tight_layout()
plt.show()
```

---

## Part 4: Model Training

```python
# Create training stage
training_stage = create_training_stage(
    config=config.to_dict(),
    repro_manager=repro_manager,
    output_dir=output_dir
)

# Execute
training_result = training_stage.execute({
    'feature_dataset_path': feature_result.outputs['feature_dataset_path'],
    'feature_names': feature_result.outputs['feature_names']
})

print(f"Trained {training_result.outputs['num_models']} models")
for model_info in training_result.outputs['trained_models']:
    print(f"  {model_info['model_type']}: R² = {model_info['performance']['r2']:.3f}")
```

### Feature Importance

```python
import pickle

# Load best model
model_path = training_result.outputs['best_model_path']
with open(model_path, 'rb') as f:
    model = pickle.load(f)

# Get feature importance
feature_importance = model.get_feature_importance()
importance_df = pd.DataFrame({
    'feature_name': feature_result.outputs['feature_names'],
    'importance': feature_importance
}).sort_values('importance', ascending=False)

# Plot
fig, ax = plt.subplots(figsize=(8, 10))
top_k = 15
ax.barh(importance_df['feature_name'][:top_k], importance_df['importance'][:top_k])
ax.set_xlabel("Importance")
ax.set_ylabel("Feature")
ax.set_title(f"Top {top_k} Feature Importances")
plt.tight_layout()
plt.show()
```

---

## Part 5: Model Evaluation

```python
# Create evaluation stage
eval_stage = create_evaluation_stage(
    config=config.to_dict(),
    repro_manager=repro_manager,
    output_dir=output_dir
)

# Execute on test graphs
eval_result = eval_stage.execute({
    'graphs': result.outputs['graphs'],
    'best_model_path': training_result.outputs['best_model_path']
})

print("Evaluation Results:")
report = eval_result.outputs['evaluation_report']
print(f"  Mean improvement vs random: {report['mean_improvement_vs_random']:.1%}")
print(f"  Mean improvement vs NN: {report['mean_improvement_vs_nn']:.1%}")
print(f"  Win rate vs random: {report['win_rate_vs_random']:.1%}")
print(f"  Win rate vs NN: {report['win_rate_vs_nn']:.1%}")
```

### Detailed Results Analysis

```python
# Load detailed results
eval_df = pd.read_csv(eval_result.outputs['report_path'])

# Performance by graph type
grouped = eval_df.groupby('graph_type').agg({
    'improvement_vs_random': 'mean',
    'improvement_vs_nn': 'mean'
})
print(grouped)

# Scatter plot: Predicted vs baselines
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

axes[0].scatter(eval_df['random_anchor_tour'], eval_df['predicted_anchor_tour'], alpha=0.6)
axes[0].plot([eval_df['random_anchor_tour'].min(), eval_df['random_anchor_tour'].max()],
             [eval_df['random_anchor_tour'].min(), eval_df['random_anchor_tour'].max()],
             'r--', label='Equal Performance')
axes[0].set_xlabel("Random Anchor Tour Weight")
axes[0].set_ylabel("Predicted Anchor Tour Weight")
axes[0].set_title("Predicted vs Random Anchor")
axes[0].legend()

axes[1].scatter(eval_df['nearest_neighbor_tour'], eval_df['predicted_anchor_tour'], alpha=0.6)
axes[1].plot([eval_df['nearest_neighbor_tour'].min(), eval_df['nearest_neighbor_tour'].max()],
             [eval_df['nearest_neighbor_tour'].min(), eval_df['nearest_neighbor_tour'].max()],
             'r--', label='Equal Performance')
axes[1].set_xlabel("Nearest Neighbor Tour Weight")
axes[1].set_ylabel("Predicted Anchor Tour Weight")
axes[1].set_title("Predicted vs Nearest Neighbor")
axes[1].legend()

plt.tight_layout()
plt.show()
```

---

## Part 6: Comprehensive Analysis

```python
# Use analysis tools
analyzer = ExperimentAnalyzer(output_dir)

# Load all results
benchmark_df = analyzer.load_benchmark_results()
eval_df = analyzer.load_evaluation_results()

# Statistical significance testing
significance_result = analyzer.compute_statistical_significance(
    algorithm_a='single_anchor',
    algorithm_b='nearest_neighbor',
    test='paired_t_test'
)

print(f"Statistical Test Results:")
print(f"  Test statistic: {significance_result['test_statistic']:.3f}")
print(f"  P-value: {significance_result['p_value']:.4f}")
print(f"  Effect size (Cohen's d): {significance_result['effect_size']:.3f}")
print(f"  Significant: {significance_result['significant']}")

# Generate summary report
summary = analyzer.generate_summary_report()
print("\n" + "="*60)
print(summary)
print("="*60)
```

### Create Publication Figures

```python
visualizer = ExperimentVisualizer(style="publication")

# Create output directory for figures
figures_dir = output_dir / 'figures'
figures_dir.mkdir(exist_ok=True)

# Algorithm comparison
visualizer.plot_algorithm_comparison(
    benchmark_df,
    metric='tour_weight',
    output_path=figures_dir / 'algorithm_comparison.png'
)

# Feature importance
visualizer.plot_feature_importance(
    importance_df,
    top_k=15,
    output_path=figures_dir / 'feature_importance.png'
)

# Performance by graph type
visualizer.plot_performance_by_graph_type(
    benchmark_df,
    algorithms=['nearest_neighbor', 'single_anchor', 'best_anchor'],
    output_path=figures_dir / 'performance_by_graph_type.png'
)

print(f"Figures saved to {figures_dir}")
```

---

## Summary and Next Steps

### Key Findings
1. **Algorithm Performance**: [Summary from results]
2. **Feature Importance**: [Top 3 features]
3. **ML Improvement**: [% improvement vs baselines]

### Research Questions Answered
- Do anchor-based heuristics outperform nearest neighbor? **[Yes/No]**
- Can ML predict good anchors? **[Yes/No]**
- Which features matter most? **[List]**

### Future Directions
- Test on larger graphs (n > 200)
- Explore additional graph types
- Investigate feature interactions
- Optimize ML model hyperparameters
```

---

## Part 6: Command-Line Experiment Runner

### File: `experiments/run_experiment.py`

#### Purpose
Single-command execution of complete pipeline.

```python
#!/usr/bin/env python3
"""
Complete experiment runner.

Usage:
    python experiments/run_experiment.py config/my_experiment.yaml
    python experiments/run_experiment.py config/my_experiment.yaml --stage feature_extraction
    python experiments/run_experiment.py config/my_experiment.yaml --resume exp_12345
"""

import sys
import argparse
from pathlib import Path

sys.path.append('src')

from pipeline.orchestrator import PipelineOrchestrator
from pipeline.config import ExperimentConfig
from pipeline.tracking import ExperimentTracker, ExperimentRegistry
from pipeline.reproducibility import ReproducibilityManager
from pipeline.stages import (
    create_graph_generation_stage,
    create_benchmarking_stage,
    create_feature_extraction_stage,
    create_training_stage,
    create_evaluation_stage
)
from pipeline.analysis import ExperimentAnalyzer
from pipeline.visualization import ExperimentVisualizer


def main():
    parser = argparse.ArgumentParser(description="Run complete TSP experiment")
    parser.add_argument('config', type=str, help="Path to experiment config YAML")
    parser.add_argument('--stage', type=str, default=None,
                        help="Run specific stage only (graph_generation, benchmarking, etc.)")
    parser.add_argument('--resume', type=str, default=None,
                        help="Resume from experiment ID")
    parser.add_argument('--dry-run', action='store_true',
                        help="Validate config without running")

    args = parser.parse_args()

    # Load config
    config = ExperimentConfig.from_yaml(args.config)

    if args.dry_run:
        print("Configuration valid!")
        print(f"Experiment: {config.get('experiment.name')}")
        print(f"Enabled stages: {[k for k, v in config.to_dict().items() if isinstance(v, dict) and v.get('enabled')]}")
        return

    # Setup experiment tracking
    output_dir = Path(config.get('experiment.output_dir'))
    output_dir.mkdir(parents=True, exist_ok=True)

    tracker = ExperimentTracker(experiment_dir=output_dir)
    tracker.start()

    # Setup reproducibility
    repro_manager = ReproducibilityManager(
        master_seed=config.get('experiment.random_seed'),
        git_tracking=True
    )

    # Create pipeline stages
    stages = []

    if config.get('graph_generation.enabled', False):
        stages.append(create_graph_generation_stage(config.to_dict(), repro_manager, output_dir))

    if config.get('benchmarking.enabled', False):
        stages.append(create_benchmarking_stage(config.to_dict(), repro_manager, output_dir))

    if config.get('feature_extraction.enabled', False):
        stages.append(create_feature_extraction_stage(config.to_dict(), repro_manager, output_dir))

    if config.get('training.enabled', False):
        stages.append(create_training_stage(config.to_dict(), repro_manager, output_dir))

    if config.get('evaluation.enabled', False):
        stages.append(create_evaluation_stage(config.to_dict(), repro_manager, output_dir))

    # Create orchestrator
    orchestrator = PipelineOrchestrator()
    for stage in stages:
        orchestrator.add_stage(stage)

    # Run pipeline
    if args.stage:
        # Run specific stage only
        print(f"Running stage: {args.stage}")
        result = orchestrator.run_stage(args.stage, {})
    else:
        # Run complete pipeline
        print(f"Running complete pipeline with {len(stages)} stages")
        result = orchestrator.run_all()

    # Complete tracking
    tracker.complete(status="success" if result.success else "failed")

    # Generate analysis if enabled
    if config.get('analysis.enabled', False):
        print("\nGenerating analysis and visualizations...")

        analyzer = ExperimentAnalyzer(output_dir)
        summary = analyzer.generate_summary_report()

        # Save report
        report_path = output_dir / 'reports' / 'summary_report.md'
        report_path.parent.mkdir(exist_ok=True)
        with open(report_path, 'w') as f:
            f.write(summary)
        print(f"Report saved to: {report_path}")

        # Generate visualizations
        if config.get('analysis.generate_visualizations', False):
            visualizer = ExperimentVisualizer(style="publication")
            visualizer.create_summary_figure(
                output_dir,
                output_path=output_dir / 'figures' / 'summary.png'
            )
            print(f"Figures saved to: {output_dir / 'figures'}")

    print("\nExperiment complete!")
    print(f"Results in: {output_dir}")


if __name__ == '__main__':
    main()
```

---

## Part 7: Testing Strategy

### Test Files Needed

#### File: `src/tests/test_phase5_prompts_5-8.py`

**Prompts to Test:**
- Prompt 5: StageValidator
- Prompt 6: PerformanceMonitor, RuntimeProfiler
- Prompt 7: ParallelExecutor, ResourceManager
- Prompt 8: ErrorHandler, retry/checkpoint patterns

**Target:** ~50 tests for complete Phase 5 coverage

#### File: `src/tests/test_pipeline_stages.py`

**What to Test:**
- Each stage factory function
- Stage inputs/outputs
- Stage execution with valid configs
- Error handling for invalid inputs
- Integration between stages

#### File: `src/tests/test_benchmark_storage.py`

**What to Test:**
- Database initialization
- Saving/loading results
- Querying by graph_id, algorithm, graph_type
- Anchor weights extraction
- Concurrent access (if using SQLite)

---

## Running Guide

### Prerequisites

```bash
# Install dependencies
pip install -r requirements.txt

# Verify installation
python -m unittest discover -s src/tests -p "test_*.py" -v
```

### Quick Start: Jupyter Notebook

```bash
# Launch notebook
jupyter notebook notebooks/01_end_to_end_workflow.ipynb

# Execute cells in order
# Each cell is self-contained with explanations
```

**Expected Behavior:**
- **Graph Generation Cell**: Should create 20-50 graphs, display 3 sample visualizations
- **Benchmarking Cell**: Should run 5-8 algorithms on all graphs, display box plot comparison
- **Feature Extraction Cell**: Should extract 50-150 features per vertex, show correlation heatmap
- **Training Cell**: Should train 2-3 models, display R² scores and feature importance
- **Evaluation Cell**: Should test predictions, show improvement vs baselines
- **Analysis Cell**: Should generate statistical tests, summary report, publication figures

**Time Estimate:** 10-30 minutes depending on graph count and sizes

### Command-Line Execution

```bash
# Dry run (validate config only)
python experiments/run_experiment.py config/complete_experiment_template.yaml --dry-run

# Run complete experiment
python experiments/run_experiment.py config/complete_experiment_template.yaml

# Run specific stage
python experiments/run_experiment.py config/complete_experiment_template.yaml --stage benchmarking

# Resume from checkpoint
python experiments/run_experiment.py config/complete_experiment_template.yaml --resume exp_12345
```

**Expected Output:**
```
Loading configuration: config/complete_experiment_template.yaml
Experiment: baseline_comparison_v1
Running complete pipeline with 5 stages

[Stage 1/5] graph_generation
  Generated 55 graphs
  Saved to: experiments/baseline_v1/graphs/
  ✓ Complete (2.3s)

[Stage 2/5] benchmarking
  Running 4 algorithms on 55 graphs (exhaustive anchors)
  Progress: [████████████████████] 100% (220/220 benchmarks)
  ✓ Complete (124.7s)

[Stage 3/5] feature_extraction
  Extracting 6 feature groups
  Progress: [████████████████████] 100% (55/55 graphs)
  ✓ Complete (18.2s)

[Stage 4/5] training
  Training 3 models with 5-fold CV
  Model performance:
    linear_ridge: R² = 0.623
    linear_lasso: R² = 0.597
    random_forest: R² = 0.681
  ✓ Complete (5.1s)

[Stage 5/5] evaluation
  Testing predictions on 55 graphs
  Mean improvement vs random: 18.3%
  Win rate vs random: 78.2%
  ✓ Complete (8.9s)

Generating analysis and visualizations...
Report saved to: experiments/baseline_v1/reports/summary_report.md
Figures saved to: experiments/baseline_v1/figures/

Experiment complete!
Results in: experiments/baseline_v1
```

### Customizing Experiments

#### Create Custom Config

```bash
# Copy template
cp config/complete_experiment_template.yaml config/my_experiment.yaml

# Edit config (change graph types, algorithms, model parameters)
nano config/my_experiment.yaml

# Run experiment
python experiments/run_experiment.py config/my_experiment.yaml
```

#### Test on Different Graph Types

```yaml
# Focus on metric graphs only
graph_generation:
  types:
    - type: "metric"
      sizes: [50, 100, 200]
      instances_per_size: 30
      strategy: "completion"
```

#### Test Different Labeling Strategies

```yaml
# Use binary classification instead of regression
feature_extraction:
  labeling_strategy: "binary"
  labeling_params:
    percentile_threshold: 20  # Top 20% = good anchors

training:
  problem_type: "classification"
  models:
    - type: "logistic_regression"
    - type: "random_forest"
      n_estimators: 200
```

### Analyzing Results

#### Load Experiment Results

```python
from pipeline.analysis import ExperimentAnalyzer

analyzer = ExperimentAnalyzer('experiments/baseline_v1')

# Compare algorithms
comparison = analyzer.compare_algorithms(
    algorithms=['nearest_neighbor', 'single_anchor', 'best_anchor'],
    groupby='graph_type'
)
print(comparison)

# Statistical test
sig_test = analyzer.compute_statistical_significance(
    algorithm_a='single_anchor',
    algorithm_b='nearest_neighbor'
)
print(f"P-value: {sig_test['p_value']}")
```

#### Create Custom Visualizations

```python
from pipeline.visualization import ExperimentVisualizer

visualizer = ExperimentVisualizer(style="publication")

# Load benchmark results
benchmark_df = analyzer.load_benchmark_results()

# Custom plot
visualizer.plot_algorithm_comparison(
    benchmark_df,
    metric='tour_weight',
    output_path='my_custom_plot.png'
)
```

---

## Expected Outcomes

### Quantitative Results

After running complete pipeline, you should obtain:

1. **Algorithm Performance:**
   - Mean tour weights for each algorithm
   - Win rates (% of graphs where each algorithm wins)
   - Statistical significance of differences

2. **Feature Importance:**
   - Top 10-15 features for predicting anchor quality
   - Feature importance scores (coefficients or Gini importance)
   - Feature correlations with labels

3. **ML Model Performance:**
   - R² scores (target: > 0.5 for regression)
   - Classification accuracy (target: > 70% for binary)
   - Top-k accuracy (target: 80% for k=5)

4. **Practical Improvement:**
   - % improvement vs random anchor (target: > 15%)
   - % improvement vs baseline algorithm (target: > 5%)
   - Computational speedup (predicted anchor vs exhaustive search)

### Qualitative Insights

Expected research findings:

1. **When Anchoring Works:**
   - "Anchor-based heuristics beat nearest neighbor on metric graphs (p < 0.01)"
   - "Advantage decreases on quasi-metric graphs (metricity < 0.6)"

2. **What Makes Good Anchors:**
   - "MST degree is strongest predictor (importance = 0.32)"
   - "Mean edge weight is second strongest (importance = 0.24)"
   - "Structural centrality matters more than geographic centrality"

3. **ML Viability:**
   - "Linear regression achieves 92% of best-anchor quality with 50× speedup"
   - "Predictions beat random anchor 78% of the time"
   - "Top-5 predictions contain optimal anchor 85% of the time"

### Publication-Ready Outputs

Generated artifacts suitable for papers:

1. **Figures:** (experiments/{exp_id}/figures/)
   - algorithm_comparison.png - Box plot of algorithm performance
   - feature_importance.png - Bar chart of top features
   - predicted_vs_actual.png - Scatter plot of model accuracy
   - performance_by_graph_type.png - Line plot of algorithm interactions

2. **Tables:** (in summary report)
   - Algorithm performance comparison table
   - Statistical test results table
   - Feature importance ranking table
   - Model evaluation metrics table

3. **Report:** (experiments/{exp_id}/reports/summary_report.md)
   - Markdown summary with all key findings
   - Convertible to LaTeX for paper submission

---

## Success Criteria

### Minimum Viable Product (MVP)

- ✅ Complete pipeline runs from config to results without errors
- ✅ All 5 stages execute in sequence
- ✅ Results are reproducible (same config + seed = same results)
- ✅ Generates at least 3 publication-quality figures
- ✅ Summary report includes statistical tests

### Research Validation

- ✅ Can answer: "Do anchor-based heuristics work?"
- ✅ Can answer: "What makes a good anchor?"
- ✅ Can answer: "Can ML predict good anchors effectively?"
- ✅ Statistical significance established (p-values < 0.05)
- ✅ Effect sizes reported (Cohen's d or similar)

### Code Quality

- ✅ 50 new tests for Phase 5 Prompts 5-8
- ✅ All 425 tests passing (375 existing + 50 new)
- ✅ No breaking changes to existing APIs
- ✅ Documentation complete (docstrings, README)

---

## Implementation Priority

### Week 1 (Critical Path)

1. **Implement Stage Factories** (`src/pipeline/stages.py`)
   - ~400-500 lines total
   - Connects all existing components
   - **Blocker for everything else**

2. **Implement Benchmark Storage** (`src/algorithms/storage.py`)
   - ~200-300 lines
   - Enables persistent results
   - Required for feature labeling

3. **Create Complete Config Template** (`config/complete_experiment_template.yaml`)
   - ~150 lines YAML
   - Documents all options
   - Validates with ExperimentConfig

### Week 2 (Analysis & Visualization)

4. **Implement Test Results Summary** (`src/pipeline/test_results_summary.py`)
   - ~400-500 lines
   - Success/failure tracking across algorithm-graph combinations
   - Automated observations and interpretations
   - Statistical analysis and anomaly detection

5. **Implement Analysis Tools** (`src/pipeline/analysis.py`)
   - ~300-400 lines
   - Load results, compute statistics
   - Generate reports

6. **Implement Visualization Tools** (`src/pipeline/visualization.py`)
   - ~200-300 lines
   - Publication-quality plots
   - Summary figures

7. **Create Jupyter Notebook** (`notebooks/01_end_to_end_workflow.ipynb`)
   - ~500-1000 lines (including markdown)
   - Interactive demonstration
   - Educational resource

### Week 3 (Polish & Testing)

8. **Create CLI Runner** (`experiments/run_experiment.py`)
   - ~200-300 lines
   - Single-command execution
   - Resume/checkpoint support
   - Integrates test results summary

9. **Write Tests for Phase 5 Prompts 5-8** (`src/tests/test_phase5_prompts_5-8.py`)
   - ~50 tests
   - Validation, profiling, parallel, error handling
   - Achieves 100% Phase 5 coverage

10. **Integration Testing**
   - Run complete notebook
   - Run CLI with multiple configs
   - Validate reproducibility
   - Benchmark performance

---

## Notes and Considerations

### What This Plan Does NOT Include

- **Writing test scripts** - You'll run `python -m unittest` manually
- **Automating tests through AI** - Tests are written for human execution
- **Real code implementations** - Only templates, pseudocode, and design
- **Actual data collection** - You'll generate graphs and run experiments

### What This Plan DOES Include

- **Architectural design** for all integration components
- **Templates and pseudocode** showing how to connect phases
- **Configuration schemas** for end-to-end experiments
- **Running instructions** for notebooks and CLI
- **Expected behaviors** when you execute the workflow
- **Success criteria** to validate completion

### Design Philosophy

1. **Leverage Existing Code:** All Phase 1-4 components are complete. Stage factories are just thin wrappers.

2. **Configuration-Driven:** Single YAML controls everything. Easy to create experiment variants.

3. **Incremental Execution:** Can run stages individually for debugging. Can resume from checkpoints.

4. **Reproducibility:** Seeds propagate through all stages. Git tracking enabled. Results deterministic.

5. **Analysis-First:** Every experiment auto-generates reports and visualizations. Research insights are first-class outputs.

### Known Limitations

- **Scalability:** Current design targets 50-200 vertex graphs. Larger graphs may need optimization.
- **Parallelization:** ParallelExecutor exists but stage factories don't use it yet (easy to add).
- **Model Comparison:** Evaluation stage tests single best model. Cross-model comparison requires manual analysis.
- **Real-World Graphs:** System tested on synthetic graphs. TSPLIB integration not included (future work).

---

## Summary

This plan provides:

1. **8 new files** to implement:
   - `src/pipeline/stages.py` - Stage factory functions
   - `src/algorithms/storage.py` - Benchmark storage
   - `config/complete_experiment_template.yaml` - Unified config
   - `src/pipeline/test_results_summary.py` - **NEW**: Test results summarizer with observations/interpretations
   - `src/pipeline/analysis.py` - Analysis tools
   - `src/pipeline/visualization.py` - Visualization tools
   - `notebooks/01_end_to_end_workflow.ipynb` - Interactive demo
   - `experiments/run_experiment.py` - CLI runner

2. **50 new tests** for Phase 5 Prompts 5-8

3. **Running instructions** for:
   - Jupyter notebook (interactive exploration)
   - Command-line runner (production experiments)
   - Custom configurations (research variants)

4. **Expected outcomes** including:
   - Quantitative results (R², improvements, win rates)
   - Qualitative insights (research findings)
   - Publication-ready artifacts (figures, tables, reports)

**Next Step:** Review this plan, then implement components in priority order. Start with stage factories (Week 1, items 1-3) as they unblock everything else.
