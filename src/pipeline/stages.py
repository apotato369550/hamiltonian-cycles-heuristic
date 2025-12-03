"""
Pipeline stage factory functions.

Each factory creates a PipelineStage that wraps phase-specific logic
into the unified pipeline interface.

This module bridges Phase 1-4 implementations with the Phase 5 orchestrator.
"""

from pathlib import Path
from typing import Dict, Any, List
import numpy as np

from .orchestrator import PipelineStage, StageResult
from .reproducibility import ReproducibilityManager


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
                  dimension: 2
                  weight_range: [1.0, 100.0]

    Outputs:
        - graphs: List[GraphInstance]
        - graph_paths: List[str]
        - batch_manifest: str (path)
        - num_graphs: int
    """
    def execute(inputs: Dict[str, Any]) -> StageResult:
        from graph_generation import EuclideanGraphGenerator, MetricGraphGenerator, QuasiMetricGraphGenerator, RandomGraphGenerator, GraphStorage

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
                        generator = EuclideanGraphGenerator(
                            seed=seed + instance,
                            dimension=graph_spec.get('dimension', 2)
                        )
                    elif graph_type == 'metric':
                        generator = MetricGraphGenerator(
                            seed=seed + instance,
                            strategy=graph_spec.get('strategy', 'completion')
                        )
                    elif graph_type == 'quasi_metric':
                        generator = QuasiMetricGraphGenerator(
                            seed=seed + instance,
                            strategy=graph_spec.get('strategy', 'completion')
                        )
                    elif graph_type == 'random':
                        generator = RandomGraphGenerator(
                            seed=seed + instance
                        )
                    else:
                        raise ValueError(f"Unknown graph type: {graph_type}")

                    # Generate graph
                    weight_range = tuple(graph_spec.get('weight_range', [1.0, 100.0]))
                    graph = generator.generate(
                        n_vertices=size,
                        weight_range=weight_range
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
                'graph_types': list(set(spec['type'] for spec in gen_config['types'])),
                'total_graphs': len(all_graphs),
                'seed': seed
            }
        )

    return PipelineStage(
        name='graph_generation',
        execute_fn=execute,
        required_inputs=[],  # No upstream dependencies
        expected_outputs=['graphs', 'graph_paths', 'batch_manifest', 'num_graphs']
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
        - graphs: List[GraphInstance] OR
        - graph_paths: List[str] (will load if graphs not provided)

    Outputs:
        - benchmark_results: List[Dict]
        - results_db_path: str
        - num_results: int
    """
    def execute(inputs: Dict[str, Any]) -> StageResult:
        from algorithms import AlgorithmRegistry
        from algorithms.storage import BenchmarkStorage
        from graph_generation import GraphStorage
        import time

        # Extract inputs
        graphs = inputs.get('graphs', [])
        if not graphs and 'graph_paths' in inputs:
            # Load graphs from paths
            storage = GraphStorage(str(output_dir / 'graphs'))
            graphs = [storage.load_graph(Path(p)) for p in inputs['graph_paths']]

        if not graphs:
            raise ValueError("No graphs provided for benchmarking")

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
            graph_type = graph.metadata.get('type', 'unknown')
            graph_size = graph.n_vertices

            for algo_spec in algo_specs:
                algo_name = algo_spec['name']
                algo_params = algo_spec.get('params', {})

                # Get algorithm from registry
                algo = AlgorithmRegistry.get_algorithm(
                    algo_name,
                    random_seed=seed,
                    **algo_params
                )

                if exhaustive_anchors and 'anchor' in algo_name:
                    # Test all possible anchors for labeling
                    for anchor_vertex in range(graph.n_vertices):
                        start_time = time.time()
                        try:
                            result = algo.solve(
                                graph.adjacency_matrix,
                                anchor_vertex=anchor_vertex
                            )
                            runtime = time.time() - start_time

                            bench_result = {
                                'graph_id': graph_id,
                                'graph_type': graph_type,
                                'graph_size': graph_size,
                                'algorithm': algo_name,
                                'anchor_vertex': anchor_vertex,
                                'tour_weight': result.weight,
                                'runtime': runtime,
                                'tour': result.tour
                            }
                        except Exception as e:
                            bench_result = {
                                'graph_id': graph_id,
                                'graph_type': graph_type,
                                'graph_size': graph_size,
                                'algorithm': algo_name,
                                'anchor_vertex': anchor_vertex,
                                'tour_weight': None,  # Indicates failure
                                'runtime': time.time() - start_time,
                                'tour': None,
                                'error': str(e)
                            }

                        all_results.append(bench_result)
                        bench_storage.save_result(bench_result)
                else:
                    # Single run per algorithm
                    start_time = time.time()
                    try:
                        result = algo.solve(graph.adjacency_matrix)
                        runtime = time.time() - start_time

                        bench_result = {
                            'graph_id': graph_id,
                            'graph_type': graph_type,
                            'graph_size': graph_size,
                            'algorithm': algo_name,
                            'anchor_vertex': None,
                            'tour_weight': result.weight,
                            'runtime': runtime,
                            'tour': result.tour
                        }
                    except Exception as e:
                        bench_result = {
                            'graph_id': graph_id,
                            'graph_type': graph_type,
                            'graph_size': graph_size,
                            'algorithm': algo_name,
                            'anchor_vertex': None,
                            'tour_weight': None,
                            'runtime': time.time() - start_time,
                            'tour': None,
                            'error': str(e)
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
                'exhaustive_anchors': exhaustive_anchors,
                'seed': seed
            }
        )

    return PipelineStage(
        name='benchmarking',
        execute_fn=execute,
        required_inputs=['graphs'],
        expected_outputs=['benchmark_results', 'results_db_path', 'num_results']
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
            labeling_params:
                percentile_top: 20
                percentile_bottom: 20
            output_format: "csv"  # or "pickle"

    Inputs:
        - graphs: List[GraphInstance]
        - benchmark_results: List[Dict] (for labeling)

    Outputs:
        - feature_dataset_path: str
        - feature_names: List[str]
        - num_features: int
        - num_vertices: int
    """
    def execute(inputs: Dict[str, Any]) -> StageResult:
        from features import (
            FeatureExtractorPipeline,
            WeightBasedExtractor,
            TopologicalExtractor,
            MSTBasedExtractor,
            NeighborhoodExtractor,
            HeuristicExtractor,
            GraphContextExtractor,
            AnchorQualityLabeler
        )
        import pandas as pd
        import pickle

        # Extract inputs
        graphs = inputs['graphs']
        benchmark_results = inputs.get('benchmark_results', [])

        # Extract config
        feat_config = config.get('feature_extraction', {})
        extractor_names = feat_config.get('extractors', ['weight_based'])
        labeling_strategy = feat_config.get('labeling_strategy', 'rank_based')
        labeling_params = feat_config.get('labeling_params', {})
        output_format = feat_config.get('output_format', 'csv')

        # Build feature extractor pipeline
        pipeline = FeatureExtractorPipeline()

        if 'weight_based' in extractor_names:
            pipeline.add_extractor(WeightBasedExtractor())
        if 'topological' in extractor_names:
            pipeline.add_extractor(TopologicalExtractor())
        if 'mst_based' in extractor_names:
            pipeline.add_extractor(MSTBasedExtractor())
        if 'neighborhood' in extractor_names:
            pipeline.add_extractor(NeighborhoodExtractor())
        if 'heuristic' in extractor_names:
            pipeline.add_extractor(HeuristicExtractor())
        if 'graph_context' in extractor_names:
            pipeline.add_extractor(GraphContextExtractor())

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
                if r['graph_id'] == graph_id and 'anchor' in str(r.get('algorithm', ''))
            ]

            if graph_bench_results:
                # Extract anchor weights (filter out None values from failures)
                anchor_weights = [
                    r['tour_weight'] for r in graph_bench_results
                    if r['tour_weight'] is not None
                ]

                if len(anchor_weights) == graph.n_vertices:
                    labeler = AnchorQualityLabeler(
                        strategy=labeling_strategy,
                        **labeling_params
                    )
                    labels = labeler.label_from_weights(anchor_weights)
                else:
                    # Not all anchors have valid results
                    labels = np.zeros(graph.n_vertices)
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
            df = pd.DataFrame(features_array, columns=feature_names)
            df['label'] = labels_array
            for key in all_metadata[0].keys():
                df[key] = [m[key] for m in all_metadata]
            df.to_csv(f"{output_path}.csv", index=False)
            final_path = f"{output_path}.csv"
        else:  # pickle
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
        expected_outputs=['feature_dataset_path', 'feature_names', 'num_features']
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
            stratify_by: "graph_type"

    Inputs:
        - feature_dataset_path: str
        - feature_names: List[str]

    Outputs:
        - trained_models: List[Dict]
        - model_paths: List[str]
        - best_model_path: str
        - evaluation_results: List[Dict]
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
        from ml.dataset import DatasetSplitter, SplitStrategy

        # Load features
        feature_path = inputs['feature_dataset_path']
        if feature_path.endswith('.csv'):
            df = pd.read_csv(feature_path)
            X = df[inputs['feature_names']].values
            y = df['label'].values
            metadata = df[['graph_id', 'graph_type', 'graph_size']].to_dict('records') if 'graph_id' in df else None
        else:  # pickle
            with open(feature_path, 'rb') as f:
                data = pickle.load(f)
            X = data['features']
            y = data['labels']
            metadata = data.get('metadata', None)

        # Extract config
        train_config = config.get('training', {})
        model_specs = train_config.get('models', [])
        test_split = train_config.get('test_split', 0.2)
        stratify_by = train_config.get('stratify_by', None)

        # Set seed for reproducibility
        seed = repro_manager.propagate_seed('training')

        # Prepare dataset
        prep = DatasetPreparator(problem_type=MLProblemType.REGRESSION)
        X_clean, y_clean, prep_metadata = prep.prepare(X, y)

        # Split data
        splitter = DatasetSplitter(strategy=SplitStrategy.RANDOM_SPLIT)
        splits = splitter.split(X_clean, y_clean, test_size=test_split, random_state=seed)
        X_train, X_test = splits['X_train'], splits['X_test']
        y_train, y_test = splits['y_train'], splits['y_test']

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
                elif 'elasticnet' in model_type_str:
                    model_type = ModelType.LINEAR_ELASTICNET
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
            elif 'tree' in model_type_str:
                model = TreeBasedModel(
                    model_type=ModelType.DECISION_TREE,
                    **{k: v for k, v in model_spec.items() if k != 'type'}
                )
            elif 'gradient' in model_type_str or 'boosting' in model_type_str:
                model = TreeBasedModel(
                    model_type=ModelType.GRADIENT_BOOSTING,
                    **{k: v for k, v in model_spec.items() if k != 'type'}
                )
            else:
                raise ValueError(f"Unknown model type: {model_type_str}")

            # Train
            model.fit(X_train, y_train)

            # Evaluate
            evaluator = ModelEvaluator()
            y_pred = model.predict(X_test)
            metrics = evaluator.evaluate_single(
                y_test,
                y_pred,
                metric_type=RegressionMetric.R2
            )

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

        # Select best model by R² score
        best_model_idx = max(range(len(trained_models)), key=lambda i: trained_models[i]['performance']['r2'])
        best_model_path = trained_models[best_model_idx]['path']

        return StageResult(
            success=True,
            outputs={
                'trained_models': trained_models,
                'model_paths': model_paths,
                'best_model_path': best_model_path,
                'evaluation_results': evaluation_results
            },
            metadata={
                'num_train': len(X_train),
                'num_test': len(X_test),
                'num_models': len(trained_models),
                'best_model': trained_models[best_model_idx]['model_type'],
                'seed': seed
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
        - best_model_path: str
        - feature_names: List[str]

    Outputs:
        - evaluation_report: Dict with comparative results
        - report_path: str
        - detailed_results: List[Dict]
    """
    def execute(inputs: Dict[str, Any]) -> StageResult:
        import pickle
        import pandas as pd
        from features import FeatureExtractorPipeline
        from algorithms import AlgorithmRegistry
        from features.extractors import (
            WeightBasedExtractor,
            TopologicalExtractor,
            MSTBasedExtractor
        )

        # Load best model
        model_path = inputs['best_model_path']
        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        # Get test graphs
        graphs = inputs['graphs']

        # Build feature extractor pipeline (simplified - should match training)
        pipeline = FeatureExtractorPipeline()
        feat_config = config.get('feature_extraction', {})
        for extractor_name in feat_config.get('extractors', ['weight_based']):
            if extractor_name == 'weight_based':
                pipeline.add_extractor(WeightBasedExtractor())
            elif extractor_name == 'topological':
                pipeline.add_extractor(TopologicalExtractor())
            elif extractor_name == 'mst_based':
                pipeline.add_extractor(MSTBasedExtractor())

        # Set seed
        seed = repro_manager.propagate_seed('evaluation')

        results = []

        for graph_idx, graph in enumerate(graphs):
            graph_id = f"graph_{graph_idx:04d}"

            # Extract features
            features, _ = pipeline.extract_features(graph.adjacency_matrix)

            # Predict best anchor
            predictions = model.predict(features)
            predicted_anchor = int(np.argmax(predictions))

            # Run algorithm with predicted anchor
            algo = AlgorithmRegistry.get_algorithm('single_anchor', random_seed=seed)
            predicted_result = algo.solve(
                graph.adjacency_matrix,
                anchor_vertex=predicted_anchor
            )

            # Compare to baselines
            nn_algo = AlgorithmRegistry.get_algorithm('nearest_neighbor', random_seed=seed)
            nn_result = nn_algo.solve(graph.adjacency_matrix)

            random_anchor = np.random.RandomState(seed).randint(0, graph.n_vertices)
            random_result = algo.solve(graph.adjacency_matrix, anchor_vertex=random_anchor)

            # Record results
            results.append({
                'graph_id': graph_id,
                'graph_type': graph.metadata.get('type', 'unknown'),
                'graph_size': graph.n_vertices,
                'predicted_anchor': predicted_anchor,
                'predicted_anchor_tour': predicted_result.weight,
                'random_anchor': random_anchor,
                'random_anchor_tour': random_result.weight,
                'nearest_neighbor_tour': nn_result.weight,
                'improvement_vs_random': (random_result.weight - predicted_result.weight) / random_result.weight if random_result.weight > 0 else 0.0,
                'improvement_vs_nn': (nn_result.weight - predicted_result.weight) / nn_result.weight if nn_result.weight > 0 else 0.0
            })

        # Generate report
        df = pd.DataFrame(results)
        report_path = output_dir / 'reports' / 'evaluation_report.csv'
        report_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(report_path, index=False)

        # Compute summary statistics
        summary = {
            'mean_improvement_vs_random': float(df['improvement_vs_random'].mean()),
            'mean_improvement_vs_nn': float(df['improvement_vs_nn'].mean()),
            'win_rate_vs_random': float((df['improvement_vs_random'] > 0).mean()),
            'win_rate_vs_nn': float((df['improvement_vs_nn'] > 0).mean()),
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
        required_inputs=['graphs', 'best_model_path', 'feature_names'],
        expected_outputs=['evaluation_report', 'report_path']
    )
