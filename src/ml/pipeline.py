"""
Prediction-to-Algorithm Pipeline (Prompt 10).

Connects ML predictions to TSP algorithm execution.

Pipeline:
1. New graph → extract features
2. Predict anchor quality for all vertices
3. Select best predicted anchor
4. Run TSP algorithm from that anchor
5. Compare to baselines (random, best anchor)
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from features.pipeline import FeatureExtractorPipeline
    from algorithms.base import TSPAlgorithm
    FEATURES_AVAILABLE = True
except ImportError:
    FEATURES_AVAILABLE = False


@dataclass
class PredictionResult:
    """
    Result of anchor prediction for a single graph.

    Attributes:
        graph_id: Identifier for the graph
        predicted_best_vertex: Vertex predicted to be best anchor
        predicted_quality: Predicted anchor quality score
        all_predictions: Predictions for all vertices
        feature_values: Feature matrix for all vertices
    """
    graph_id: str
    predicted_best_vertex: int
    predicted_quality: float
    all_predictions: np.ndarray
    feature_values: Optional[pd.DataFrame] = None


@dataclass
class AlgorithmExecutionResult:
    """
    Result of running TSP algorithm with predicted anchor.

    Attributes:
        graph_id: Identifier for the graph
        predicted_anchor: Vertex used as anchor (from ML prediction)
        predicted_tour_weight: Tour weight from predicted anchor
        random_anchor: Random vertex used (for comparison)
        random_tour_weight: Tour weight from random anchor
        best_anchor: Best vertex (from exhaustive search, if available)
        best_tour_weight: Tour weight from best anchor (if available)
        metadata: Additional information (algorithm name, runtime, etc.)
    """
    graph_id: str
    predicted_anchor: int
    predicted_tour_weight: float
    random_anchor: int
    random_tour_weight: float
    best_anchor: Optional[int] = None
    best_tour_weight: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def compute_metrics(self) -> Dict[str, float]:
        """
        Compute performance metrics.

        Returns:
            Dictionary with:
            - beat_random: 1 if predicted beat random, 0 otherwise
            - improvement_over_random: (random - predicted) / random
            - optimality_gap: (predicted - best) / best (if best available)
            - relative_quality: predicted / random
        """
        metrics = {}

        # Did predicted beat random?
        metrics['beat_random'] = 1.0 if self.predicted_tour_weight < self.random_tour_weight else 0.0

        # Improvement over random
        if self.random_tour_weight > 0:
            metrics['improvement_over_random'] = (
                (self.random_tour_weight - self.predicted_tour_weight) / self.random_tour_weight
            )
        else:
            metrics['improvement_over_random'] = 0.0

        # Relative quality
        if self.random_tour_weight > 0:
            metrics['relative_quality'] = self.predicted_tour_weight / self.random_tour_weight
        else:
            metrics['relative_quality'] = 1.0

        # Optimality gap (if best available)
        if self.best_tour_weight is not None and self.best_tour_weight > 0:
            metrics['optimality_gap'] = (
                (self.predicted_tour_weight - self.best_tour_weight) / self.best_tour_weight
            )
            metrics['gap_to_best_pct'] = metrics['optimality_gap'] * 100.0
        else:
            metrics['optimality_gap'] = None
            metrics['gap_to_best_pct'] = None

        return metrics


@dataclass
class BatchPredictionResult:
    """
    Results from batch prediction on multiple graphs.

    Attributes:
        graph_ids: List of graph identifiers
        execution_results: List of AlgorithmExecutionResult
        aggregate_metrics: Aggregated performance metrics
    """
    graph_ids: List[str]
    execution_results: List[AlgorithmExecutionResult]
    aggregate_metrics: Dict[str, float] = field(default_factory=dict)

    def compute_aggregate_metrics(self) -> Dict[str, float]:
        """
        Compute aggregate metrics across all graphs.

        Returns:
            Dictionary with:
            - mean_predicted_weight: Mean tour weight from predicted anchors
            - mean_random_weight: Mean tour weight from random anchors
            - mean_best_weight: Mean tour weight from best anchors (if available)
            - beat_random_rate: Fraction of graphs where predicted beat random
            - mean_improvement_over_random: Mean improvement over random
            - mean_optimality_gap: Mean gap to best (if available)
            - success_rate: Fraction within X% of best (if available)
        """
        if not self.execution_results:
            return {}

        # Collect metrics from each result
        all_metrics = [r.compute_metrics() for r in self.execution_results]

        # Aggregate
        agg = {}

        # Mean tour weights
        agg['mean_predicted_weight'] = np.mean([r.predicted_tour_weight for r in self.execution_results])
        agg['mean_random_weight'] = np.mean([r.random_tour_weight for r in self.execution_results])

        if any(r.best_tour_weight is not None for r in self.execution_results):
            best_weights = [r.best_tour_weight for r in self.execution_results if r.best_tour_weight is not None]
            if best_weights:
                agg['mean_best_weight'] = np.mean(best_weights)

        # Beat random rate
        agg['beat_random_rate'] = np.mean([m['beat_random'] for m in all_metrics])

        # Mean improvement
        agg['mean_improvement_over_random'] = np.mean([m['improvement_over_random'] for m in all_metrics])

        # Mean optimality gap
        gaps = [m['optimality_gap'] for m in all_metrics if m['optimality_gap'] is not None]
        if gaps:
            agg['mean_optimality_gap'] = np.mean(gaps)
            agg['mean_gap_to_best_pct'] = np.mean(gaps) * 100.0

        # Success rate (within 10% of best)
        if gaps:
            success_count = sum(1 for g in gaps if g <= 0.10)
            agg['success_rate_10pct'] = success_count / len(gaps)

            success_count_15 = sum(1 for g in gaps if g <= 0.15)
            agg['success_rate_15pct'] = success_count_15 / len(gaps)

        self.aggregate_metrics = agg
        return agg

    def get_summary(self) -> str:
        """Get human-readable summary."""
        if not self.aggregate_metrics:
            self.compute_aggregate_metrics()

        summary = []
        summary.append(f"Batch Prediction Results ({len(self.graph_ids)} graphs):")

        metrics = self.aggregate_metrics

        if 'mean_predicted_weight' in metrics:
            summary.append(f"  Mean predicted tour weight: {metrics['mean_predicted_weight']:.2f}")

        if 'mean_random_weight' in metrics:
            summary.append(f"  Mean random tour weight: {metrics['mean_random_weight']:.2f}")

        if 'mean_best_weight' in metrics:
            summary.append(f"  Mean best tour weight: {metrics['mean_best_weight']:.2f}")

        if 'beat_random_rate' in metrics:
            summary.append(f"  Beat random rate: {metrics['beat_random_rate']:.1%}")

        if 'mean_improvement_over_random' in metrics:
            summary.append(f"  Mean improvement over random: {metrics['mean_improvement_over_random']:.1%}")

        if 'mean_gap_to_best_pct' in metrics:
            summary.append(f"  Mean gap to best: {metrics['mean_gap_to_best_pct']:.1f}%")

        if 'success_rate_10pct' in metrics:
            summary.append(f"  Success rate (within 10% of best): {metrics['success_rate_10pct']:.1%}")

        if 'success_rate_15pct' in metrics:
            summary.append(f"  Success rate (within 15% of best): {metrics['success_rate_15pct']:.1%}")

        return "\n".join(summary)


class MLPipeline:
    """
    End-to-end ML prediction pipeline.

    Pipeline:
    1. Graph → feature extraction
    2. Features → anchor quality prediction
    3. Select best predicted anchor
    4. Run TSP algorithm
    5. Compare to baselines
    """

    def __init__(
        self,
        model: Any,
        feature_pipeline: Optional[Any] = None,
        scaler: Optional[Any] = None
    ):
        """
        Initialize ML pipeline.

        Args:
            model: Trained ML model for anchor prediction
            feature_pipeline: FeatureExtractorPipeline for extracting features
            scaler: Feature scaler (fitted on training data)
        """
        self.model = model
        self.feature_pipeline = feature_pipeline
        self.scaler = scaler

    def predict_best_anchor(
        self,
        graph: np.ndarray,
        graph_id: str = "graph"
    ) -> PredictionResult:
        """
        Predict best anchor for a graph.

        Args:
            graph: Adjacency matrix (weighted, complete)
            graph_id: Identifier for the graph

        Returns:
            PredictionResult
        """
        n = len(graph)

        # Extract features for all vertices
        if self.feature_pipeline is not None:
            features_df = self.feature_pipeline.extract_all_features(graph)
        else:
            # Dummy features if no pipeline provided
            features_df = pd.DataFrame(
                np.random.randn(n, 5),
                columns=[f'feat_{i}' for i in range(5)]
            )

        # Scale features
        if self.scaler is not None:
            features_scaled = self.scaler.transform(features_df)
        else:
            features_scaled = features_df

        # Predict anchor quality for all vertices
        predictions = self.model.predict(features_scaled)

        # Select vertex with highest predicted quality
        best_vertex = int(np.argmax(predictions))
        best_quality = float(predictions[best_vertex])

        return PredictionResult(
            graph_id=graph_id,
            predicted_best_vertex=best_vertex,
            predicted_quality=best_quality,
            all_predictions=predictions,
            feature_values=features_df
        )

    def execute_algorithm(
        self,
        graph: np.ndarray,
        algorithm: Any,
        predicted_anchor: int,
        compare_to_random: bool = True,
        compare_to_best: bool = False,
        random_seed: Optional[int] = None,
        graph_id: str = "graph"
    ) -> AlgorithmExecutionResult:
        """
        Execute TSP algorithm with predicted anchor and baselines.

        Args:
            graph: Adjacency matrix
            algorithm: TSP algorithm instance (must support start_vertex parameter)
            predicted_anchor: Vertex to use as anchor (from ML prediction)
            compare_to_random: Whether to compare to random anchor
            compare_to_best: Whether to compare to best anchor (expensive!)
            random_seed: Random seed for random anchor selection
            graph_id: Identifier for the graph

        Returns:
            AlgorithmExecutionResult
        """
        n = len(graph)

        # Run algorithm with predicted anchor
        try:
            # Assume algorithm supports start_vertex parameter
            result_predicted = algorithm.solve(graph, start_vertex=predicted_anchor)
            predicted_weight = result_predicted.quality
        except Exception as e:
            predicted_weight = float('inf')

        # Compare to random anchor
        random_anchor = None
        random_weight = None

        if compare_to_random:
            if random_seed is not None:
                np.random.seed(random_seed)
            random_anchor = np.random.randint(0, n)

            try:
                result_random = algorithm.solve(graph, start_vertex=random_anchor)
                random_weight = result_random.quality
            except:
                random_weight = float('inf')

        # Compare to best anchor (exhaustive search)
        best_anchor = None
        best_weight = None

        if compare_to_best:
            best_weight = float('inf')
            best_anchor = 0

            for v in range(n):
                try:
                    result_v = algorithm.solve(graph, start_vertex=v)
                    if result_v.quality < best_weight:
                        best_weight = result_v.quality
                        best_anchor = v
                except:
                    pass

        return AlgorithmExecutionResult(
            graph_id=graph_id,
            predicted_anchor=predicted_anchor,
            predicted_tour_weight=predicted_weight,
            random_anchor=random_anchor if random_anchor is not None else -1,
            random_tour_weight=random_weight if random_weight is not None else 0.0,
            best_anchor=best_anchor,
            best_tour_weight=best_weight,
            metadata={
                'algorithm': algorithm.__class__.__name__ if hasattr(algorithm, '__class__') else 'unknown',
                'graph_size': n
            }
        )

    def predict_and_execute(
        self,
        graph: np.ndarray,
        algorithm: Any,
        compare_to_random: bool = True,
        compare_to_best: bool = False,
        random_seed: Optional[int] = None,
        graph_id: str = "graph"
    ) -> Tuple[PredictionResult, AlgorithmExecutionResult]:
        """
        End-to-end: predict best anchor and execute algorithm.

        Args:
            graph: Adjacency matrix
            algorithm: TSP algorithm
            compare_to_random: Compare to random anchor
            compare_to_best: Compare to best anchor (expensive!)
            random_seed: Random seed
            graph_id: Graph identifier

        Returns:
            (PredictionResult, AlgorithmExecutionResult)
        """
        # Step 1: Predict best anchor
        prediction = self.predict_best_anchor(graph, graph_id)

        # Step 2: Execute algorithm
        execution = self.execute_algorithm(
            graph=graph,
            algorithm=algorithm,
            predicted_anchor=prediction.predicted_best_vertex,
            compare_to_random=compare_to_random,
            compare_to_best=compare_to_best,
            random_seed=random_seed,
            graph_id=graph_id
        )

        return prediction, execution

    def batch_predict_and_execute(
        self,
        graphs: List[np.ndarray],
        graph_ids: List[str],
        algorithm: Any,
        compare_to_random: bool = True,
        compare_to_best: bool = False,
        random_seed: Optional[int] = None
    ) -> BatchPredictionResult:
        """
        Batch prediction and execution on multiple graphs.

        Args:
            graphs: List of adjacency matrices
            graph_ids: List of graph identifiers
            algorithm: TSP algorithm
            compare_to_random: Compare to random anchors
            compare_to_best: Compare to best anchors (expensive!)
            random_seed: Random seed

        Returns:
            BatchPredictionResult
        """
        execution_results = []

        for i, (graph, graph_id) in enumerate(zip(graphs, graph_ids)):
            # Use different random seed for each graph if provided
            seed = random_seed + i if random_seed is not None else None

            _, execution = self.predict_and_execute(
                graph=graph,
                algorithm=algorithm,
                compare_to_random=compare_to_random,
                compare_to_best=compare_to_best,
                random_seed=seed,
                graph_id=graph_id
            )

            execution_results.append(execution)

        # Create batch result
        batch_result = BatchPredictionResult(
            graph_ids=graph_ids,
            execution_results=execution_results
        )

        # Compute aggregate metrics
        batch_result.compute_aggregate_metrics()

        return batch_result


class ErrorAnalyzer:
    """
    Analyze prediction errors to identify patterns and failure modes.

    Useful for understanding:
    - Which graph types cause problems
    - Feature patterns correlated with errors
    - When to trust predictions
    """

    @staticmethod
    def analyze_errors(
        predictions: np.ndarray,
        actual_qualities: np.ndarray,
        feature_matrix: pd.DataFrame,
        graph_types: Optional[np.ndarray] = None,
        graph_sizes: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Analyze prediction errors.

        Args:
            predictions: Predicted anchor qualities
            actual_qualities: Actual anchor qualities
            feature_matrix: Feature values for each vertex
            graph_types: Graph type for each vertex (if available)
            graph_sizes: Graph size for each vertex (if available)

        Returns:
            Dictionary with error analysis:
            - mean_error: Mean prediction error
            - median_error: Median prediction error
            - std_error: Std dev of errors
            - overestimate_rate: Fraction where prediction > actual
            - error_by_graph_type: Errors broken down by graph type
            - error_by_size: Errors broken down by graph size
            - features_correlated_with_error: Features most correlated with error
        """
        errors = predictions - actual_qualities

        analysis = {
            'mean_error': float(np.mean(errors)),
            'median_error': float(np.median(errors)),
            'std_error': float(np.std(errors)),
            'mae': float(np.mean(np.abs(errors))),
            'rmse': float(np.sqrt(np.mean(errors ** 2))),
            'overestimate_rate': float(np.mean(errors > 0))
        }

        # Error by graph type
        if graph_types is not None:
            error_by_type = {}
            for gt in np.unique(graph_types):
                mask = graph_types == gt
                error_by_type[gt] = {
                    'mean_error': float(np.mean(errors[mask])),
                    'mae': float(np.mean(np.abs(errors[mask]))),
                    'count': int(np.sum(mask))
                }
            analysis['error_by_graph_type'] = error_by_type

        # Error by size
        if graph_sizes is not None:
            # Bin sizes
            size_bins = [0, 50, 100, 200, 1000]
            error_by_size = {}

            for i in range(len(size_bins) - 1):
                low, high = size_bins[i], size_bins[i + 1]
                mask = (graph_sizes > low) & (graph_sizes <= high)

                if np.sum(mask) > 0:
                    bin_label = f"{low+1}-{high}"
                    error_by_size[bin_label] = {
                        'mean_error': float(np.mean(errors[mask])),
                        'mae': float(np.mean(np.abs(errors[mask]))),
                        'count': int(np.sum(mask))
                    }

            analysis['error_by_size'] = error_by_size

        # Features correlated with error
        correlations = {}
        for col in feature_matrix.columns:
            try:
                corr = np.corrcoef(feature_matrix[col], np.abs(errors))[0, 1]
                correlations[col] = float(corr) if not np.isnan(corr) else 0.0
            except:
                correlations[col] = 0.0

        # Top correlated features
        sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
        analysis['features_correlated_with_error'] = dict(sorted_corr[:10])

        return analysis
