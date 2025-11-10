"""
Feature Engineering Pipeline for ML-Ready Dataset Generation.

This module provides an end-to-end pipeline that transforms raw graphs into
ML-ready datasets combining features and labels. Supports batch processing,
caching, progress tracking, and resumable execution.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import pandas as pd
import time
import os
import json
import pickle
from pathlib import Path

from .labeling import AnchorQualityLabeler, LabelingStrategy
from .pipeline import FeatureExtractorPipeline
from .extractors import (
    WeightFeatureExtractor,
    TopologicalFeatureExtractor,
    MSTFeatureExtractor,
    NeighborhoodFeatureExtractor,
    HeuristicFeatureExtractor,
    GraphContextFeatureExtractor
)


@dataclass
class DatasetConfig:
    """
    Configuration for dataset generation pipeline.

    Attributes:
        labeling_strategy: How to assign quality scores
        algorithm_name: Algorithm to use for labeling
        feature_extractors: Which extractors to use (None = use defaults)
        cache_dir: Directory for caching intermediate results
        use_cache: Whether to use caching
        validate_features: Whether to validate features
        show_progress: Whether to show progress bars
        save_intermediate: Whether to save intermediate results
    """
    labeling_strategy: LabelingStrategy = LabelingStrategy.RANK_BASED
    algorithm_name: str = 'single_anchor'
    feature_extractors: Optional[List[str]] = None
    cache_dir: Optional[str] = None
    use_cache: bool = False
    validate_features: bool = True
    show_progress: bool = False
    save_intermediate: bool = False
    top_k_percent: float = 10.0  # For binary labeling
    random_seed: Optional[int] = None

    def __post_init__(self):
        """Set defaults for feature extractors."""
        if self.feature_extractors is None:
            # Use all extractors by default (lightweight configuration)
            self.feature_extractors = [
                'weight',
                'topological_minimal',  # Without betweenness (expensive)
                'mst',
                'neighborhood',
                'heuristic',
                'graph_context'
            ]


@dataclass
class DatasetResult:
    """
    Result of dataset generation.

    Attributes:
        features: DataFrame with features, labels, and metadata columns
        metadata: Summary statistics and processing information
    """
    features: pd.DataFrame
    metadata: Dict[str, Any]

    def save_csv(self, path: str):
        """Save dataset to CSV file."""
        self.features.to_csv(path, index=False)

    def save_pickle(self, path: str):
        """Save dataset to pickle file (preserves dtypes)."""
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load_pickle(path: str) -> 'DatasetResult':
        """Load dataset from pickle file."""
        with open(path, 'rb') as f:
            return pickle.load(f)

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics of the dataset."""
        df = self.features

        summary = {
            'num_vertices': len(df),
            'num_features': len([c for c in df.columns if c not in ['label', 'graph_id', 'vertex_id']]),
            'feature_columns': [c for c in df.columns if c not in ['label', 'graph_id', 'vertex_id']],
            'label_distribution': {
                'mean': float(df['label'].mean()),
                'std': float(df['label'].std()),
                'min': float(df['label'].min()),
                'max': float(df['label'].max())
            },
            'missing_values': int(df.isnull().sum().sum()),
            'graphs': sorted(df['graph_id'].unique().tolist()) if 'graph_id' in df.columns else []
        }

        return summary


class FeatureDatasetPipeline:
    """
    End-to-end pipeline for generating ML-ready datasets from graphs.

    Pipeline steps:
    1. Extract features from graph (all vertices)
    2. Run labeling algorithm to get quality scores
    3. Combine features + labels into DataFrame
    4. Add metadata columns (graph_id, vertex_id, etc.)
    5. Validate and clean
    6. Save/cache results

    Supports:
    - Batch processing of multiple graphs
    - Progress tracking
    - Caching and resumption
    - Graceful error handling
    """

    def __init__(self, config: DatasetConfig):
        """
        Initialize the pipeline.

        Args:
            config: Pipeline configuration
        """
        self.config = config

        # Initialize feature extraction pipeline
        self.feature_pipeline = self._build_feature_pipeline()

        # Initialize labeler
        self.labeler = AnchorQualityLabeler(
            strategy=config.labeling_strategy,
            algorithm_name=config.algorithm_name,
            top_k_percent=config.top_k_percent,
            random_seed=config.random_seed
        )

        # Setup caching
        if config.use_cache and config.cache_dir:
            Path(config.cache_dir).mkdir(parents=True, exist_ok=True)
            self._load_cache_index()
        else:
            self._cache_index = {}

    def _build_feature_pipeline(self) -> FeatureExtractorPipeline:
        """Build feature extraction pipeline from config."""
        pipeline = FeatureExtractorPipeline()

        for extractor_name in self.config.feature_extractors:
            if extractor_name == 'weight':
                pipeline.add_extractor(WeightFeatureExtractor(
                    include_asymmetric_features=True
                ))
            elif extractor_name == 'topological_minimal':
                pipeline.add_extractor(TopologicalFeatureExtractor(
                    include_betweenness=False,  # Expensive
                    include_eigenvector=True,
                    include_clustering=True
                ))
            elif extractor_name == 'topological_full':
                pipeline.add_extractor(TopologicalFeatureExtractor(
                    include_betweenness=True,
                    include_eigenvector=True,
                    include_clustering=True
                ))
            elif extractor_name == 'mst':
                pipeline.add_extractor(MSTFeatureExtractor())
            elif extractor_name == 'neighborhood':
                pipeline.add_extractor(NeighborhoodFeatureExtractor())
            elif extractor_name == 'heuristic':
                pipeline.add_extractor(HeuristicFeatureExtractor(
                    include_tour_estimates=True,
                    include_baseline_comparison=True
                ))
            elif extractor_name == 'graph_context':
                pipeline.add_extractor(GraphContextFeatureExtractor())
            else:
                raise ValueError(f"Unknown extractor: {extractor_name}")

        return pipeline

    def process_graph(
        self,
        graph: np.ndarray,
        graph_id: str,
        graph_metadata: Optional[Dict[str, Any]] = None,
        optimal_weight: Optional[float] = None
    ) -> DatasetResult:
        """
        Process a single graph to extract features and labels.

        Args:
            graph: NxN adjacency matrix
            graph_id: Unique identifier for this graph
            graph_metadata: Optional metadata (graph_type, size, etc.)
            optimal_weight: Known optimal tour weight (if available)

        Returns:
            DatasetResult with features and labels
        """
        # Check cache
        if self.config.use_cache and graph_id in self._cache_index:
            return self._load_from_cache(graph_id)

        start_time = time.time()
        n = len(graph)

        # Step 1: Extract features
        cache = {}
        features, feature_names = self.feature_pipeline.extract_features(graph, cache)

        # Step 2: Assign labels
        labeling_result = self.labeler.label_vertices(graph, optimal_weight)

        # Step 3: Combine into DataFrame
        df = pd.DataFrame(features, columns=feature_names)
        df['label'] = labeling_result.labels
        df['graph_id'] = graph_id
        df['vertex_id'] = range(n)

        # Add graph metadata columns
        if graph_metadata:
            for key, value in graph_metadata.items():
                df[f'graph_{key}'] = value

        # Step 4: Validate if requested
        if self.config.validate_features:
            self._validate_dataframe(df, feature_names)

        # Build metadata
        metadata = {
            'graph_id': graph_id,
            'num_vertices': n,
            'num_features': len(feature_names),
            'processing_time': time.time() - start_time,
            'labeling_metadata': labeling_result.metadata,
            'successful_labels': len(labeling_result.successful_vertices),
            'failed_labels': len(labeling_result.failed_vertices)
        }

        result = DatasetResult(features=df, metadata=metadata)

        # Cache if requested
        if self.config.use_cache:
            self._save_to_cache(graph_id, result)

        return result

    def process_batch(
        self,
        graphs_with_metadata: List[Tuple[np.ndarray, str, Dict[str, Any]]],
        optimal_weights: Optional[Dict[str, float]] = None
    ) -> DatasetResult:
        """
        Process a batch of graphs.

        Args:
            graphs_with_metadata: List of (graph, graph_id, metadata) tuples
            optimal_weights: Dict mapping graph_id to optimal weights

        Returns:
            Combined DatasetResult with all graphs
        """
        start_time = time.time()
        all_dataframes = []
        graphs_processed = 0
        vertices_processed = 0

        if self.config.show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(graphs_with_metadata, desc="Processing graphs")
            except ImportError:
                iterator = graphs_with_metadata
                print(f"Processing {len(graphs_with_metadata)} graphs...")
        else:
            iterator = graphs_with_metadata

        for graph, graph_id, metadata in iterator:
            try:
                # Get optimal weight if available
                opt_weight = optimal_weights.get(graph_id) if optimal_weights else None

                # Process graph
                result = self.process_graph(graph, graph_id, metadata, opt_weight)

                all_dataframes.append(result.features)
                graphs_processed += 1
                vertices_processed += len(graph)

            except Exception as e:
                print(f"Warning: Failed to process graph {graph_id}: {e}")
                continue

        # Combine all dataframes
        if all_dataframes:
            combined_df = pd.concat(all_dataframes, ignore_index=True)
        else:
            combined_df = pd.DataFrame()

        # Build metadata
        metadata = {
            'graphs_processed': graphs_processed,
            'vertices_processed': vertices_processed,
            'processing_time': time.time() - start_time,
            'config': {
                'labeling_strategy': self.config.labeling_strategy.value,
                'algorithm': self.config.algorithm_name,
                'extractors': self.config.feature_extractors
            }
        }

        return DatasetResult(features=combined_df, metadata=metadata)

    def _validate_dataframe(self, df: pd.DataFrame, feature_names: List[str]):
        """
        Validate the feature dataframe.

        Args:
            df: DataFrame to validate
            feature_names: Expected feature column names

        Raises:
            ValueError: If validation fails
        """
        # Check for NaN values in features (labels can be NaN for failed vertices)
        feature_cols = [c for c in feature_names if c in df.columns]
        if df[feature_cols].isnull().any().any():
            raise ValueError("NaN values found in features")

        # Check for infinite values
        numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns
        if np.isinf(df[numeric_cols].values).any():
            raise ValueError("Infinite values found in features")

    def _load_cache_index(self):
        """Load cache index from disk."""
        if not self.config.cache_dir:
            self._cache_index = {}
            return

        index_path = os.path.join(self.config.cache_dir, 'cache_index.json')
        if os.path.exists(index_path):
            with open(index_path, 'r') as f:
                self._cache_index = json.load(f)
        else:
            self._cache_index = {}

    def _save_cache_index(self):
        """Save cache index to disk."""
        if not self.config.cache_dir:
            return

        index_path = os.path.join(self.config.cache_dir, 'cache_index.json')
        with open(index_path, 'w') as f:
            json.dump(self._cache_index, f)

    def _save_to_cache(self, graph_id: str, result: DatasetResult):
        """Save result to cache."""
        if not self.config.cache_dir:
            return

        cache_file = os.path.join(self.config.cache_dir, f'{graph_id}.pkl')
        result.save_pickle(cache_file)

        self._cache_index[graph_id] = {
            'file': cache_file,
            'timestamp': time.time()
        }
        self._save_cache_index()

    def _load_from_cache(self, graph_id: str) -> DatasetResult:
        """Load result from cache."""
        cache_info = self._cache_index[graph_id]
        return DatasetResult.load_pickle(cache_info['file'])

    def clear_cache(self):
        """Clear all cached results."""
        if not self.config.cache_dir:
            return

        for graph_id, info in self._cache_index.items():
            if os.path.exists(info['file']):
                os.remove(info['file'])

        self._cache_index = {}
        self._save_cache_index()

    def __repr__(self) -> str:
        return (
            f"FeatureDatasetPipeline("
            f"strategy={self.config.labeling_strategy.value}, "
            f"extractors={len(self.config.feature_extractors)})"
        )
