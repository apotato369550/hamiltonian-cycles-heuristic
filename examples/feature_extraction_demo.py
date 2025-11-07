"""
Feature Extraction Demo

Demonstrates the Phase 3 feature extraction system on a sample graph.
Shows how to:
1. Create a feature extraction pipeline
2. Add multiple extractors
3. Extract features from a graph
4. Inspect results
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from src.features import FeatureExtractorPipeline
from src.features.extractors import (
    WeightFeatureExtractor,
    TopologicalFeatureExtractor,
    MSTFeatureExtractor
)


def create_sample_graph(n=10, seed=42):
    """Create a small random symmetric graph for demonstration."""
    np.random.seed(seed)

    # Generate random symmetric matrix
    graph = np.random.uniform(1, 100, size=(n, n))
    graph = (graph + graph.T) / 2  # Make symmetric
    np.fill_diagonal(graph, 0)  # Zero diagonal

    return graph


def main():
    print("=" * 70)
    print("Feature Extraction System Demo - Phase 3")
    print("=" * 70)
    print()

    # Create sample graph
    print("Step 1: Creating a 10-vertex random symmetric graph...")
    graph = create_sample_graph(n=10)
    print(f"   Graph shape: {graph.shape}")
    print(f"   Weight range: [{graph[graph > 0].min():.2f}, {graph.max():.2f}]")
    print()

    # Create pipeline
    print("Step 2: Building feature extraction pipeline...")
    pipeline = FeatureExtractorPipeline()

    # Add extractors
    print("   Adding WeightFeatureExtractor...")
    pipeline.add_extractor(WeightFeatureExtractor())

    print("   Adding TopologicalFeatureExtractor (without expensive features)...")
    pipeline.add_extractor(TopologicalFeatureExtractor(
        include_betweenness=False,  # Disable for demo speed
        include_eigenvector=False
    ))

    print("   Adding MSTFeatureExtractor...")
    pipeline.add_extractor(MSTFeatureExtractor())

    print(f"   Total extractors: {len(pipeline.extractors)}")
    print(f"   Total features: {pipeline.get_feature_count()}")
    print()

    # Extract features
    print("Step 3: Extracting features...")
    features, feature_names = pipeline.extract_features(graph)

    print(f"   Feature matrix shape: {features.shape}")
    print(f"   Number of vertices: {features.shape[0]}")
    print(f"   Number of features per vertex: {features.shape[1]}")
    print()

    # Display sample features
    print("Step 4: Sample features for vertex 0:")
    print("-" * 70)

    # Show first 10 features
    for i in range(min(10, len(feature_names))):
        name = feature_names[i]
        value = features[0, i]
        print(f"   {name:50s} = {value:10.4f}")

    if len(feature_names) > 10:
        print(f"   ... and {len(feature_names) - 10} more features")

    print()

    # Show feature statistics
    print("Step 5: Feature statistics across all vertices:")
    print("-" * 70)

    # Find features with highest variance (most informative)
    variances = np.var(features, axis=0)
    top_variance_idx = np.argsort(variances)[-5:][::-1]

    print("   Top 5 features by variance:")
    for idx in top_variance_idx:
        name = feature_names[idx]
        mean_val = np.mean(features[:, idx])
        std_val = np.std(features[:, idx])
        print(f"   {name:50s} mean={mean_val:8.2f} std={std_val:8.2f}")

    print()

    # Feature breakdown by extractor
    print("Step 6: Features by extractor:")
    print("-" * 70)

    weight_features = [n for n in feature_names if n.startswith('weight_based.')]
    topo_features = [n for n in feature_names if n.startswith('topological.')]
    mst_features = [n for n in feature_names if n.startswith('mst_based.')]

    print(f"   Weight-based features: {len(weight_features)}")
    print(f"   Topological features:  {len(topo_features)}")
    print(f"   MST-based features:    {len(mst_features)}")
    print(f"   Total:                 {len(feature_names)}")
    print()

    # Validation
    print("Step 7: Feature validation:")
    print("-" * 70)

    has_nan = np.any(np.isnan(features))
    has_inf = np.any(np.isinf(features))

    print(f"   Contains NaN values:      {has_nan}")
    print(f"   Contains infinite values: {has_inf}")
    print(f"   All features valid:       {not has_nan and not has_inf}")
    print()

    print("=" * 70)
    print("Demo complete! Feature extraction system is working correctly.")
    print("=" * 70)


if __name__ == '__main__':
    main()
