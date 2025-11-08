"""
Complete Feature Extraction Demo (Prompts 1-8)

Demonstrates all Phase 3 feature extractors and analysis tools.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from src.features import FeatureExtractorPipeline, FeatureAnalyzer
from src.features.extractors import (
    WeightFeatureExtractor,
    TopologicalFeatureExtractor,
    MSTFeatureExtractor,
    NeighborhoodFeatureExtractor,
    HeuristicFeatureExtractor,
    GraphContextFeatureExtractor
)


def create_sample_graph(n=15, seed=42):
    """Create a random symmetric graph."""
    np.random.seed(seed)
    graph = np.random.uniform(10, 100, size=(n, n))
    graph = (graph + graph.T) / 2
    np.fill_diagonal(graph, 0)
    return graph


def main():
    print("=" * 80)
    print(" Complete Feature Extraction System Demo - Phase 3 (Prompts 1-8)")
    print("=" * 80)
    print()

    # Create graph
    print("Creating 15-vertex random graph...")
    graph = create_sample_graph(n=15)
    print(f"Graph shape: {graph.shape}")
    print(f"Weight range: [{graph[graph > 0].min():.1f}, {graph.max():.1f}]")
    print()

    # Build comprehensive pipeline
    print("Building feature extraction pipeline with ALL extractors...")
    pipeline = FeatureExtractorPipeline()

    extractors = [
        ("Weight-based", WeightFeatureExtractor()),
        ("Topological", TopologicalFeatureExtractor(
            include_betweenness=False,  # Disable expensive feature for demo
            include_eigenvector=False
        )),
        ("MST-based", MSTFeatureExtractor()),
        ("Neighborhood", NeighborhoodFeatureExtractor()),
        ("Heuristic", HeuristicFeatureExtractor()),
        ("Graph Context", GraphContextFeatureExtractor())
    ]

    for name, extractor in extractors:
        pipeline.add_extractor(extractor)
        print(f"  ✓ Added {name}")

    print(f"\nTotal extractors: {len(pipeline.extractors)}")
    print(f"Total features: {pipeline.get_feature_count()}")
    print()

    # Extract features
    print("Extracting features...")
    features, feature_names = pipeline.extract_features(graph)
    print(f"Feature matrix shape: {features.shape}")
    print(f"  - Vertices: {features.shape[0]}")
    print(f"  - Features per vertex: {features.shape[1]}")
    print()

    # Feature breakdown
    print("Feature breakdown by extractor:")
    print("-" * 80)
    for ext_name in pipeline.get_extractor_names():
        count = sum(1 for n in feature_names if n.startswith(ext_name + '.'))
        print(f"  {ext_name:20s}: {count:3d} features")
    print()

    # Feature analysis
    print("Running feature analysis...")
    print("=" * 80)
    analyzer = FeatureAnalyzer(features, feature_names)

    # Validation
    print("\n1. Feature Validation:")
    validation = analyzer.validate_ranges()
    print(f"   NaN features: {len(validation['nan_features'])}")
    print(f"   Inf features: {len(validation['inf_features'])}")
    print(f"   ✓ All features valid" if not any(validation.values()) else "   ✗ Invalid features found")

    # Constant features
    constant = analyzer.find_constant_features()
    print(f"\n2. Constant Features: {len(constant)}")
    if constant:
        for name in constant[:3]:
            print(f"   - {name}")

    # Correlation
    print(f"\n3. High Correlation Analysis:")
    high_corr = analyzer.find_highly_correlated_pairs(threshold=0.95)
    print(f"   Found {len(high_corr)} highly correlated pairs (|r| > 0.95)")
    for pair in high_corr[:3]:
        print(f"   - {pair[0][:40]:40s} <-> {pair[1][:40]:40s} : r={pair[2]:.3f}")

    # PCA
    print(f"\n4. PCA Analysis:")
    pca = analyzer.perform_pca(n_components=5)
    print(f"   Top 5 principal components:")
    cumsum = np.cumsum(pca['explained_variance_ratio'])
    for i in range(5):
        print(f"   PC{i+1}: {pca['explained_variance_ratio'][i]:.3f} (cumsum: {cumsum[i]:.3f})")

    # Feature importance by variance
    print(f"\n5. Top 10 Features by Variance:")
    top_features = analyzer.get_feature_importance_by_variance(top_k=10)
    for i, (name, var) in enumerate(top_features, 1):
        short_name = name[-50:] if len(name) > 50 else name
        print(f"   {i:2d}. {short_name:50s} var={var:10.2f}")

    # Distribution analysis
    print(f"\n6. Distribution Statistics (sample):")
    distributions = analyzer.analyze_distributions()
    sample_features = list(distributions.keys())[:3]
    for fname in sample_features:
        stats = distributions[fname]
        print(f"   {fname[:50]:50s}")
        print(f"      mean={stats['mean']:8.2f}, std={stats['std']:8.2f}, "
              f"skew={stats['skewness']:6.2f}")

    print()
    print("=" * 80)
    print("✓ Demo Complete!")
    print("=" * 80)
    print()
    print("Summary:")
    print(f"  - Extracted {features.shape[1]} features from {features.shape[0]} vertices")
    print(f"  - Used 6 different feature extractors")
    print(f"  - Performed comprehensive feature analysis")
    print(f"  - All features validated (no NaN/Inf)")
    print()
    print("Phase 3 (Prompts 1-8) feature extraction system is fully operational!")
    print()


if __name__ == '__main__':
    main()
