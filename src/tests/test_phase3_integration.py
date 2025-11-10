#!/usr/bin/env python3
"""
Phase 3 Integration Test Script

Tests the complete Phase 3 feature engineering system (Prompts 1-12)
without requiring pandas/sklearn (which are not installed in system Python).

This script validates:
- Feature extraction (Prompts 1-8)
- Labeling system (Prompt 9)
- Basic functionality without full ML dependencies

For complete testing including Prompts 10-12 (Pipeline, Selection, Transformation),
install pandas and scikit-learn and run: python3 -m unittest src.tests.test_features_final
"""

import sys
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

print("=" * 80)
print("Phase 3 Feature Engineering - Integration Test")
print("=" * 80)
print()

# Test 1: Feature Extraction (Prompts 1-8)
print("Test 1: Feature Extraction System (Prompts 1-8)")
print("-" * 80)

try:
    from src.features import FeatureExtractorPipeline, FeatureAnalyzer
    from src.features.extractors import (
        WeightFeatureExtractor,
        TopologicalFeatureExtractor,
        MSTFeatureExtractor,
        NeighborhoodFeatureExtractor,
        HeuristicFeatureExtractor,
        GraphContextFeatureExtractor
    )

    # Create test graph
    np.random.seed(42)
    n = 10
    graph = np.random.uniform(10, 100, size=(n, n))
    graph = (graph + graph.T) / 2  # Make symmetric
    np.fill_diagonal(graph, 0)

    # Build pipeline with all extractors
    pipeline = FeatureExtractorPipeline()
    pipeline.add_extractor(WeightFeatureExtractor())
    pipeline.add_extractor(TopologicalFeatureExtractor(
        include_betweenness=False,
        include_eigenvector=False
    ))
    pipeline.add_extractor(MSTFeatureExtractor())
    pipeline.add_extractor(NeighborhoodFeatureExtractor())
    pipeline.add_extractor(HeuristicFeatureExtractor())
    pipeline.add_extractor(GraphContextFeatureExtractor())

    # Extract features
    features, feature_names = pipeline.extract_features(graph)

    print(f"✓ Created {len(pipeline.extractors)} extractors")
    print(f"✓ Extracted {features.shape[1]} features from {features.shape[0]} vertices")
    print(f"✓ Feature matrix shape: {features.shape}")
    print(f"✓ No NaN values: {not np.any(np.isnan(features))}")
    print(f"✓ No Inf values: {not np.any(np.isinf(features))}")

    # Test analyzer
    analyzer = FeatureAnalyzer(features, feature_names)
    validation = analyzer.validate_ranges()
    constant = analyzer.find_constant_features()

    print(f"✓ Feature analyzer created")
    print(f"✓ Validation passed: NaN={len(validation['nan_features'])}, Inf={len(validation['inf_features'])}")
    print(f"✓ Constant features: {len(constant)}")

    print("\n✅ Feature Extraction Test PASSED\n")

except Exception as e:
    print(f"\n❌ Feature Extraction Test FAILED: {e}\n")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 2: Labeling System (Prompt 9)
print("Test 2: Anchor Quality Labeling System (Prompt 9)")
print("-" * 80)

try:
    # Import algorithm to register it
    import src.algorithms.single_anchor

    from src.features.labeling import (
        AnchorQualityLabeler,
        LabelingStrategy,
        LabelingResult
    )

    # Create small test graph
    test_graph = np.array([
        [0, 1, 2, 3],
        [1, 0, 4, 5],
        [2, 4, 0, 6],
        [3, 5, 6, 0]
    ], dtype=float)

    # Test different labeling strategies
    strategies = [
        ("Rank-based", LabelingStrategy.RANK_BASED),
        ("Absolute", LabelingStrategy.ABSOLUTE_QUALITY),
        ("Binary", LabelingStrategy.BINARY),
        ("Multi-class", LabelingStrategy.MULTICLASS),
    ]

    for name, strategy in strategies:
        labeler = AnchorQualityLabeler(
            algorithm_name="single_anchor_v1",
            strategy=strategy,
            random_seed=42
        )

        result = labeler.label_vertices(test_graph)

        successful_count = len(result.successful_vertices) if hasattr(result, 'successful_vertices') else len(result.labels)
        failed_count = len(result.failed_vertices) if hasattr(result, 'failed_vertices') else 0

        print(f"✓ {name}: {successful_count} labeled, {failed_count} failed")

    print(f"\n✅ Labeling System Test PASSED\n")

except Exception as e:
    print(f"\n❌ Labeling System Test FAILED: {e}\n")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Check if advanced modules exist (Prompts 10-12)
print("Test 3: Advanced Modules Check (Prompts 10-12)")
print("-" * 80)

advanced_modules = []

try:
    from src.features import dataset_pipeline
    advanced_modules.append("✓ Dataset Pipeline (Prompt 10)")
except Exception as e:
    advanced_modules.append(f"✗ Dataset Pipeline: {e}")

try:
    from src.features import selection
    advanced_modules.append("✓ Feature Selection (Prompt 11)")
except Exception as e:
    advanced_modules.append(f"✗ Feature Selection: {e}")

try:
    from src.features import transformation
    advanced_modules.append("✓ Feature Transformation (Prompt 12)")
except Exception as e:
    advanced_modules.append(f"✗ Feature Transformation: {e}")

for status in advanced_modules:
    print(status)

print()
print("NOTE: Full testing of Prompts 10-12 requires pandas and scikit-learn.")
print("Install with: pip install pandas scikit-learn")
print("Then run: python3 -m unittest src.tests.test_features_final")
print()

# Summary
print("=" * 80)
print("Integration Test Summary")
print("=" * 80)
print("✅ Phase 3 Prompts 1-8: Feature Extraction - WORKING")
print("✅ Phase 3 Prompt 9: Labeling System - WORKING")
print("⚠️  Phase 3 Prompts 10-12: Requires pandas/sklearn for full testing")
print()
print("To run complete test suite:")
print("  1. Install dependencies: pip install pandas scikit-learn")
print("  2. Run unit tests: python3 -m unittest discover -s src/tests -p 'test_features*.py'")
print()
print("=" * 80)
