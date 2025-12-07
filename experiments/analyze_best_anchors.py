#!/usr/bin/env python3
"""
Best Anchor Analysis Script

Analyzes what makes a good anchor vertex using feature engineering.
Answers: "What structural properties predict anchor quality?"

Usage:
    python experiments/analyze_best_anchors.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict
from src.algorithms.registry import AlgorithmRegistry
from src.graph_generation.random_generator import RandomGraphGenerator
from src.features.pipeline import FeatureExtractorPipeline
from src.features.extractors.weight_based import WeightFeatureExtractor
from src.features.extractors.topological import TopologicalFeatureExtractor
from src.features.extractors.mst_based import MSTFeatureExtractor
from src.features.extractors.heuristic import HeuristicFeatureExtractor
from src.features.labeling import AnchorQualityLabeler, LabelingStrategy

# ===== CONFIGURATION =====
GRAPH_SIZES = [20, 50, 100]
GRAPHS_PER_SIZE = 5
SEED = 42
SHOW_PLOTS = True

# ===== MAIN ANALYSIS =====

def analyze_anchors():
    """Run anchor analysis: extract features, label quality, find correlations."""
    print("=" * 80)
    print("BEST ANCHOR ANALYSIS")
    print("=" * 80)

    # Setup feature extraction
    pipeline = FeatureExtractorPipeline()
    pipeline.add_extractor(WeightFeatureExtractor())
    pipeline.add_extractor(TopologicalFeatureExtractor())
    pipeline.add_extractor(MSTFeatureExtractor())
    pipeline.add_extractor(HeuristicFeatureExtractor())

    labeler = AnchorQualityLabeler(strategy=LabelingStrategy.RANK_BASED, algorithm_name='single_anchor_v2')

    all_features = []
    all_labels = []
    all_metadata = []

    # Generate graphs and analyze
    for size in GRAPH_SIZES:
        print(f"\n{'='*80}")
        print(f"Graph Size: {size}")
        print(f"{'='*80}")

        for i in range(GRAPHS_PER_SIZE):
            print(f"\nGraph {i+1}/{GRAPHS_PER_SIZE}...")

            # Generate graph
            gen = RandomGraphGenerator(random_seed=SEED + i)
            graph = gen.generate(num_vertices=size)
            graph = np.array(graph)  # Convert to numpy array

            # Extract features
            features, names = pipeline.extract_features(graph)

            # Label anchor quality
            result = labeler.label_vertices(graph)
            labels = result.labels

            # Store
            all_features.append(features)
            all_labels.append(labels)
            all_metadata.append({'size': size, 'graph_id': i})

            print(f"  Extracted {len(names)} features for {size} vertices")

    # Combine data
    X = np.vstack(all_features)
    y = np.concatenate(all_labels)

    print(f"\n{'='*80}")
    print(f"Dataset: {X.shape[0]} vertices, {X.shape[1]} features")
    print(f"{'='*80}")

    # Analyze feature importance
    correlations = compute_correlations(X, y, names)
    display_top_features(correlations, top_k=10)

    if SHOW_PLOTS:
        plot_feature_importance(correlations)
        plot_top_features_vs_quality(X, y, names, correlations, top_k=6)

    return X, y, names, correlations


def compute_correlations(X, y, feature_names):
    """Compute Pearson correlation between each feature and anchor quality."""
    correlations = []
    for i, name in enumerate(feature_names):
        # Handle constant features
        if np.std(X[:, i]) < 1e-10:
            corr = 0.0
        else:
            corr = np.corrcoef(X[:, i], y)[0, 1]
        correlations.append({'name': name, 'correlation': corr})

    # Sort by absolute correlation
    correlations.sort(key=lambda x: abs(x['correlation']), reverse=True)
    return correlations


def display_top_features(correlations, top_k=10):
    """Display top predictive features."""
    print(f"\nTop {top_k} Most Predictive Features:")
    print("-" * 80)
    print(f"{'Feature':<40} {'Correlation':>15}")
    print("-" * 80)

    for i, item in enumerate(correlations[:top_k], 1):
        print(f"{i:2}. {item['name']:<37} {item['correlation']:>12.4f}")


def plot_feature_importance(correlations, top_k=15):
    """Bar chart of top feature correlations."""
    top = correlations[:top_k]
    names = [c['name'][:30] for c in top]  # Truncate long names
    corrs = [c['correlation'] for c in top]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['green' if c > 0 else 'red' for c in corrs]
    ax.barh(range(len(names)), corrs, color=colors, alpha=0.7)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.set_xlabel('Correlation with Anchor Quality')
    ax.set_title('Top Feature Importance (Higher Rank = Better Anchor)')
    ax.axvline(0, color='black', linewidth=0.5)
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()

    output_path = os.path.join(os.path.dirname(__file__), 'feature_importance.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nFeature importance plot saved to: {output_path}")
    plt.show()


def plot_top_features_vs_quality(X, y, names, correlations, top_k=6):
    """Scatter plots of top features vs anchor quality."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    for i in range(top_k):
        feature_name = correlations[i]['name']
        feature_idx = names.index(feature_name)

        axes[i].scatter(X[:, feature_idx], y, alpha=0.5, s=10)
        axes[i].set_xlabel(feature_name[:40])
        axes[i].set_ylabel('Anchor Quality (Rank)')
        axes[i].set_title(f"r={correlations[i]['correlation']:.3f}")
        axes[i].grid(alpha=0.3)

    plt.tight_layout()

    output_path = os.path.join(os.path.dirname(__file__), 'top_features_scatter.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Feature scatter plots saved to: {output_path}")
    plt.show()


def main():
    """Main entry point."""
    X, y, names, correlations = analyze_anchors()

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE!")
    print("=" * 80)
    print("\nKey Insights:")
    print("- Features with positive correlation: prefer vertices with HIGHER values")
    print("- Features with negative correlation: prefer vertices with LOWER values")
    print("- Look for MST, weight statistics, and heuristic features in top rankings")


if __name__ == '__main__':
    main()
