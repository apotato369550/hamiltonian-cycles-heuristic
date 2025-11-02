"""
Graph collection analysis tools.

Provides utilities for analyzing collections of generated graphs
to understand diversity, coverage, and detect anomalies.
"""

import json
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from .graph_instance import GraphInstance
from .storage import GraphStorage


class CollectionAnalyzer:
    """
    Analyzer for graph collections.

    Provides summary statistics, coverage metrics, and diversity analysis
    for collections of generated graphs.
    """

    def __init__(self, storage: Optional[GraphStorage] = None):
        """
        Initialize the analyzer.

        Args:
            storage: GraphStorage instance (creates default if None)
        """
        self.storage = storage or GraphStorage()

    def analyze_collection(
        self,
        graphs: Optional[List[GraphInstance]] = None,
        batch_name: Optional[str] = None,
        subdirectory: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze a collection of graphs.

        Args:
            graphs: List of graphs to analyze (loads from storage if None)
            batch_name: Batch name to load (if graphs is None)
            subdirectory: Subdirectory to search (if graphs is None)

        Returns:
            Analysis results dictionary
        """
        # Load graphs if not provided
        if graphs is None:
            if batch_name:
                graphs = self.storage.load_batch(batch_name)
            elif subdirectory:
                graphs = self.storage.find_graphs(subdirectory=subdirectory)
            else:
                # Load all graphs
                graphs = self.storage.find_graphs()

        if not graphs:
            return {'error': 'No graphs found'}

        analysis = {
            'summary': self._compute_summary(graphs),
            'coverage': self._compute_coverage(graphs),
            'property_distribution': self._compute_property_distribution(graphs),
            'diversity_metrics': self._compute_diversity_metrics(graphs),
            'outliers': self._detect_outliers(graphs)
        }

        return analysis

    def _compute_summary(self, graphs: List[GraphInstance]) -> Dict[str, Any]:
        """Compute basic summary statistics."""
        sizes = [g.metadata.size for g in graphs]
        types = [g.metadata.graph_type for g in graphs]

        return {
            'total_graphs': len(graphs),
            'unique_types': len(set(types)),
            'unique_sizes': len(set(sizes)),
            'size_range': (min(sizes), max(sizes)),
            'avg_size': sum(sizes) / len(sizes)
        }

    def _compute_coverage(self, graphs: List[GraphInstance]) -> Dict[str, Any]:
        """Compute coverage metrics across type/size combinations."""
        # Count instances per (type, size) combination
        coverage_matrix = defaultdict(lambda: defaultdict(int))

        for graph in graphs:
            coverage_matrix[graph.metadata.graph_type][graph.metadata.size] += 1

        # Convert to regular dict
        coverage = {
            graph_type: dict(sizes)
            for graph_type, sizes in coverage_matrix.items()
        }

        # Calculate coverage statistics
        all_counts = [count for sizes in coverage.values() for count in sizes.values()]

        return {
            'by_type_and_size': coverage,
            'total_combinations': len(all_counts),
            'min_instances': min(all_counts) if all_counts else 0,
            'max_instances': max(all_counts) if all_counts else 0,
            'avg_instances': sum(all_counts) / len(all_counts) if all_counts else 0
        }

    def _compute_property_distribution(self, graphs: List[GraphInstance]) -> Dict[str, Any]:
        """Compute distribution of graph properties."""
        metric_graphs = sum(1 for g in graphs if g.properties.is_metric)
        symmetric_graphs = sum(1 for g in graphs if g.properties.is_symmetric)

        # Collect metricity scores
        metricity_scores = [
            g.properties.metricity_score
            for g in graphs
            if g.properties.metricity_score is not None
        ]

        # Collect weight statistics
        weight_means = [g.properties.weight_mean for g in graphs]
        weight_stds = [g.properties.weight_std for g in graphs]
        weight_ranges = [g.properties.weight_range for g in graphs]

        return {
            'metric_count': metric_graphs,
            'metric_percentage': metric_graphs / len(graphs) * 100,
            'symmetric_count': symmetric_graphs,
            'symmetric_percentage': symmetric_graphs / len(graphs) * 100,
            'metricity_scores': {
                'count': len(metricity_scores),
                'mean': np.mean(metricity_scores) if metricity_scores else None,
                'std': np.std(metricity_scores) if metricity_scores else None,
                'min': min(metricity_scores) if metricity_scores else None,
                'max': max(metricity_scores) if metricity_scores else None
            },
            'weight_statistics': {
                'mean_of_means': np.mean(weight_means),
                'mean_of_stds': np.mean(weight_stds),
                'range_mins': [r[0] for r in weight_ranges],
                'range_maxs': [r[1] for r in weight_ranges]
            }
        }

    def _compute_diversity_metrics(self, graphs: List[GraphInstance]) -> Dict[str, Any]:
        """
        Compute diversity metrics for the collection.

        Measures how diverse the collection is in terms of properties.
        """
        # Extract features for diversity calculation
        features = []
        for graph in graphs:
            feature_vector = [
                graph.metadata.size,
                graph.properties.weight_mean,
                graph.properties.weight_std,
                graph.properties.weight_range[0],
                graph.properties.weight_range[1],
                1.0 if graph.properties.is_metric else 0.0,
                graph.properties.metricity_score if graph.properties.metricity_score else 0.0
            ]
            features.append(feature_vector)

        features = np.array(features)

        # Normalize features
        feature_stds = np.std(features, axis=0)
        feature_means = np.mean(features, axis=0)

        # Avoid division by zero
        feature_stds = np.where(feature_stds == 0, 1, feature_stds)

        normalized_features = (features - feature_means) / feature_stds

        # Calculate pairwise distances (sample if too many)
        max_pairs = 10000
        n = len(graphs)

        if n * (n - 1) // 2 > max_pairs:
            # Sample pairs
            distances = []
            for _ in range(max_pairs):
                i, j = np.random.choice(n, 2, replace=False)
                dist = np.linalg.norm(normalized_features[i] - normalized_features[j])
                distances.append(dist)
        else:
            # Calculate all pairs
            distances = []
            for i in range(n):
                for j in range(i + 1, n):
                    dist = np.linalg.norm(normalized_features[i] - normalized_features[j])
                    distances.append(dist)

        return {
            'avg_pairwise_distance': np.mean(distances),
            'std_pairwise_distance': np.std(distances),
            'min_pairwise_distance': min(distances),
            'max_pairwise_distance': max(distances),
            'diversity_score': np.mean(distances)  # Higher = more diverse
        }

    def _detect_outliers(self, graphs: List[GraphInstance]) -> Dict[str, List[str]]:
        """
        Detect outlier graphs based on properties.

        Returns:
            Dictionary of outlier categories with graph IDs
        """
        outliers = {
            'unusual_metricity': [],
            'unusual_weight_distribution': [],
            'extreme_size': []
        }

        # Collect statistics
        sizes = [g.metadata.size for g in graphs]
        weight_means = [g.properties.weight_mean for g in graphs]
        weight_stds = [g.properties.weight_std for g in graphs]

        size_mean = np.mean(sizes)
        size_std = np.std(sizes)
        mean_mean = np.mean(weight_means)
        mean_std = np.std(weight_means)
        std_mean = np.mean(weight_stds)
        std_std = np.std(weight_stds)

        for graph in graphs:
            # Check for unusual metricity
            if (graph.properties.metricity_score is not None and
                0.4 < graph.properties.metricity_score < 0.6):
                # Borderline metric - unusual
                outliers['unusual_metricity'].append(graph.id)

            # Check for unusual weight distribution
            if size_std > 0:
                z_score_mean = abs(graph.properties.weight_mean - mean_mean) / (mean_std + 1e-9)
                z_score_std = abs(graph.properties.weight_std - std_mean) / (std_std + 1e-9)

                if z_score_mean > 3 or z_score_std > 3:
                    outliers['unusual_weight_distribution'].append(graph.id)

            # Check for extreme size
            if size_std > 0:
                z_score_size = abs(graph.metadata.size - size_mean) / (size_std + 1e-9)
                if z_score_size > 2.5:
                    outliers['extreme_size'].append(graph.id)

        return outliers

    def visualize_collection(
        self,
        graphs: List[GraphInstance],
        output_file: str = "collection_analysis.png"
    ) -> str:
        """
        Create comprehensive visualization of graph collection.

        Args:
            graphs: List of graphs to visualize
            output_file: Output filename

        Returns:
            Path to saved file
        """
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

        # 1. Coverage heatmap (type x size)
        ax1 = fig.add_subplot(gs[0, :2])
        self._plot_coverage_heatmap(graphs, ax1)

        # 2. Property distribution pie charts
        ax2 = fig.add_subplot(gs[0, 2])
        self._plot_property_pies(graphs, ax2)

        # 3. Size distribution
        ax3 = fig.add_subplot(gs[1, 0])
        self._plot_size_distribution(graphs, ax3)

        # 4. Metricity score distribution
        ax4 = fig.add_subplot(gs[1, 1])
        self._plot_metricity_distribution(graphs, ax4)

        # 5. Weight statistics scatter
        ax5 = fig.add_subplot(gs[1, 2])
        self._plot_weight_scatter(graphs, ax5)

        # 6. Type distribution bar chart
        ax6 = fig.add_subplot(gs[2, 0])
        self._plot_type_distribution(graphs, ax6)

        # 7. Weight range box plots
        ax7 = fig.add_subplot(gs[2, 1:])
        self._plot_weight_ranges(graphs, ax7)

        fig.suptitle(f"Graph Collection Analysis (n={len(graphs)})",
                     fontsize=16, fontweight='bold')

        output_path = Path(self.storage.base_directory) / output_file
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        return str(output_path)

    def _plot_coverage_heatmap(self, graphs: List[GraphInstance], ax):
        """Plot coverage heatmap of type x size combinations."""
        coverage = defaultdict(lambda: defaultdict(int))

        for graph in graphs:
            coverage[graph.metadata.graph_type][graph.metadata.size] += 1

        types = sorted(coverage.keys())
        sizes = sorted(set(s for type_sizes in coverage.values() for s in type_sizes.keys()))

        # Create matrix
        matrix = np.zeros((len(types), len(sizes)))
        for i, graph_type in enumerate(types):
            for j, size in enumerate(sizes):
                matrix[i, j] = coverage[graph_type].get(size, 0)

        im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto')
        ax.set_xticks(np.arange(len(sizes)))
        ax.set_yticks(np.arange(len(types)))
        ax.set_xticklabels(sizes)
        ax.set_yticklabels(types)
        ax.set_xlabel('Graph Size')
        ax.set_ylabel('Graph Type')
        ax.set_title('Coverage Heatmap')

        # Add text annotations
        for i in range(len(types)):
            for j in range(len(sizes)):
                text = ax.text(j, i, int(matrix[i, j]),
                             ha="center", va="center", color="black", fontsize=8)

        plt.colorbar(im, ax=ax, label='Instance Count')

    def _plot_property_pies(self, graphs: List[GraphInstance], ax):
        """Plot pie charts for graph properties."""
        metric_count = sum(1 for g in graphs if g.properties.is_metric)
        non_metric_count = len(graphs) - metric_count

        colors = ['#2ecc71', '#e74c3c']
        ax.pie([metric_count, non_metric_count], labels=['Metric', 'Non-Metric'],
               autopct='%1.1f%%', colors=colors, startangle=90)
        ax.set_title('Metric vs Non-Metric')

    def _plot_size_distribution(self, graphs: List[GraphInstance], ax):
        """Plot distribution of graph sizes."""
        sizes = [g.metadata.size for g in graphs]
        ax.hist(sizes, bins=20, edgecolor='black', alpha=0.7, color='steelblue')
        ax.set_xlabel('Graph Size (vertices)')
        ax.set_ylabel('Frequency')
        ax.set_title('Size Distribution')
        ax.grid(True, alpha=0.3)

    def _plot_metricity_distribution(self, graphs: List[GraphInstance], ax):
        """Plot distribution of metricity scores."""
        scores = [g.properties.metricity_score for g in graphs
                 if g.properties.metricity_score is not None]

        if scores:
            ax.hist(scores, bins=20, edgecolor='black', alpha=0.7, color='orange')
            ax.set_xlabel('Metricity Score')
            ax.set_ylabel('Frequency')
            ax.set_title('Metricity Score Distribution')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No metricity scores available',
                   ha='center', va='center', transform=ax.transAxes)

    def _plot_weight_scatter(self, graphs: List[GraphInstance], ax):
        """Plot scatter of weight mean vs std."""
        means = [g.properties.weight_mean for g in graphs]
        stds = [g.properties.weight_std for g in graphs]
        types = [g.metadata.graph_type for g in graphs]

        # Color by type
        type_colors = {}
        colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']
        for i, t in enumerate(set(types)):
            type_colors[t] = colors[i % len(colors)]

        point_colors = [type_colors[t] for t in types]

        ax.scatter(means, stds, c=point_colors, alpha=0.6, s=50)
        ax.set_xlabel('Weight Mean')
        ax.set_ylabel('Weight Std Dev')
        ax.set_title('Weight Statistics')
        ax.grid(True, alpha=0.3)

        # Add legend
        handles = [plt.Line2D([0], [0], marker='o', color='w',
                             markerfacecolor=color, markersize=8, label=graph_type)
                  for graph_type, color in type_colors.items()]
        ax.legend(handles=handles, loc='best', fontsize=8)

    def _plot_type_distribution(self, graphs: List[GraphInstance], ax):
        """Plot bar chart of graph types."""
        type_counts = defaultdict(int)
        for graph in graphs:
            type_counts[graph.metadata.graph_type] += 1

        types = list(type_counts.keys())
        counts = [type_counts[t] for t in types]

        ax.bar(types, counts, edgecolor='black', alpha=0.7, color='teal')
        ax.set_xlabel('Graph Type')
        ax.set_ylabel('Count')
        ax.set_title('Distribution by Type')
        ax.grid(True, alpha=0.3, axis='y')
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

    def _plot_weight_ranges(self, graphs: List[GraphInstance], ax):
        """Plot box plots of weight ranges by type."""
        type_weights = defaultdict(list)

        for graph in graphs:
            weights = [graph.adjacency_matrix[i][j]
                      for i in range(graph.metadata.size)
                      for j in range(i+1, graph.metadata.size)]
            type_weights[graph.metadata.graph_type].extend(weights)

        types = list(type_weights.keys())
        data = [type_weights[t] for t in types]

        ax.boxplot(data, labels=types)
        ax.set_ylabel('Edge Weight')
        ax.set_title('Weight Ranges by Type')
        ax.grid(True, alpha=0.3, axis='y')
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

    def export_analysis_report(
        self,
        graphs: List[GraphInstance],
        output_file: str = "analysis_report.json"
    ) -> str:
        """
        Export comprehensive analysis report to JSON.

        Args:
            graphs: List of graphs to analyze
            output_file: Output filename

        Returns:
            Path to saved file
        """
        analysis = self.analyze_collection(graphs=graphs)

        output_path = Path(self.storage.base_directory) / output_file
        with open(output_path, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)

        return str(output_path)


def analyze_collection(
    batch_name: Optional[str] = None,
    output_dir: str = "data/graphs",
    create_visualizations: bool = True
) -> Dict[str, Any]:
    """
    Convenience function to analyze a graph collection.

    Args:
        batch_name: Name of batch to analyze (analyzes all if None)
        output_dir: Base directory for graphs
        create_visualizations: Whether to create visualizations

    Returns:
        Analysis results
    """
    storage = GraphStorage(base_directory=output_dir)
    analyzer = CollectionAnalyzer(storage=storage)

    if batch_name:
        graphs = storage.load_batch(batch_name)
    else:
        graphs = storage.find_graphs()

    analysis = analyzer.analyze_collection(graphs=graphs)

    if create_visualizations and graphs:
        viz_file = analyzer.visualize_collection(graphs)
        analysis['visualization_file'] = viz_file

        report_file = analyzer.export_analysis_report(graphs)
        analysis['report_file'] = report_file

    return analysis
