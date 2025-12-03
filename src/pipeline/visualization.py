"""
Visualization tools for TSP experiments.

Provides publication-quality plots for algorithm comparison, feature importance,
model performance, and research insights.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List


class ExperimentVisualizer:
    """Create publication-quality visualizations."""

    def __init__(self, style: str = "publication"):
        """
        Initialize visualizer with specific style.

        Args:
            style: Visualization style ('publication', 'notebook', 'presentation')
        """
        if style == "publication":
            # High DPI, colorblind-friendly palette
            plt.rcParams['figure.dpi'] = 300
            plt.rcParams['savefig.dpi'] = 300
            plt.rcParams['font.size'] = 10
            plt.rcParams['axes.linewidth'] = 0.8
            plt.rcParams['grid.linewidth'] = 0.5
            sns.set_palette("colorblind")
            sns.set_style("whitegrid")
        elif style == "notebook":
            plt.rcParams['figure.dpi'] = 100
            sns.set_palette("deep")
            sns.set_style("darkgrid")
        elif style == "presentation":
            plt.rcParams['figure.dpi'] = 150
            plt.rcParams['font.size'] = 14
            sns.set_palette("bright")
            sns.set_style("whitegrid")

        self.style = style

    def plot_algorithm_comparison(
        self,
        results_df: pd.DataFrame,
        metric: str = "tour_weight",
        output_path: Optional[Path] = None
    ):
        """
        Box plot comparing algorithm performance.

        Args:
            results_df: DataFrame with 'algorithm' and metric columns
            metric: Column name to compare (default: 'tour_weight')
            output_path: Optional path to save figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        # Create box plot
        sns.boxplot(data=results_df, x='algorithm', y=metric, ax=ax)

        ax.set_xlabel("Algorithm", fontsize=12)
        ax.set_ylabel(f"{metric.replace('_', ' ').title()}", fontsize=12)
        ax.set_title("Algorithm Performance Comparison", fontsize=14, fontweight='bold')

        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def plot_feature_importance(
        self,
        importance_df: pd.DataFrame,
        top_k: int = 15,
        output_path: Optional[Path] = None
    ):
        """
        Horizontal bar chart of top-k feature importances.

        Args:
            importance_df: DataFrame with 'feature_name' and 'importance' columns
            top_k: Number of top features to display
            output_path: Optional path to save figure
        """
        top_features = importance_df.nlargest(top_k, 'importance')

        fig, ax = plt.subplots(figsize=(8, 10))

        # Create horizontal bar chart
        y_pos = np.arange(len(top_features))
        ax.barh(y_pos, top_features['importance'], align='center')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_features['feature_name'])
        ax.invert_yaxis()  # Top features at the top

        ax.set_xlabel("Importance", fontsize=12)
        ax.set_ylabel("Feature", fontsize=12)
        ax.set_title(f"Top {top_k} Feature Importances", fontsize=14, fontweight='bold')

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def plot_predicted_vs_actual(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
        output_path: Optional[Path] = None
    ):
        """
        Scatter plot: predicted vs actual anchor quality.

        Args:
            predictions: Predicted values
            actuals: Actual values
            output_path: Optional path to save figure
        """
        fig, ax = plt.subplots(figsize=(8, 8))

        # Scatter plot
        ax.scatter(actuals, predictions, alpha=0.5, s=20)

        # Diagonal line for perfect prediction
        min_val = min(actuals.min(), predictions.min())
        max_val = max(actuals.max(), predictions.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')

        ax.set_xlabel("Actual Anchor Quality", fontsize=12)
        ax.set_ylabel("Predicted Anchor Quality", fontsize=12)
        ax.set_title("Model Prediction Accuracy", fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def plot_performance_by_graph_type(
        self,
        results_df: pd.DataFrame,
        algorithms: List[str],
        output_path: Optional[Path] = None
    ):
        """
        Line plot: algorithm performance across graph types.

        Args:
            results_df: DataFrame with 'graph_type', 'algorithm', and 'tour_weight' columns
            algorithms: List of algorithms to plot
            output_path: Optional path to save figure
        """
        # Aggregate data
        grouped = results_df.groupby(['graph_type', 'algorithm'])['tour_weight'].mean().reset_index()

        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot line for each algorithm
        for algo in algorithms:
            algo_data = grouped[grouped['algorithm'] == algo]
            ax.plot(algo_data['graph_type'], algo_data['tour_weight'],
                    marker='o', linewidth=2, markersize=8, label=algo)

        ax.set_xlabel("Graph Type", fontsize=12)
        ax.set_ylabel("Mean Tour Weight", fontsize=12)
        ax.set_title("Performance by Graph Type", fontsize=14, fontweight='bold')
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def plot_interaction_heatmap(
        self,
        results_df: pd.DataFrame,
        output_path: Optional[Path] = None
    ):
        """
        Heatmap: algorithm × graph type interaction matrix.

        Args:
            results_df: DataFrame with 'algorithm', 'graph_type', and 'tour_weight' columns
            output_path: Optional path to save figure
        """
        # Create pivot table
        interaction_matrix = results_df.pivot_table(
            values='tour_weight',
            index='algorithm',
            columns='graph_type',
            aggfunc='mean'
        )

        fig, ax = plt.subplots(figsize=(10, 8))

        # Create heatmap
        sns.heatmap(interaction_matrix, annot=True, fmt='.2f', cmap='YlOrRd',
                    ax=ax, cbar_kws={'label': 'Mean Tour Weight'})

        ax.set_xlabel("Graph Type", fontsize=12)
        ax.set_ylabel("Algorithm", fontsize=12)
        ax.set_title("Algorithm × Graph Type Performance Matrix", fontsize=14, fontweight='bold')

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def create_summary_figure(
        self,
        experiment_dir: Path,
        output_path: Optional[Path] = None
    ):
        """
        Multi-panel figure with:
        - Algorithm comparison (top-left)
        - Feature importance (top-right)
        - Predicted vs actual (bottom-left)
        - Performance by type (bottom-right)

        Args:
            experiment_dir: Path to experiment directory
            output_path: Optional path to save figure
        """
        from pipeline.analysis import ExperimentAnalyzer

        analyzer = ExperimentAnalyzer(experiment_dir)
        results_df = analyzer.load_benchmark_results()
        eval_df = analyzer.load_evaluation_results()

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Top-left: Algorithm comparison
        if results_df is not None:
            sns.boxplot(data=results_df, x='algorithm', y='tour_weight', ax=axes[0, 0])
            axes[0, 0].set_title("Algorithm Performance Comparison", fontsize=12, fontweight='bold')
            axes[0, 0].set_xlabel("Algorithm")
            axes[0, 0].set_ylabel("Tour Weight")
            axes[0, 0].tick_params(axis='x', rotation=45)

        # Top-right: Feature importance (placeholder if not available)
        model_files = list((experiment_dir / 'models').glob('*.pkl')) if (experiment_dir / 'models').exists() else []
        if model_files:
            try:
                importance_df = analyzer.analyze_feature_importance(model_files[0])
                top_features = importance_df.head(10)
                axes[0, 1].barh(range(len(top_features)), top_features['importance'])
                axes[0, 1].set_yticks(range(len(top_features)))
                axes[0, 1].set_yticklabels(top_features['feature_name'], fontsize=8)
                axes[0, 1].invert_yaxis()
                axes[0, 1].set_title("Top 10 Feature Importances", fontsize=12, fontweight='bold')
                axes[0, 1].set_xlabel("Importance")
            except:
                axes[0, 1].text(0.5, 0.5, "Feature importance\nnot available",
                               ha='center', va='center', fontsize=10)
                axes[0, 1].set_title("Feature Importance", fontsize=12, fontweight='bold')
        else:
            axes[0, 1].text(0.5, 0.5, "No model found", ha='center', va='center', fontsize=10)
            axes[0, 1].set_title("Feature Importance", fontsize=12, fontweight='bold')

        # Bottom-left: Predicted vs actual (if evaluation exists)
        if eval_df is not None and 'predicted_anchor_tour' in eval_df.columns:
            axes[1, 0].scatter(eval_df['random_anchor_tour'], eval_df['predicted_anchor_tour'], alpha=0.5)
            min_val = min(eval_df['random_anchor_tour'].min(), eval_df['predicted_anchor_tour'].min())
            max_val = max(eval_df['random_anchor_tour'].max(), eval_df['predicted_anchor_tour'].max())
            axes[1, 0].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
            axes[1, 0].set_title("Predicted vs Random Anchor", fontsize=12, fontweight='bold')
            axes[1, 0].set_xlabel("Random Anchor Tour Weight")
            axes[1, 0].set_ylabel("Predicted Anchor Tour Weight")
            axes[1, 0].grid(True, alpha=0.3)
        else:
            axes[1, 0].text(0.5, 0.5, "Evaluation data\nnot available",
                           ha='center', va='center', fontsize=10)
            axes[1, 0].set_title("Predicted vs Actual", fontsize=12, fontweight='bold')

        # Bottom-right: Performance by graph type
        if results_df is not None and 'graph_type' in results_df.columns:
            grouped = results_df.groupby(['graph_type', 'algorithm'])['tour_weight'].mean().reset_index()
            for algo in results_df['algorithm'].unique():
                algo_data = grouped[grouped['algorithm'] == algo]
                axes[1, 1].plot(algo_data['graph_type'], algo_data['tour_weight'],
                               marker='o', label=algo)
            axes[1, 1].set_title("Performance by Graph Type", fontsize=12, fontweight='bold')
            axes[1, 1].set_xlabel("Graph Type")
            axes[1, 1].set_ylabel("Mean Tour Weight")
            axes[1, 1].legend(fontsize=8, loc='best')
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, "Graph type data\nnot available",
                           ha='center', va='center', fontsize=10)
            axes[1, 1].set_title("Performance by Graph Type", fontsize=12, fontweight='bold')

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
