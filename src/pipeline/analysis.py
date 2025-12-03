"""
Results analysis tools for TSP experiments.

Provides statistical analysis, comparison tools, and report generation.
"""

import pandas as pd
import pickle
from pathlib import Path
from typing import Dict, Any, List, Optional
from scipy import stats
import numpy as np


class ExperimentAnalyzer:
    """Analyze results from completed experiments."""

    def __init__(self, experiment_dir: Path):
        """
        Initialize analyzer with experiment directory.

        Args:
            experiment_dir: Path to experiment output directory
        """
        self.experiment_dir = Path(experiment_dir)
        self.benchmark_results = None
        self.evaluation_results = None

    def load_benchmark_results(self) -> pd.DataFrame:
        """
        Load benchmark results from SQLite database.

        Returns:
            DataFrame with all benchmark results
        """
        from algorithms.storage import BenchmarkStorage

        storage = BenchmarkStorage(str(self.experiment_dir / 'benchmarks'))
        results = storage.load_results()
        self.benchmark_results = pd.DataFrame(results)
        return self.benchmark_results

    def load_evaluation_results(self) -> pd.DataFrame:
        """
        Load evaluation results from CSV.

        Returns:
            DataFrame with evaluation results
        """
        eval_path = self.experiment_dir / 'reports' / 'evaluation_report.csv'
        if eval_path.exists():
            self.evaluation_results = pd.read_csv(eval_path)
            return self.evaluation_results
        return None

    def compare_algorithms(
        self,
        algorithms: Optional[List[str]] = None,
        groupby: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Compare algorithm performance.

        Args:
            algorithms: List of algorithm names to compare (None = all)
            groupby: Optional grouping column ('graph_type', 'graph_size')

        Returns:
            DataFrame with mean, std, median for each algorithm
        """
        if self.benchmark_results is None:
            self.load_benchmark_results()

        df = self.benchmark_results.copy()

        if algorithms:
            df = df[df['algorithm'].isin(algorithms)]

        if groupby:
            comparison = df.groupby(['algorithm', groupby])['tour_weight'].agg([
                ('mean', 'mean'),
                ('std', 'std'),
                ('median', 'median'),
                ('count', 'count')
            ]).reset_index()
        else:
            comparison = df.groupby('algorithm')['tour_weight'].agg([
                ('mean', 'mean'),
                ('std', 'std'),
                ('median', 'median'),
                ('count', 'count')
            ]).reset_index()

        return comparison

    def compute_statistical_significance(
        self,
        algorithm_a: str,
        algorithm_b: str,
        test: str = "paired_t_test"
    ) -> Dict[str, float]:
        """
        Test if algorithm A significantly outperforms B.

        Args:
            algorithm_a: First algorithm name
            algorithm_b: Second algorithm name
            test: Statistical test ('paired_t_test', 'wilcoxon')

        Returns:
            Dict with test_statistic, p_value, effect_size, significant
        """
        if self.benchmark_results is None:
            self.load_benchmark_results()

        # Get paired results (same graphs)
        df_a = self.benchmark_results[self.benchmark_results['algorithm'] == algorithm_a]
        df_b = self.benchmark_results[self.benchmark_results['algorithm'] == algorithm_b]

        # Merge on graph_id to get pairs
        merged = df_a.merge(df_b, on='graph_id', suffixes=('_a', '_b'))
        weights_a = merged['tour_weight_a'].dropna()
        weights_b = merged['tour_weight_b'].dropna()

        if len(weights_a) == 0 or len(weights_b) == 0:
            return {
                'test_statistic': np.nan,
                'p_value': np.nan,
                'effect_size': np.nan,
                'significant': False,
                'error': 'Insufficient paired data'
            }

        # Perform test
        if test == "paired_t_test":
            stat, p_value = stats.ttest_rel(weights_a, weights_b)
        elif test == "wilcoxon":
            stat, p_value = stats.wilcoxon(weights_a, weights_b)
        else:
            raise ValueError(f"Unknown test: {test}")

        # Compute Cohen's d effect size
        diff = weights_a - weights_b
        effect_size = diff.mean() / diff.std() if diff.std() > 0 else 0.0

        return {
            'test_statistic': float(stat),
            'p_value': float(p_value),
            'effect_size': float(effect_size),
            'significant': p_value < 0.05,
            'n_pairs': len(weights_a)
        }

    def analyze_feature_importance(self, model_path: Path) -> pd.DataFrame:
        """
        Extract feature importance from trained model.

        Args:
            model_path: Path to saved model pickle file

        Returns:
            DataFrame with feature names and importance scores
        """
        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        # Try to get feature importance
        if hasattr(model, 'get_feature_importance'):
            importance = model.get_feature_importance()
        elif hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importance = np.abs(model.coef_)
        else:
            return pd.DataFrame({'error': ['Model does not support feature importance']})

        # Load feature names from features dataset
        feature_path = self.experiment_dir / 'features' / 'feature_dataset.csv'
        if feature_path.exists():
            df = pd.read_csv(feature_path, nrows=1)
            feature_names = [col for col in df.columns if col not in ['label', 'graph_id', 'vertex_id', 'graph_type', 'graph_size']]
        else:
            feature_names = [f"feature_{i}" for i in range(len(importance))]

        importance_df = pd.DataFrame({
            'feature_name': feature_names[:len(importance)],
            'importance': importance
        }).sort_values('importance', ascending=False)

        return importance_df

    def compute_ml_improvement(self) -> Dict[str, float]:
        """
        Compute practical improvement from ML predictions.

        Returns:
            Dict with improvement metrics
        """
        if self.evaluation_results is None:
            self.load_evaluation_results()

        if self.evaluation_results is None:
            return {'error': 'No evaluation results found'}

        df = self.evaluation_results

        return {
            'mean_improvement_vs_random': float(df['improvement_vs_random'].mean()),
            'std_improvement_vs_random': float(df['improvement_vs_random'].std()),
            'win_rate_vs_random': float((df['improvement_vs_random'] > 0).mean()),
            'mean_improvement_vs_nn': float(df['improvement_vs_nn'].mean()),
            'std_improvement_vs_nn': float(df['improvement_vs_nn'].std()),
            'win_rate_vs_nn': float((df['improvement_vs_nn'] > 0).mean()),
            'num_graphs': len(df)
        }

    def generate_summary_report(self) -> str:
        """
        Generate markdown summary report.

        Returns:
            Markdown string with experiment summary
        """
        report_lines = []

        # Header
        report_lines.append("# Experiment Summary Report")
        report_lines.append("")
        report_lines.append(f"**Experiment Directory**: {self.experiment_dir}")
        report_lines.append("")

        # Algorithm Comparison
        report_lines.append("## Algorithm Performance Comparison")
        report_lines.append("")

        comparison = self.compare_algorithms()
        report_lines.append("| Algorithm | Mean Weight | Std Dev | Median | Count |")
        report_lines.append("|-----------|-------------|---------|--------|-------|")
        for _, row in comparison.iterrows():
            report_lines.append(
                f"| {row['algorithm']} | {row['mean']:.2f} | {row['std']:.2f} | "
                f"{row['median']:.2f} | {int(row['count'])} |"
            )
        report_lines.append("")

        # Statistical Tests
        report_lines.append("## Statistical Significance Tests")
        report_lines.append("")

        algorithms = comparison['algorithm'].tolist()
        if len(algorithms) >= 2:
            for i in range(len(algorithms)):
                for j in range(i + 1, len(algorithms)):
                    sig_test = self.compute_statistical_significance(
                        algorithms[i],
                        algorithms[j]
                    )

                    if 'error' not in sig_test:
                        report_lines.append(f"### {algorithms[i]} vs {algorithms[j]}")
                        report_lines.append(f"- **Test statistic**: {sig_test['test_statistic']:.3f}")
                        report_lines.append(f"- **P-value**: {sig_test['p_value']:.4f}")
                        report_lines.append(f"- **Effect size (Cohen's d)**: {sig_test['effect_size']:.3f}")
                        report_lines.append(f"- **Significant**: {'Yes' if sig_test['significant'] else 'No'} (α=0.05)")
                        report_lines.append("")

        # ML Performance
        if self.evaluation_results is not None:
            report_lines.append("## ML Model Performance")
            report_lines.append("")

            ml_metrics = self.compute_ml_improvement()
            report_lines.append(f"- **Mean improvement vs random anchor**: {ml_metrics['mean_improvement_vs_random']:.1%}")
            report_lines.append(f"- **Win rate vs random**: {ml_metrics['win_rate_vs_random']:.1%}")
            report_lines.append(f"- **Mean improvement vs nearest neighbor**: {ml_metrics['mean_improvement_vs_nn']:.1%}")
            report_lines.append(f"- **Win rate vs nearest neighbor**: {ml_metrics['win_rate_vs_nn']:.1%}")
            report_lines.append("")

        # Feature Importance
        model_path = self.experiment_dir / 'models'
        if model_path.exists():
            model_files = list(model_path.glob('*.pkl'))
            if model_files:
                report_lines.append("## Feature Importance (Best Model)")
                report_lines.append("")

                try:
                    importance_df = self.analyze_feature_importance(model_files[0])
                    report_lines.append("| Rank | Feature | Importance |")
                    report_lines.append("|------|---------|------------|")

                    for idx, row in importance_df.head(15).iterrows():
                        report_lines.append(f"| {idx+1} | {row['feature_name']} | {row['importance']:.4f} |")
                    report_lines.append("")
                except Exception as e:
                    report_lines.append(f"Error extracting feature importance: {e}")
                    report_lines.append("")

        return "\n".join(report_lines)

    def export_summary(self, output_path: Path):
        """
        Export summary report to file.

        Args:
            output_path: Path to save markdown report
        """
        report = self.generate_summary_report()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(report)
