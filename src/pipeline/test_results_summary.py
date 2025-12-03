"""
Test results summary and analysis module.

Analyzes benchmark results to provide:
- Success/failure rates per algorithm and graph type
- Statistical summaries of performance metrics
- Anomaly detection and outlier identification
- Initial observations and interpretations
- Data quality assessment
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum

from algorithms.storage import BenchmarkStorage


class TestStatus(Enum):
    """Test execution status."""
    SUCCESS = "success"
    FAILURE = "failure"
    TIMEOUT = "timeout"
    ERROR = "error"


@dataclass
class TestSummary:
    """Summary statistics for a test category."""
    total_tests: int
    successes: int
    failures: int
    success_rate: float
    mean_tour_weight: float
    std_tour_weight: float
    median_tour_weight: float
    mean_runtime: float
    outlier_count: int


@dataclass
class Observation:
    """Single observation or interpretation."""
    category: str  # "performance", "reliability", "anomaly", "pattern"
    severity: str  # "info", "warning", "critical"
    description: str
    affected_items: List[str]  # Graph IDs, algorithm names, etc.
    supporting_data: Dict[str, Any]


class TestResultsSummarizer:
    """
    Comprehensive test results analysis and summarization.

    Analyzes benchmark results to extract insights about:
    - Algorithm performance across graph types
    - Success/failure patterns
    - Data quality issues
    - Performance anomalies
    - Statistical patterns
    """

    def __init__(self, benchmark_storage: BenchmarkStorage):
        """
        Initialize summarizer with benchmark storage.

        Args:
            benchmark_storage: BenchmarkStorage instance containing results
        """
        self.storage = benchmark_storage
        self.results_df = None
        self.observations = []

    def load_results(self) -> pd.DataFrame:
        """Load all benchmark results into DataFrame."""
        results = self.storage.load_results()
        self.results_df = pd.DataFrame(results)
        return self.results_df

    def compute_success_rates(self) -> Dict[str, TestSummary]:
        """
        Compute success/failure rates by algorithm and graph type.

        Returns:
            Dict mapping (algorithm, graph_type) → TestSummary
        """
        if self.results_df is None:
            self.load_results()

        summaries = {}

        # Overall summary
        summaries['overall'] = self._compute_summary(self.results_df)

        # Per algorithm
        for algorithm in self.results_df['algorithm'].unique():
            algo_df = self.results_df[self.results_df['algorithm'] == algorithm]
            summaries[f"algorithm:{algorithm}"] = self._compute_summary(algo_df)

        # Per graph type
        for graph_type in self.results_df['graph_type'].unique():
            type_df = self.results_df[self.results_df['graph_type'] == graph_type]
            summaries[f"graph_type:{graph_type}"] = self._compute_summary(type_df)

        # Per algorithm-graph_type combination
        for algorithm in self.results_df['algorithm'].unique():
            for graph_type in self.results_df['graph_type'].unique():
                combo_df = self.results_df[
                    (self.results_df['algorithm'] == algorithm) &
                    (self.results_df['graph_type'] == graph_type)
                ]
                if len(combo_df) > 0:
                    key = f"{algorithm}@{graph_type}"
                    summaries[key] = self._compute_summary(combo_df)

        return summaries

    def _compute_summary(self, df: pd.DataFrame) -> TestSummary:
        """Compute summary statistics for a subset of results."""
        # Detect failures (NaN tour weights, extreme values, etc.)
        valid_mask = df['tour_weight'].notna() & (df['tour_weight'] > 0)

        total = len(df)
        successes = valid_mask.sum()
        failures = total - successes

        # Outlier detection (values > 3 standard deviations)
        if successes > 0:
            weights = df.loc[valid_mask, 'tour_weight']
            mean = weights.mean()
            std = weights.std()
            outliers = ((weights - mean).abs() > 3 * std).sum()
        else:
            weights = pd.Series([])
            outliers = 0

        return TestSummary(
            total_tests=total,
            successes=successes,
            failures=failures,
            success_rate=successes / total if total > 0 else 0.0,
            mean_tour_weight=weights.mean() if successes > 0 else np.nan,
            std_tour_weight=weights.std() if successes > 0 else np.nan,
            median_tour_weight=weights.median() if successes > 0 else np.nan,
            mean_runtime=df['runtime'].mean() if 'runtime' in df else np.nan,
            outlier_count=outliers
        )

    def identify_patterns(self) -> List[Observation]:
        """
        Identify interesting patterns in the test results.

        Looks for:
        - Algorithms that consistently beat others
        - Graph types that are particularly hard/easy
        - Anomalous results
        - Performance degradation patterns
        """
        observations = []

        if self.results_df is None:
            self.load_results()

        # Pattern 1: Algorithm dominance
        algo_performance = self.results_df.groupby('algorithm')['tour_weight'].mean()
        best_algo = algo_performance.idxmin()
        worst_algo = algo_performance.idxmax()

        if algo_performance[best_algo] < algo_performance[worst_algo] * 0.9:
            observations.append(Observation(
                category="performance",
                severity="info",
                description=f"Algorithm '{best_algo}' consistently outperforms others",
                affected_items=[best_algo],
                supporting_data={
                    'mean_weight': float(algo_performance[best_algo]),
                    'improvement_over_worst': float((algo_performance[worst_algo] - algo_performance[best_algo]) / algo_performance[worst_algo])
                }
            ))

        # Pattern 2: Graph type difficulty
        type_performance = self.results_df.groupby('graph_type')['tour_weight'].mean()
        easiest_type = type_performance.idxmin()
        hardest_type = type_performance.idxmax()

        observations.append(Observation(
            category="pattern",
            severity="info",
            description=f"Graph type '{hardest_type}' produces tours {type_performance[hardest_type]/type_performance[easiest_type]:.2f}× longer than '{easiest_type}'",
            affected_items=[hardest_type, easiest_type],
            supporting_data={
                'hardest': hardest_type,
                'easiest': easiest_type,
                'difficulty_ratio': float(type_performance[hardest_type] / type_performance[easiest_type])
            }
        ))

        # Pattern 3: Failures detection
        failed_mask = self.results_df['tour_weight'].isna() | (self.results_df['tour_weight'] <= 0)
        if failed_mask.any():
            failed_algos = self.results_df.loc[failed_mask, 'algorithm'].unique()
            observations.append(Observation(
                category="reliability",
                severity="warning",
                description=f"Found {failed_mask.sum()} failed tests across {len(failed_algos)} algorithms",
                affected_items=list(failed_algos),
                supporting_data={
                    'failure_count': int(failed_mask.sum()),
                    'failure_rate': float(failed_mask.mean())
                }
            ))

        # Pattern 4: Performance scaling with graph size
        size_corr = self.results_df[['graph_size', 'tour_weight']].corr().iloc[0, 1]
        if size_corr > 0.8:
            observations.append(Observation(
                category="pattern",
                severity="info",
                description=f"Strong correlation (r={size_corr:.3f}) between graph size and tour weight",
                affected_items=[],
                supporting_data={'correlation': float(size_corr)}
            ))

        # Pattern 5: Outlier identification
        summaries = self.compute_success_rates()
        high_outlier_categories = [
            (key, summary.outlier_count)
            for key, summary in summaries.items()
            if summary.outlier_count > 0
        ]

        if high_outlier_categories:
            observations.append(Observation(
                category="anomaly",
                severity="warning",
                description=f"Detected statistical outliers in {len(high_outlier_categories)} test categories",
                affected_items=[key for key, _ in high_outlier_categories],
                supporting_data={
                    'categories': dict(high_outlier_categories)
                }
            ))

        # Pattern 6: Algorithm-graph type interactions
        interaction_matrix = self.results_df.pivot_table(
            values='tour_weight',
            index='algorithm',
            columns='graph_type',
            aggfunc='mean'
        )

        # Find best algorithm for each graph type
        best_per_type = {}
        for graph_type in interaction_matrix.columns:
            best_algo = interaction_matrix[graph_type].idxmin()
            best_per_type[graph_type] = best_algo

        if len(set(best_per_type.values())) > 1:
            observations.append(Observation(
                category="pattern",
                severity="info",
                description="Different graph types favor different algorithms",
                affected_items=list(best_per_type.keys()),
                supporting_data={
                    'best_per_type': best_per_type
                }
            ))

        self.observations = observations
        return observations

    def generate_summary_report(self, output_path: Optional[Path] = None) -> str:
        """
        Generate comprehensive summary report in Markdown format.

        Args:
            output_path: Optional path to save report

        Returns:
            Markdown-formatted report string
        """
        if self.results_df is None:
            self.load_results()

        summaries = self.compute_success_rates()
        observations = self.identify_patterns() if not self.observations else self.observations

        report_lines = []

        # Header
        report_lines.append("# Test Results Summary Report")
        report_lines.append("")
        report_lines.append(f"**Generated**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"**Total Tests**: {len(self.results_df)}")
        report_lines.append("")

        # Overall statistics
        report_lines.append("## Overall Statistics")
        report_lines.append("")
        overall = summaries['overall']
        report_lines.append(f"- **Total Tests**: {overall.total_tests}")
        report_lines.append(f"- **Successes**: {overall.successes} ({overall.success_rate:.1%})")
        report_lines.append(f"- **Failures**: {overall.failures}")
        report_lines.append(f"- **Mean Tour Weight**: {overall.mean_tour_weight:.2f} ± {overall.std_tour_weight:.2f}")
        report_lines.append(f"- **Median Tour Weight**: {overall.median_tour_weight:.2f}")
        report_lines.append(f"- **Mean Runtime**: {overall.mean_runtime:.3f}s")
        report_lines.append(f"- **Outliers Detected**: {overall.outlier_count}")
        report_lines.append("")

        # Per-algorithm summary
        report_lines.append("## Performance by Algorithm")
        report_lines.append("")
        report_lines.append("| Algorithm | Tests | Success Rate | Mean Weight | Std Dev | Median Weight | Outliers |")
        report_lines.append("|-----------|-------|--------------|-------------|---------|---------------|----------|")

        for key, summary in summaries.items():
            if key.startswith('algorithm:'):
                algo_name = key.split(':')[1]
                report_lines.append(
                    f"| {algo_name} | {summary.total_tests} | {summary.success_rate:.1%} | "
                    f"{summary.mean_tour_weight:.2f} | {summary.std_tour_weight:.2f} | "
                    f"{summary.median_tour_weight:.2f} | {summary.outlier_count} |"
                )
        report_lines.append("")

        # Per-graph-type summary
        report_lines.append("## Performance by Graph Type")
        report_lines.append("")
        report_lines.append("| Graph Type | Tests | Success Rate | Mean Weight | Std Dev | Median Weight | Outliers |")
        report_lines.append("|------------|-------|--------------|-------------|---------|---------------|----------|")

        for key, summary in summaries.items():
            if key.startswith('graph_type:'):
                type_name = key.split(':')[1]
                report_lines.append(
                    f"| {type_name} | {summary.total_tests} | {summary.success_rate:.1%} | "
                    f"{summary.mean_tour_weight:.2f} | {summary.std_tour_weight:.2f} | "
                    f"{summary.median_tour_weight:.2f} | {summary.outlier_count} |"
                )
        report_lines.append("")

        # Algorithm-GraphType interaction matrix
        report_lines.append("## Algorithm × Graph Type Performance Matrix")
        report_lines.append("")
        report_lines.append("Mean tour weights for each combination:")
        report_lines.append("")

        interaction_matrix = self.results_df.pivot_table(
            values='tour_weight',
            index='algorithm',
            columns='graph_type',
            aggfunc='mean'
        )

        # Format as markdown table
        header = "| Algorithm | " + " | ".join(interaction_matrix.columns) + " |"
        separator = "|" + "|".join(["---"] * (len(interaction_matrix.columns) + 1)) + "|"
        report_lines.append(header)
        report_lines.append(separator)

        for algo in interaction_matrix.index:
            row = f"| {algo} |"
            for graph_type in interaction_matrix.columns:
                value = interaction_matrix.loc[algo, graph_type]
                row += f" {value:.2f} |"
            report_lines.append(row)
        report_lines.append("")

        # Observations and interpretations
        report_lines.append("## Observations and Interpretations")
        report_lines.append("")

        # Group by category
        obs_by_category = defaultdict(list)
        for obs in observations:
            obs_by_category[obs.category].append(obs)

        for category in ['performance', 'reliability', 'pattern', 'anomaly']:
            if category in obs_by_category:
                report_lines.append(f"### {category.title()}")
                report_lines.append("")

                for obs in obs_by_category[category]:
                    severity_icon = {"info": "ℹ️", "warning": "⚠️", "critical": "❌"}
                    report_lines.append(f"{severity_icon.get(obs.severity, '•')} **{obs.description}**")

                    if obs.affected_items:
                        report_lines.append(f"  - Affected: {', '.join(obs.affected_items)}")

                    if obs.supporting_data:
                        report_lines.append(f"  - Data: {obs.supporting_data}")

                    report_lines.append("")

        # Data quality assessment
        report_lines.append("## Data Quality Assessment")
        report_lines.append("")

        missing_weights = self.results_df['tour_weight'].isna().sum()
        missing_runtimes = self.results_df['runtime'].isna().sum() if 'runtime' in self.results_df else 0

        report_lines.append(f"- **Missing tour weights**: {missing_weights} ({missing_weights/len(self.results_df):.1%})")
        report_lines.append(f"- **Missing runtimes**: {missing_runtimes} ({missing_runtimes/len(self.results_df):.1%})")

        # Check for duplicate tests
        duplicates = self.results_df.duplicated(subset=['graph_id', 'algorithm', 'anchor_vertex']).sum()
        report_lines.append(f"- **Duplicate tests**: {duplicates}")

        report_lines.append("")

        # Key findings summary
        report_lines.append("## Key Findings")
        report_lines.append("")

        # Best algorithm
        algo_perf = self.results_df.groupby('algorithm')['tour_weight'].mean()
        best_algo = algo_perf.idxmin()
        report_lines.append(f"1. **Best performing algorithm**: {best_algo} (mean weight: {algo_perf[best_algo]:.2f})")

        # Hardest graph type
        type_perf = self.results_df.groupby('graph_type')['tour_weight'].mean()
        hardest_type = type_perf.idxmax()
        report_lines.append(f"2. **Most challenging graph type**: {hardest_type} (mean weight: {type_perf[hardest_type]:.2f})")

        # Reliability
        success_rate = summaries['overall'].success_rate
        report_lines.append(f"3. **Overall reliability**: {success_rate:.1%} success rate")

        # Variance
        report_lines.append(f"4. **Result consistency**: Std dev = {summaries['overall'].std_tour_weight:.2f} (CV = {summaries['overall'].std_tour_weight/summaries['overall'].mean_tour_weight:.1%})")

        report_lines.append("")

        # Compile report
        report_text = "\n".join(report_lines)

        # Save if path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(report_text)

        return report_text

    def export_data_summary(self, output_path: Path):
        """
        Export structured data summary as JSON for programmatic access.

        Args:
            output_path: Path to save JSON file
        """
        import json

        summaries = self.compute_success_rates()
        observations = self.identify_patterns() if not self.observations else self.observations

        # Convert to serializable format
        summary_dict = {}
        for key, summary in summaries.items():
            summary_dict[key] = {
                'total_tests': summary.total_tests,
                'successes': summary.successes,
                'failures': summary.failures,
                'success_rate': summary.success_rate,
                'mean_tour_weight': float(summary.mean_tour_weight) if not np.isnan(summary.mean_tour_weight) else None,
                'std_tour_weight': float(summary.std_tour_weight) if not np.isnan(summary.std_tour_weight) else None,
                'median_tour_weight': float(summary.median_tour_weight) if not np.isnan(summary.median_tour_weight) else None,
                'mean_runtime': float(summary.mean_runtime) if not np.isnan(summary.mean_runtime) else None,
                'outlier_count': summary.outlier_count
            }

        observations_dict = [
            {
                'category': obs.category,
                'severity': obs.severity,
                'description': obs.description,
                'affected_items': obs.affected_items,
                'supporting_data': obs.supporting_data
            }
            for obs in observations
        ]

        output_data = {
            'generated_at': pd.Timestamp.now().isoformat(),
            'total_tests': len(self.results_df),
            'summaries': summary_dict,
            'observations': observations_dict
        }

        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)


# Convenience function for quick usage
def summarize_test_results(
    benchmark_storage: BenchmarkStorage,
    output_dir: Path
) -> Tuple[str, List[Observation]]:
    """
    Quick function to generate test summary report.

    Args:
        benchmark_storage: BenchmarkStorage with results
        output_dir: Directory to save reports

    Returns:
        (report_text, observations)
    """
    summarizer = TestResultsSummarizer(benchmark_storage)

    # Generate markdown report
    report_path = output_dir / 'test_results_summary.md'
    report_text = summarizer.generate_summary_report(report_path)

    # Export JSON data
    json_path = output_dir / 'test_results_summary.json'
    summarizer.export_data_summary(json_path)

    return report_text, summarizer.observations
