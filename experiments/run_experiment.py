#!/usr/bin/env python3
"""
Complete experiment runner.

Usage:
    python experiments/run_experiment.py config/my_experiment.yaml
    python experiments/run_experiment.py config/my_experiment.yaml --stage feature_extraction
    python experiments/run_experiment.py config/my_experiment.yaml --dry-run
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from pipeline.orchestrator import PipelineOrchestrator
from pipeline.config import ExperimentConfig
from pipeline.tracking import ExperimentTracker
from pipeline.reproducibility import ReproducibilityManager
from pipeline.stages import (
    create_graph_generation_stage,
    create_benchmarking_stage,
    create_feature_extraction_stage,
    create_training_stage,
    create_evaluation_stage
)
from pipeline.analysis import ExperimentAnalyzer
from pipeline.visualization import ExperimentVisualizer
from pipeline.test_results_summary import summarize_test_results
from algorithms.storage import BenchmarkStorage


def main():
    """Main entry point for experiment runner."""
    parser = argparse.ArgumentParser(description="Run complete TSP experiment")
    parser.add_argument('config', type=str, help="Path to experiment config YAML")
    parser.add_argument('--stage', type=str, default=None,
                        help="Run specific stage only (graph_generation, benchmarking, etc.)")
    parser.add_argument('--dry-run', action='store_true',
                        help="Validate config without running")

    args = parser.parse_args()

    # Load config
    print(f"Loading configuration from: {args.config}")
    config = ExperimentConfig.from_yaml(args.config)

    if args.dry_run:
        print("\n✓ Configuration valid!")
        print(f"Experiment: {config.get('experiment.name')}")
        print(f"Random seed: {config.get('experiment.random_seed')}")

        enabled_stages = []
        for stage in ['graph_generation', 'benchmarking', 'feature_extraction', 'training', 'evaluation']:
            if config.get(f'{stage}.enabled', False):
                enabled_stages.append(stage)

        print(f"Enabled stages: {', '.join(enabled_stages) if enabled_stages else 'None'}")
        return

    # Setup experiment tracking
    from pipeline.tracking import ExperimentRegistry

    registry = ExperimentRegistry(Path("experiments/registry.json"))
    exp_id = registry.generate_experiment_id(config.name)

    output_dir = Path(config.output_dir) / exp_id
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nExperiment: {config.name}")
    print(f"Experiment ID: {exp_id}")
    print(f"Output directory: {output_dir}")

    tracker = ExperimentTracker(
        experiment_id=exp_id,
        name=config.name,
        description=config.description,
        config=config.to_dict(),
        output_dir=output_dir
    )
    tracker.start()

    # Setup reproducibility
    repro_manager = ReproducibilityManager(master_seed=config.random_seed)
    repro_manager.initialize()  # Set all random seeds

    print(f"Random seed: {repro_manager.seed_manager.master_seed}")
    print(f"Git commit: {repro_manager.git_commit[:8] if repro_manager.git_commit else 'N/A'}")

    # Create pipeline stages
    stages = []

    if config.get('graph_generation.enabled', False):
        print("  + graph_generation stage")
        stages.append(create_graph_generation_stage(config.to_dict(), repro_manager, output_dir))

    if config.get('benchmarking.enabled', False):
        print("  + benchmarking stage")
        stages.append(create_benchmarking_stage(config.to_dict(), repro_manager, output_dir))

    if config.get('feature_extraction.enabled', False):
        print("  + feature_extraction stage")
        stages.append(create_feature_extraction_stage(config.to_dict(), repro_manager, output_dir))

    if config.get('training.enabled', False):
        print("  + training stage")
        stages.append(create_training_stage(config.to_dict(), repro_manager, output_dir))

    if config.get('evaluation.enabled', False):
        print("  + evaluation stage")
        stages.append(create_evaluation_stage(config.to_dict(), repro_manager, output_dir))

    if not stages:
        print("\n⚠️ Warning: No stages enabled in configuration")
        return

    # Create orchestrator
    orchestrator = PipelineOrchestrator(experiment_dir=output_dir)
    for stage in stages:
        orchestrator.add_stage(stage)

    # Run pipeline
    print(f"\n{'='*60}")
    print(f"Running pipeline with {len(stages)} stages")
    print(f"{'='*60}\n")

    if args.stage:
        # Run specific stage only
        print(f"Running stage: {args.stage}")
        result = orchestrator.run_stage(args.stage, {})
    else:
        # Run complete pipeline
        result = orchestrator.run()

    # Complete tracking
    tracker.complete(status="success" if result.success else "failed")

    if not result.success:
        print(f"\n❌ Pipeline failed")
        if hasattr(result, 'error'):
            print(f"Error: {result.error}")
        return 1

    print(f"\n✓ Pipeline completed successfully")

    # Generate analysis if enabled
    if config.get('analysis.enabled', False):
        print(f"\n{'='*60}")
        print("Generating analysis and reports")
        print(f"{'='*60}\n")

        # Test results summary
        if config.get('analysis.generate_test_summary', True):
            print("  → Generating test results summary...")
            try:
                bench_storage = BenchmarkStorage(str(output_dir / 'benchmarks'))
                report_text, observations = summarize_test_results(
                    bench_storage,
                    output_dir / 'reports'
                )

                print(f"     ✓ Test results summary saved to: reports/test_results_summary.md")
                print(f"     ✓ JSON data saved to: reports/test_results_summary.json")

                # Print critical observations
                critical_obs = [obs for obs in observations if obs.severity == 'critical']
                if critical_obs:
                    print("\n     ⚠️ CRITICAL ISSUES DETECTED:")
                    for obs in critical_obs:
                        print(f"       - {obs.description}")
            except Exception as e:
                print(f"     ⚠️ Could not generate test summary: {e}")

        # Analyzer report
        print("  → Generating analysis report...")
        try:
            analyzer = ExperimentAnalyzer(output_dir)
            analyzer.load_benchmark_results()
            analyzer.load_evaluation_results()

            summary_path = output_dir / 'reports' / 'summary_report.md'
            analyzer.export_summary(summary_path)
            print(f"     ✓ Analysis report saved to: {summary_path}")
        except Exception as e:
            print(f"     ⚠️ Could not generate analysis report: {e}")

        # Generate visualizations
        if config.get('analysis.visualizations', []):
            print("  → Generating visualizations...")
            try:
                visualizer = ExperimentVisualizer(
                    style="publication" if config.get('analysis.publication_quality', True) else "notebook"
                )

                figures_dir = output_dir / 'figures'
                figures_dir.mkdir(exist_ok=True)

                # Load results
                analyzer = ExperimentAnalyzer(output_dir)
                results_df = analyzer.load_benchmark_results()

                viz_list = config.get('analysis.visualizations', [])

                if 'algorithm_comparison_boxplot' in viz_list and results_df is not None:
                    visualizer.plot_algorithm_comparison(
                        results_df,
                        output_path=figures_dir / 'algorithm_comparison.png'
                    )
                    print(f"     ✓ Algorithm comparison saved to: figures/algorithm_comparison.png")

                if 'feature_importance_barplot' in viz_list:
                    model_files = list((output_dir / 'models').glob('*.pkl')) if (output_dir / 'models').exists() else []
                    if model_files:
                        importance_df = analyzer.analyze_feature_importance(model_files[0])
                        visualizer.plot_feature_importance(
                            importance_df,
                            output_path=figures_dir / 'feature_importance.png'
                        )
                        print(f"     ✓ Feature importance saved to: figures/feature_importance.png")

                if 'performance_by_graph_type' in viz_list and results_df is not None:
                    algorithms = results_df['algorithm'].unique().tolist()
                    visualizer.plot_performance_by_graph_type(
                        results_df,
                        algorithms=algorithms,
                        output_path=figures_dir / 'performance_by_graph_type.png'
                    )
                    print(f"     ✓ Performance by graph type saved to: figures/performance_by_graph_type.png")

                if 'interaction_heatmap' in viz_list and results_df is not None:
                    visualizer.plot_interaction_heatmap(
                        results_df,
                        output_path=figures_dir / 'interaction_heatmap.png'
                    )
                    print(f"     ✓ Interaction heatmap saved to: figures/interaction_heatmap.png")

                # Create summary figure
                visualizer.create_summary_figure(
                    output_dir,
                    output_path=figures_dir / 'summary_figure.png'
                )
                print(f"     ✓ Summary figure saved to: figures/summary_figure.png")

            except Exception as e:
                print(f"     ⚠️ Could not generate visualizations: {e}")
                import traceback
                traceback.print_exc()

    print(f"\n{'='*60}")
    print("Experiment complete!")
    print(f"{'='*60}")
    print(f"Results in: {output_dir}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
