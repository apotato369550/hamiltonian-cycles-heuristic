"""
Pipeline Validation (Prompt 5).

Validates outputs from each pipeline stage to ensure data integrity
and catch errors before they propagate through the pipeline.
"""

from pathlib import Path
from typing import Dict, List, Any, Optional
import json
import numpy as np


class ValidationError(Exception):
    """Raised when pipeline stage output validation fails."""
    pass


class StageValidator:
    """
    Validates outputs from pipeline stages.

    Ensures each stage produces outputs in the expected format
    with valid data before passing to the next stage.
    """

    @staticmethod
    def validate_graph_generation_output(output_dir: Path) -> Dict[str, Any]:
        """
        Validate graph generation stage output.

        Args:
            output_dir: Directory containing generated graphs

        Returns:
            Validation report with counts and any errors

        Raises:
            ValidationError: If output is invalid
        """
        errors = []
        warnings = []

        if not output_dir.exists():
            raise ValidationError(f"Graph output directory does not exist: {output_dir}")

        # Find all graph files
        graph_files = list(output_dir.glob("*.json"))

        if len(graph_files) == 0:
            raise ValidationError(f"No graph files found in {output_dir}")

        # Validate each graph file
        valid_graphs = 0
        for graph_file in graph_files:
            try:
                with open(graph_file) as f:
                    graph_data = json.load(f)

                # Check required fields
                required_fields = ['n', 'adjacency_matrix', 'metadata']
                missing = [f for f in required_fields if f not in graph_data]
                if missing:
                    errors.append(f"{graph_file.name}: Missing fields {missing}")
                    continue

                # Validate adjacency matrix
                adj = np.array(graph_data['adjacency_matrix'])
                n = graph_data['n']

                if adj.shape != (n, n):
                    errors.append(f"{graph_file.name}: Adjacency matrix shape {adj.shape} != ({n}, {n})")
                    continue

                # Check for valid weights (no NaN, no negative)
                if np.any(np.isnan(adj)):
                    errors.append(f"{graph_file.name}: Contains NaN values")
                    continue

                if np.any(adj < 0):
                    errors.append(f"{graph_file.name}: Contains negative weights")
                    continue

                valid_graphs += 1

            except Exception as e:
                errors.append(f"{graph_file.name}: {str(e)}")

        if errors:
            raise ValidationError(f"Graph validation failed:\n" + "\n".join(errors))

        return {
            'stage': 'graph_generation',
            'valid': True,
            'total_graphs': len(graph_files),
            'valid_graphs': valid_graphs,
            'errors': errors,
            'warnings': warnings
        }

    @staticmethod
    def validate_benchmarking_output(output_dir: Path) -> Dict[str, Any]:
        """
        Validate benchmarking stage output.

        Args:
            output_dir: Directory containing benchmark results

        Returns:
            Validation report

        Raises:
            ValidationError: If output is invalid
        """
        errors = []
        warnings = []

        if not output_dir.exists():
            raise ValidationError(f"Benchmark output directory does not exist: {output_dir}")

        result_files = list(output_dir.glob("*.json"))

        if len(result_files) == 0:
            raise ValidationError(f"No benchmark result files found in {output_dir}")

        valid_results = 0
        for result_file in result_files:
            try:
                with open(result_file) as f:
                    results = json.load(f)

                # Check structure
                if not isinstance(results, dict):
                    errors.append(f"{result_file.name}: Not a dictionary")
                    continue

                # Check for algorithm results
                if 'algorithms' not in results:
                    errors.append(f"{result_file.name}: Missing 'algorithms' field")
                    continue

                # Validate each algorithm result
                for alg_name, alg_result in results.get('algorithms', {}).items():
                    required = ['tour', 'weight', 'runtime', 'success']
                    missing = [f for f in required if f not in alg_result]
                    if missing:
                        warnings.append(f"{result_file.name}/{alg_name}: Missing {missing}")

                valid_results += 1

            except Exception as e:
                errors.append(f"{result_file.name}: {str(e)}")

        if errors:
            raise ValidationError(f"Benchmark validation failed:\n" + "\n".join(errors))

        return {
            'stage': 'benchmarking',
            'valid': True,
            'total_results': len(result_files),
            'valid_results': valid_results,
            'errors': errors,
            'warnings': warnings
        }

    @staticmethod
    def validate_features_output(output_file: Path) -> Dict[str, Any]:
        """
        Validate feature extraction stage output.

        Args:
            output_file: CSV/parquet file containing features

        Returns:
            Validation report

        Raises:
            ValidationError: If output is invalid
        """
        errors = []
        warnings = []

        if not output_file.exists():
            raise ValidationError(f"Feature output file does not exist: {output_file}")

        try:
            import pandas as pd

            # Load features
            if output_file.suffix == '.csv':
                df = pd.read_csv(output_file)
            elif output_file.suffix == '.parquet':
                df = pd.read_parquet(output_file)
            else:
                raise ValidationError(f"Unsupported file format: {output_file.suffix}")

            # Check for required columns
            required = ['graph_id', 'vertex_id']
            missing = [c for c in required if c not in df.columns]
            if missing:
                errors.append(f"Missing required columns: {missing}")

            # Check for NaN values
            nan_cols = df.columns[df.isna().any()].tolist()
            if nan_cols:
                warnings.append(f"Columns with NaN values: {nan_cols}")

            # Check for infinite values
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            inf_cols = [col for col in numeric_cols if np.any(np.isinf(df[col]))]
            if inf_cols:
                warnings.append(f"Columns with infinite values: {inf_cols}")

            # Check for constant features
            constant_cols = [col for col in numeric_cols if df[col].std() == 0]
            if constant_cols:
                warnings.append(f"Constant features (std=0): {constant_cols}")

            if errors:
                raise ValidationError(f"Feature validation failed:\n" + "\n".join(errors))

            return {
                'stage': 'feature_extraction',
                'valid': True,
                'total_rows': len(df),
                'total_features': len(df.columns) - 2,  # Exclude graph_id, vertex_id
                'errors': errors,
                'warnings': warnings
            }

        except ImportError:
            warnings.append("pandas not available, skipping detailed validation")
            return {
                'stage': 'feature_extraction',
                'valid': True,
                'errors': errors,
                'warnings': warnings
            }
        except Exception as e:
            raise ValidationError(f"Feature validation failed: {str(e)}")

    @staticmethod
    def validate_model_output(output_dir: Path) -> Dict[str, Any]:
        """
        Validate model training stage output.

        Args:
            output_dir: Directory containing trained models

        Returns:
            Validation report

        Raises:
            ValidationError: If output is invalid
        """
        errors = []
        warnings = []

        if not output_dir.exists():
            raise ValidationError(f"Model output directory does not exist: {output_dir}")

        # Check for model files (pickle or joblib)
        model_files = list(output_dir.glob("*.pkl")) + list(output_dir.glob("*.joblib"))

        if len(model_files) == 0:
            warnings.append(f"No model files found in {output_dir}")

        # Check for evaluation results
        eval_files = list(output_dir.glob("*_evaluation.json"))

        if len(eval_files) == 0:
            warnings.append("No evaluation result files found")

        valid_evals = 0
        for eval_file in eval_files:
            try:
                with open(eval_file) as f:
                    eval_data = json.load(f)

                # Check for metrics
                if 'metrics' not in eval_data:
                    warnings.append(f"{eval_file.name}: Missing 'metrics' field")
                else:
                    valid_evals += 1

            except Exception as e:
                errors.append(f"{eval_file.name}: {str(e)}")

        if errors:
            raise ValidationError(f"Model validation failed:\n" + "\n".join(errors))

        return {
            'stage': 'model_training',
            'valid': True,
            'total_models': len(model_files),
            'total_evaluations': len(eval_files),
            'valid_evaluations': valid_evals,
            'errors': errors,
            'warnings': warnings
        }

    @staticmethod
    def validate_pipeline_run(experiment_dir: Path) -> Dict[str, List[Dict[str, Any]]]:
        """
        Validate entire pipeline run.

        Args:
            experiment_dir: Root experiment directory

        Returns:
            Dictionary mapping stage names to validation reports
        """
        validation_results = {}

        stages = [
            ('graph_generation', 'data/graphs'),
            ('benchmarking', 'data/benchmarks'),
            ('feature_extraction', 'data/features/features.csv'),
            ('model_training', 'models')
        ]

        for stage_name, rel_path in stages:
            path = experiment_dir / rel_path

            try:
                if stage_name == 'graph_generation':
                    result = StageValidator.validate_graph_generation_output(path)
                elif stage_name == 'benchmarking':
                    result = StageValidator.validate_benchmarking_output(path)
                elif stage_name == 'feature_extraction':
                    result = StageValidator.validate_features_output(path)
                elif stage_name == 'model_training':
                    result = StageValidator.validate_model_output(path)
                else:
                    result = {'stage': stage_name, 'valid': False, 'errors': ['Unknown stage']}

                validation_results[stage_name] = result

            except ValidationError as e:
                validation_results[stage_name] = {
                    'stage': stage_name,
                    'valid': False,
                    'errors': [str(e)]
                }
            except Exception as e:
                validation_results[stage_name] = {
                    'stage': stage_name,
                    'valid': False,
                    'errors': [f"Unexpected error: {str(e)}"]
                }

        return validation_results
