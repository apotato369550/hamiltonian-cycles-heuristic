"""
Configuration Management System (Prompt 2).

Provides comprehensive configuration system for specifying experiments with
validation, inheritance, and documentation.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import yaml


@dataclass
class GraphGenConfig:
    """Configuration for graph generation stage."""
    enabled: bool = True
    graph_types: List[Dict[str, Any]] = field(default_factory=list)
    output_dir: str = "data/graphs"
    save_format: str = "json"  # or "pickle"
    batch_name: Optional[str] = None  # Optional batch name for organizing graphs

    def validate(self) -> List[str]:
        """Validate configuration. Returns list of errors."""
        errors = []

        if not self.graph_types and self.enabled:
            errors.append("graph_generation.graph_types cannot be empty when enabled")

        for i, gtype in enumerate(self.graph_types):
            if 'type' not in gtype:
                errors.append(f"graph_types[{i}] missing 'type' field")
            if 'sizes' not in gtype:
                errors.append(f"graph_types[{i}] missing 'sizes' field")
            if 'instances_per_size' not in gtype:
                errors.append(f"graph_types[{i}] missing 'instances_per_size' field")

            valid_types = ['euclidean', 'metric', 'quasi_metric', 'random']
            if gtype.get('type') not in valid_types:
                errors.append(
                    f"graph_types[{i}].type must be one of {valid_types}"
                )

        return errors


@dataclass
class BenchmarkConfig:
    """Configuration for algorithm benchmarking stage."""
    enabled: bool = True
    algorithms: List[Dict[str, Any]] = field(default_factory=list)
    timeout_seconds: Optional[float] = 300.0
    output_dir: str = "results/benchmarks"
    save_format: str = "json"  # or "csv", "sqlite"
    exhaustive_anchors: bool = False  # Test all possible anchors for label generation
    storage_format: Optional[str] = None  # Alternative to save_format

    def validate(self) -> List[str]:
        """Validate configuration. Returns list of errors."""
        errors = []

        if not self.algorithms and self.enabled:
            errors.append("benchmarking.algorithms cannot be empty when enabled")

        for i, alg in enumerate(self.algorithms):
            if 'name' not in alg:
                errors.append(f"algorithms[{i}] missing 'name' field")

        if self.timeout_seconds is not None and self.timeout_seconds <= 0:
            errors.append("benchmarking.timeout_seconds must be > 0")

        return errors


@dataclass
class FeatureConfig:
    """Configuration for feature engineering stage."""
    enabled: bool = True
    feature_groups: List[str] = field(default_factory=list)
    labeling_strategy: str = "rank_based"
    labeling_params: Optional[Dict[str, Any]] = None  # Parameters for labeling strategy
    output_dir: str = "results/features"
    save_format: str = "csv"  # or "parquet"
    output_format: Optional[str] = None  # Alternative to save_format

    def __post_init__(self):
        """Handle alternative field names."""
        # Use output_format if save_format not explicitly set
        if self.output_format and self.save_format == "csv":
            self.save_format = self.output_format

    def validate(self) -> List[str]:
        """Validate configuration. Returns list of errors."""
        errors = []

        valid_groups = [
            'weight_based',
            'topological',
            'mst_based',
            'neighborhood',
            'heuristic',
            'graph_context'
        ]

        for group in self.feature_groups:
            if group not in valid_groups:
                errors.append(
                    f"feature_groups contains invalid group '{group}'. "
                    f"Valid groups: {valid_groups}"
                )

        valid_strategies = [
            'rank_based',
            'absolute_percentile',
            'binary',
            'multiclass',
            'relative_gap'
        ]
        if self.labeling_strategy not in valid_strategies:
            errors.append(
                f"labeling_strategy must be one of {valid_strategies}"
            )

        return errors


@dataclass
class ModelConfig:
    """Configuration for model training stage."""
    enabled: bool = True
    models: List[Dict[str, Any]] = field(default_factory=list)
    split_strategy: str = "stratified_graph"
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    output_dir: str = "models"
    save_models: bool = True
    problem_type: Optional[str] = "regression"  # regression, classification, ranking
    test_split: Optional[float] = None  # Alternative to test_ratio
    stratify_by: Optional[str] = None  # Stratification column
    cross_validation: Optional[Dict[str, Any]] = None  # CV configuration

    def validate(self) -> List[str]:
        """Validate configuration. Returns list of errors."""
        errors = []

        if not self.models and self.enabled:
            errors.append("model_training.models cannot be empty when enabled")

        for i, model in enumerate(self.models):
            if 'type' not in model:
                errors.append(f"models[{i}] missing 'type' field")

        valid_strategies = [
            'random',
            'graph_based',
            'stratified_graph',
            'graph_type_holdout',
            'size_holdout'
        ]
        if self.split_strategy not in valid_strategies:
            errors.append(
                f"split_strategy must be one of {valid_strategies}"
            )

        # Check ratios sum to 1.0 (if all specified)
        total_ratio = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total_ratio - 1.0) > 1e-6:
            errors.append(
                f"train_ratio + val_ratio + test_ratio must equal 1.0, "
                f"got {total_ratio}"
            )

        if self.train_ratio <= 0 or self.val_ratio < 0 or self.test_ratio <= 0:
            errors.append("All split ratios must be non-negative, train and test > 0")

        return errors


@dataclass
class ExperimentConfig:
    """
    Complete experiment configuration.

    Specifies all parameters for a multi-stage TSP research experiment.
    """
    name: str
    description: str = ""
    random_seed: int = 42
    output_dir: str = "experiments"

    graph_generation: GraphGenConfig = field(default_factory=GraphGenConfig)
    benchmarking: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    feature_engineering: FeatureConfig = field(default_factory=FeatureConfig)
    model_training: ModelConfig = field(default_factory=ModelConfig)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ExperimentConfig':
        """
        Create ExperimentConfig from dictionary.

        Args:
            config_dict: Configuration dictionary (from YAML)

        Returns:
            ExperimentConfig instance
        """
        # Handle 'experiment' wrapper if present
        if 'experiment' in config_dict:
            exp_data = config_dict['experiment']
            name = exp_data.get('name', 'unnamed_experiment')
            description = exp_data.get('description', '')
            random_seed = exp_data.get('random_seed', 42)
            output_dir = exp_data.get('output_dir', 'experiments')
        else:
            # Extract top-level fields
            name = config_dict.get('name', 'unnamed_experiment')
            description = config_dict.get('description', '')
            random_seed = config_dict.get('random_seed', 42)
            output_dir = config_dict.get('output_dir', 'experiments')

        # Extract stage configs, handling 'types' alias for 'graph_types'
        graph_gen_dict = config_dict.get('graph_generation', {})
        if 'types' in graph_gen_dict and 'graph_types' not in graph_gen_dict:
            graph_gen_dict['graph_types'] = graph_gen_dict.pop('types')
        graph_gen = GraphGenConfig(**graph_gen_dict)

        benchmark = BenchmarkConfig(**config_dict.get('benchmarking', {}))

        # Handle 'feature_extraction' alias for 'feature_engineering'
        feature_dict = config_dict.get('feature_engineering', config_dict.get('feature_extraction', {}))
        # Handle 'extractors' alias for 'feature_groups'
        if 'extractors' in feature_dict and 'feature_groups' not in feature_dict:
            feature_dict['feature_groups'] = feature_dict.pop('extractors')
        feature = FeatureConfig(**feature_dict)

        # Handle 'training' alias for 'model_training'
        model_dict = config_dict.get('model_training', config_dict.get('training', {}))
        model = ModelConfig(**model_dict)

        return cls(
            name=name,
            description=description,
            random_seed=random_seed,
            output_dir=output_dir,
            graph_generation=graph_gen,
            benchmarking=benchmark,
            feature_engineering=feature,
            model_training=model
        )

    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> 'ExperimentConfig':
        """
        Load configuration from YAML file.

        Args:
            yaml_path: Path to YAML configuration file

        Returns:
            ExperimentConfig instance
        """
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)

        return cls.from_dict(config_dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)

    def to_yaml(self, yaml_path: Union[str, Path]):
        """
        Save configuration to YAML file.

        Args:
            yaml_path: Path to save YAML file
        """
        with open(yaml_path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, indent=2)

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.

        Args:
            key: Configuration key in dot notation (e.g., 'experiment.name' or 'name')
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        # Handle 'experiment.' prefix by stripping it (for backward compatibility)
        if key.startswith('experiment.'):
            key = key[11:]  # Remove 'experiment.' prefix

        # Handle field aliases
        aliases = {
            'feature_extraction': 'feature_engineering',
            'training': 'model_training'
        }

        parts = key.split('.')
        if parts[0] in aliases:
            parts[0] = aliases[parts[0]]

        value = self

        for part in parts:
            if hasattr(value, part):
                value = getattr(value, part)
            elif isinstance(value, dict):
                value = value.get(part, default)
                if value == default:
                    return default
            else:
                return default

        return value

    def validate(self) -> List[str]:
        """
        Validate entire configuration.

        Returns:
            List of error messages (empty if valid)
        """
        errors = []

        # Validate name
        if not self.name or not self.name.strip():
            errors.append("experiment.name cannot be empty")

        # Validate random seed
        if not isinstance(self.random_seed, int):
            errors.append("random_seed must be an integer")

        # Validate each stage
        errors.extend(self.graph_generation.validate())
        errors.extend(self.benchmarking.validate())
        errors.extend(self.feature_engineering.validate())
        errors.extend(self.model_training.validate())

        return errors


class ConfigValidator:
    """
    Validates experiment configurations and provides helpful error messages.
    """

    @staticmethod
    def validate(config: ExperimentConfig) -> bool:
        """
        Validate configuration.

        Args:
            config: ExperimentConfig to validate

        Returns:
            True if valid, False otherwise

        Raises:
            ValueError: If configuration is invalid (with detailed error messages)
        """
        errors = config.validate()

        if errors:
            error_msg = "Configuration validation failed:\n"
            error_msg += "\n".join(f"  - {err}" for err in errors)
            raise ValueError(error_msg)

        return True

    @staticmethod
    def validate_file(yaml_path: Union[str, Path]) -> ExperimentConfig:
        """
        Validate and load configuration from YAML file.

        Args:
            yaml_path: Path to YAML configuration file

        Returns:
            ExperimentConfig if valid

        Raises:
            ValueError: If configuration is invalid
            FileNotFoundError: If file doesn't exist
        """
        if not Path(yaml_path).exists():
            raise FileNotFoundError(f"Configuration file not found: {yaml_path}")

        config = ExperimentConfig.from_yaml(yaml_path)
        ConfigValidator.validate(config)
        return config

    @staticmethod
    def create_template(output_path: Union[str, Path]):
        """
        Create a template configuration file with documentation.

        Args:
            output_path: Where to save the template
        """
        template = {
            'name': 'example_experiment',
            'description': 'Example TSP experiment configuration',
            'random_seed': 42,
            'output_dir': 'experiments',

            'graph_generation': {
                'enabled': True,
                'graph_types': [
                    {
                        'type': 'euclidean',
                        'sizes': [20, 50, 100],
                        'instances_per_size': 10,
                        'dimension': 2
                    },
                    {
                        'type': 'metric',
                        'sizes': [50, 100],
                        'instances_per_size': 5,
                        'strategy': 'completion'
                    }
                ],
                'output_dir': 'data/graphs',
                'save_format': 'json'
            },

            'benchmarking': {
                'enabled': True,
                'algorithms': [
                    {'name': 'nearest_neighbor', 'params': {'strategy': 'best_start'}},
                    {'name': 'single_anchor', 'params': {}},
                    {'name': 'best_anchor', 'params': {}}
                ],
                'timeout_seconds': 300,
                'output_dir': 'results/benchmarks',
                'save_format': 'json'
            },

            'feature_engineering': {
                'enabled': True,
                'feature_groups': [
                    'weight_based',
                    'topological',
                    'mst_based',
                    'neighborhood',
                    'heuristic',
                    'graph_context'
                ],
                'labeling_strategy': 'rank_based',
                'output_dir': 'results/features',
                'save_format': 'csv'
            },

            'model_training': {
                'enabled': True,
                'models': [
                    {
                        'type': 'linear_regression',
                        'model_variant': 'ridge',
                        'params': {'alpha': 1.0}
                    },
                    {
                        'type': 'random_forest',
                        'params': {'n_estimators': 100, 'max_depth': 10}
                    }
                ],
                'split_strategy': 'stratified_graph',
                'train_ratio': 0.7,
                'val_ratio': 0.15,
                'test_ratio': 0.15,
                'output_dir': 'models',
                'save_models': True
            }
        }

        with open(output_path, 'w') as f:
            yaml.dump(template, f, default_flow_style=False, indent=2)
