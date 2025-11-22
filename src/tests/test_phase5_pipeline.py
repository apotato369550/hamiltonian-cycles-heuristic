"""
Phase 5: Pipeline Integration Tests (Prompts 1-4).

Tests pipeline orchestration, configuration management, experiment tracking,
and reproducibility infrastructure.
"""

import unittest
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
import json
import yaml
import random
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline import (
    # Prompt 1: Orchestrator
    PipelineStage,
    PipelineOrchestrator,
    StageResult,
    StageStatus,

    # Prompt 2: Configuration
    ExperimentConfig,
    ConfigValidator,
    GraphGenConfig,
    BenchmarkConfig,
    FeatureConfig,
    ModelConfig,

    # Prompt 3: Tracking
    ExperimentTracker,
    ExperimentRegistry,
    ExperimentMetadata,
    ExperimentStatus,

    # Prompt 4: Reproducibility
    ReproducibilityManager,
    EnvironmentInfo,
    SeedManager
)


class TestPipelineStage(unittest.TestCase):
    """Test PipelineStage (Prompt 1)."""

    def test_stage_initialization(self):
        """Test stage creation."""
        def dummy_execute(inputs):
            return StageResult(
                stage_name="test",
                status=StageStatus.COMPLETED,
                start_time=datetime.now()
            )

        stage = PipelineStage(
            name="test_stage",
            execute_fn=dummy_execute,
            required_inputs=['input1'],
            output_keys=['output1']
        )

        self.assertEqual(stage.name, "test_stage")
        self.assertEqual(stage.required_inputs, ['input1'])
        self.assertEqual(stage.output_keys, ['output1'])

    def test_validate_inputs(self):
        """Test input validation."""
        def dummy_execute(inputs):
            return StageResult(
                stage_name="test",
                status=StageStatus.COMPLETED,
                start_time=datetime.now()
            )

        stage = PipelineStage(
            name="test",
            execute_fn=dummy_execute,
            required_inputs=['a', 'b']
        )

        # Valid inputs
        self.assertTrue(stage.validate_inputs({'a': 1, 'b': 2, 'c': 3}))

        # Missing input
        self.assertFalse(stage.validate_inputs({'a': 1}))

    def test_stage_execution(self):
        """Test stage execution."""
        def execute_fn(inputs):
            result = StageResult(
                stage_name="test",
                status=StageStatus.COMPLETED,
                start_time=datetime.now()
            )
            result.outputs = {'result': inputs['x'] * 2}
            result.metadata = {'operation': 'double'}
            return result

        stage = PipelineStage(
            name="test",
            execute_fn=execute_fn,
            required_inputs=['x'],
            skip_if_exists=False  # Don't skip for tests
        )

        result = stage.execute({'x': 5})

        self.assertEqual(result.status, StageStatus.COMPLETED)
        self.assertEqual(result.outputs['result'], 10)
        self.assertEqual(result.metadata['operation'], 'double')

    def test_stage_failure(self):
        """Test stage failure handling."""
        def failing_execute(inputs):
            raise ValueError("Intentional failure")

        stage = PipelineStage(
            name="test",
            execute_fn=failing_execute,
            skip_if_exists=False
        )

        result = stage.execute({})

        self.assertEqual(result.status, StageStatus.FAILED)
        self.assertIn("Intentional failure", result.error)


class TestPipelineOrchestrator(unittest.TestCase):
    """Test PipelineOrchestrator (Prompt 1)."""

    def setUp(self):
        """Set up temporary directory for tests."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir)

    def test_orchestrator_initialization(self):
        """Test orchestrator creation."""
        orchestrator = PipelineOrchestrator(
            experiment_dir=Path(self.temp_dir)
        )

        self.assertEqual(orchestrator.experiment_dir, Path(self.temp_dir))
        self.assertEqual(len(orchestrator.stages), 0)

    def test_add_stage(self):
        """Test adding stages."""
        orchestrator = PipelineOrchestrator(Path(self.temp_dir))

        def dummy_execute(inputs):
            return StageResult(
                stage_name="test",
                status=StageStatus.COMPLETED,
                start_time=datetime.now()
            )

        stage = PipelineStage("stage1", dummy_execute)
        orchestrator.add_stage(stage)

        self.assertEqual(len(orchestrator.stages), 1)
        self.assertEqual(orchestrator.stages[0].name, "stage1")

    def test_pipeline_execution(self):
        """Test full pipeline execution."""
        orchestrator = PipelineOrchestrator(Path(self.temp_dir))

        # Stage 1: Add 10
        def stage1_execute(inputs):
            result = StageResult("stage1", StageStatus.COMPLETED, datetime.now())
            result.outputs = {'value': inputs.get('initial', 0) + 10}
            return result

        # Stage 2: Multiply by 2
        def stage2_execute(inputs):
            result = StageResult("stage2", StageStatus.COMPLETED, datetime.now())
            result.outputs = {'value': inputs['value'] * 2}
            return result

        orchestrator.add_stage(PipelineStage("stage1", stage1_execute, skip_if_exists=False))
        orchestrator.add_stage(PipelineStage("stage2", stage2_execute, ['value'], skip_if_exists=False))

        results = orchestrator.run({'initial': 5})

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].status, StageStatus.COMPLETED)
        self.assertEqual(results[1].status, StageStatus.COMPLETED)
        self.assertEqual(results[1].outputs['value'], 30)  # (5 + 10) * 2

    def test_pipeline_stops_on_failure(self):
        """Test pipeline stops if stage fails."""
        orchestrator = PipelineOrchestrator(Path(self.temp_dir))

        def stage1_execute(inputs):
            raise ValueError("Stage 1 failed")

        def stage2_execute(inputs):
            return StageResult("stage2", StageStatus.COMPLETED, datetime.now())

        orchestrator.add_stage(PipelineStage("stage1", stage1_execute, skip_if_exists=False))
        orchestrator.add_stage(PipelineStage("stage2", stage2_execute, skip_if_exists=False))

        results = orchestrator.run()

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].status, StageStatus.FAILED)

    def test_save_manifest(self):
        """Test manifest saving."""
        orchestrator = PipelineOrchestrator(Path(self.temp_dir))

        def dummy_execute(inputs):
            return StageResult("test", StageStatus.COMPLETED, datetime.now())

        orchestrator.add_stage(PipelineStage("stage1", dummy_execute))
        orchestrator.run()

        orchestrator.save_manifest()

        manifest_path = Path(self.temp_dir) / "manifest.json"
        self.assertTrue(manifest_path.exists())

        with open(manifest_path) as f:
            manifest = json.load(f)

        self.assertEqual(manifest['total_stages'], 1)
        self.assertEqual(manifest['executed_stages'], 1)


class TestExperimentConfig(unittest.TestCase):
    """Test ExperimentConfig (Prompt 2)."""

    def test_config_initialization(self):
        """Test config creation."""
        config = ExperimentConfig(
            name="test_experiment",
            description="Test config",
            random_seed=42
        )

        self.assertEqual(config.name, "test_experiment")
        self.assertEqual(config.random_seed, 42)

    def test_config_from_dict(self):
        """Test config creation from dictionary."""
        config_dict = {
            'name': 'test',
            'description': 'Test experiment',
            'random_seed': 123,
            'graph_generation': {
                'enabled': True,
                'graph_types': [
                    {'type': 'euclidean', 'sizes': [10, 20], 'instances_per_size': 5}
                ]
            }
        }

        config = ExperimentConfig.from_dict(config_dict)

        self.assertEqual(config.name, 'test')
        self.assertEqual(config.random_seed, 123)
        self.assertTrue(config.graph_generation.enabled)

    def test_config_to_dict(self):
        """Test config serialization."""
        config = ExperimentConfig(name="test", random_seed=42)
        config_dict = config.to_dict()

        self.assertIsInstance(config_dict, dict)
        self.assertEqual(config_dict['name'], 'test')
        self.assertEqual(config_dict['random_seed'], 42)

    def test_config_yaml_roundtrip(self):
        """Test YAML save/load."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            temp_path = f.name

        try:
            config = ExperimentConfig(name="test", random_seed=99)
            config.to_yaml(temp_path)

            loaded_config = ExperimentConfig.from_yaml(temp_path)

            self.assertEqual(loaded_config.name, "test")
            self.assertEqual(loaded_config.random_seed, 99)
        finally:
            Path(temp_path).unlink()


class TestConfigValidation(unittest.TestCase):
    """Test configuration validation (Prompt 2)."""

    def test_valid_config(self):
        """Test validation of valid config."""
        config = ExperimentConfig(
            name="valid_experiment",
            random_seed=42
        )
        config.graph_generation.graph_types = [
            {'type': 'euclidean', 'sizes': [10], 'instances_per_size': 5}
        ]
        config.benchmarking.algorithms = [
            {'name': 'nearest_neighbor'}
        ]
        # Disable feature and model stages (or they need models/features specified)
        config.feature_engineering.enabled = False
        config.model_training.enabled = False

        errors = config.validate()
        self.assertEqual(len(errors), 0)

    def test_missing_name(self):
        """Test validation catches missing name."""
        config = ExperimentConfig(name="", random_seed=42)

        errors = config.validate()
        self.assertTrue(any('name' in err.lower() for err in errors))

    def test_invalid_graph_type(self):
        """Test validation catches invalid graph type."""
        config = ExperimentConfig(name="test", random_seed=42)
        config.graph_generation.graph_types = [
            {'type': 'invalid_type', 'sizes': [10], 'instances_per_size': 5}
        ]

        errors = config.validate()
        self.assertTrue(any('type' in err for err in errors))

    def test_invalid_split_ratios(self):
        """Test validation catches invalid split ratios."""
        config = ExperimentConfig(name="test", random_seed=42)
        config.model_training.train_ratio = 0.5
        config.model_training.val_ratio = 0.3
        config.model_training.test_ratio = 0.3  # Sum > 1.0

        errors = config.validate()
        self.assertTrue(any('ratio' in err.lower() for err in errors))

    def test_validator_raises_on_invalid(self):
        """Test ConfigValidator raises ValueError."""
        config = ExperimentConfig(name="", random_seed=42)

        with self.assertRaises(ValueError):
            ConfigValidator.validate(config)

    def test_create_template(self):
        """Test template creation."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            temp_path = f.name

        try:
            ConfigValidator.create_template(temp_path)

            self.assertTrue(Path(temp_path).exists())

            # Validate created template
            config = ExperimentConfig.from_yaml(temp_path)
            errors = config.validate()
            self.assertEqual(len(errors), 0)
        finally:
            Path(temp_path).unlink()


class TestExperimentTracker(unittest.TestCase):
    """Test ExperimentTracker (Prompt 3)."""

    def setUp(self):
        """Set up temporary directory."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up."""
        shutil.rmtree(self.temp_dir)

    def test_tracker_initialization(self):
        """Test tracker creation."""
        tracker = ExperimentTracker(
            experiment_id="test_123",
            name="test_experiment",
            description="Test",
            config={},
            output_dir=Path(self.temp_dir)
        )

        self.assertEqual(tracker.metadata.experiment_id, "test_123")
        self.assertEqual(tracker.metadata.name, "test_experiment")
        self.assertEqual(tracker.metadata.status, ExperimentStatus.PENDING)

    def test_tracker_creates_directories(self):
        """Test tracker creates output directories."""
        tracker = ExperimentTracker(
            experiment_id="test",
            name="test",
            description="",
            config={},
            output_dir=Path(self.temp_dir) / "exp"
        )

        exp_dir = Path(self.temp_dir) / "exp"
        self.assertTrue(exp_dir.exists())
        self.assertTrue((exp_dir / "logs").exists())
        self.assertTrue((exp_dir / "data").exists())
        self.assertTrue((exp_dir / "models").exists())
        self.assertTrue((exp_dir / "reports").exists())

    def test_tracker_lifecycle(self):
        """Test experiment lifecycle (start -> complete)."""
        tracker = ExperimentTracker(
            experiment_id="test",
            name="test",
            description="",
            config={},
            output_dir=Path(self.temp_dir)
        )

        # Initially pending
        self.assertEqual(tracker.metadata.status, ExperimentStatus.PENDING)

        # Start
        tracker.start()
        self.assertEqual(tracker.metadata.status, ExperimentStatus.RUNNING)
        self.assertIsNotNone(tracker.metadata.started_at)

        # Complete
        tracker.complete({'metric': 0.95})
        self.assertEqual(tracker.metadata.status, ExperimentStatus.COMPLETED)
        self.assertIsNotNone(tracker.metadata.completed_at)
        self.assertEqual(tracker.metadata.results_summary['metric'], 0.95)

    def test_tracker_fail(self):
        """Test experiment failure tracking."""
        tracker = ExperimentTracker(
            experiment_id="test",
            name="test",
            description="",
            config={},
            output_dir=Path(self.temp_dir)
        )

        tracker.start()
        tracker.fail("Something went wrong")

        self.assertEqual(tracker.metadata.status, ExperimentStatus.FAILED)
        self.assertEqual(tracker.metadata.error_message, "Something went wrong")

    def test_save_metadata(self):
        """Test metadata saving."""
        tracker = ExperimentTracker(
            experiment_id="test",
            name="test",
            description="Test experiment",
            config={'seed': 42},
            output_dir=Path(self.temp_dir)
        )

        tracker.save_metadata()

        metadata_path = Path(self.temp_dir) / "metadata.json"
        self.assertTrue(metadata_path.exists())

        with open(metadata_path) as f:
            metadata = json.load(f)

        self.assertEqual(metadata['experiment_id'], 'test')
        self.assertEqual(metadata['name'], 'test')


class TestExperimentRegistry(unittest.TestCase):
    """Test ExperimentRegistry (Prompt 3)."""

    def setUp(self):
        """Set up temporary directory."""
        self.temp_dir = tempfile.mkdtemp()
        self.registry_path = Path(self.temp_dir) / "registry.json"

    def tearDown(self):
        """Clean up."""
        shutil.rmtree(self.temp_dir)

    def test_registry_initialization(self):
        """Test registry creation."""
        registry = ExperimentRegistry(self.registry_path)

        self.assertEqual(len(registry.experiments), 0)

    def test_register_experiment(self):
        """Test registering experiments."""
        registry = ExperimentRegistry(self.registry_path)

        metadata = ExperimentMetadata(
            experiment_id="exp1",
            name="test1",
            description="Test",
            status=ExperimentStatus.COMPLETED,
            created_at=datetime.now()
        )

        registry.register(metadata)

        self.assertEqual(len(registry.experiments), 1)
        self.assertEqual(registry.get("exp1").name, "test1")

    def test_registry_persistence(self):
        """Test registry saves and loads."""
        # Create and save
        registry1 = ExperimentRegistry(self.registry_path)
        metadata = ExperimentMetadata(
            experiment_id="exp1",
            name="test",
            description="",
            status=ExperimentStatus.COMPLETED,
            created_at=datetime.now()
        )
        registry1.register(metadata)

        # Load in new instance
        registry2 = ExperimentRegistry(self.registry_path)

        self.assertEqual(len(registry2.experiments), 1)
        self.assertEqual(registry2.get("exp1").name, "test")

    def test_list_by_status(self):
        """Test filtering by status."""
        registry = ExperimentRegistry(self.registry_path)

        for i, status in enumerate([ExperimentStatus.COMPLETED, ExperimentStatus.FAILED, ExperimentStatus.RUNNING]):
            metadata = ExperimentMetadata(
                experiment_id=f"exp{i}",
                name=f"test{i}",
                description="",
                status=status,
                created_at=datetime.now()
            )
            registry.register(metadata)

        completed = registry.list_by_status(ExperimentStatus.COMPLETED)
        self.assertEqual(len(completed), 1)

        failed = registry.list_by_status(ExperimentStatus.FAILED)
        self.assertEqual(len(failed), 1)

    def test_list_by_name(self):
        """Test filtering by name pattern."""
        registry = ExperimentRegistry(self.registry_path)

        for name in ["baseline_v1", "baseline_v2", "improved_v1"]:
            metadata = ExperimentMetadata(
                experiment_id=name,
                name=name,
                description="",
                status=ExperimentStatus.COMPLETED,
                created_at=datetime.now()
            )
            registry.register(metadata)

        baseline = registry.list_by_name("baseline")
        self.assertEqual(len(baseline), 2)

    def test_generate_experiment_id(self):
        """Test experiment ID generation."""
        registry = ExperimentRegistry(self.registry_path)

        exp_id = registry.generate_experiment_id("test")

        self.assertIn("test", exp_id)
        self.assertTrue(len(exp_id) > len("test"))

    def test_get_summary(self):
        """Test summary generation."""
        registry = ExperimentRegistry(self.registry_path)

        # Add some experiments
        for i in range(3):
            metadata = ExperimentMetadata(
                experiment_id=f"exp{i}",
                name=f"test{i}",
                description="",
                status=ExperimentStatus.COMPLETED,
                created_at=datetime.now()
            )
            registry.register(metadata)

        summary = registry.get_summary()

        self.assertEqual(summary['total_experiments'], 3)
        self.assertEqual(summary['by_status']['completed'], 3)


class TestSeedManager(unittest.TestCase):
    """Test SeedManager (Prompt 4)."""

    def test_seed_manager_initialization(self):
        """Test seed manager creation."""
        manager = SeedManager(master_seed=42)

        self.assertEqual(manager.master_seed, 42)

    def test_set_global_seed(self):
        """Test setting global seeds."""
        manager = SeedManager(master_seed=12345)
        manager.set_global_seed()

        # Check Python random
        val1 = random.random()
        manager.set_global_seed()
        val2 = random.random()
        self.assertEqual(val1, val2)

        # Check NumPy
        manager.set_global_seed()
        arr1 = np.random.random(5)
        manager.set_global_seed()
        arr2 = np.random.random(5)
        np.testing.assert_array_equal(arr1, arr2)

    def test_stage_seeds_deterministic(self):
        """Test stage seeds are deterministic."""
        manager = SeedManager(master_seed=42)

        seed1 = manager.get_stage_seed("graph_generation")
        seed2 = manager.get_stage_seed("graph_generation")

        self.assertEqual(seed1, seed2)

    def test_different_stages_different_seeds(self):
        """Test different stages get different seeds."""
        manager = SeedManager(master_seed=42)

        seed_gen = manager.get_stage_seed("graph_generation")
        seed_bench = manager.get_stage_seed("benchmarking")

        self.assertNotEqual(seed_gen, seed_bench)

    def test_graph_seed(self):
        """Test graph-specific seeds."""
        manager = SeedManager(master_seed=42)

        seed0 = manager.get_graph_seed(0)
        seed1 = manager.get_graph_seed(1)

        self.assertNotEqual(seed0, seed1)

        # Deterministic
        seed0_again = manager.get_graph_seed(0)
        self.assertEqual(seed0, seed0_again)

    def test_seed_summary(self):
        """Test seed summary."""
        manager = SeedManager(master_seed=42)

        manager.get_stage_seed("stage1")
        manager.get_stage_seed("stage2")

        summary = manager.get_seed_summary()

        self.assertEqual(summary['master_seed'], 42)
        self.assertIn('stage1', summary['stage_seeds'])
        self.assertIn('stage2', summary['stage_seeds'])


class TestEnvironmentInfo(unittest.TestCase):
    """Test EnvironmentInfo (Prompt 4)."""

    def test_capture_environment(self):
        """Test environment capture."""
        env = EnvironmentInfo.capture()

        self.assertIsNotNone(env.python_version)
        self.assertIsNotNone(env.platform)
        self.assertIsNotNone(env.numpy_version)

    def test_environment_to_dict(self):
        """Test environment serialization."""
        env = EnvironmentInfo.capture()
        env_dict = env.to_dict()

        self.assertIn('python_version', env_dict)
        self.assertIn('numpy_version', env_dict)
        self.assertIn('platform', env_dict)

    def test_get_key_versions(self):
        """Test key version extraction."""
        env = EnvironmentInfo.capture()
        key_versions = env.get_key_versions()

        self.assertIn('python', key_versions)
        self.assertIn('numpy', key_versions)


class TestReproducibilityManager(unittest.TestCase):
    """Test ReproducibilityManager (Prompt 4)."""

    def test_manager_initialization(self):
        """Test reproducibility manager creation."""
        manager = ReproducibilityManager(master_seed=42)

        self.assertEqual(manager.seed_manager.master_seed, 42)
        self.assertIsNotNone(manager.environment_info)

    def test_initialize_sets_seeds(self):
        """Test initialize sets all seeds."""
        manager = ReproducibilityManager(master_seed=999)
        manager.initialize()

        # Test reproducibility
        val1 = random.random()
        manager.initialize()
        val2 = random.random()

        self.assertEqual(val1, val2)

    def test_get_reproducibility_info(self):
        """Test getting complete reproducibility info."""
        manager = ReproducibilityManager(master_seed=42)
        info = manager.get_reproducibility_info()

        self.assertEqual(info['master_seed'], 42)
        self.assertIn('environment', info)
        self.assertIn('reproducibility_check', info)

    def test_save_reproducibility_info(self):
        """Test saving reproducibility info."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = ReproducibilityManager(master_seed=42)
            output_path = Path(temp_dir) / "repro.json"

            manager.save_reproducibility_info(output_path)

            self.assertTrue(output_path.exists())

            with open(output_path) as f:
                info = json.load(f)

            self.assertEqual(info['master_seed'], 42)

    def test_verify_environment(self):
        """Test environment verification."""
        manager = ReproducibilityManager(master_seed=42)
        current_env = manager.environment_info.to_dict()

        # Same environment should match
        result = manager.verify_environment(current_env)
        self.assertTrue(result['environment_matches'])

        # Different environment should not match
        fake_env = current_env.copy()
        fake_env['numpy_version'] = '0.0.0'

        result = manager.verify_environment(fake_env)
        self.assertFalse(result['environment_matches'])
        self.assertIn('numpy', result['differences'])


if __name__ == '__main__':
    unittest.main()
