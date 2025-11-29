"""
Phase 5: Pipeline Integration Tests (Prompts 1-8).

Tests pipeline orchestration, configuration management, experiment tracking,
reproducibility infrastructure, validation, profiling, parallel execution,
and error handling.
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
import time
import logging

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
    SeedManager,

    # Prompt 5: Validation
    StageValidator,
    ValidationError,

    # Prompt 6: Profiling
    PerformanceMonitor,
    PerformanceMetrics,
    RuntimeProfiler,
    profile_stage,

    # Prompt 7: Parallel
    ParallelExecutor,
    ParallelConfig,
    ResourceManager,
    create_parallel_executor,

    # Prompt 8: Error Handling
    ErrorHandler,
    ErrorRecord,
    Checkpoint,
    retry_with_backoff,
    try_continue,
    graceful_degradation
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


# ============================================================================
# Prompt 5: Stage Validation Tests
# ============================================================================

class TestStageValidator(unittest.TestCase):
    """Test StageValidator (Prompt 5)."""

    def setUp(self):
        """Set up temporary directory."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up."""
        shutil.rmtree(self.temp_dir)

    # Graph Generation Validation Tests (3 tests)

    def test_validate_graph_generation_valid(self):
        """Test validation of valid graph directory."""
        graph_dir = Path(self.temp_dir) / "graphs"
        graph_dir.mkdir()

        # Create valid graph files
        for i in range(3):
            graph_data = {
                'n': 5,
                'adjacency_matrix': np.random.rand(5, 5).tolist(),
                'metadata': {'type': 'euclidean', 'seed': i}
            }
            with open(graph_dir / f"graph_{i}.json", 'w') as f:
                json.dump(graph_data, f)

        report = StageValidator.validate_graph_generation_output(graph_dir)

        self.assertTrue(report['valid'])
        self.assertEqual(report['total_graphs'], 3)
        self.assertEqual(report['valid_graphs'], 3)

    def test_validate_graph_generation_missing_directory(self):
        """Test validation catches missing directory."""
        missing_dir = Path(self.temp_dir) / "nonexistent"

        with self.assertRaises(ValidationError) as cm:
            StageValidator.validate_graph_generation_output(missing_dir)

        self.assertIn("does not exist", str(cm.exception))

    def test_validate_graph_generation_empty_directory(self):
        """Test validation catches empty directory."""
        empty_dir = Path(self.temp_dir) / "empty"
        empty_dir.mkdir()

        with self.assertRaises(ValidationError) as cm:
            StageValidator.validate_graph_generation_output(empty_dir)

        self.assertIn("No graph files", str(cm.exception))

    # Benchmarking Validation Tests (3 tests)

    def test_validate_benchmarking_valid(self):
        """Test validation of valid benchmark results."""
        bench_dir = Path(self.temp_dir) / "benchmarks"
        bench_dir.mkdir()

        # Create valid benchmark files
        for i in range(2):
            result_data = {
                'graph_id': f'graph_{i}',
                'algorithms': {
                    'nearest_neighbor': {
                        'tour': list(range(5)),
                        'weight': 10.5,
                        'runtime': 0.001,
                        'success': True
                    }
                }
            }
            with open(bench_dir / f"result_{i}.json", 'w') as f:
                json.dump(result_data, f)

        report = StageValidator.validate_benchmarking_output(bench_dir)

        self.assertTrue(report['valid'])
        self.assertEqual(report['total_results'], 2)

    def test_validate_benchmarking_invalid_tours(self):
        """Test validation catches invalid tours."""
        bench_dir = Path(self.temp_dir) / "benchmarks"
        bench_dir.mkdir()

        # Create result with missing required fields
        result_data = {
            'algorithms': {
                'nearest_neighbor': {
                    'tour': [0, 1, 2]  # Missing weight, runtime, success
                }
            }
        }
        with open(bench_dir / f"result_0.json", 'w') as f:
            json.dump(result_data, f)

        report = StageValidator.validate_benchmarking_output(bench_dir)
        self.assertTrue(len(report['warnings']) > 0)

    def test_validate_benchmarking_missing_algorithms(self):
        """Test validation detects missing algorithm field."""
        bench_dir = Path(self.temp_dir) / "benchmarks"
        bench_dir.mkdir()

        # Create result without algorithms field
        result_data = {'graph_id': 'graph_0'}
        with open(bench_dir / f"result_0.json", 'w') as f:
            json.dump(result_data, f)

        with self.assertRaises(ValidationError):
            StageValidator.validate_benchmarking_output(bench_dir)

    # Feature Validation Tests (3 tests)

    def test_validate_features_valid(self):
        """Test validation of valid features."""
        try:
            import pandas as pd
        except ImportError:
            self.skipTest("pandas not available")

        features_file = Path(self.temp_dir) / "features.csv"

        # Create valid features
        df = pd.DataFrame({
            'graph_id': ['g1', 'g1', 'g2', 'g2'],
            'vertex_id': [0, 1, 0, 1],
            'feature1': [1.0, 2.0, 3.0, 4.0],
            'feature2': [0.5, 0.6, 0.7, 0.8]
        })
        df.to_csv(features_file, index=False)

        report = StageValidator.validate_features_output(features_file)

        self.assertTrue(report['valid'])
        self.assertEqual(report['total_rows'], 4)
        self.assertEqual(report['total_features'], 2)

    def test_validate_features_nan_values(self):
        """Test validation detects NaN values."""
        try:
            import pandas as pd
        except ImportError:
            self.skipTest("pandas not available")

        features_file = Path(self.temp_dir) / "features.csv"

        # Create features with NaN
        df = pd.DataFrame({
            'graph_id': ['g1', 'g1'],
            'vertex_id': [0, 1],
            'feature1': [1.0, np.nan]
        })
        df.to_csv(features_file, index=False)

        report = StageValidator.validate_features_output(features_file)
        self.assertTrue(len(report['warnings']) > 0)
        self.assertTrue(any('NaN' in w for w in report['warnings']))

    def test_validate_features_missing_file(self):
        """Test validation catches missing feature file."""
        missing_file = Path(self.temp_dir) / "missing.csv"

        with self.assertRaises(ValidationError) as cm:
            StageValidator.validate_features_output(missing_file)

        self.assertIn("does not exist", str(cm.exception))

    # Model Training Validation Tests (3 tests)

    def test_validate_model_output_valid(self):
        """Test validation of valid model output."""
        model_dir = Path(self.temp_dir) / "models"
        model_dir.mkdir()

        # Create mock model file
        with open(model_dir / "model.pkl", 'w') as f:
            f.write("mock model")

        # Create evaluation file
        eval_data = {
            'metrics': {'r2': 0.85, 'mae': 0.12}
        }
        with open(model_dir / "model_evaluation.json", 'w') as f:
            json.dump(eval_data, f)

        report = StageValidator.validate_model_output(model_dir)

        self.assertTrue(report['valid'])
        self.assertEqual(report['total_models'], 1)
        self.assertEqual(report['valid_evaluations'], 1)

    def test_validate_model_missing_directory(self):
        """Test validation catches missing model directory."""
        missing_dir = Path(self.temp_dir) / "nonexistent"

        with self.assertRaises(ValidationError):
            StageValidator.validate_model_output(missing_dir)

    def test_validate_model_no_files(self):
        """Test validation handles directory with no model files."""
        model_dir = Path(self.temp_dir) / "models"
        model_dir.mkdir()

        report = StageValidator.validate_model_output(model_dir)

        # Should have warnings but not fail
        self.assertTrue(report['valid'])
        self.assertTrue(len(report['warnings']) > 0)


# ============================================================================
# Prompt 6: Performance Profiling Tests
# ============================================================================

class TestPerformanceMonitor(unittest.TestCase):
    """Test PerformanceMonitor (Prompt 6)."""

    # Basic Monitoring Tests (3 tests)

    def test_monitor_basic_timing(self):
        """Test basic timing monitoring."""
        monitor = PerformanceMonitor()

        metrics = monitor.start_monitoring("test_op")
        time.sleep(0.1)  # Sleep for predictable duration
        monitor.stop_monitoring(metrics)

        self.assertGreaterEqual(metrics.duration_seconds, 0.1)
        self.assertLess(metrics.duration_seconds, 0.2)

    def test_monitor_memory_tracking(self):
        """Test memory delta tracking."""
        monitor = PerformanceMonitor()

        metrics = monitor.start_monitoring("test_op")
        # Memory should be tracked
        self.assertGreater(metrics.memory_start_mb, 0)
        monitor.stop_monitoring(metrics)

        self.assertGreater(metrics.memory_end_mb, 0)

    def test_monitor_cpu_tracking(self):
        """Test CPU usage capture."""
        monitor = PerformanceMonitor()

        metrics = monitor.start_monitoring("test_op")
        monitor.stop_monitoring(metrics)

        # CPU percent should be non-negative
        self.assertGreaterEqual(metrics.cpu_percent, 0.0)

    # Multiple Operations Tests (2 tests)

    def test_monitor_multiple_operations(self):
        """Test tracking multiple named operations."""
        monitor = PerformanceMonitor()

        for i in range(3):
            metrics = monitor.start_monitoring(f"op_{i}")
            time.sleep(0.01)
            monitor.stop_monitoring(metrics)

        self.assertEqual(len(monitor.metrics), 3)

    def test_get_all_metrics(self):
        """Test get_all_metrics returns all tracked operations."""
        monitor = PerformanceMonitor()

        metrics1 = monitor.start_monitoring("op1")
        monitor.stop_monitoring(metrics1)

        metrics2 = monitor.start_monitoring("op2")
        monitor.stop_monitoring(metrics2)

        summary = monitor.get_summary()
        self.assertEqual(summary['total_operations'], 2)
        self.assertIn('op1', summary['by_stage'])
        self.assertIn('op2', summary['by_stage'])

    # Persistence Tests (2 tests)

    def test_save_metrics(self):
        """Test save_metrics to JSON."""
        with tempfile.TemporaryDirectory() as temp_dir:
            monitor = PerformanceMonitor()

            metrics = monitor.start_monitoring("test")
            time.sleep(0.01)
            monitor.stop_monitoring(metrics)

            output_path = Path(temp_dir) / "metrics.json"
            monitor.save_report(output_path)

            self.assertTrue(output_path.exists())

            with open(output_path) as f:
                report = json.load(f)

            self.assertIn('summary', report)
            self.assertIn('all_metrics', report)

    def test_metrics_serialization(self):
        """Test metrics serialization."""
        monitor = PerformanceMonitor()

        metrics = monitor.start_monitoring("test")
        monitor.stop_monitoring(metrics)

        metrics_dict = metrics.to_dict()

        self.assertEqual(metrics_dict['name'], 'test')
        self.assertIn('duration_seconds', metrics_dict)
        self.assertIn('memory_delta_mb', metrics_dict)


class TestRuntimeProfiler(unittest.TestCase):
    """Test RuntimeProfiler (Prompt 6)."""

    # Context Manager Profiling Tests (2 tests)

    def test_profiler_record_sample(self):
        """Test recording profiling samples."""
        profiler = RuntimeProfiler()

        profiler.record_sample("op1", input_size=10, duration_seconds=0.5)
        profiler.record_sample("op1", input_size=20, duration_seconds=1.0)

        self.assertEqual(len(profiler.samples), 2)

    def test_profiler_multiple_operations(self):
        """Test profiling multiple operations."""
        profiler = RuntimeProfiler()

        profiler.record_sample("op1", input_size=10, duration_seconds=0.5)
        profiler.record_sample("op2", input_size=10, duration_seconds=0.3)

        operations = set(s['operation'] for s in profiler.samples)
        self.assertEqual(len(operations), 2)

    # Statistics Generation Tests (2 tests)

    def test_analyze_complexity_basic(self):
        """Test complexity analysis with sufficient samples."""
        profiler = RuntimeProfiler()

        # Linear growth
        for n in [10, 20, 30, 40, 50]:
            profiler.record_sample("linear_op", input_size=n, duration_seconds=n * 0.01)

        analysis = profiler.analyze_complexity("linear_op")

        self.assertEqual(analysis['samples'], 5)
        self.assertIn('estimated_complexity', analysis)

    def test_analyze_complexity_insufficient_samples(self):
        """Test complexity analysis with too few samples."""
        profiler = RuntimeProfiler()

        profiler.record_sample("op", input_size=10, duration_seconds=0.5)

        analysis = profiler.analyze_complexity("op")

        self.assertIn('error', analysis)

    # Decorator Profiling Test (1 test)

    def test_profile_stage_decorator(self):
        """Test @profile_stage decorator."""
        monitor = PerformanceMonitor()

        @profile_stage(monitor, "decorated_op")
        def dummy_function():
            time.sleep(0.01)
            return 42

        result = dummy_function()

        self.assertEqual(result, 42)
        self.assertEqual(len(monitor.metrics), 1)
        self.assertEqual(monitor.metrics[0].name, "decorated_op")


# ============================================================================
# Prompt 7: Parallel Execution Tests
# ============================================================================

class TestParallelExecutor(unittest.TestCase):
    """Test ParallelExecutor (Prompt 7)."""

    # Configuration Test (1 test)

    def test_parallel_config_initialization(self):
        """Test ParallelConfig initialization and validation."""
        config = ParallelConfig(n_jobs=4, backend='loky', max_memory_mb=2000)

        self.assertEqual(config.n_jobs, 4)
        self.assertEqual(config.backend, 'loky')
        self.assertEqual(config.max_memory_mb, 2000)

        # Test get_n_workers
        n_workers = config.get_n_workers()
        self.assertGreater(n_workers, 0)

    # Parallel Execution Tests (2 tests)

    def test_parallel_map_basic(self):
        """Test basic parallel map."""
        config = ParallelConfig(n_jobs=2, verbose=0)
        executor = ParallelExecutor(config)

        def square(x):
            return x * x

        items = [1, 2, 3, 4, 5]
        results = executor.map(square, items, description="Test square")

        self.assertEqual(results, [1, 4, 9, 16, 25])

    def test_parallel_starmap(self):
        """Test parallel starmap with tuples."""
        config = ParallelConfig(n_jobs=2, verbose=0)
        executor = ParallelExecutor(config)

        def add(a, b):
            return a + b

        items = [(1, 2), (3, 4), (5, 6)]
        results = executor.starmap(add, items, description="Test add")

        self.assertEqual(results, [3, 7, 11])

    # Fallback Tests (2 tests)

    def test_parallel_map_without_joblib(self):
        """Test fallback to sequential when joblib unavailable."""
        # This tests the ImportError handling in the map method
        executor = ParallelExecutor()

        def identity(x):
            return x

        # Even if joblib is available, this should work
        results = executor.map(identity, [1, 2, 3])
        self.assertEqual(results, [1, 2, 3])

    def test_create_parallel_executor(self):
        """Test factory function."""
        executor = create_parallel_executor(n_jobs=2, max_memory_mb=1000, verbose=0)

        self.assertIsInstance(executor, ParallelExecutor)
        self.assertEqual(executor.config.n_jobs, 2)

    # Stage Execution Tests (2 tests)

    def test_parallel_stage_execution(self):
        """Test parallel stage execution."""
        executor = ParallelExecutor(ParallelConfig(n_jobs=2, verbose=0))

        def process_item(x):
            return x * 2

        inputs = [1, 2, 3, 4]
        results = executor.parallel_stage_execution(
            stage_func=process_item,
            inputs=inputs,
            stage_name="test_stage"
        )

        self.assertEqual(results, [2, 4, 6, 8])

    def test_parallel_map_with_progress(self):
        """Test parallel map with progress callback."""
        executor = ParallelExecutor(ParallelConfig(n_jobs=2, verbose=0))

        progress_calls = []

        def progress_callback(current, total):
            progress_calls.append((current, total))

        def identity(x):
            return x

        results = executor.map_with_progress(
            identity,
            list(range(20)),
            description="Test",
            progress_callback=progress_callback
        )

        self.assertEqual(len(results), 20)
        # Should have received progress updates
        self.assertGreater(len(progress_calls), 0)


class TestResourceManager(unittest.TestCase):
    """Test ResourceManager (Prompt 7)."""

    # Resource Tracking Tests (2 tests)

    def test_resource_manager_initialization(self):
        """Test resource manager initialization."""
        manager = ResourceManager(max_memory_gb=8.0, max_cpu_percent=80.0)

        self.assertEqual(manager.max_memory_gb, 8.0)
        self.assertEqual(manager.max_cpu_percent, 80.0)

    def test_get_recommended_workers(self):
        """Test recommended workers calculation."""
        manager = ResourceManager(max_memory_gb=8.0)

        # Task requiring 1GB per worker
        workers = manager.get_recommended_workers(task_memory_mb=1000, min_workers=1)

        self.assertGreaterEqual(workers, 1)
        self.assertLessEqual(workers, 8)  # Should not exceed memory limit

    # System Monitoring Tests (1 test)

    def test_check_resources(self):
        """Test system resource checking."""
        manager = ResourceManager()

        resources = manager.check_resources()

        self.assertIn('memory', resources)
        self.assertIn('cpu', resources)
        self.assertIn('total_gb', resources['memory'])
        self.assertIn('percent_used', resources['cpu'])


# ============================================================================
# Prompt 8: Error Handling Tests
# ============================================================================

class TestErrorHandler(unittest.TestCase):
    """Test ErrorHandler (Prompt 8)."""

    # Error Recording Tests (3 tests)

    def test_error_handler_record_error(self):
        """Test error recording."""
        handler = ErrorHandler()

        error = ValueError("Test error")
        handler.record_error(
            stage_name="test_stage",
            error=error,
            recoverable=True,
            context={'graph_id': 'g1'}
        )

        self.assertEqual(len(handler.error_log), 1)
        self.assertEqual(handler.error_log[0].stage_name, "test_stage")
        self.assertEqual(handler.error_log[0].error_type, "ValueError")

    def test_error_categorization(self):
        """Test error categorization (recoverable vs fatal)."""
        handler = ErrorHandler()

        handler.record_error("stage1", ValueError("Recoverable"), recoverable=True)
        handler.record_error("stage2", RuntimeError("Fatal"), recoverable=False)

        summary = handler.get_error_summary()
        self.assertEqual(summary['total_errors'], 2)
        self.assertEqual(summary['recoverable'], 1)
        self.assertEqual(summary['fatal'], 1)

    def test_error_summary_generation(self):
        """Test error summary statistics."""
        handler = ErrorHandler()

        handler.record_error("stage1", ValueError("Error 1"), recoverable=True)
        handler.record_error("stage1", ValueError("Error 2"), recoverable=True)
        handler.record_error("stage2", TypeError("Error 3"), recoverable=False)

        summary = handler.get_error_summary()

        self.assertEqual(summary['by_stage']['stage1'], 2)
        self.assertEqual(summary['by_stage']['stage2'], 1)
        self.assertEqual(summary['by_type']['ValueError'], 2)
        self.assertEqual(summary['by_type']['TypeError'], 1)

    # Error Retrieval Tests (2 tests)

    def test_error_persistence(self):
        """Test saving errors to JSON."""
        with tempfile.TemporaryDirectory() as temp_dir:
            handler = ErrorHandler()

            handler.record_error("stage1", ValueError("Test"), recoverable=True)

            output_path = Path(temp_dir) / "errors.json"
            handler.save_error_log(output_path)

            self.assertTrue(output_path.exists())

            with open(output_path) as f:
                log_data = json.load(f)

            self.assertIn('summary', log_data)
            self.assertIn('errors', log_data)
            self.assertEqual(len(log_data['errors']), 1)

    def test_error_handler_multiple_stages(self):
        """Test handling many errors across stages."""
        handler = ErrorHandler()

        for i in range(10):
            stage = f"stage{i % 3}"
            handler.record_error(stage, ValueError(f"Error {i}"), recoverable=True)

        summary = handler.get_error_summary()
        self.assertEqual(summary['total_errors'], 10)
        self.assertEqual(len(summary['by_stage']), 3)


class TestCheckpoint(unittest.TestCase):
    """Test Checkpoint (Prompt 8)."""

    # Save/Load Tests (2 tests)

    def test_checkpoint_save_load(self):
        """Test checkpoint save and load."""
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = Checkpoint(Path(temp_dir))

            # Save checkpoint
            checkpoint.save(
                completed_stages=['stage1', 'stage2'],
                current_outputs={'result': 42},
                metadata={'note': 'test'}
            )

            # Load checkpoint
            data = checkpoint.load()

            self.assertIsNotNone(data)
            self.assertEqual(data['completed_stages'], ['stage1', 'stage2'])
            self.assertEqual(data['outputs']['result'], 42)

    def test_checkpoint_exists(self):
        """Test checkpoint existence check."""
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = Checkpoint(Path(temp_dir))

            # Initially doesn't exist
            self.assertIsNone(checkpoint.load())

            # After save, exists
            checkpoint.save([], {})
            self.assertIsNotNone(checkpoint.load())

            # After clear, doesn't exist
            checkpoint.clear()
            self.assertIsNone(checkpoint.load())


class TestRetryDecorators(unittest.TestCase):
    """Test Retry Decorators (Prompt 8)."""

    # Retry with Backoff Test (1 test)

    def test_retry_with_backoff(self):
        """Test retry_with_backoff eventually succeeds."""
        attempt_count = [0]

        @retry_with_backoff(max_retries=3, initial_delay=0.01, backoff_factor=2.0)
        def flaky_function():
            attempt_count[0] += 1
            if attempt_count[0] < 3:
                raise ValueError("Not yet")
            return "success"

        result = flaky_function()

        self.assertEqual(result, "success")
        self.assertEqual(attempt_count[0], 3)

    # Try Continue Test (1 test)

    def test_try_continue_pattern(self):
        """Test try_continue continues on failures."""
        handler = ErrorHandler()

        def process_item(x):
            if x == 2:
                raise ValueError("Item 2 fails")
            return x * 2

        items = [1, 2, 3, 4]
        results = try_continue(
            func=process_item,
            items=items,
            error_handler=handler,
            stage_name="test_stage"
        )

        # Should have 3 results (item 2 failed)
        self.assertEqual(len(results), 3)
        self.assertEqual(results, [2, 6, 8])

        # Should have 1 error recorded
        self.assertEqual(len(handler.error_log), 1)

    # Graceful Degradation Test (1 test)

    def test_graceful_degradation(self):
        """Test graceful_degradation returns default on failure."""
        def failing_func():
            raise RuntimeError("Failed")

        def fallback_func():
            return "fallback"

        result = graceful_degradation(failing_func, fallback_func)

        self.assertEqual(result, "fallback")


if __name__ == '__main__':
    unittest.main()
