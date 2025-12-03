"""
Pipeline Integration Module (Phase 5).

This module provides infrastructure for orchestrating multi-stage TSP research
experiments, including configuration management, experiment tracking, and
reproducibility guarantees.

Key Components:
- PipelineStage: Individual pipeline stages (generation, benchmarking, etc.)
- PipelineOrchestrator: Coordinates execution of multi-stage experiments
- ExperimentConfig: Configuration management and validation
- ExperimentTracker: Tracks experiment metadata and results
- ReproducibilityManager: Ensures reproducible experiments
"""

from .orchestrator import (
    PipelineStage,
    PipelineOrchestrator,
    StageResult,
    StageStatus
)

from .config import (
    ExperimentConfig,
    ConfigValidator,
    GraphGenConfig,
    BenchmarkConfig,
    FeatureConfig,
    ModelConfig
)

from .tracking import (
    ExperimentTracker,
    ExperimentRegistry,
    ExperimentMetadata,
    ExperimentStatus
)

from .reproducibility import (
    ReproducibilityManager,
    EnvironmentInfo,
    SeedManager
)

from .validation import (
    StageValidator,
    ValidationError
)

from .profiling import (
    PerformanceMonitor,
    PerformanceMetrics,
    RuntimeProfiler,
    profile_stage
)

from .parallel import (
    ParallelExecutor,
    ParallelConfig,
    ResourceManager,
    create_parallel_executor
)

from .error_handling import (
    ErrorHandler,
    ErrorRecord,
    Checkpoint,
    retry_with_backoff,
    try_continue,
    graceful_degradation
)

# Analysis and Visualization (New Integration Components)
from .test_results_summary import (
    TestResultsSummarizer,
    TestSummary,
    Observation,
    TestStatus,
    summarize_test_results
)

from .analysis import (
    ExperimentAnalyzer
)

from .visualization import (
    ExperimentVisualizer
)

__all__ = [
    # Orchestrator
    'PipelineStage',
    'PipelineOrchestrator',
    'StageResult',
    'StageStatus',

    # Configuration
    'ExperimentConfig',
    'ConfigValidator',
    'GraphGenConfig',
    'BenchmarkConfig',
    'FeatureConfig',
    'ModelConfig',

    # Tracking
    'ExperimentTracker',
    'ExperimentRegistry',
    'ExperimentMetadata',
    'ExperimentStatus',

    # Reproducibility
    'ReproducibilityManager',
    'EnvironmentInfo',
    'SeedManager',

    # Validation (Prompt 5)
    'StageValidator',
    'ValidationError',

    # Profiling (Prompt 6)
    'PerformanceMonitor',
    'PerformanceMetrics',
    'RuntimeProfiler',
    'profile_stage',

    # Parallel Execution (Prompt 7)
    'ParallelExecutor',
    'ParallelConfig',
    'ResourceManager',
    'create_parallel_executor',

    # Error Handling (Prompt 8)
    'ErrorHandler',
    'ErrorRecord',
    'Checkpoint',
    'retry_with_backoff',
    'try_continue',
    'graceful_degradation',

    # Analysis and Visualization (Integration Components)
    'TestResultsSummarizer',
    'TestSummary',
    'Observation',
    'TestStatus',
    'summarize_test_results',
    'ExperimentAnalyzer',
    'ExperimentVisualizer'
]
