"""
Pipeline Orchestrator (Prompt 1).

Coordinates execution of multi-stage TSP research pipeline with modularity,
idempotency, and resumability.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
import time
import logging
from datetime import datetime


class StageStatus(Enum):
    """Status of a pipeline stage."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class StageResult:
    """
    Result of executing a pipeline stage.

    Attributes:
        stage_name: Name of the stage
        status: Final status of the stage
        start_time: When stage started
        end_time: When stage completed
        duration_seconds: How long stage took
        outputs: Dictionary of output paths/data produced
        metadata: Additional metadata (counts, statistics, etc.)
        error: Error message if failed
    """
    stage_name: str
    status: StageStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_seconds: float = 0.0
    outputs: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'stage_name': self.stage_name,
            'status': self.status.value,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'duration_seconds': self.duration_seconds,
            'outputs': self.outputs,
            'metadata': self.metadata,
            'error': self.error
        }


class PipelineStage:
    """
    Individual pipeline stage (graph generation, benchmarking, etc.).

    Each stage:
    - Has a unique name
    - Takes inputs and produces outputs
    - Can be run independently
    - Is idempotent (same inputs → same outputs)
    - Supports resumption (skips if outputs already exist)
    """

    def __init__(
        self,
        name: str,
        execute_fn: Callable[[Dict[str, Any]], StageResult],
        required_inputs: Optional[List[str]] = None,
        output_keys: Optional[List[str]] = None,
        skip_if_exists: bool = True,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize pipeline stage.

        Args:
            name: Stage name (e.g., "graph_generation")
            execute_fn: Function that executes the stage
            required_inputs: List of required input keys
            output_keys: List of output keys this stage produces
            skip_if_exists: If True, skip if outputs already exist
            logger: Logger instance
        """
        self.name = name
        self.execute_fn = execute_fn
        self.required_inputs = required_inputs or []
        self.output_keys = output_keys or []
        self.skip_if_exists = skip_if_exists
        self.logger = logger or logging.getLogger(__name__)

    def validate_inputs(self, inputs: Dict[str, Any]) -> bool:
        """
        Check if all required inputs are present.

        Args:
            inputs: Dictionary of available inputs

        Returns:
            True if all required inputs present, False otherwise
        """
        missing = [key for key in self.required_inputs if key not in inputs]
        if missing:
            self.logger.error(f"Stage '{self.name}' missing inputs: {missing}")
            return False
        return True

    def check_outputs_exist(self, outputs: Dict[str, Any]) -> bool:
        """
        Check if outputs already exist (for resumption).

        Args:
            outputs: Dictionary of expected outputs

        Returns:
            True if all outputs exist, False otherwise
        """
        if not self.skip_if_exists:
            return False

        for key in self.output_keys:
            if key not in outputs:
                return False

            value = outputs[key]
            # Check if it's a file path
            if isinstance(value, (str, Path)):
                if not Path(value).exists():
                    return False
            # For other types, assume exists if key present
        return True

    def execute(self, inputs: Dict[str, Any]) -> StageResult:
        """
        Execute the stage.

        Args:
            inputs: Dictionary of inputs (from previous stages or config)

        Returns:
            StageResult containing status and outputs
        """
        start_time = datetime.now()
        result = StageResult(
            stage_name=self.name,
            status=StageStatus.RUNNING,
            start_time=start_time
        )

        try:
            # Validate inputs
            if not self.validate_inputs(inputs):
                result.status = StageStatus.FAILED
                result.error = f"Missing required inputs: {self.required_inputs}"
                result.end_time = datetime.now()
                return result

            # Check if can skip (outputs already exist)
            if self.check_outputs_exist(inputs):
                self.logger.info(f"Stage '{self.name}' outputs already exist, skipping")
                result.status = StageStatus.SKIPPED
                result.end_time = datetime.now()
                result.duration_seconds = (result.end_time - start_time).total_seconds()
                return result

            # Execute stage function
            self.logger.info(f"Executing stage '{self.name}'")
            result = self.execute_fn(inputs)
            result.end_time = datetime.now()
            result.duration_seconds = (result.end_time - start_time).total_seconds()

            if result.status == StageStatus.RUNNING:
                result.status = StageStatus.COMPLETED

            self.logger.info(
                f"Stage '{self.name}' completed in {result.duration_seconds:.2f}s"
            )
            return result

        except Exception as e:
            self.logger.exception(f"Stage '{self.name}' failed: {e}")
            result.status = StageStatus.FAILED
            result.error = str(e)
            result.end_time = datetime.now()
            result.duration_seconds = (result.end_time - start_time).total_seconds()
            return result


class PipelineOrchestrator:
    """
    Orchestrates execution of multi-stage research pipeline.

    Coordinates: graph generation → benchmarking → feature engineering →
    model training → evaluation

    Features:
    - Modular: stages run independently
    - Resumable: skip stages with existing outputs
    - Observable: clear logging at each step
    - Configurable: all parameters from config
    """

    def __init__(
        self,
        experiment_dir: Path,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize pipeline orchestrator.

        Args:
            experiment_dir: Root directory for experiment outputs
            logger: Logger instance
        """
        self.experiment_dir = Path(experiment_dir)
        self.logger = logger or logging.getLogger(__name__)
        self.stages: List[PipelineStage] = []
        self.stage_results: List[StageResult] = []

    def add_stage(self, stage: PipelineStage):
        """
        Add a stage to the pipeline.

        Args:
            stage: PipelineStage to add
        """
        self.stages.append(stage)
        self.logger.debug(f"Added stage: {stage.name}")

    def run(
        self,
        initial_inputs: Optional[Dict[str, Any]] = None,
        start_from_stage: Optional[str] = None,
        stop_at_stage: Optional[str] = None
    ) -> List[StageResult]:
        """
        Run the pipeline.

        Args:
            initial_inputs: Initial inputs (e.g., config, data paths)
            start_from_stage: Optional stage name to start from
            stop_at_stage: Optional stage name to stop at

        Returns:
            List of StageResult for each executed stage
        """
        self.logger.info("=" * 60)
        self.logger.info("Starting pipeline execution")
        self.logger.info("=" * 60)

        pipeline_start = time.time()
        inputs = initial_inputs or {}
        self.stage_results = []

        # Determine which stages to run
        start_idx = 0
        end_idx = len(self.stages)

        if start_from_stage:
            for i, stage in enumerate(self.stages):
                if stage.name == start_from_stage:
                    start_idx = i
                    break

        if stop_at_stage:
            for i, stage in enumerate(self.stages):
                if stage.name == stop_at_stage:
                    end_idx = i + 1
                    break

        # Execute stages
        for i in range(start_idx, end_idx):
            stage = self.stages[i]
            self.logger.info(f"\nStage {i+1}/{len(self.stages)}: {stage.name}")
            self.logger.info("-" * 60)

            # Execute stage
            result = stage.execute(inputs)
            self.stage_results.append(result)

            # Update inputs with outputs for next stage
            if result.outputs:
                inputs.update(result.outputs)

            # Stop if stage failed
            if result.status == StageStatus.FAILED:
                self.logger.error(f"Stage '{stage.name}' failed, stopping pipeline")
                self.logger.error(f"Error: {result.error}")
                break

        # Summary
        pipeline_duration = time.time() - pipeline_start
        self.logger.info("\n" + "=" * 60)
        self.logger.info("Pipeline execution complete")
        self.logger.info(f"Total duration: {pipeline_duration:.2f}s")
        self.logger.info("=" * 60)

        self._log_summary()

        return self.stage_results

    def _log_summary(self):
        """Log summary of pipeline execution."""
        self.logger.info("\nStage Summary:")
        self.logger.info("-" * 60)

        for result in self.stage_results:
            status_symbol = {
                StageStatus.COMPLETED: "✓",
                StageStatus.SKIPPED: "⊙",
                StageStatus.FAILED: "✗",
                StageStatus.RUNNING: "→"
            }.get(result.status, "?")

            self.logger.info(
                f"{status_symbol} {result.stage_name:20s} "
                f"{result.status.value:10s} "
                f"{result.duration_seconds:6.2f}s"
            )

    def get_manifest(self) -> Dict[str, Any]:
        """
        Get manifest of pipeline execution.

        Returns:
            Dictionary with all stage results and outputs
        """
        return {
            'experiment_dir': str(self.experiment_dir),
            'total_stages': len(self.stages),
            'executed_stages': len(self.stage_results),
            'stages': [result.to_dict() for result in self.stage_results]
        }

    def save_manifest(self, path: Optional[Path] = None):
        """
        Save pipeline manifest to JSON.

        Args:
            path: Path to save manifest (default: experiment_dir/manifest.json)
        """
        import json

        if path is None:
            path = self.experiment_dir / "manifest.json"

        manifest = self.get_manifest()

        with open(path, 'w') as f:
            json.dump(manifest, f, indent=2)

        self.logger.info(f"Saved manifest to {path}")
