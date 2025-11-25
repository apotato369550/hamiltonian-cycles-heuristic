"""
Error Handling and Fault Tolerance (Prompt 8).

Provides robust error handling, retry logic, and checkpointing for
pipeline resilience.
"""

from dataclasses import dataclass, field
from typing import Callable, Any, Optional, List, Dict
from pathlib import Path
import time
import logging
import json
import functools
from datetime import datetime


@dataclass
class ErrorRecord:
    """Record of an error that occurred during pipeline execution."""
    stage_name: str
    timestamp: datetime
    error_type: str
    error_message: str
    recoverable: bool
    retry_count: int = 0
    context: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'stage_name': self.stage_name,
            'timestamp': self.timestamp.isoformat(),
            'error_type': self.error_type,
            'error_message': self.error_message,
            'recoverable': self.recoverable,
            'retry_count': self.retry_count,
            'context': self.context
        }


class ErrorHandler:
    """
    Manages error handling and recovery strategies.

    Provides try-continue patterns, retry logic, and error reporting.
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize error handler.

        Args:
            logger: Logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
        self.error_log: List[ErrorRecord] = []

    def record_error(
        self,
        stage_name: str,
        error: Exception,
        recoverable: bool = True,
        retry_count: int = 0,
        context: Optional[Dict[str, Any]] = None
    ):
        """
        Record an error.

        Args:
            stage_name: Name of the stage where error occurred
            error: The exception that occurred
            recoverable: Whether the error is recoverable
            retry_count: Number of retries attempted
            context: Additional context information
        """
        record = ErrorRecord(
            stage_name=stage_name,
            timestamp=datetime.now(),
            error_type=type(error).__name__,
            error_message=str(error),
            recoverable=recoverable,
            retry_count=retry_count,
            context=context or {}
        )

        self.error_log.append(record)

        if recoverable:
            self.logger.warning(
                f"{stage_name}: Recoverable error ({type(error).__name__}): {error}"
            )
        else:
            self.logger.error(
                f"{stage_name}: Fatal error ({type(error).__name__}): {error}"
            )

    def get_error_summary(self) -> Dict[str, Any]:
        """
        Get summary of all errors.

        Returns:
            Error summary statistics
        """
        total_errors = len(self.error_log)
        recoverable = sum(1 for e in self.error_log if e.recoverable)
        fatal = total_errors - recoverable

        # Group by stage
        by_stage: Dict[str, int] = {}
        for error in self.error_log:
            by_stage[error.stage_name] = by_stage.get(error.stage_name, 0) + 1

        # Group by error type
        by_type: Dict[str, int] = {}
        for error in self.error_log:
            by_type[error.error_type] = by_type.get(error.error_type, 0) + 1

        return {
            'total_errors': total_errors,
            'recoverable': recoverable,
            'fatal': fatal,
            'by_stage': by_stage,
            'by_type': by_type
        }

    def save_error_log(self, output_path: Path):
        """
        Save error log to JSON.

        Args:
            output_path: Path to save error log
        """
        log_data = {
            'summary': self.get_error_summary(),
            'errors': [e.to_dict() for e in self.error_log]
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(log_data, f, indent=2)

    def print_error_summary(self):
        """Print error summary to console."""
        summary = self.get_error_summary()

        if summary['total_errors'] == 0:
            self.logger.info("No errors occurred")
            return

        self.logger.info("=" * 60)
        self.logger.info("Error Summary")
        self.logger.info("=" * 60)
        self.logger.info(f"Total errors: {summary['total_errors']}")
        self.logger.info(f"  Recoverable: {summary['recoverable']}")
        self.logger.info(f"  Fatal: {summary['fatal']}")

        if summary['by_stage']:
            self.logger.info("\nErrors by stage:")
            for stage, count in summary['by_stage'].items():
                self.logger.info(f"  {stage}: {count}")

        if summary['by_type']:
            self.logger.info("\nErrors by type:")
            for error_type, count in summary['by_type'].items():
                self.logger.info(f"  {error_type}: {count}")


def retry_with_backoff(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: tuple = (Exception,),
    logger: Optional[logging.Logger] = None
):
    """
    Decorator for retry with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds
        backoff_factor: Multiplier for delay after each retry
        exceptions: Tuple of exceptions to catch and retry
        logger: Logger instance

    Returns:
        Decorated function
    """
    log = logger or logging.getLogger(__name__)

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_retries:
                        log.error(f"{func.__name__} failed after {max_retries} retries")
                        raise

                    log.warning(
                        f"{func.__name__} failed (attempt {attempt + 1}/{max_retries}), "
                        f"retrying in {delay:.1f}s: {e}"
                    )
                    time.sleep(delay)
                    delay *= backoff_factor

        return wrapper
    return decorator


def try_continue(
    func: Callable,
    items: List[Any],
    error_handler: ErrorHandler,
    stage_name: str,
    context_fn: Optional[Callable[[Any], Dict]] = None
) -> List[Any]:
    """
    Execute function on items with try-continue pattern.

    Continues processing remaining items even if some fail.

    Args:
        func: Function to apply to each item
        items: List of items to process
        error_handler: ErrorHandler to record errors
        stage_name: Name of the stage for error tracking
        context_fn: Optional function to extract context from item

    Returns:
        List of successful results (may be shorter than items if some failed)
    """
    results = []

    for item in items:
        try:
            result = func(item)
            results.append(result)
        except Exception as e:
            context = context_fn(item) if context_fn else {}
            error_handler.record_error(
                stage_name=stage_name,
                error=e,
                recoverable=True,
                context=context
            )
            # Continue with next item

    return results


class Checkpoint:
    """
    Manages checkpointing for pipeline resumption.

    Allows pipeline to resume from last successful stage after failure.
    """

    def __init__(self, checkpoint_dir: Path):
        """
        Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory to store checkpoints
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_file = self.checkpoint_dir / "checkpoint.json"

    def save(
        self,
        completed_stages: List[str],
        current_outputs: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Save checkpoint.

        Args:
            completed_stages: List of completed stage names
            current_outputs: Current pipeline outputs
            metadata: Optional metadata
        """
        checkpoint_data = {
            'timestamp': datetime.now().isoformat(),
            'completed_stages': completed_stages,
            'outputs': self._serialize_outputs(current_outputs),
            'metadata': metadata or {}
        }

        with open(self.checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)

    def load(self) -> Optional[Dict[str, Any]]:
        """
        Load checkpoint if it exists.

        Returns:
            Checkpoint data or None if no checkpoint exists
        """
        if not self.checkpoint_file.exists():
            return None

        try:
            with open(self.checkpoint_file) as f:
                return json.load(f)
        except Exception:
            return None

    def clear(self):
        """Clear checkpoint file."""
        if self.checkpoint_file.exists():
            self.checkpoint_file.unlink()

    def _serialize_outputs(self, outputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Serialize outputs for checkpointing.

        Converts Path objects to strings.
        """
        serialized = {}
        for key, value in outputs.items():
            if isinstance(value, Path):
                serialized[key] = str(value)
            elif isinstance(value, dict):
                serialized[key] = self._serialize_outputs(value)
            else:
                serialized[key] = value
        return serialized


def graceful_degradation(
    primary_func: Callable,
    fallback_func: Callable,
    error_types: tuple = (Exception,),
    logger: Optional[logging.Logger] = None
) -> Any:
    """
    Try primary function, fall back to simpler version on failure.

    Args:
        primary_func: Primary function to try
        fallback_func: Fallback function if primary fails
        error_types: Exception types to catch
        logger: Logger instance

    Returns:
        Result from primary or fallback function
    """
    log = logger or logging.getLogger(__name__)

    try:
        return primary_func()
    except error_types as e:
        log.warning(f"Primary function failed: {e}, using fallback")
        return fallback_func()
