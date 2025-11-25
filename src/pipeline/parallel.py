"""
Parallel Execution and Scaling (Prompt 7).

Provides parallelization strategies for pipeline stages to maximize
throughput on multi-core machines.
"""

from dataclasses import dataclass
from typing import List, Callable, Any, Optional, Dict
from pathlib import Path
import multiprocessing as mp
import logging


@dataclass
class ParallelConfig:
    """Configuration for parallel execution."""
    n_jobs: int = -1  # -1 = use all cores
    backend: str = 'loky'  # 'loky', 'threading', 'multiprocessing'
    max_memory_mb: Optional[float] = None  # Memory limit per job
    batch_size: Optional[int] = None  # Batch size for chunked processing
    verbose: int = 0  # Verbosity level (0-10)

    def get_n_workers(self) -> int:
        """Get actual number of workers."""
        if self.n_jobs == -1:
            return mp.cpu_count()
        elif self.n_jobs <= 0:
            return max(1, mp.cpu_count() + self.n_jobs)
        else:
            return min(self.n_jobs, mp.cpu_count())


class ParallelExecutor:
    """
    Executes operations in parallel using joblib.

    Supports parallel execution of independent tasks with resource management.
    """

    def __init__(
        self,
        config: Optional[ParallelConfig] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize parallel executor.

        Args:
            config: Parallel configuration
            logger: Logger instance
        """
        self.config = config or ParallelConfig()
        self.logger = logger or logging.getLogger(__name__)

    def map(
        self,
        func: Callable,
        items: List[Any],
        description: str = "Processing"
    ) -> List[Any]:
        """
        Apply function to items in parallel.

        Args:
            func: Function to apply to each item
            items: List of items to process
            description: Description for progress messages

        Returns:
            List of results in same order as items
        """
        try:
            from joblib import Parallel, delayed
        except ImportError:
            self.logger.warning("joblib not available, falling back to sequential execution")
            return [func(item) for item in items]

        n_workers = self.config.get_n_workers()
        self.logger.info(f"{description}: Processing {len(items)} items with {n_workers} workers")

        try:
            results = Parallel(
                n_jobs=n_workers,
                backend=self.config.backend,
                verbose=self.config.verbose
            )(delayed(func)(item) for item in items)

            return results

        except Exception as e:
            self.logger.error(f"Parallel execution failed: {e}")
            self.logger.info("Falling back to sequential execution")
            return [func(item) for item in items]

    def starmap(
        self,
        func: Callable,
        items: List[tuple],
        description: str = "Processing"
    ) -> List[Any]:
        """
        Apply function to tuples of arguments in parallel.

        Args:
            func: Function to apply
            items: List of tuples (args for each function call)
            description: Description for progress messages

        Returns:
            List of results
        """
        try:
            from joblib import Parallel, delayed
        except ImportError:
            self.logger.warning("joblib not available, falling back to sequential execution")
            return [func(*args) for args in items]

        n_workers = self.config.get_n_workers()
        self.logger.info(f"{description}: Processing {len(items)} items with {n_workers} workers")

        try:
            results = Parallel(
                n_jobs=n_workers,
                backend=self.config.backend,
                verbose=self.config.verbose
            )(delayed(func)(*args) for args in items)

            return results

        except Exception as e:
            self.logger.error(f"Parallel execution failed: {e}")
            self.logger.info("Falling back to sequential execution")
            return [func(*args) for args in items]

    def map_with_progress(
        self,
        func: Callable,
        items: List[Any],
        description: str = "Processing",
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> List[Any]:
        """
        Map with progress tracking.

        Args:
            func: Function to apply
            items: Items to process
            description: Description
            progress_callback: Callback(current, total) called periodically

        Returns:
            List of results
        """
        results = []
        total = len(items)

        # For small tasks, just use sequential
        if total < 10:
            for i, item in enumerate(items):
                result = func(item)
                results.append(result)
                if progress_callback:
                    progress_callback(i + 1, total)
            return results

        # For larger tasks, use parallel with progress tracking
        try:
            from joblib import Parallel, delayed

            n_workers = self.config.get_n_workers()
            batch_size = self.config.batch_size or max(1, total // (n_workers * 4))

            self.logger.info(
                f"{description}: {total} items, {n_workers} workers, "
                f"batch size {batch_size}"
            )

            # Process in batches to allow progress updates
            for batch_start in range(0, total, batch_size):
                batch_end = min(batch_start + batch_size, total)
                batch = items[batch_start:batch_end]

                batch_results = Parallel(
                    n_jobs=n_workers,
                    backend=self.config.backend
                )(delayed(func)(item) for item in batch)

                results.extend(batch_results)

                if progress_callback:
                    progress_callback(batch_end, total)

            return results

        except ImportError:
            self.logger.warning("joblib not available, using sequential execution")
            for i, item in enumerate(items):
                result = func(item)
                results.append(result)
                if progress_callback:
                    progress_callback(i + 1, total)
            return results

    def parallel_stage_execution(
        self,
        stage_func: Callable[[Any], Any],
        inputs: List[Any],
        stage_name: str = "stage"
    ) -> List[Any]:
        """
        Execute a pipeline stage in parallel over multiple inputs.

        Args:
            stage_func: Stage function to execute
            inputs: List of inputs for the stage
            stage_name: Name of the stage for logging

        Returns:
            List of outputs
        """
        self.logger.info(f"Executing '{stage_name}' stage in parallel")

        def progress_callback(current: int, total: int):
            if current % max(1, total // 10) == 0 or current == total:
                percent = 100 * current / total
                self.logger.info(f"{stage_name}: {current}/{total} ({percent:.1f}%)")

        results = self.map_with_progress(
            stage_func,
            inputs,
            description=stage_name,
            progress_callback=progress_callback
        )

        self.logger.info(f"'{stage_name}' stage complete: {len(results)} outputs")
        return results


class ResourceManager:
    """
    Manages computational resources for parallel execution.

    Prevents oversubscription and monitors resource usage.
    """

    def __init__(
        self,
        max_memory_gb: Optional[float] = None,
        max_cpu_percent: float = 90.0
    ):
        """
        Initialize resource manager.

        Args:
            max_memory_gb: Maximum memory to use (None = no limit)
            max_cpu_percent: Maximum CPU utilization percentage
        """
        self.max_memory_gb = max_memory_gb
        self.max_cpu_percent = max_cpu_percent

    def get_recommended_workers(
        self,
        task_memory_mb: float,
        min_workers: int = 1
    ) -> int:
        """
        Get recommended number of workers based on resource constraints.

        Args:
            task_memory_mb: Memory required per task (MB)
            min_workers: Minimum number of workers

        Returns:
            Recommended number of workers
        """
        import psutil

        # Get available resources
        available_memory_gb = psutil.virtual_memory().available / (1024 ** 3)
        cpu_count = mp.cpu_count()

        # Calculate memory-constrained workers
        if self.max_memory_gb is not None:
            memory_limit_gb = min(self.max_memory_gb, available_memory_gb)
        else:
            memory_limit_gb = available_memory_gb

        memory_workers = int((memory_limit_gb * 1024) / task_memory_mb)

        # Calculate CPU-constrained workers
        cpu_workers = int(cpu_count * (self.max_cpu_percent / 100))

        # Take minimum of constraints
        recommended = max(min_workers, min(memory_workers, cpu_workers, cpu_count))

        return recommended

    def check_resources(self) -> Dict[str, Any]:
        """
        Check current resource availability.

        Returns:
            Dictionary with resource information
        """
        import psutil

        memory = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=1)

        return {
            'memory': {
                'total_gb': memory.total / (1024 ** 3),
                'available_gb': memory.available / (1024 ** 3),
                'percent_used': memory.percent
            },
            'cpu': {
                'count': mp.cpu_count(),
                'percent_used': cpu_percent
            }
        }


def create_parallel_executor(
    n_jobs: int = -1,
    max_memory_mb: Optional[float] = None,
    verbose: int = 0
) -> ParallelExecutor:
    """
    Convenience function to create a parallel executor.

    Args:
        n_jobs: Number of parallel jobs (-1 = all cores)
        max_memory_mb: Memory limit per job
        verbose: Verbosity level

    Returns:
        Configured ParallelExecutor
    """
    config = ParallelConfig(
        n_jobs=n_jobs,
        max_memory_mb=max_memory_mb,
        verbose=verbose
    )

    return ParallelExecutor(config=config)
