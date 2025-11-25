"""
Performance Monitoring and Profiling (Prompt 6).

Tools for tracking runtime, memory usage, and identifying bottlenecks
in the pipeline execution.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from pathlib import Path
import time
import functools
import json
from datetime import datetime
import psutil
import os


@dataclass
class PerformanceMetrics:
    """Performance metrics for a stage or operation."""
    name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_seconds: float = 0.0
    memory_start_mb: float = 0.0
    memory_end_mb: float = 0.0
    memory_peak_mb: float = 0.0
    memory_delta_mb: float = 0.0
    cpu_percent: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'duration_seconds': self.duration_seconds,
            'memory_start_mb': self.memory_start_mb,
            'memory_end_mb': self.memory_end_mb,
            'memory_peak_mb': self.memory_peak_mb,
            'memory_delta_mb': self.memory_delta_mb,
            'cpu_percent': self.cpu_percent,
            'metadata': self.metadata
        }


class PerformanceMonitor:
    """
    Monitors performance of pipeline stages and operations.

    Tracks timing, memory usage, and generates performance reports.
    """

    def __init__(self):
        """Initialize performance monitor."""
        self.metrics: List[PerformanceMetrics] = []
        self.process = psutil.Process(os.getpid())

    def _get_memory_mb(self) -> float:
        """Get current memory usage in MB."""
        return self.process.memory_info().rss / 1024 / 1024

    def start_monitoring(self, name: str, metadata: Optional[Dict[str, Any]] = None) -> PerformanceMetrics:
        """
        Start monitoring a stage/operation.

        Args:
            name: Name of the stage/operation
            metadata: Optional metadata about the operation

        Returns:
            PerformanceMetrics object to track this operation
        """
        metrics = PerformanceMetrics(
            name=name,
            start_time=datetime.now(),
            memory_start_mb=self._get_memory_mb(),
            metadata=metadata or {}
        )
        return metrics

    def stop_monitoring(self, metrics: PerformanceMetrics):
        """
        Stop monitoring and record final metrics.

        Args:
            metrics: The PerformanceMetrics object from start_monitoring
        """
        metrics.end_time = datetime.now()
        metrics.duration_seconds = (metrics.end_time - metrics.start_time).total_seconds()
        metrics.memory_end_mb = self._get_memory_mb()
        metrics.memory_delta_mb = metrics.memory_end_mb - metrics.memory_start_mb

        # Get CPU percent (over the duration)
        try:
            metrics.cpu_percent = self.process.cpu_percent()
        except:
            metrics.cpu_percent = 0.0

        # Add to history
        self.metrics.append(metrics)

    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary of all monitored operations.

        Returns:
            Summary statistics
        """
        if not self.metrics:
            return {'total_operations': 0}

        total_time = sum(m.duration_seconds for m in self.metrics)
        total_memory = sum(m.memory_delta_mb for m in self.metrics)

        # Group by name
        by_name: Dict[str, List[PerformanceMetrics]] = {}
        for m in self.metrics:
            if m.name not in by_name:
                by_name[m.name] = []
            by_name[m.name].append(m)

        stage_summaries = {}
        for name, metrics_list in by_name.items():
            durations = [m.duration_seconds for m in metrics_list]
            memories = [m.memory_delta_mb for m in metrics_list]

            stage_summaries[name] = {
                'count': len(metrics_list),
                'total_time': sum(durations),
                'mean_time': sum(durations) / len(durations),
                'min_time': min(durations),
                'max_time': max(durations),
                'total_memory_delta': sum(memories),
                'mean_memory_delta': sum(memories) / len(memories)
            }

        return {
            'total_operations': len(self.metrics),
            'total_time_seconds': total_time,
            'total_memory_delta_mb': total_memory,
            'by_stage': stage_summaries
        }

    def get_bottlenecks(self, top_n: int = 5) -> List[Dict[str, Any]]:
        """
        Identify the slowest operations.

        Args:
            top_n: Number of top bottlenecks to return

        Returns:
            List of slowest operations
        """
        sorted_metrics = sorted(
            self.metrics,
            key=lambda m: m.duration_seconds,
            reverse=True
        )

        return [
            {
                'name': m.name,
                'duration_seconds': m.duration_seconds,
                'memory_delta_mb': m.memory_delta_mb,
                'start_time': m.start_time.isoformat(),
                'metadata': m.metadata
            }
            for m in sorted_metrics[:top_n]
        ]

    def save_report(self, output_path: Path):
        """
        Save performance report to JSON.

        Args:
            output_path: Path to save report
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': self.get_summary(),
            'bottlenecks': self.get_bottlenecks(),
            'all_metrics': [m.to_dict() for m in self.metrics]
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)

    def print_summary(self):
        """Print summary to console."""
        summary = self.get_summary()

        print("=" * 60)
        print("Performance Summary")
        print("=" * 60)
        print(f"Total operations: {summary['total_operations']}")
        print(f"Total time: {summary['total_time_seconds']:.2f}s")
        print(f"Total memory delta: {summary['total_memory_delta_mb']:.2f} MB")
        print()

        print("By Stage:")
        print("-" * 60)
        for name, stats in summary.get('by_stage', {}).items():
            print(f"{name:30s} {stats['total_time']:8.2f}s  {stats['count']:3d} ops")

        print()
        print("Top 5 Bottlenecks:")
        print("-" * 60)
        for i, bottleneck in enumerate(self.get_bottlenecks(5), 1):
            print(f"{i}. {bottleneck['name']:30s} {bottleneck['duration_seconds']:8.2f}s")


def profile_stage(monitor: PerformanceMonitor, stage_name: str, metadata: Optional[Dict] = None):
    """
    Decorator to automatically profile a stage function.

    Args:
        monitor: PerformanceMonitor instance
        stage_name: Name of the stage
        metadata: Optional metadata

    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            metrics = monitor.start_monitoring(stage_name, metadata)
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                monitor.stop_monitoring(metrics)
        return wrapper
    return decorator


class RuntimeProfiler:
    """
    Profiles runtime complexity and scaling behavior.

    Useful for understanding how algorithms scale with graph size.
    """

    def __init__(self):
        """Initialize runtime profiler."""
        self.samples: List[Dict[str, Any]] = []

    def record_sample(
        self,
        operation: str,
        input_size: int,
        duration_seconds: float,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Record a profiling sample.

        Args:
            operation: Name of the operation
            input_size: Size of the input (e.g., graph size)
            duration_seconds: How long it took
            metadata: Optional additional information
        """
        self.samples.append({
            'operation': operation,
            'input_size': input_size,
            'duration_seconds': duration_seconds,
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {}
        })

    def analyze_complexity(self, operation: str) -> Dict[str, Any]:
        """
        Analyze runtime complexity for an operation.

        Args:
            operation: Name of the operation to analyze

        Returns:
            Analysis results including estimated complexity
        """
        # Filter samples for this operation
        op_samples = [s for s in self.samples if s['operation'] == operation]

        if len(op_samples) < 3:
            return {
                'operation': operation,
                'samples': len(op_samples),
                'error': 'Need at least 3 samples for complexity analysis'
            }

        # Sort by input size
        op_samples.sort(key=lambda s: s['input_size'])

        sizes = [s['input_size'] for s in op_samples]
        times = [s['duration_seconds'] for s in op_samples]

        # Try to fit different complexity models
        import numpy as np

        n = np.array(sizes)
        t = np.array(times)

        # Linear: O(n)
        linear_fit = np.polyfit(n, t, 1)
        linear_error = np.mean((np.polyval(linear_fit, n) - t) ** 2)

        # Quadratic: O(n²)
        quad_fit = np.polyfit(n, t, 2)
        quad_error = np.mean((np.polyval(quad_fit, n) - t) ** 2)

        # Log-linear: O(n log n)
        try:
            log_n = n * np.log(n)
            loglinear_fit = np.polyfit(log_n, t, 1)
            loglinear_error = np.mean((np.polyval(loglinear_fit, log_n) - t) ** 2)
        except:
            loglinear_error = float('inf')

        # Determine best fit
        fits = {
            'O(n)': linear_error,
            'O(n²)': quad_error,
            'O(n log n)': loglinear_error
        }

        best_complexity = min(fits, key=fits.get)

        return {
            'operation': operation,
            'samples': len(op_samples),
            'size_range': (min(sizes), max(sizes)),
            'time_range': (min(times), max(times)),
            'estimated_complexity': best_complexity,
            'fit_errors': fits,
            'scaling_factor': times[-1] / times[0] if times[0] > 0 else None
        }

    def save_analysis(self, output_path: Path):
        """
        Save runtime analysis to JSON.

        Args:
            output_path: Path to save analysis
        """
        # Get unique operations
        operations = set(s['operation'] for s in self.samples)

        analyses = {}
        for op in operations:
            analyses[op] = self.analyze_complexity(op)

        report = {
            'timestamp': datetime.now().isoformat(),
            'total_samples': len(self.samples),
            'analyses': analyses,
            'raw_samples': self.samples
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
