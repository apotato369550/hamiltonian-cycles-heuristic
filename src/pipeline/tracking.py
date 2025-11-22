"""
Experiment Tracking and Metadata (Prompt 3).

Tracks experiments, their configurations, and results for reproducibility
and comparison.
"""

from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import json
import uuid


class ExperimentStatus(Enum):
    """Status of an experiment."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ExperimentMetadata:
    """
    Metadata for a single experiment.

    Tracks everything needed to understand and reproduce an experiment.
    """
    experiment_id: str
    name: str
    description: str
    status: ExperimentStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    config: Dict[str, Any] = field(default_factory=dict)
    git_commit: Optional[str] = None
    environment: Dict[str, Any] = field(default_factory=dict)
    results_summary: Dict[str, Any] = field(default_factory=dict)
    output_dir: Optional[str] = None
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'experiment_id': self.experiment_id,
            'name': self.name,
            'description': self.description,
            'status': self.status.value,
            'created_at': self.created_at.isoformat(),
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'config': self.config,
            'git_commit': self.git_commit,
            'environment': self.environment,
            'results_summary': self.results_summary,
            'output_dir': self.output_dir,
            'error_message': self.error_message
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExperimentMetadata':
        """Create from dictionary."""
        # Parse datetimes
        created_at = datetime.fromisoformat(data['created_at'])
        started_at = (
            datetime.fromisoformat(data['started_at'])
            if data.get('started_at') else None
        )
        completed_at = (
            datetime.fromisoformat(data['completed_at'])
            if data.get('completed_at') else None
        )

        # Parse status
        status = ExperimentStatus(data['status'])

        return cls(
            experiment_id=data['experiment_id'],
            name=data['name'],
            description=data.get('description', ''),
            status=status,
            created_at=created_at,
            started_at=started_at,
            completed_at=completed_at,
            config=data.get('config', {}),
            git_commit=data.get('git_commit'),
            environment=data.get('environment', {}),
            results_summary=data.get('results_summary', {}),
            output_dir=data.get('output_dir'),
            error_message=data.get('error_message')
        )


class ExperimentTracker:
    """
    Tracks individual experiment execution and metadata.

    Logs progress, captures metadata, and produces structured logs.
    """

    def __init__(
        self,
        experiment_id: str,
        name: str,
        description: str,
        config: Dict[str, Any],
        output_dir: Path
    ):
        """
        Initialize experiment tracker.

        Args:
            experiment_id: Unique experiment ID
            name: Experiment name
            description: Experiment description
            config: Full experiment configuration
            output_dir: Directory for experiment outputs
        """
        self.metadata = ExperimentMetadata(
            experiment_id=experiment_id,
            name=name,
            description=description,
            status=ExperimentStatus.PENDING,
            created_at=datetime.now(),
            config=config,
            output_dir=str(output_dir)
        )

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        (self.output_dir / "logs").mkdir(exist_ok=True)
        (self.output_dir / "data").mkdir(exist_ok=True)
        (self.output_dir / "models").mkdir(exist_ok=True)
        (self.output_dir / "reports").mkdir(exist_ok=True)

    def start(self):
        """Mark experiment as started."""
        self.metadata.status = ExperimentStatus.RUNNING
        self.metadata.started_at = datetime.now()
        self.save_metadata()

    def complete(self, results_summary: Optional[Dict[str, Any]] = None):
        """
        Mark experiment as completed.

        Args:
            results_summary: Dictionary of key results/metrics
        """
        self.metadata.status = ExperimentStatus.COMPLETED
        self.metadata.completed_at = datetime.now()
        if results_summary:
            self.metadata.results_summary = results_summary
        self.save_metadata()

    def fail(self, error_message: str):
        """
        Mark experiment as failed.

        Args:
            error_message: Description of the error
        """
        self.metadata.status = ExperimentStatus.FAILED
        self.metadata.completed_at = datetime.now()
        self.metadata.error_message = error_message
        self.save_metadata()

    def cancel(self):
        """Mark experiment as cancelled."""
        self.metadata.status = ExperimentStatus.CANCELLED
        self.metadata.completed_at = datetime.now()
        self.save_metadata()

    def update_environment(self, env_info: Dict[str, Any]):
        """
        Update environment information.

        Args:
            env_info: Dictionary with Python version, packages, OS, etc.
        """
        self.metadata.environment = env_info
        self.save_metadata()

    def update_git_commit(self, commit_hash: str):
        """
        Update git commit hash.

        Args:
            commit_hash: Git commit hash for reproducibility
        """
        self.metadata.git_commit = commit_hash
        self.save_metadata()

    def save_metadata(self):
        """Save metadata to JSON file."""
        metadata_path = self.output_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata.to_dict(), f, indent=2)

    def get_log_path(self, stage_name: str) -> Path:
        """
        Get log file path for a specific stage.

        Args:
            stage_name: Name of the pipeline stage

        Returns:
            Path to log file
        """
        return self.output_dir / "logs" / f"{stage_name}.log"

    def get_data_path(self, subdir: str) -> Path:
        """
        Get path to data subdirectory.

        Args:
            subdir: Subdirectory name (e.g., 'graphs', 'benchmarks')

        Returns:
            Path to subdirectory
        """
        path = self.output_dir / "data" / subdir
        path.mkdir(parents=True, exist_ok=True)
        return path

    def get_model_path(self) -> Path:
        """Get path to models directory."""
        return self.output_dir / "models"

    def get_report_path(self) -> Path:
        """Get path to reports directory."""
        return self.output_dir / "reports"


class ExperimentRegistry:
    """
    Registry of all experiments for querying and comparison.

    Maintains database (JSON or SQLite) of all experiments.
    """

    def __init__(self, registry_path: Path):
        """
        Initialize experiment registry.

        Args:
            registry_path: Path to registry file (JSON)
        """
        self.registry_path = Path(registry_path)
        self.experiments: Dict[str, ExperimentMetadata] = {}
        self._load()

    def _load(self):
        """Load registry from disk."""
        if self.registry_path.exists():
            with open(self.registry_path, 'r') as f:
                data = json.load(f)

            for exp_data in data.get('experiments', []):
                exp = ExperimentMetadata.from_dict(exp_data)
                self.experiments[exp.experiment_id] = exp

    def _save(self):
        """Save registry to disk."""
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            'experiments': [
                exp.to_dict() for exp in self.experiments.values()
            ]
        }

        with open(self.registry_path, 'w') as f:
            json.dump(data, f, indent=2)

    def register(self, metadata: ExperimentMetadata):
        """
        Register a new experiment.

        Args:
            metadata: ExperimentMetadata to register
        """
        self.experiments[metadata.experiment_id] = metadata
        self._save()

    def update(self, experiment_id: str, metadata: ExperimentMetadata):
        """
        Update existing experiment metadata.

        Args:
            experiment_id: Experiment ID to update
            metadata: Updated ExperimentMetadata
        """
        if experiment_id not in self.experiments:
            raise KeyError(f"Experiment {experiment_id} not found in registry")

        self.experiments[experiment_id] = metadata
        self._save()

    def get(self, experiment_id: str) -> Optional[ExperimentMetadata]:
        """
        Get experiment by ID.

        Args:
            experiment_id: Experiment ID

        Returns:
            ExperimentMetadata if found, None otherwise
        """
        return self.experiments.get(experiment_id)

    def list_all(self) -> List[ExperimentMetadata]:
        """Get list of all experiments."""
        return list(self.experiments.values())

    def list_by_status(self, status: ExperimentStatus) -> List[ExperimentMetadata]:
        """
        Get experiments by status.

        Args:
            status: ExperimentStatus to filter by

        Returns:
            List of matching experiments
        """
        return [
            exp for exp in self.experiments.values()
            if exp.status == status
        ]

    def list_by_name(self, name_pattern: str) -> List[ExperimentMetadata]:
        """
        Get experiments by name pattern.

        Args:
            name_pattern: Name pattern to match (substring match)

        Returns:
            List of matching experiments
        """
        return [
            exp for exp in self.experiments.values()
            if name_pattern.lower() in exp.name.lower()
        ]

    def generate_experiment_id(self, name: str) -> str:
        """
        Generate unique experiment ID.

        Args:
            name: Experiment name

        Returns:
            Unique experiment ID (name + timestamp + short UUID)
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        short_uuid = str(uuid.uuid4())[:8]
        return f"{name}_{timestamp}_{short_uuid}"

    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics of all experiments.

        Returns:
            Dictionary with counts by status, recent experiments, etc.
        """
        total = len(self.experiments)
        by_status = {}
        for status in ExperimentStatus:
            count = len(self.list_by_status(status))
            by_status[status.value] = count

        # Most recent experiments
        recent = sorted(
            self.experiments.values(),
            key=lambda x: x.created_at,
            reverse=True
        )[:5]

        return {
            'total_experiments': total,
            'by_status': by_status,
            'recent_experiments': [
                {
                    'id': exp.experiment_id,
                    'name': exp.name,
                    'status': exp.status.value,
                    'created_at': exp.created_at.isoformat()
                }
                for exp in recent
            ]
        }
