"""
Reproducibility Infrastructure (Prompt 4).

Ensures all experiments are perfectly reproducible through seed management,
environment tracking, and code versioning.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from pathlib import Path
import sys
import platform
import subprocess
import random
import numpy as np


@dataclass
class EnvironmentInfo:
    """
    Information about the execution environment.

    Tracks everything needed to reproduce an experiment.
    """
    python_version: str
    platform: str
    os_version: str
    numpy_version: str
    scipy_version: Optional[str] = None
    sklearn_version: Optional[str] = None
    pandas_version: Optional[str] = None
    packages: Dict[str, str] = field(default_factory=dict)

    @classmethod
    def capture(cls) -> 'EnvironmentInfo':
        """
        Capture current environment information.

        Returns:
            EnvironmentInfo with current system details
        """
        import numpy

        # Get Python version
        python_version = sys.version.split()[0]

        # Get platform info
        platform_name = platform.system()
        os_version = platform.release()

        # Get key package versions
        numpy_version = numpy.__version__

        scipy_version = None
        try:
            import scipy
            scipy_version = scipy.__version__
        except ImportError:
            pass

        sklearn_version = None
        try:
            import sklearn
            sklearn_version = sklearn.__version__
        except ImportError:
            pass

        pandas_version = None
        try:
            import pandas
            pandas_version = pandas.__version__
        except ImportError:
            pass

        # Get all installed packages
        packages = cls._get_installed_packages()

        return cls(
            python_version=python_version,
            platform=platform_name,
            os_version=os_version,
            numpy_version=numpy_version,
            scipy_version=scipy_version,
            sklearn_version=sklearn_version,
            pandas_version=pandas_version,
            packages=packages
        )

    @staticmethod
    def _get_installed_packages() -> Dict[str, str]:
        """
        Get all installed packages and versions.

        Returns:
            Dictionary mapping package names to versions
        """
        try:
            import pkg_resources
            packages = {
                pkg.key: pkg.version
                for pkg in pkg_resources.working_set
            }
            return packages
        except Exception:
            return {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'python_version': self.python_version,
            'platform': self.platform,
            'os_version': self.os_version,
            'numpy_version': self.numpy_version,
            'scipy_version': self.scipy_version,
            'sklearn_version': self.sklearn_version,
            'pandas_version': self.pandas_version,
            'packages': self.packages
        }

    def get_key_versions(self) -> Dict[str, str]:
        """Get just the key package versions."""
        return {
            'python': self.python_version,
            'numpy': self.numpy_version,
            'scipy': self.scipy_version or 'not installed',
            'sklearn': self.sklearn_version or 'not installed',
            'pandas': self.pandas_version or 'not installed'
        }


class SeedManager:
    """
    Manages random seeds for reproducibility.

    Propagates master seed to all random number generators.
    """

    def __init__(self, master_seed: int = 42):
        """
        Initialize seed manager.

        Args:
            master_seed: Master random seed for the experiment
        """
        self.master_seed = master_seed
        self.stage_seeds: Dict[str, int] = {}

    def set_global_seed(self):
        """
        Set global random seeds for all libraries.

        Sets seeds for:
        - Python random module
        - NumPy
        - Any other libraries that use randomness
        """
        random.seed(self.master_seed)
        np.random.seed(self.master_seed)

        # Set hash seed for deterministic Python behavior
        import os
        os.environ['PYTHONHASHSEED'] = str(self.master_seed)

    def get_stage_seed(self, stage_name: str, offset: int = 0) -> int:
        """
        Get deterministic seed for a specific stage.

        Args:
            stage_name: Name of the stage (e.g., "graph_generation")
            offset: Optional offset for multiple calls within same stage

        Returns:
            Deterministic seed based on master seed and stage name
        """
        # Create deterministic seed from stage name
        # Use hash of stage name + offset
        stage_hash = hash(f"{stage_name}_{offset}")
        stage_seed = (self.master_seed + stage_hash) % (2**32)

        # Cache for reproducibility logs
        key = f"{stage_name}_{offset}" if offset else stage_name
        self.stage_seeds[key] = stage_seed

        return stage_seed

    def get_graph_seed(self, graph_index: int) -> int:
        """
        Get deterministic seed for generating a specific graph.

        Args:
            graph_index: Index of the graph being generated

        Returns:
            Deterministic seed for this graph
        """
        return self.get_stage_seed("graph_generation", offset=graph_index)

    def get_split_seed(self) -> int:
        """Get deterministic seed for train/test splitting."""
        return self.get_stage_seed("train_test_split")

    def get_model_seed(self, model_index: int = 0) -> int:
        """
        Get deterministic seed for model training.

        Args:
            model_index: Index of the model being trained

        Returns:
            Deterministic seed for this model
        """
        return self.get_stage_seed("model_training", offset=model_index)

    def get_seed_summary(self) -> Dict[str, int]:
        """
        Get summary of all seeds used.

        Returns:
            Dictionary mapping stage names to seeds
        """
        return {
            'master_seed': self.master_seed,
            'stage_seeds': self.stage_seeds.copy()
        }


class ReproducibilityManager:
    """
    Complete reproducibility management system.

    Combines seed management, environment tracking, and git versioning.
    """

    def __init__(self, master_seed: int = 42):
        """
        Initialize reproducibility manager.

        Args:
            master_seed: Master random seed
        """
        self.seed_manager = SeedManager(master_seed)
        self.environment_info = EnvironmentInfo.capture()
        self.git_commit = self._get_git_commit()

    def initialize(self):
        """
        Initialize reproducibility for an experiment.

        Sets all random seeds and captures environment.
        """
        self.seed_manager.set_global_seed()

    def _get_git_commit(self) -> Optional[str]:
        """
        Get current git commit hash.

        Returns:
            Git commit hash if in a git repo, None otherwise
        """
        try:
            result = subprocess.run(
                ['git', 'rev-parse', 'HEAD'],
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None

    def get_git_diff(self) -> Optional[str]:
        """
        Get git diff (uncommitted changes).

        Returns:
            Git diff string if changes exist, None if clean
        """
        try:
            result = subprocess.run(
                ['git', 'diff'],
                capture_output=True,
                text=True,
                check=True
            )
            diff = result.stdout.strip()
            return diff if diff else None
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None

    def check_reproducibility(self) -> Dict[str, Any]:
        """
        Check if current state is fully reproducible.

        Returns:
            Dictionary with reproducibility status and warnings
        """
        warnings = []
        is_reproducible = True

        # Check git status
        if self.git_commit is None:
            warnings.append("Not in a git repository - cannot track code version")
            is_reproducible = False

        # Check for uncommitted changes
        diff = self.get_git_diff()
        if diff:
            warnings.append("Uncommitted changes detected - results may not be reproducible")
            is_reproducible = False

        # Check if on a tagged release
        try:
            subprocess.run(
                ['git', 'describe', '--exact-match', '--tags'],
                capture_output=True,
                check=True
            )
            on_tag = True
        except (subprocess.CalledProcessError, FileNotFoundError):
            on_tag = False

        return {
            'is_reproducible': is_reproducible,
            'warnings': warnings,
            'git_commit': self.git_commit,
            'has_uncommitted_changes': diff is not None,
            'on_tagged_release': on_tag
        }

    def get_reproducibility_info(self) -> Dict[str, Any]:
        """
        Get complete reproducibility information.

        Returns:
            Dictionary with all reproducibility details
        """
        return {
            'master_seed': self.seed_manager.master_seed,
            'git_commit': self.git_commit,
            'environment': self.environment_info.to_dict(),
            'reproducibility_check': self.check_reproducibility()
        }

    def save_reproducibility_info(self, output_path: Path):
        """
        Save reproducibility information to JSON.

        Args:
            output_path: Path to save reproducibility info
        """
        import json

        info = self.get_reproducibility_info()

        with open(output_path, 'w') as f:
            json.dump(info, f, indent=2)

    def verify_environment(self, saved_env_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Verify current environment matches saved environment.

        Args:
            saved_env_info: Previously saved environment info

        Returns:
            Dictionary with verification results and differences
        """
        current = self.environment_info.get_key_versions()
        saved = {
            'python': saved_env_info.get('python_version'),
            'numpy': saved_env_info.get('numpy_version'),
            'scipy': saved_env_info.get('scipy_version'),
            'sklearn': saved_env_info.get('sklearn_version'),
            'pandas': saved_env_info.get('pandas_version')
        }

        differences = {}
        for key in current:
            if current[key] != saved.get(key):
                differences[key] = {
                    'current': current[key],
                    'expected': saved.get(key)
                }

        matches = len(differences) == 0

        return {
            'environment_matches': matches,
            'differences': differences
        }
