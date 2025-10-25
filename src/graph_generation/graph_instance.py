"""
Core data structure for representing TSP graph instances.

This module provides the fundamental GraphInstance class that stores
graph data, metadata, and verified properties with support for
serialization and reproducibility.
"""

import json
import hashlib
import uuid
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import numpy as np


@dataclass
class GraphMetadata:
    """Metadata for a graph instance."""
    graph_type: str  # 'euclidean', 'metric', 'random'
    size: int  # Number of vertices
    generation_params: Dict[str, Any]  # Parameters used to generate the graph
    random_seed: Optional[int]  # Seed for reproducibility
    timestamp: str  # ISO format timestamp
    code_version: Optional[str] = None  # Git commit hash or version

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GraphMetadata':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class GraphProperties:
    """Verified properties of a graph instance."""
    is_metric: bool  # Satisfies triangle inequality
    is_symmetric: bool  # weight(i,j) == weight(j,i)
    weight_range: Tuple[float, float]  # (min, max) weights
    weight_mean: float
    weight_std: float
    density: float  # For sparse graphs, ratio of edges to possible edges
    metricity_score: Optional[float] = None  # Percentage of triplets satisfying triangle inequality
    triangle_violations: int = 0  # Number of triangle inequality violations

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data['weight_range'] = list(data['weight_range'])  # Convert tuple to list for JSON
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GraphProperties':
        """Create from dictionary."""
        data['weight_range'] = tuple(data['weight_range'])  # Convert list back to tuple
        return cls(**data)


class GraphInstance:
    """
    A complete TSP graph instance with adjacency matrix, metadata, and properties.

    This class represents a single graph instance with full reproducibility support,
    property verification, and serialization capabilities.

    Attributes:
        adjacency_matrix: 2D list/array of edge weights
        metadata: Graph generation metadata
        properties: Verified graph properties
        id: Unique identifier for this graph instance
        coordinates: Optional vertex coordinates (for Euclidean graphs)
    """

    def __init__(
        self,
        adjacency_matrix: List[List[float]],
        metadata: GraphMetadata,
        properties: GraphProperties,
        coordinates: Optional[List[Tuple[float, ...]]] = None,
        graph_id: Optional[str] = None
    ):
        """
        Initialize a graph instance.

        Args:
            adjacency_matrix: 2D adjacency matrix of edge weights
            metadata: Graph metadata
            properties: Verified graph properties
            coordinates: Optional vertex coordinates for Euclidean graphs
            graph_id: Optional unique identifier (generated if not provided)
        """
        self.adjacency_matrix = adjacency_matrix
        self.metadata = metadata
        self.properties = properties
        self.coordinates = coordinates
        self.id = graph_id or self._generate_id()

        # Validate size consistency
        if len(adjacency_matrix) != metadata.size:
            raise ValueError(
                f"Adjacency matrix size ({len(adjacency_matrix)}) "
                f"doesn't match metadata size ({metadata.size})"
            )

    def _generate_id(self) -> str:
        """
        Generate a unique identifier based on graph content.

        Uses a hash of the adjacency matrix and metadata to create
        a deterministic but unique ID.
        """
        # Create a deterministic hash from adjacency matrix
        matrix_str = json.dumps(self.adjacency_matrix, sort_keys=True)
        metadata_str = json.dumps(self.metadata.to_dict(), sort_keys=True)

        combined = f"{matrix_str}{metadata_str}"
        hash_digest = hashlib.sha256(combined.encode()).hexdigest()

        # Use first 12 characters for readability
        return hash_digest[:12]

    def get_edge_weight(self, i: int, j: int) -> float:
        """Get the weight of edge between vertices i and j."""
        return self.adjacency_matrix[i][j]

    def get_num_vertices(self) -> int:
        """Get the number of vertices in the graph."""
        return len(self.adjacency_matrix)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert graph instance to dictionary for serialization.

        Returns:
            Dictionary representation suitable for JSON serialization
        """
        data = {
            'id': self.id,
            'adjacency_matrix': self.adjacency_matrix,
            'metadata': self.metadata.to_dict(),
            'properties': self.properties.to_dict(),
        }

        if self.coordinates is not None:
            data['coordinates'] = [list(coord) for coord in self.coordinates]

        return data

    def to_json(self, filepath: str, indent: int = 2) -> None:
        """
        Save graph instance to JSON file.

        Args:
            filepath: Path to save the JSON file
            indent: Indentation level for pretty printing (None for compact)
        """
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GraphInstance':
        """
        Create graph instance from dictionary.

        Args:
            data: Dictionary representation of graph instance

        Returns:
            GraphInstance object
        """
        metadata = GraphMetadata.from_dict(data['metadata'])
        properties = GraphProperties.from_dict(data['properties'])

        coordinates = None
        if 'coordinates' in data and data['coordinates'] is not None:
            coordinates = [tuple(coord) for coord in data['coordinates']]

        return cls(
            adjacency_matrix=data['adjacency_matrix'],
            metadata=metadata,
            properties=properties,
            coordinates=coordinates,
            graph_id=data.get('id')
        )

    @classmethod
    def from_json(cls, filepath: str) -> 'GraphInstance':
        """
        Load graph instance from JSON file.

        Args:
            filepath: Path to JSON file

        Returns:
            GraphInstance object
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)

    def get_filename(self) -> str:
        """
        Generate a standard filename for this graph instance.

        Format: {graph_type}_{size}_{seed}_{id}.json
        """
        seed_str = str(self.metadata.random_seed) if self.metadata.random_seed is not None else 'noseed'
        return f"{self.metadata.graph_type}_{self.metadata.size}_{seed_str}_{self.id}.json"

    def __repr__(self) -> str:
        """String representation of graph instance."""
        return (
            f"GraphInstance(id={self.id}, type={self.metadata.graph_type}, "
            f"size={self.metadata.size}, metric={self.properties.is_metric})"
        )

    def __eq__(self, other: 'GraphInstance') -> bool:
        """Check equality based on adjacency matrix and metadata."""
        if not isinstance(other, GraphInstance):
            return False

        return (
            self.adjacency_matrix == other.adjacency_matrix and
            self.metadata == other.metadata
        )

    def summary(self) -> str:
        """
        Generate a human-readable summary of the graph instance.

        Returns:
            Multi-line string with graph details
        """
        lines = [
            f"Graph Instance: {self.id}",
            f"Type: {self.metadata.graph_type}",
            f"Size: {self.metadata.size} vertices",
            f"Seed: {self.metadata.random_seed}",
            f"",
            f"Properties:",
            f"  - Metric: {self.properties.is_metric}",
            f"  - Symmetric: {self.properties.is_symmetric}",
            f"  - Weight Range: {self.properties.weight_range}",
            f"  - Weight Mean: {self.properties.weight_mean:.2f}",
            f"  - Weight Std Dev: {self.properties.weight_std:.2f}",
        ]

        if self.properties.metricity_score is not None:
            lines.append(f"  - Metricity Score: {self.properties.metricity_score:.2%}")

        if self.properties.triangle_violations > 0:
            lines.append(f"  - Triangle Violations: {self.properties.triangle_violations}")

        if self.coordinates is not None:
            lines.append(f"  - Has Coordinates: Yes ({len(self.coordinates[0])}D)")

        return "\n".join(lines)


def create_graph_instance(
    adjacency_matrix: List[List[float]],
    graph_type: str,
    generation_params: Dict[str, Any],
    random_seed: Optional[int] = None,
    coordinates: Optional[List[Tuple[float, ...]]] = None,
    verify: bool = True
) -> GraphInstance:
    """
    Factory function to create a graph instance with automatic property computation.

    Args:
        adjacency_matrix: 2D adjacency matrix
        graph_type: Type of graph ('euclidean', 'metric', 'random')
        generation_params: Parameters used to generate the graph
        random_seed: Random seed used for generation
        coordinates: Optional vertex coordinates
        verify: Whether to verify properties (can be slow for large graphs)

    Returns:
        GraphInstance with computed properties
    """
    from .verification import verify_graph_properties

    size = len(adjacency_matrix)

    # Create metadata
    metadata = GraphMetadata(
        graph_type=graph_type,
        size=size,
        generation_params=generation_params,
        random_seed=random_seed,
        timestamp=datetime.utcnow().isoformat(),
        code_version=None  # Can be set to git commit hash
    )

    # Verify properties
    if verify:
        properties = verify_graph_properties(adjacency_matrix, coordinates)
    else:
        # Create placeholder properties (will need manual verification)
        weights = [adjacency_matrix[i][j] for i in range(size) for j in range(i+1, size)]
        properties = GraphProperties(
            is_metric=False,
            is_symmetric=True,
            weight_range=(min(weights) if weights else 0, max(weights) if weights else 0),
            weight_mean=sum(weights) / len(weights) if weights else 0,
            weight_std=0.0,
            density=1.0
        )

    return GraphInstance(
        adjacency_matrix=adjacency_matrix,
        metadata=metadata,
        properties=properties,
        coordinates=coordinates
    )
