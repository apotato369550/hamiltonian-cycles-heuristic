"""
Graph instance storage and retrieval system.

Provides functionality to save, load, and query graph instances
with support for batching and integrity verification.
"""

import os
import json
import glob
from typing import List, Optional, Dict, Any, Callable
from pathlib import Path
from datetime import datetime
from .graph_instance import GraphInstance


class GraphStorage:
    """
    Storage system for graph instances.

    Handles saving graphs to disk, loading them back, and querying
    collections of graphs based on properties.
    """

    def __init__(self, base_directory: str = "data/graphs"):
        """
        Initialize the storage system.

        Args:
            base_directory: Root directory for storing graphs
        """
        self.base_directory = Path(base_directory)
        self.base_directory.mkdir(parents=True, exist_ok=True)

    def save_graph(
        self,
        graph: GraphInstance,
        subdirectory: Optional[str] = None,
        filename: Optional[str] = None,
        compress: bool = False
    ) -> str:
        """
        Save a graph instance to disk.

        Args:
            graph: Graph instance to save
            subdirectory: Optional subdirectory within base_directory
            filename: Optional custom filename (uses graph.get_filename() if None)
            compress: Whether to compress the JSON (not implemented yet)

        Returns:
            Path to saved file
        """
        # Determine save directory
        if subdirectory:
            save_dir = self.base_directory / subdirectory
        else:
            save_dir = self.base_directory
        save_dir.mkdir(parents=True, exist_ok=True)

        # Determine filename
        if filename is None:
            filename = graph.get_filename()

        filepath = save_dir / filename

        # Save to JSON
        graph.to_json(str(filepath), indent=2)

        return str(filepath)

    def load_graph(self, filepath: str, verify: bool = False) -> GraphInstance:
        """
        Load a graph instance from disk.

        Args:
            filepath: Path to JSON file
            verify: Whether to re-verify properties on load

        Returns:
            GraphInstance object
        """
        graph = GraphInstance.from_json(filepath)

        if verify:
            # Re-verify properties
            from .verification import GraphVerifier
            verifier = GraphVerifier(fast_mode=True)
            results = verifier.verify_all(graph.adjacency_matrix, graph.coordinates)

            # Check if verification failed
            failed = [r for r in results if not r.passed]
            if failed:
                print(f"Warning: Graph {graph.id} failed verification on load:")
                for result in failed:
                    print(f"  - {result.property_name}: {result.errors[0] if result.errors else 'unknown error'}")

        return graph

    def save_batch(
        self,
        graphs: List[GraphInstance],
        batch_name: str,
        create_manifest: bool = True
    ) -> Dict[str, Any]:
        """
        Save a batch of graphs with a manifest file.

        Args:
            graphs: List of graph instances
            batch_name: Name for this batch
            create_manifest: Whether to create a manifest file

        Returns:
            Dictionary with batch metadata
        """
        # Create batch subdirectory
        batch_dir = self.base_directory / batch_name
        batch_dir.mkdir(parents=True, exist_ok=True)

        # Save all graphs
        saved_files = []
        for graph in graphs:
            filepath = self.save_graph(graph, subdirectory=batch_name)
            saved_files.append(os.path.basename(filepath))

        # Create manifest
        manifest = {
            'batch_name': batch_name,
            'timestamp': datetime.utcnow().isoformat(),
            'num_graphs': len(graphs),
            'graphs': []
        }

        for graph, filename in zip(graphs, saved_files):
            manifest['graphs'].append({
                'id': graph.id,
                'filename': filename,
                'type': graph.metadata.graph_type,
                'size': graph.metadata.size,
                'seed': graph.metadata.random_seed,
                'is_metric': graph.properties.is_metric,
                'weight_range': list(graph.properties.weight_range)
            })

        if create_manifest:
            manifest_path = batch_dir / 'manifest.json'
            with open(manifest_path, 'w') as f:
                json.dump(manifest, f, indent=2)

        return manifest

    def load_batch(
        self,
        batch_name: str,
        verify: bool = False
    ) -> List[GraphInstance]:
        """
        Load all graphs from a batch.

        Args:
            batch_name: Name of the batch
            verify: Whether to verify graphs on load

        Returns:
            List of graph instances
        """
        batch_dir = self.base_directory / batch_name

        if not batch_dir.exists():
            raise FileNotFoundError(f"Batch directory not found: {batch_dir}")

        # Load manifest if it exists
        manifest_path = batch_dir / 'manifest.json'
        if manifest_path.exists():
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
            filenames = [item['filename'] for item in manifest['graphs']]
        else:
            # Find all JSON files in directory
            filenames = [f.name for f in batch_dir.glob('*.json') if f.name != 'manifest.json']

        # Load all graphs
        graphs = []
        for filename in filenames:
            filepath = batch_dir / filename
            try:
                graph = self.load_graph(str(filepath), verify=verify)
                graphs.append(graph)
            except Exception as e:
                print(f"Error loading {filename}: {e}")

        return graphs

    def find_graphs(
        self,
        graph_type: Optional[str] = None,
        size_range: Optional[tuple] = None,
        is_metric: Optional[bool] = None,
        custom_filter: Optional[Callable[[GraphInstance], bool]] = None,
        subdirectory: Optional[str] = None
    ) -> List[GraphInstance]:
        """
        Find graphs matching specified criteria.

        Args:
            graph_type: Filter by graph type ('euclidean', 'metric', 'random')
            size_range: Filter by size range (min_size, max_size)
            is_metric: Filter by metric property
            custom_filter: Custom filter function
            subdirectory: Search in specific subdirectory only

        Returns:
            List of matching graph instances
        """
        # Determine search directory
        if subdirectory:
            search_dir = self.base_directory / subdirectory
        else:
            search_dir = self.base_directory

        if not search_dir.exists():
            return []

        # Find all JSON files
        json_files = []
        if subdirectory:
            json_files = list(search_dir.glob('*.json'))
        else:
            json_files = list(search_dir.rglob('*.json'))

        # Filter out manifest files
        json_files = [f for f in json_files if f.name != 'manifest.json']

        # Load and filter graphs
        matching_graphs = []
        for filepath in json_files:
            try:
                graph = self.load_graph(str(filepath), verify=False)

                # Apply filters
                if graph_type is not None and graph.metadata.graph_type != graph_type:
                    continue

                if size_range is not None:
                    min_size, max_size = size_range
                    if not (min_size <= graph.metadata.size <= max_size):
                        continue

                if is_metric is not None and graph.properties.is_metric != is_metric:
                    continue

                if custom_filter is not None and not custom_filter(graph):
                    continue

                matching_graphs.append(graph)

            except Exception as e:
                print(f"Error loading {filepath}: {e}")

        return matching_graphs

    def get_graph_by_id(self, graph_id: str) -> Optional[GraphInstance]:
        """
        Find and load a graph by its ID.

        Args:
            graph_id: Graph ID to search for

        Returns:
            GraphInstance if found, None otherwise
        """
        # Search all JSON files
        for filepath in self.base_directory.rglob('*.json'):
            if filepath.name == 'manifest.json':
                continue

            try:
                # Quick check: see if ID is in filename
                if graph_id in filepath.name:
                    graph = self.load_graph(str(filepath), verify=False)
                    if graph.id == graph_id:
                        return graph
            except Exception:
                continue

        return None

    def delete_graph(self, graph_id: str) -> bool:
        """
        Delete a graph by ID.

        Args:
            graph_id: Graph ID to delete

        Returns:
            True if deleted, False if not found
        """
        for filepath in self.base_directory.rglob('*.json'):
            if filepath.name == 'manifest.json':
                continue

            try:
                if graph_id in filepath.name:
                    graph = GraphInstance.from_json(str(filepath))
                    if graph.id == graph_id:
                        filepath.unlink()
                        return True
            except Exception:
                continue

        return False

    def get_storage_stats(self) -> Dict[str, Any]:
        """
        Get statistics about stored graphs.

        Returns:
            Dictionary with storage statistics
        """
        total_graphs = 0
        total_size = 0
        by_type = {}
        by_size = {}

        for filepath in self.base_directory.rglob('*.json'):
            if filepath.name == 'manifest.json':
                continue

            try:
                total_size += filepath.stat().st_size
                total_graphs += 1

                # Load minimal info from JSON without full parsing
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    graph_type = data.get('metadata', {}).get('graph_type', 'unknown')
                    size = data.get('metadata', {}).get('size', 0)

                    by_type[graph_type] = by_type.get(graph_type, 0) + 1
                    by_size[size] = by_size.get(size, 0) + 1

            except Exception:
                continue

        return {
            'total_graphs': total_graphs,
            'total_size_bytes': total_size,
            'total_size_mb': total_size / (1024 * 1024),
            'by_type': by_type,
            'by_size': sorted(by_size.items())
        }

    def list_batches(self) -> List[str]:
        """
        List all batch directories.

        Returns:
            List of batch names
        """
        batches = []
        for item in self.base_directory.iterdir():
            if item.is_dir():
                batches.append(item.name)
        return batches

    def export_manifest(self, output_file: str) -> None:
        """
        Export a complete manifest of all stored graphs.

        Args:
            output_file: Path to output manifest file
        """
        graphs_info = []

        for filepath in self.base_directory.rglob('*.json'):
            if filepath.name == 'manifest.json':
                continue

            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)

                graphs_info.append({
                    'id': data.get('id'),
                    'filepath': str(filepath.relative_to(self.base_directory)),
                    'type': data.get('metadata', {}).get('graph_type'),
                    'size': data.get('metadata', {}).get('size'),
                    'seed': data.get('metadata', {}).get('random_seed'),
                    'is_metric': data.get('properties', {}).get('is_metric'),
                    'is_symmetric': data.get('properties', {}).get('is_symmetric'),
                })
            except Exception:
                continue

        manifest = {
            'timestamp': datetime.utcnow().isoformat(),
            'total_graphs': len(graphs_info),
            'graphs': graphs_info
        }

        with open(output_file, 'w') as f:
            json.dump(manifest, f, indent=2)


# Convenience functions

def save_graph(
    graph: GraphInstance,
    directory: str = "data/graphs",
    subdirectory: Optional[str] = None
) -> str:
    """
    Convenience function to save a graph.

    Args:
        graph: Graph instance to save
        directory: Base directory
        subdirectory: Optional subdirectory

    Returns:
        Path to saved file
    """
    storage = GraphStorage(base_directory=directory)
    return storage.save_graph(graph, subdirectory=subdirectory)


def load_graph(filepath: str, verify: bool = False) -> GraphInstance:
    """
    Convenience function to load a graph.

    Args:
        filepath: Path to graph JSON file
        verify: Whether to verify properties

    Returns:
        GraphInstance
    """
    return GraphInstance.from_json(filepath)
