"""
Benchmark results storage system.

Stores algorithm performance results for later analysis and label generation.
Similar to GraphStorage but for algorithm results.
"""

import json
import sqlite3
from pathlib import Path
from typing import Dict, Any, List, Optional


class BenchmarkStorage:
    """
    Store and retrieve benchmark results.

    Storage format: SQLite database for efficient querying

    Schema:
        benchmarks (
            id INTEGER PRIMARY KEY,
            graph_id TEXT,
            graph_type TEXT,
            graph_size INTEGER,
            algorithm TEXT,
            anchor_vertex INTEGER,
            tour_weight REAL,
            runtime REAL,
            tour TEXT,  -- JSON serialized
            metadata TEXT,  -- JSON serialized (includes error messages if any)
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """

    def __init__(self, storage_dir: str):
        """
        Initialize benchmark storage.

        Args:
            storage_dir: Directory to store benchmark database
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.storage_dir / 'benchmarks.db'
        self._init_database()

    def _init_database(self):
        """Initialize database schema if it doesn't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS benchmarks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                graph_id TEXT NOT NULL,
                graph_type TEXT,
                graph_size INTEGER,
                algorithm TEXT NOT NULL,
                anchor_vertex INTEGER,
                tour_weight REAL,
                runtime REAL,
                tour TEXT,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Create indices for common queries
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_graph_id ON benchmarks(graph_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_algorithm ON benchmarks(algorithm)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_graph_type ON benchmarks(graph_type)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_graph_size ON benchmarks(graph_size)")

        conn.commit()
        conn.close()

    def save_result(self, result: Dict[str, Any]) -> int:
        """
        Save single benchmark result.

        Args:
            result: Dictionary with keys:
                - graph_id: str
                - graph_type: str (optional)
                - graph_size: int (optional)
                - algorithm: str
                - anchor_vertex: int (optional, None if not anchor-based)
                - tour_weight: float (or None if failed)
                - runtime: float (optional)
                - tour: List[int] (optional)
                - error: str (optional, if execution failed)

        Returns:
            Result ID in database
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Prepare metadata (includes errors if any)
        metadata = {}
        if 'error' in result:
            metadata['error'] = result['error']
        if 'metadata' in result:
            metadata.update(result['metadata'])

        cursor.execute("""
            INSERT INTO benchmarks
            (graph_id, graph_type, graph_size, algorithm, anchor_vertex,
             tour_weight, runtime, tour, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            result['graph_id'],
            result.get('graph_type'),
            result.get('graph_size'),
            result['algorithm'],
            result.get('anchor_vertex'),
            result.get('tour_weight'),
            result.get('runtime', 0.0),
            json.dumps(result.get('tour', [])) if result.get('tour') is not None else None,
            json.dumps(metadata) if metadata else None
        ))

        result_id = cursor.lastrowid
        conn.commit()
        conn.close()

        return result_id

    def load_results(
        self,
        graph_id: Optional[str] = None,
        algorithm: Optional[str] = None,
        graph_type: Optional[str] = None,
        graph_size: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Load benchmark results with optional filtering.

        Args:
            graph_id: Filter by graph ID
            algorithm: Filter by algorithm name
            graph_type: Filter by graph type
            graph_size: Filter by graph size

        Returns:
            List of result dictionaries
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        query = "SELECT * FROM benchmarks WHERE 1=1"
        params = []

        if graph_id:
            query += " AND graph_id = ?"
            params.append(graph_id)
        if algorithm:
            query += " AND algorithm = ?"
            params.append(algorithm)
        if graph_type:
            query += " AND graph_type = ?"
            params.append(graph_type)
        if graph_size:
            query += " AND graph_size = ?"
            params.append(graph_size)

        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()

        results = []
        for row in rows:
            result = {
                'id': row[0],
                'graph_id': row[1],
                'graph_type': row[2],
                'graph_size': row[3],
                'algorithm': row[4],
                'anchor_vertex': row[5],
                'tour_weight': row[6],
                'runtime': row[7],
                'tour': json.loads(row[8]) if row[8] else None,
                'metadata': json.loads(row[9]) if row[9] else {},
                'created_at': row[10]
            }

            # Extract error from metadata if present
            if result['metadata'] and 'error' in result['metadata']:
                result['error'] = result['metadata']['error']

            results.append(result)

        return results

    def get_anchor_weights(self, graph_id: str, algorithm: str = 'single_anchor') -> List[float]:
        """
        Get tour weights for all anchors on a specific graph.

        This is useful for generating anchor quality labels.

        Args:
            graph_id: Graph identifier
            algorithm: Anchor algorithm name (default: 'single_anchor')

        Returns:
            List of tour weights, one per anchor vertex (sorted by anchor_vertex)
            None values indicate failed runs
        """
        results = self.load_results(graph_id=graph_id, algorithm=algorithm)
        results.sort(key=lambda r: r.get('anchor_vertex', 0))
        return [r['tour_weight'] for r in results]

    def get_graph_statistics(self, graph_id: str) -> Dict[str, Any]:
        """
        Get statistical summary for all algorithms on a specific graph.

        Args:
            graph_id: Graph identifier

        Returns:
            Dictionary with statistics per algorithm
        """
        results = self.load_results(graph_id=graph_id)

        stats = {}
        for result in results:
            algo = result['algorithm']
            if algo not in stats:
                stats[algo] = {
                    'runs': 0,
                    'successes': 0,
                    'failures': 0,
                    'mean_weight': 0.0,
                    'mean_runtime': 0.0,
                    'best_weight': float('inf')
                }

            stats[algo]['runs'] += 1
            if result['tour_weight'] is not None:
                stats[algo]['successes'] += 1
                stats[algo]['mean_weight'] += result['tour_weight']
                stats[algo]['mean_runtime'] += result.get('runtime', 0.0)
                stats[algo]['best_weight'] = min(stats[algo]['best_weight'], result['tour_weight'])
            else:
                stats[algo]['failures'] += 1

        # Compute averages
        for algo in stats:
            if stats[algo]['successes'] > 0:
                stats[algo]['mean_weight'] /= stats[algo]['successes']
                stats[algo]['mean_runtime'] /= stats[algo]['successes']
            else:
                stats[algo]['mean_weight'] = None
                stats[algo]['mean_runtime'] = None
                stats[algo]['best_weight'] = None

        return stats

    def get_algorithm_statistics(self, algorithm: str) -> Dict[str, Any]:
        """
        Get statistical summary for a specific algorithm across all graphs.

        Args:
            algorithm: Algorithm name

        Returns:
            Dictionary with overall statistics
        """
        results = self.load_results(algorithm=algorithm)

        successes = [r for r in results if r['tour_weight'] is not None]
        failures = [r for r in results if r['tour_weight'] is None]

        if successes:
            weights = [r['tour_weight'] for r in successes]
            runtimes = [r.get('runtime', 0.0) for r in successes]

            return {
                'total_runs': len(results),
                'successes': len(successes),
                'failures': len(failures),
                'success_rate': len(successes) / len(results),
                'mean_weight': sum(weights) / len(weights),
                'min_weight': min(weights),
                'max_weight': max(weights),
                'mean_runtime': sum(runtimes) / len(runtimes)
            }
        else:
            return {
                'total_runs': len(results),
                'successes': 0,
                'failures': len(failures),
                'success_rate': 0.0,
                'mean_weight': None,
                'min_weight': None,
                'max_weight': None,
                'mean_runtime': None
            }

    def clear(self):
        """Clear all benchmark results from database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM benchmarks")
        conn.commit()
        conn.close()

    def save_database(self) -> Path:
        """
        Return path to database.

        Database is saved incrementally, so this just returns the path.
        Useful for pipeline stage outputs.

        Returns:
            Path to SQLite database file
        """
        return self.db_path

    def export_to_csv(self, output_path: Path):
        """
        Export all benchmark results to CSV for analysis.

        Args:
            output_path: Path to save CSV file
        """
        import pandas as pd

        results = self.load_results()

        # Flatten results for DataFrame
        rows = []
        for r in results:
            row = {
                'id': r['id'],
                'graph_id': r['graph_id'],
                'graph_type': r['graph_type'],
                'graph_size': r['graph_size'],
                'algorithm': r['algorithm'],
                'anchor_vertex': r['anchor_vertex'],
                'tour_weight': r['tour_weight'],
                'runtime': r['runtime'],
                'created_at': r['created_at'],
                'has_error': 'error' in r
            }
            rows.append(row)

        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)

    def get_comparison_matrix(self) -> Dict[str, Any]:
        """
        Generate algorithm comparison matrix across all graph types.

        Returns:
            Dictionary with comparison statistics suitable for visualization
        """
        results = self.load_results()

        # Group by (graph_type, algorithm)
        groups = {}
        for r in results:
            key = (r['graph_type'], r['algorithm'])
            if key not in groups:
                groups[key] = []
            if r['tour_weight'] is not None:
                groups[key].append(r['tour_weight'])

        # Compute averages
        matrix = {}
        for (graph_type, algorithm), weights in groups.items():
            if graph_type not in matrix:
                matrix[graph_type] = {}
            matrix[graph_type][algorithm] = {
                'mean': sum(weights) / len(weights) if weights else None,
                'count': len(weights)
            }

        return matrix
