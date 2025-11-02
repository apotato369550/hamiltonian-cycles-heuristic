"""
Graph visualization utilities.

Provides tools for visualizing graph instances including graph layouts,
weight distributions, and adjacency matrix heatmaps.
"""

import math
from typing import List, Tuple, Optional, Dict, Any
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from pathlib import Path

from .graph_instance import GraphInstance


class GraphVisualizer:
    """
    Visualization tool for graph instances.

    Provides multiple visualization types for understanding graph structure
    and properties.
    """

    def __init__(self, output_dir: str = "visualizations"):
        """
        Initialize visualizer.

        Args:
            output_dir: Directory to save visualizations
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def visualize_all(
        self,
        graph: GraphInstance,
        prefix: Optional[str] = None
    ) -> List[str]:
        """
        Create all applicable visualizations for a graph.

        Args:
            graph: Graph instance to visualize
            prefix: Optional filename prefix

        Returns:
            List of saved file paths
        """
        saved_files = []

        if prefix is None:
            prefix = f"{graph.metadata.graph_type}_{graph.metadata.size}_{graph.id}"

        # Graph layout (for small graphs)
        if graph.metadata.size <= 20:
            filepath = self.visualize_graph_layout(graph, save_as=f"{prefix}_layout.png")
            if filepath:
                saved_files.append(filepath)

        # Weight distribution
        filepath = self.visualize_weight_distribution(graph, save_as=f"{prefix}_weights.png")
        saved_files.append(filepath)

        # Adjacency matrix heatmap
        filepath = self.visualize_adjacency_heatmap(graph, save_as=f"{prefix}_heatmap.png")
        saved_files.append(filepath)

        # Summary statistics
        filepath = self.visualize_summary_stats(graph, save_as=f"{prefix}_summary.png")
        saved_files.append(filepath)

        return saved_files

    def visualize_graph_layout(
        self,
        graph: GraphInstance,
        save_as: Optional[str] = None,
        show: bool = False
    ) -> Optional[str]:
        """
        Visualize graph as a network layout.

        For Euclidean graphs, uses actual coordinates.
        For others, uses force-directed layout.

        Args:
            graph: Graph instance
            save_as: Filename to save (relative to output_dir)
            show: Whether to display interactively

        Returns:
            Path to saved file if save_as is provided
        """
        if graph.metadata.size > 20:
            print("Graph too large for layout visualization (max 20 vertices)")
            return None

        fig, ax = plt.subplots(figsize=(10, 10))

        # Get vertex positions
        if graph.coordinates is not None:
            positions = self._coordinates_to_2d(graph.coordinates)
        else:
            positions = self._force_directed_layout(graph)

        # Normalize edge weights for coloring
        weights = []
        for i in range(graph.metadata.size):
            for j in range(i + 1, graph.metadata.size):
                weights.append(graph.adjacency_matrix[i][j])

        if weights:
            min_weight = min(weights)
            max_weight = max(weights)
            weight_range = max_weight - min_weight if max_weight > min_weight else 1.0

            # Draw edges
            for i in range(graph.metadata.size):
                for j in range(i + 1, graph.metadata.size):
                    x1, y1 = positions[i]
                    x2, y2 = positions[j]

                    weight = graph.adjacency_matrix[i][j]
                    normalized_weight = (weight - min_weight) / weight_range

                    # Color from blue (low) to red (high)
                    color = plt.cm.coolwarm(normalized_weight)
                    linewidth = 0.5 + 2.0 * (1 - normalized_weight)  # Thicker for cheaper edges

                    ax.plot([x1, x2], [y1, y2], color=color, linewidth=linewidth, alpha=0.6, zorder=1)

        # Draw vertices
        x_coords = [p[0] for p in positions]
        y_coords = [p[1] for p in positions]
        ax.scatter(x_coords, y_coords, s=300, c='white', edgecolors='black', linewidths=2, zorder=2)

        # Add vertex labels
        for i, (x, y) in enumerate(positions):
            ax.text(x, y, str(i), ha='center', va='center', fontsize=10, fontweight='bold', zorder=3)

        # Add colorbar for edge weights
        if weights:
            sm = plt.cm.ScalarMappable(cmap=plt.cm.coolwarm, norm=plt.Normalize(vmin=min_weight, vmax=max_weight))
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Edge Weight', rotation=270, labelpad=20)

        ax.set_title(f"Graph Layout: {graph.metadata.graph_type} (n={graph.metadata.size})", fontsize=14, fontweight='bold')
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

        if save_as:
            filepath = self.output_dir / save_as
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close()
            return str(filepath)
        elif show:
            plt.show()
        else:
            plt.close()

        return None

    def visualize_weight_distribution(
        self,
        graph: GraphInstance,
        save_as: Optional[str] = None,
        show: bool = False
    ) -> Optional[str]:
        """
        Visualize edge weight distribution as histogram.

        Args:
            graph: Graph instance
            save_as: Filename to save
            show: Whether to display interactively

        Returns:
            Path to saved file
        """
        # Extract weights
        weights = []
        for i in range(graph.metadata.size):
            for j in range(i + 1, graph.metadata.size):
                weights.append(graph.adjacency_matrix[i][j])

        if not weights:
            print("No edges to visualize")
            return None

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Histogram
        ax1.hist(weights, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
        ax1.axvline(graph.properties.weight_mean, color='red', linestyle='--', linewidth=2, label=f'Mean: {graph.properties.weight_mean:.2f}')
        ax1.set_xlabel('Edge Weight')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Weight Distribution (Histogram)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Box plot
        ax2.boxplot(weights, vert=True)
        ax2.set_ylabel('Edge Weight')
        ax2.set_title('Weight Distribution (Box Plot)')
        ax2.grid(True, alpha=0.3, axis='y')

        fig.suptitle(f"Edge Weight Analysis: {graph.metadata.graph_type} (n={graph.metadata.size})",
                     fontsize=14, fontweight='bold')

        if save_as:
            filepath = self.output_dir / save_as
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close()
            return str(filepath)
        elif show:
            plt.show()
        else:
            plt.close()

        return None

    def visualize_adjacency_heatmap(
        self,
        graph: GraphInstance,
        save_as: Optional[str] = None,
        show: bool = False
    ) -> Optional[str]:
        """
        Visualize adjacency matrix as heatmap.

        Args:
            graph: Graph instance
            save_as: Filename to save
            show: Whether to display interactively

        Returns:
            Path to saved file
        """
        matrix = np.array(graph.adjacency_matrix)

        fig, ax = plt.subplots(figsize=(10, 8))

        im = ax.imshow(matrix, cmap='viridis', aspect='auto')

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Edge Weight', rotation=270, labelpad=20)

        # Set ticks
        if graph.metadata.size <= 50:
            ax.set_xticks(np.arange(graph.metadata.size))
            ax.set_yticks(np.arange(graph.metadata.size))
            ax.set_xticklabels(np.arange(graph.metadata.size))
            ax.set_yticklabels(np.arange(graph.metadata.size))
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        ax.set_title(f"Adjacency Matrix Heatmap: {graph.metadata.graph_type} (n={graph.metadata.size})",
                     fontsize=14, fontweight='bold')
        ax.set_xlabel('Vertex')
        ax.set_ylabel('Vertex')

        if save_as:
            filepath = self.output_dir / save_as
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close()
            return str(filepath)
        elif show:
            plt.show()
        else:
            plt.close()

        return None

    def visualize_summary_stats(
        self,
        graph: GraphInstance,
        save_as: Optional[str] = None,
        show: bool = False
    ) -> Optional[str]:
        """
        Create a summary visualization with key statistics.

        Args:
            graph: Graph instance
            save_as: Filename to save
            show: Whether to display interactively

        Returns:
            Path to saved file
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.axis('off')

        # Prepare summary text
        summary_lines = [
            f"Graph ID: {graph.id}",
            f"Type: {graph.metadata.graph_type}",
            f"Size: {graph.metadata.size} vertices",
            f"Seed: {graph.metadata.random_seed}",
            "",
            "Properties:",
            f"  • Metric: {'Yes' if graph.properties.is_metric else 'No'}",
            f"  • Symmetric: {'Yes' if graph.properties.is_symmetric else 'No'}",
            f"  • Weight Range: [{graph.properties.weight_range[0]:.2f}, {graph.properties.weight_range[1]:.2f}]",
            f"  • Weight Mean: {graph.properties.weight_mean:.2f}",
            f"  • Weight Std Dev: {graph.properties.weight_std:.2f}",
        ]

        if graph.properties.metricity_score is not None:
            summary_lines.append(f"  • Metricity Score: {graph.properties.metricity_score:.2%}")

        if graph.properties.triangle_violations > 0:
            summary_lines.append(f"  • Triangle Violations: {graph.properties.triangle_violations}")

        if graph.coordinates is not None:
            summary_lines.append(f"  • Coordinates: {len(graph.coordinates[0])}D space")

        # Add generation parameters
        summary_lines.append("")
        summary_lines.append("Generation Parameters:")
        for key, value in graph.metadata.generation_params.items():
            summary_lines.append(f"  • {key}: {value}")

        # Draw text
        y_position = 0.95
        for line in summary_lines:
            ax.text(0.05, y_position, line, transform=ax.transAxes,
                   fontsize=11, verticalalignment='top', family='monospace')
            y_position -= 0.05

        ax.set_title(f"Graph Summary: {graph.metadata.graph_type}_{graph.metadata.size}",
                     fontsize=14, fontweight='bold', pad=20)

        if save_as:
            filepath = self.output_dir / save_as
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close()
            return str(filepath)
        elif show:
            plt.show()
        else:
            plt.close()

        return None

    def _coordinates_to_2d(self, coordinates: List[Tuple[float, ...]]) -> List[Tuple[float, float]]:
        """Convert coordinates to 2D (project if 3D)."""
        if len(coordinates[0]) == 2:
            return [(x, y) for x, y in coordinates]
        elif len(coordinates[0]) == 3:
            # Simple projection: drop z-coordinate
            return [(x, y) for x, y, z in coordinates]
        else:
            raise ValueError("Coordinates must be 2D or 3D")

    def _force_directed_layout(
        self,
        graph: GraphInstance,
        iterations: int = 100
    ) -> List[Tuple[float, float]]:
        """
        Compute force-directed layout for graph without coordinates.

        Simple implementation of Fruchterman-Reingold algorithm.
        """
        n = graph.metadata.size

        # Initialize random positions
        positions = [(np.random.rand() * 100, np.random.rand() * 100) for _ in range(n)]

        # Parameters
        area = 100.0 * 100.0
        k = math.sqrt(area / n)  # Optimal distance

        def attractive_force(distance):
            return distance * distance / k

        def repulsive_force(distance):
            return k * k / (distance + 1e-6)

        # Iterate
        for iteration in range(iterations):
            # Calculate temperature
            temperature = 100.0 * (1.0 - iteration / iterations)

            # Calculate forces
            forces = [(0.0, 0.0) for _ in range(n)]

            # Repulsive forces between all pairs
            for i in range(n):
                for j in range(i + 1, n):
                    dx = positions[i][0] - positions[j][0]
                    dy = positions[i][1] - positions[j][1]
                    distance = math.sqrt(dx * dx + dy * dy)

                    if distance > 0:
                        force = repulsive_force(distance)
                        fx = (dx / distance) * force
                        fy = (dy / distance) * force

                        forces[i] = (forces[i][0] + fx, forces[i][1] + fy)
                        forces[j] = (forces[j][0] - fx, forces[j][1] - fy)

            # Attractive forces for edges (all pairs in complete graph)
            for i in range(n):
                for j in range(i + 1, n):
                    dx = positions[i][0] - positions[j][0]
                    dy = positions[i][1] - positions[j][1]
                    distance = math.sqrt(dx * dx + dy * dy)

                    if distance > 0:
                        # Weight by edge weight (lighter edges pull more)
                        weight_factor = 1.0 / (graph.adjacency_matrix[i][j] + 1.0)
                        force = attractive_force(distance) * weight_factor
                        fx = (dx / distance) * force
                        fy = (dy / distance) * force

                        forces[i] = (forces[i][0] - fx, forces[i][1] - fy)
                        forces[j] = (forces[j][0] + fx, forces[j][1] + fy)

            # Update positions
            for i in range(n):
                force_magnitude = math.sqrt(forces[i][0] ** 2 + forces[i][1] ** 2)
                if force_magnitude > 0:
                    displacement = min(force_magnitude, temperature)
                    dx = (forces[i][0] / force_magnitude) * displacement
                    dy = (forces[i][1] / force_magnitude) * displacement

                    positions[i] = (
                        max(0, min(100, positions[i][0] + dx)),
                        max(0, min(100, positions[i][1] + dy))
                    )

        return positions


def visualize_graph(
    graph: GraphInstance,
    output_dir: str = "visualizations",
    create_all: bool = True
) -> List[str]:
    """
    Convenience function to visualize a graph.

    Args:
        graph: Graph instance to visualize
        output_dir: Output directory
        create_all: Whether to create all visualizations

    Returns:
        List of saved file paths
    """
    visualizer = GraphVisualizer(output_dir=output_dir)

    if create_all:
        return visualizer.visualize_all(graph)
    else:
        # Just create basic visualizations
        files = []
        files.append(visualizer.visualize_weight_distribution(
            graph, save_as=f"{graph.id}_weights.png"
        ))
        files.append(visualizer.visualize_adjacency_heatmap(
            graph, save_as=f"{graph.id}_heatmap.png"
        ))
        return [f for f in files if f is not None]
