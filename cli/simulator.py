import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import random
from typing import List, Tuple, Dict
import time

# Enhanced versions of your heuristic functions with step tracking
def construct_greedy_cycle_verbose(graph, start, anchor1, anchor2, verbose=True):
    """Enhanced version that tracks each step of the construction process."""
    vertices_count = len(graph)
    steps = []
    
    # Initialize with only the start point in the visited set
    visited = set([start])
    path = [start]
    current_vertex = start
    total_weight = 0
    
    steps.append({
        'action': 'initialize',
        'current_vertex': current_vertex,
        'path': path.copy(),
        'visited': visited.copy(),
        'total_weight': total_weight,
        'description': f"Starting at vertex {start}"
    })
    
    # First, we need to visit anchor1
    if anchor1 not in visited:
        path.append(anchor1)
        visited.add(anchor1)
        weight_added = graph[current_vertex][anchor1]
        total_weight += weight_added
        steps.append({
            'action': 'add_anchor1',
            'current_vertex': current_vertex,
            'next_vertex': anchor1,
            'path': path.copy(),
            'visited': visited.copy(),
            'weight_added': weight_added,
            'total_weight': total_weight,
            'description': f"Moving to first anchor {anchor1} (weight: {weight_added})"
        })
        current_vertex = anchor1
    
    # Visit all remaining vertices except anchor2
    while len(visited) < vertices_count - 1:  # -1 because we'll add anchor2 last
        next_vertex = None
        lowest_weight = float("inf")
        
        # Find nearest unvisited vertex (excluding anchor2)
        candidates = []
        for i in range(vertices_count):
            if i not in visited and i != anchor2:
                candidates.append((i, graph[current_vertex][i]))
                if graph[current_vertex][i] < lowest_weight:
                    next_vertex = i
                    lowest_weight = graph[current_vertex][i]
        
        # If we can't find a next vertex, break the loop
        if next_vertex is None:
            break
            
        visited.add(next_vertex)
        path.append(next_vertex)
        total_weight += lowest_weight
        
        steps.append({
            'action': 'greedy_choice',
            'current_vertex': current_vertex,
            'next_vertex': next_vertex,
            'candidates': candidates,
            'path': path.copy(),
            'visited': visited.copy(),
            'weight_added': lowest_weight,
            'total_weight': total_weight,
            'description': f"Greedy choice: {current_vertex} → {next_vertex} (weight: {lowest_weight})"
        })
        
        current_vertex = next_vertex
    
    # Now add anchor2 if it's not already visited
    if anchor2 not in visited:
        path.append(anchor2)
        visited.add(anchor2)
        weight_added = graph[current_vertex][anchor2]
        total_weight += weight_added
        steps.append({
            'action': 'add_anchor2',
            'current_vertex': current_vertex,
            'next_vertex': anchor2,
            'path': path.copy(),
            'visited': visited.copy(),
            'weight_added': weight_added,
            'total_weight': total_weight,
            'description': f"Moving to second anchor {anchor2} (weight: {weight_added})"
        })
        current_vertex = anchor2
    
    # Complete the cycle by returning to start
    weight_added = graph[current_vertex][start]
    total_weight += weight_added
    path.append(start)
    
    steps.append({
        'action': 'close_cycle',
        'current_vertex': current_vertex,
        'next_vertex': start,
        'path': path.copy(),
        'visited': visited.copy(),
        'weight_added': weight_added,
        'total_weight': total_weight,
        'description': f"Closing cycle: {current_vertex} → {start} (weight: {weight_added})"
    })
    
    return path, total_weight, steps

def low_anchor_heuristic_verbose(graph, vertex):
    """Enhanced version that tracks the anchor selection and both construction attempts."""
    def find_two_lowest_indices(values, vertex):
        if len(values) < 2:
            raise ValueError("List must contain at least two elements.")
        sorted_indices = sorted((i for i in range(len(values)) if i != vertex), key=lambda i: values[i])
        return sorted_indices[:2]
    
    # Find anchors
    anchors = find_two_lowest_indices(graph[vertex], vertex)
    anchor_weights = [graph[vertex][anchors[0]], graph[vertex][anchors[1]]]
    
    # Try both anchor orders
    cycle_1, weight_1, steps_1 = construct_greedy_cycle_verbose(graph, vertex, anchors[0], anchors[1])
    cycle_2, weight_2, steps_2 = construct_greedy_cycle_verbose(graph, vertex, anchors[1], anchors[0])
    
    # Choose the better one
    if weight_1 < weight_2:
        return {
            'best_cycle': cycle_1,
            'best_weight': weight_1,
            'best_steps': steps_1,
            'anchors': anchors,
            'anchor_weights': anchor_weights,
            'anchor_order_used': f"{anchors[0]} then {anchors[1]}",
            'alternative_weight': weight_2,
            'comparison': f"Order {anchors[0]}→{anchors[1]} (weight {weight_1}) beats {anchors[1]}→{anchors[0]} (weight {weight_2})"
        }
    else:
        return {
            'best_cycle': cycle_2,
            'best_weight': weight_2,
            'best_steps': steps_2,
            'anchors': anchors,
            'anchor_weights': anchor_weights,
            'anchor_order_used': f"{anchors[1]} then {anchors[0]}",
            'alternative_weight': weight_1,
            'comparison': f"Order {anchors[1]}→{anchors[0]} (weight {weight_2}) beats {anchors[0]}→{anchors[1]} (weight {weight_1})"
        }

# Your original functions for finding the best starting vertex
def construct_greedy_cycle(graph, start, anchor1, anchor2):
    """Original function for finding best starting vertex."""
    vertices_count = len(graph)
    visited = set([start])
    path = [start]
    current_vertex = start
    total_weight = 0
    
    if anchor1 not in visited:
        path.append(anchor1)
        visited.add(anchor1)
        total_weight += graph[current_vertex][anchor1]
        current_vertex = anchor1
    
    while len(visited) < vertices_count - 1:
        next_vertex = None
        lowest_weight = float("inf")
        
        for i in range(vertices_count):
            if i not in visited and i != anchor2 and graph[current_vertex][i] < lowest_weight:
                next_vertex = i
                lowest_weight = graph[current_vertex][i]
        
        if next_vertex is None:
            break
            
        visited.add(next_vertex)
        path.append(next_vertex)
        total_weight += lowest_weight
        current_vertex = next_vertex
    
    if anchor2 not in visited:
        path.append(anchor2)
        visited.add(anchor2)
        total_weight += graph[current_vertex][anchor2]
        current_vertex = anchor2
    
    total_weight += graph[current_vertex][start]
    path.append(start)
    
    return path, total_weight

def low_anchor_heuristic(graph, vertex):
    """Original function for finding best starting vertex."""
    def find_two_lowest_indices(values, vertex):
        if len(values) < 2:
            raise ValueError("List must contain at least two elements.")
        sorted_indices = sorted((i for i in range(len(values)) if i != vertex), key=lambda i: values[i])
        return sorted_indices[:2]
    
    anchors = find_two_lowest_indices(graph[vertex], vertex)
    cycle_1, weight_1 = construct_greedy_cycle(graph, vertex, anchors[0], anchors[1])
    cycle_2, weight_2 = construct_greedy_cycle(graph, vertex, anchors[1], anchors[0])

    if weight_1 < weight_2:
        return cycle_1, weight_1 
    return cycle_2, weight_2

def best_anchor_heuristic(graph):
    """Find the best starting vertex."""
    vertices_count = len(graph)
    best_cycle = None
    best_weight = float('inf')
    best_start_vertex = None
    
    for vertex in range(vertices_count):
        try:
            cycle, weight = low_anchor_heuristic(graph, vertex)
            if weight < best_weight:
                best_weight = weight
                best_cycle = cycle
                best_start_vertex = vertex
        except (ValueError, IndexError):
            continue
    
    if best_cycle is None:
        raise ValueError("No valid cycle could be constructed from any starting vertex")
    
    return best_cycle, best_weight, best_start_vertex

class TSPStepByStepVisualizer:
    def __init__(self, num_vertices: int = 6):
        self.num_vertices = num_vertices
        self.graph = None
        self.G = None
        self.pos = None
        self.current_step = 0
        self.steps = []
        self.fig = None
        self.waiting_for_input = False
        
    def generate_random_graph(self, seed: int = None) -> np.ndarray:
        """Generate a random complete graph with weights."""
        if seed:
            np.random.seed(seed)
            random.seed(seed)
        
        self.graph = np.random.randint(5, 25, size=(self.num_vertices, self.num_vertices))
        self.graph = (self.graph + self.graph.T) // 2
        np.fill_diagonal(self.graph, 0)
        return self.graph
    
    def generate_euclidean_graph(self, seed: int = None) -> np.ndarray:
        """Generate a graph based on Euclidean distances."""
        if seed:
            np.random.seed(seed)
            random.seed(seed)
        
        points = np.random.rand(self.num_vertices, 2) * 100
        self.graph = np.zeros((self.num_vertices, self.num_vertices))
        
        for i in range(self.num_vertices):
            for j in range(self.num_vertices):
                if i != j:
                    dist = np.sqrt((points[i][0] - points[j][0])**2 + (points[i][1] - points[j][1])**2)
                    self.graph[i][j] = int(dist * 0.5)  # Scale down for readability
        
        self.pos = {i: points[i] for i in range(self.num_vertices)}
        return self.graph
    
    def create_networkx_graph(self):
        """Create NetworkX graph from adjacency matrix."""
        self.G = nx.Graph()
        
        for i in range(self.num_vertices):
            self.G.add_node(i)
        
        for i in range(self.num_vertices):
            for j in range(i + 1, self.num_vertices):
                self.G.add_edge(i, j, weight=self.graph[i][j])
        
        if self.pos is None:
            self.pos = nx.spring_layout(self.G, seed=42)
    
    def on_next_click(self, event):
        """Handle next button click."""
        if self.waiting_for_input:
            self.waiting_for_input = False
    
    def on_prev_click(self, event):
        """Handle previous button click."""
        if self.current_step > 0:
            self.current_step -= 1
            self.update_visualization()
    
    def on_auto_click(self, event):
        """Handle auto-play button click."""
        if hasattr(self, 'auto_playing') and self.auto_playing:
            self.auto_playing = False
            self.auto_button.label.set_text('Auto Play')
        else:
            self.auto_playing = True
            self.auto_button.label.set_text('Stop Auto')
            self.auto_play_steps()
    
    def auto_play_steps(self):
        """Automatically advance through steps."""
        while self.auto_playing and self.current_step < len(self.steps) - 1:
            plt.pause(2.0)  # Wait 2 seconds between steps
            if self.auto_playing:  # Check again in case user stopped
                self.current_step += 1
                self.update_visualization()
        
        if self.auto_playing:
            self.auto_playing = False
            self.auto_button.label.set_text('Auto Play')
    
    def visualize_step(self, step_info, step_num, total_steps, title_prefix=""):
        """Visualize a single step and wait for user input."""
        
        # Create figure if it doesn't exist or create a new one for this step
        if self.fig is None or not plt.fignum_exists(self.fig.number):
            self.fig = plt.figure(figsize=(16, 10))
        else:
            self.fig.clear()
        
        # Create layout
        gs = plt.GridSpec(2, 3, height_ratios=[3, 1], width_ratios=[2, 2, 1], 
                        hspace=0.3, wspace=0.3)
        
        ax_graph = plt.subplot(gs[0, :2])
        ax_info = plt.subplot(gs[1, :2])
        ax_weights = plt.subplot(gs[:, 2])
        
        # Draw graph visualization
        plt.sca(ax_graph)
        
        # Draw all edges in light gray
        nx.draw_networkx_edges(self.G, self.pos, alpha=0.2, edge_color='lightgray', width=1)
        
        # Draw path edges in order with different colors
        path = step_info['path']
        if len(path) > 1:
            for i in range(len(path) - 1):
                edge_color = 'red' if i == len(path) - 2 else 'blue'  # Current edge in red
                edge_width = 4 if i == len(path) - 2 else 2
                nx.draw_networkx_edges(self.G, self.pos, 
                                    edgelist=[(path[i], path[i+1])], 
                                    edge_color=edge_color, width=edge_width, alpha=0.8)
        
        # Highlight current vertex
        current = step_info.get('current_vertex')
        next_vertex = step_info.get('next_vertex')
        
        # Draw all nodes
        node_colors = []
        for node in range(self.num_vertices):
            if node == current:
                node_colors.append('red')
            elif node == next_vertex and 'next_vertex' in step_info:
                node_colors.append('orange')
            elif node in step_info['visited']:
                node_colors.append('lightgreen')
            else:
                node_colors.append('lightblue')
        
        nx.draw_networkx_nodes(self.G, self.pos, node_color=node_colors, 
                            node_size=600, alpha=0.9)
        nx.draw_networkx_labels(self.G, self.pos, font_size=12, font_weight='bold')
        
        # Draw edge weights
        edge_labels = nx.get_edge_attributes(self.G, 'weight')
        nx.draw_networkx_edge_labels(self.G, self.pos, edge_labels, font_size=8)
        
        ax_graph.set_title(f"{title_prefix}Step {step_num}/{total_steps}: {step_info['description']}", 
                        fontsize=14, fontweight='bold')
        ax_graph.axis('off')
        
        # Step information
        plt.sca(ax_info)
        
        info_text = f"Action: {step_info['action']}\n"
        info_text += f"Current Path: {' → '.join(map(str, path))}\n"
        info_text += f"Visited: {sorted(list(step_info['visited']))}\n"
        info_text += f"Total Weight: {step_info['total_weight']}\n"
        
        if 'weight_added' in step_info:
            info_text += f"Weight Added: {step_info['weight_added']}\n"
        
        if 'candidates' in step_info:
            candidates_str = ", ".join([f"{v}({w})" for v, w in step_info['candidates']])
            info_text += f"Candidates: {candidates_str}\n"
        
        ax_info.text(0.05, 0.95, info_text, transform=ax_info.transAxes, 
                    fontsize=10, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
        ax_info.axis('off')
        
        # Adjacency matrix
        plt.sca(ax_weights)
        im = ax_weights.imshow(self.graph, cmap='Blues', alpha=0.7)
        ax_weights.set_title("Weight Matrix", fontsize=12)
        
        # Add text annotations
        for i in range(self.num_vertices):
            for j in range(self.num_vertices):
                if i != j:
                    ax_weights.text(j, i, str(self.graph[i][j]), ha='center', va='center', 
                                fontsize=8, fontweight='bold')
        
        ax_weights.set_xticks(range(self.num_vertices))
        ax_weights.set_yticks(range(self.num_vertices))
        
        # Add legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Current'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=10, label='Next'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgreen', markersize=10, label='Visited'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue', markersize=10, label='Unvisited'),
            plt.Line2D([0], [0], color='red', linewidth=3, label='Current Edge'),
            plt.Line2D([0], [0], color='blue', linewidth=2, label='Path Edge')
        ]
        ax_graph.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1))
        
        plt.tight_layout()
        plt.show(block=False)
        plt.draw()
        
        # Force display update
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def update_visualization(self):
        """Update the visualization for the current step."""
        if self.current_step >= len(self.steps):
            return
        
        step_info = self.steps[self.current_step]
        
        # Clear previous drawings
        for ax in [self.ax_graph, self.ax_info, self.ax_weights]:
            ax.clear()
        
        # Update graph visualization
        plt.sca(self.ax_graph)
        
        # Draw all edges in light gray
        nx.draw_networkx_edges(self.G, self.pos, alpha=0.2, edge_color='lightgray', width=1)
        
        # Draw path edges in order with different colors
        path = step_info['path']
        if len(path) > 1:
            for i in range(len(path) - 1):
                edge_color = 'red' if i == len(path) - 2 else 'blue'  # Current edge in red
                edge_width = 4 if i == len(path) - 2 else 2
                nx.draw_networkx_edges(self.G, self.pos, 
                                     edgelist=[(path[i], path[i+1])], 
                                     edge_color=edge_color, width=edge_width, alpha=0.8)
        
        # Highlight current vertex
        current = step_info.get('current_vertex')
        next_vertex = step_info.get('next_vertex')
        
        # Draw all nodes
        node_colors = []
        for node in range(self.num_vertices):
            if node == current:
                node_colors.append('red')
            elif node == next_vertex and 'next_vertex' in step_info:
                node_colors.append('orange')
            elif node in step_info['visited']:
                node_colors.append('lightgreen')
            else:
                node_colors.append('lightblue')
        
        nx.draw_networkx_nodes(self.G, self.pos, node_color=node_colors, 
                              node_size=600, alpha=0.9)
        nx.draw_networkx_labels(self.G, self.pos, font_size=12, font_weight='bold')
        
        # Draw edge weights
        edge_labels = nx.get_edge_attributes(self.G, 'weight')
        nx.draw_networkx_edge_labels(self.G, self.pos, edge_labels, font_size=8)
        
        step_num = self.current_step + 1
        total_steps = len(self.steps)
        self.ax_graph.set_title(f"Step {step_num}/{total_steps}: {step_info['description']}", 
                               fontsize=14, fontweight='bold')
        self.ax_graph.axis('off')
        
        # Update step information
        plt.sca(self.ax_info)
        
        info_text = f"Action: {step_info['action']}\n"
        info_text += f"Current Path: {' → '.join(map(str, path))}\n"
        info_text += f"Visited: {sorted(list(step_info['visited']))}\n"
        info_text += f"Total Weight: {step_info['total_weight']}\n"
        
        if 'weight_added' in step_info:
            info_text += f"Weight Added: {step_info['weight_added']}\n"
        
        if 'candidates' in step_info:
            candidates_str = ", ".join([f"{v}({w})" for v, w in step_info['candidates']])
            info_text += f"Candidates: {candidates_str}\n"
        
        self.ax_info.text(0.05, 0.95, info_text, transform=self.ax_info.transAxes, 
                         fontsize=10, verticalalignment='top', fontfamily='monospace',
                         bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
        self.ax_info.axis('off')
        
        # Update adjacency matrix
        plt.sca(self.ax_weights)
        im = self.ax_weights.imshow(self.graph, cmap='Blues', alpha=0.7)
        self.ax_weights.set_title("Weight Matrix", fontsize=12)
        
        # Add text annotations
        for i in range(self.num_vertices):
            for j in range(self.num_vertices):
                if i != j:
                    self.ax_weights.text(j, i, str(self.graph[i][j]), ha='center', va='center', 
                                        fontsize=8, fontweight='bold')
        
        self.ax_weights.set_xticks(range(self.num_vertices))
        self.ax_weights.set_yticks(range(self.num_vertices))
        
        # Update button states
        self.prev_button.ax.set_visible(self.current_step > 0)
        self.next_button.ax.set_visible(self.current_step < len(self.steps) - 1)
        
        # Update step counter
        if hasattr(self, 'step_text'):
            self.step_text.set_text(f"Step {step_num} of {total_steps}")
        
        plt.draw()
    
    def visualize_interactive_steps(self, steps, title_prefix=""):
        """Create an interactive visualization with navigation buttons."""
        self.steps = steps
        self.current_step = 0
        self.auto_playing = False
        
        # Create figure with more space for buttons
        self.fig = plt.figure(figsize=(16, 12))
        
        # Create layout
        gs = plt.GridSpec(3, 4, height_ratios=[3, 1, 0.3], width_ratios=[2, 2, 1, 1], 
                         hspace=0.3, wspace=0.3)
        
        self.ax_graph = plt.subplot(gs[0, :3])
        self.ax_info = plt.subplot(gs[1, :2])
        self.ax_weights = plt.subplot(gs[1, 2:])
        
        # Create button axes
        ax_prev = plt.subplot(gs[2, 0])
        ax_next = plt.subplot(gs[2, 1])
        ax_auto = plt.subplot(gs[2, 2])
        ax_reset = plt.subplot(gs[2, 3])
        
        # Create buttons
        self.prev_button = Button(ax_prev, '← Previous')
        self.next_button = Button(ax_next, 'Next →')
        self.auto_button = Button(ax_auto, 'Auto Play')
        self.reset_button = Button(ax_reset, 'Reset')
        
        # Connect button events
        self.prev_button.on_clicked(self.on_prev_click)
        self.next_button.on_clicked(self.on_next_click)
        self.auto_button.on_clicked(self.on_auto_click)
        self.reset_button.on_clicked(lambda x: self.reset_visualization())
        
        # Add step counter text
        self.step_text = self.fig.text(0.5, 0.02, f"Step 1 of {len(steps)}", 
                                      ha='center', fontsize=12, fontweight='bold')
        
        # Initial visualization
        self.update_visualization()
        
        # Add instructions
        instructions = ("Use buttons to navigate:\n"
                       "• Next/Previous: Manual step control\n"
                       "• Auto Play: Automatic progression\n"
                       "• Reset: Return to first step")
        
        self.fig.text(0.02, 0.98, instructions, fontsize=10, 
                     verticalalignment='top', 
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))
        
        # Setup keyboard shortcuts
        def on_key(event):
            if event.key == 'right' or event.key == ' ':
                if self.current_step < len(self.steps) - 1:
                    self.current_step += 1
                    self.update_visualization()
            elif event.key == 'left':
                if self.current_step > 0:
                    self.current_step -= 1
                    self.update_visualization()
            elif event.key == 'r':
                self.reset_visualization()
        
        self.fig.canvas.mpl_connect('key_press_event', on_key)
        
        plt.show()
        
        return self.fig
    
    def reset_visualization(self):
        """Reset to the first step."""
        self.current_step = 0
        self.auto_playing = False
        self.auto_button.label.set_text('Auto Play')
        self.update_visualization()
    
    def run_complete_simulation(self, use_euclidean=False, seed=42):
        """Run the complete simulation with step-by-step visualization."""
        print("[INFO] TSP Low Anchor Heuristic - Complete Step-by-Step Simulation")
        print("=" * 70)

        # Generate graph
        if use_euclidean:
            self.generate_euclidean_graph(seed)
            graph_type = "Euclidean Distance"
        else:
            self.generate_random_graph(seed)
            graph_type = "Random Weights"

        self.create_networkx_graph()

        print(f"\n[GRAPH] Generated {graph_type} Graph ({self.num_vertices} vertices)")
        print("\nAdjacency Matrix:")
        for i, row in enumerate(self.graph):
            print(f"  {i}: {row}")

        # Phase 1: Find best starting vertex
        print(f"\n[PHASE 1] Testing all {self.num_vertices} vertices to find best starting point")
        print("-" * 60)

        all_results = []
        for vertex in range(self.num_vertices):
            try:
                cycle, weight = low_anchor_heuristic(self.graph, vertex)
                all_results.append((vertex, cycle, weight))

                # Show anchor selection for this vertex
                row = self.graph[vertex]
                sorted_indices = sorted((i for i in range(len(row)) if i != vertex), key=lambda i: row[i])
                anchors = sorted_indices[:2]
                anchor_weights = [row[anchors[0]], row[anchors[1]]]

                print(f"  Vertex {vertex}: Weight = {weight:3d} | Anchors: {anchors[0]}({anchor_weights[0]}) & {anchors[1]}({anchor_weights[1]})")

            except Exception as e:
                print(f"  Vertex {vertex}: Error - {e}")

        if not all_results:
            print("[ERROR] No valid solutions found!")
            return

        # Find best result
        best_vertex, best_cycle, best_weight = min(all_results, key=lambda x: x[2])

        print(f"\n[BEST] Best starting vertex: {best_vertex} (Total weight: {best_weight})")
        print(f"   Best cycle: {' -> '.join(map(str, best_cycle))}")

        # Phase 2: Detailed step-by-step visualization
        print(f"\n[PHASE 2] Step-by-step visualization starting from vertex {best_vertex}")
        print("-" * 60)

        input("\nPress Enter to start step-by-step visualization...")

        # Get detailed steps for the best starting vertex
        result = low_anchor_heuristic_verbose(self.graph, best_vertex)

        print(f"\n[ANCHORS] Anchor Selection for vertex {best_vertex}:")
        print(f"   Anchors found: {result['anchors']} with weights {result['anchor_weights']}")
        print(f"   Strategy: {result['comparison']}")
        print(f"   Using anchor order: {result['anchor_order_used']}")

        # Visualize each step
        steps = result['best_steps']
        total_steps = len(steps)

        print(f"\n[VISUALIZE] Visualizing {total_steps} steps...")

        for i, step in enumerate(steps, 1):
            print(f"\nStep {i}/{total_steps}: {step['description']}")
            self.visualize_step(step, i, total_steps, f"Start={best_vertex} | ")

            if i < total_steps:
                input("Press Enter for next step...")

        # Final summary
        print(f"\n[SUCCESS] Algorithm Complete!")
        print(f"   Final cycle: {' -> '.join(map(str, result['best_cycle']))}")
        print(f"   Total weight: {result['best_weight']}")
        print(f"   Anchors used: {result['anchors']} in order {result['anchor_order_used']}")

        return result

# Quick demo function
def demo_step_by_step():
    """Run a quick demonstration."""
    visualizer = TSPStepByStepVisualizer(num_vertices=6)
    
    print("Choose graph type:")
    print("1. Random weights (faster)")
    print("2. Euclidean distances (more realistic)")
    
    choice = input("Enter choice (1 or 2): ").strip()
    use_euclidean = choice == "2"
    
    result = visualizer.run_complete_simulation(use_euclidean=use_euclidean, seed=42)
    
    return visualizer, result

if __name__ == "__main__":
    # Run the step-by-step demonstration
    visualizer, result = demo_step_by_step()