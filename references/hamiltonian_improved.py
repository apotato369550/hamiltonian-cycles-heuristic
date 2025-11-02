import random
import heapq
from typing import List, Tuple, Set, Dict, Optional

class ImprovedHamiltonianSolver:
    """
    Improved anchor-based Hamiltonian cycle solver with better architecture.
    
    Key improvements:
    1. Anchors as regional hubs rather than strict waypoints
    2. Adaptive clustering based on graph structure
    3. Intelligent bridge planning using MST insights
    4. Local optimization with 2-opt improvements
    """
    
    def __init__(self, graph: List[List[int]]):
        self.graph = graph
        self.n = len(graph)
        self.mst_edges = self._compute_mst()
        
    def solve(self, start: int = 0, num_anchors: Optional[int] = None, 
              local_search: bool = True) -> Tuple[List[int], int]:
        """
        Main solving method with improved architecture.
        
        Args:
            start: Starting vertex
            num_anchors: Number of anchors (auto-calculated if None)
            local_search: Whether to apply local optimization
            
        Returns:
            Tuple of (cycle, total_weight)
        """
        # 1. Intelligent anchor selection based on graph structure
        anchors = self._select_anchors_intelligently(start, num_anchors)
        
        # 2. Create regions around each anchor
        regions = self._create_anchor_regions(anchors)
        
        # 3. Plan optimal anchor tour
        anchor_tour = self._plan_anchor_tour(anchors, start)
        
        # 4. Build cycle by connecting regions
        cycle, weight = self._build_regional_cycle(anchor_tour, regions)
        
        # 5. Apply local optimization if requested
        if local_search:
            cycle, weight = self._local_optimization(cycle)
            
        return cycle, weight
    
    def _compute_mst(self) -> List[Tuple[int, int, int]]:
        """Compute MST to understand graph structure."""
        edges = []
        for i in range(self.n):
            for j in range(i + 1, self.n):
                edges.append((self.graph[i][j], i, j))
        
        edges.sort()
        parent = list(range(self.n))
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        mst_edges = []
        for weight, u, v in edges:
            pu, pv = find(u), find(v)
            if pu != pv:
                parent[pu] = pv
                mst_edges.append((weight, u, v))
                if len(mst_edges) == self.n - 1:
                    break
                    
        return mst_edges
    
    def _select_anchors_intelligently(self, start: int, num_anchors: Optional[int]) -> List[int]:
        """
        Select anchors based on graph structure rather than randomly.
        
        Strategy:
        1. Use vertices with high degree in MST (structural importance)
        2. Ensure good spatial distribution
        3. Include vertices that are far from each other
        """
        if num_anchors is None:
            num_anchors = max(2, min(8, self.n // 6))  # More conservative anchor count
        
        # Calculate vertex importance based on MST degree and centrality
        mst_degree = [0] * self.n
        for _, u, v in self.mst_edges:
            mst_degree[u] += 1
            mst_degree[v] += 1
        
        # Calculate average distance to all other vertices (centrality)
        centrality = []
        for i in range(self.n):
            avg_dist = sum(self.graph[i]) / (self.n - 1)
            centrality.append(avg_dist)
        
        # Combine metrics: prefer high MST degree and low centrality (close to others)
        importance = []
        for i in range(self.n):
            # Normalize metrics
            degree_score = mst_degree[i] / max(mst_degree) if max(mst_degree) > 0 else 0
            centrality_score = 1.0 - (centrality[i] / max(centrality)) if max(centrality) > 0 else 0
            importance.append((degree_score + centrality_score, i))
        
        importance.sort(reverse=True)
        
        # Select anchors ensuring start is included and good distribution
        selected = [start]
        candidates = [v for _, v in importance if v != start]
        
        while len(selected) < num_anchors and candidates:
            # Find candidate that maximizes minimum distance to existing anchors
            best_candidate = None
            best_min_dist = -1
            
            for candidate in candidates:
                min_dist = min(self.graph[candidate][anchor] for anchor in selected)
                if min_dist > best_min_dist:
                    best_min_dist = min_dist
                    best_candidate = candidate
            
            if best_candidate is not None:
                selected.append(best_candidate)
                candidates.remove(best_candidate)
            else:
                break
        
        return selected
    
    def _create_anchor_regions(self, anchors: List[int]) -> Dict[int, List[int]]:
        """
        Assign each non-anchor vertex to the closest anchor, creating regions.
        """
        regions = {anchor: [anchor] for anchor in anchors}
        anchor_set = set(anchors)
        
        for v in range(self.n):
            if v not in anchor_set:
                # Find closest anchor
                closest_anchor = min(anchors, key=lambda a: self.graph[v][a])
                regions[closest_anchor].append(v)
        
        return regions
    
    def _plan_anchor_tour(self, anchors: List[int], start: int) -> List[int]:
        """
        Find good tour through anchors using nearest neighbor with 2-opt.
        """
        if len(anchors) <= 2:
            return anchors + [start] if anchors[-1] != start else anchors
        
        # Nearest neighbor starting from start
        unvisited = set(anchors)
        tour = [start]
        unvisited.remove(start)
        current = start
        
        while unvisited:
            nearest = min(unvisited, key=lambda v: self.graph[current][v])
            tour.append(nearest)
            unvisited.remove(nearest)
            current = nearest
        
        # Add return to start if not already there
        if tour[-1] != start:
            tour.append(start)
        
        # Apply 2-opt improvement to anchor tour
        return self._two_opt_anchors(tour, anchors)
    
    def _two_opt_anchors(self, tour: List[int], anchors: List[int]) -> List[int]:
        """Apply 2-opt optimization to anchor tour."""
        if len(tour) <= 3:
            return tour
            
        improved = True
        while improved:
            improved = False
            for i in range(1, len(tour) - 2):
                for j in range(i + 1, len(tour) - 1):
                    # Calculate current cost
                    current_cost = (self.graph[tour[i-1]][tour[i]] + 
                                  self.graph[tour[j]][tour[j+1]])
                    
                    # Calculate cost after 2-opt swap
                    new_cost = (self.graph[tour[i-1]][tour[j]] + 
                              self.graph[tour[i]][tour[j+1]])
                    
                    if new_cost < current_cost:
                        # Reverse the segment between i and j
                        tour[i:j+1] = tour[i:j+1][::-1]
                        improved = True
                        break
                if improved:
                    break
        
        return tour
    
    def _build_regional_cycle(self, anchor_tour: List[int], 
                            regions: Dict[int, List[int]]) -> Tuple[List[int], int]:
        """
        Build cycle by visiting each anchor region in tour order.
        """
        cycle = []
        total_weight = 0
        visited_global = set()
        
        for i in range(len(anchor_tour) - 1):  # Exclude the return to start
            current_anchor = anchor_tour[i]
            region_vertices = [v for v in regions[current_anchor] if v not in visited_global]
            
            if not region_vertices:
                continue
                
            # Visit vertices in this region using nearest neighbor
            if not cycle:
                # First region - start with the anchor
                cycle.append(current_anchor)
                visited_global.add(current_anchor)
                region_vertices.remove(current_anchor)
            
            current_pos = cycle[-1]
            
            # Visit remaining vertices in region using greedy nearest neighbor
            while region_vertices:
                nearest = min(region_vertices, key=lambda v: self.graph[current_pos][v])
                total_weight += self.graph[current_pos][nearest]
                cycle.append(nearest)
                visited_global.add(nearest)
                region_vertices.remove(nearest)
                current_pos = nearest
        
        # Handle any remaining unvisited vertices
        remaining = set(range(self.n)) - visited_global
        current_pos = cycle[-1] if cycle else anchor_tour[0]
        
        while remaining:
            nearest = min(remaining, key=lambda v: self.graph[current_pos][v])
            total_weight += self.graph[current_pos][nearest]
            cycle.append(nearest)
            remaining.remove(nearest)
            current_pos = nearest
        
        # Complete the cycle
        if cycle and cycle[-1] != anchor_tour[0]:
            total_weight += self.graph[cycle[-1]][anchor_tour[0]]
            cycle.append(anchor_tour[0])
        
        return cycle, total_weight
    
    def _local_optimization(self, cycle: List[int]) -> Tuple[List[int], int]:
        """
        Apply local search optimization (2-opt) to improve the cycle.
        """
        if len(cycle) <= 3:
            return cycle, self._calculate_cycle_weight(cycle)
        
        improved = True
        iteration = 0
        max_iterations = min(100, len(cycle))  # Limit iterations
        
        while improved and iteration < max_iterations:
            improved = False
            iteration += 1
            
            for i in range(len(cycle) - 1):
                for j in range(i + 2, len(cycle) - 1):
                    # Avoid adjacent edges
                    if j == i + 1:
                        continue
                    
                    # Calculate improvement from 2-opt swap
                    current_edges = (self.graph[cycle[i]][cycle[i+1]] + 
                                   self.graph[cycle[j]][cycle[j+1]])
                    new_edges = (self.graph[cycle[i]][cycle[j]] + 
                               self.graph[cycle[i+1]][cycle[j+1]])
                    
                    if new_edges < current_edges:
                        # Perform 2-opt swap
                        cycle[i+1:j+1] = cycle[i+1:j+1][::-1]
                        improved = True
                        break
                if improved:
                    break
        
        return cycle, self._calculate_cycle_weight(cycle)
    
    def _calculate_cycle_weight(self, cycle: List[int]) -> int:
        """Calculate total weight of a cycle."""
        if len(cycle) <= 1:
            return 0
        
        weight = 0
        for i in range(len(cycle) - 1):
            weight += self.graph[cycle[i]][cycle[i+1]]
        return weight


# Wrapper function to maintain compatibility with original interface
def hamiltonian_cycle_heuristic_improved(graph, start, anchors=None, max_depth=-1, early_exit=False):
    """
    Improved Hamiltonian cycle heuristic with better architectural design.
    
    Args:
        graph: A complete, weighted graph represented as an adjacency matrix
        start: The starting vertex
        anchors: Ignored in new implementation (auto-selected intelligently)
        max_depth: Ignored in new implementation  
        early_exit: Ignored in new implementation
    
    Returns:
        A tuple containing:
        - cycle: A list of vertex indices representing the Hamiltonian cycle
        - total_weight: The sum of weights in the cycle
    """
    solver = ImprovedHamiltonianSolver(graph)
    # disable local search
    cycle, weight = solver.solve(start=start, local_search=False)
    '''
    print("Improved Hamiltonian Results: ")
    print(f"Cycle: {cycle} Weight: {weight}")
    print(f"Length of cycle: {len(cycle)}")
    '''
    return (cycle, weight)