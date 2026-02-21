#@title Network topology
import networkx as nx
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Set, Optional
from dataclasses import dataclass, field
from copy import deepcopy
from collections import defaultdict
import re
import pandas as pd
import heapq, random
from TP_config import *
from traffic_generator import *
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from tqdm import tqdm  # Add this import at the top
from pymoo.indicators.hv import Hypervolume
import plotly.graph_objects as go
@dataclass
class NetworkTopology:
    """Simple container for any network topology"""
    PE_count: int
    level_names: List[str] = field(default_factory=list)
    routers: Dict[str, List[str]] = field(default_factory=dict)
    edges: List[Tuple[str, str, Dict]] = field(default_factory=list)

    def __post_init__(self):
        self.G = nx.DiGraph()
        self._built = False

    def add_level(self, name: str, router_count: int):
        """Add a level with N routers"""
        self.level_names.append(name)
        self.routers[name] = [f"{name}_{i}" for i in range(router_count)]
        return self

    def connect(self, src: str, dst: str, link_type: str = "down", **attrs):
        """Add any connection between any two nodes"""
        self.edges.append((src, dst, {"type": link_type, **attrs}))
        return self
    def get_reachable_PEs(self, level_name: str) -> List[List[str]]:
        """
        Get list of PEs reachable from each router in a level (downward only).

        Parameters
        ----------
        level_name : name of the level (e.g., "L1", "L2", "L3")

        Returns
        -------
        List of lists, where each inner list contains PE IDs reachable from one router.
        Index i corresponds to router i in the level.

        Example
        -------
        >>> net.get_reachable_PEs("L2")
        [['PE_0', 'PE_1', 'PE_2', 'PE_3', 'PE_4', 'PE_5', 'PE_6', 'PE_7'],
        ['PE_8', 'PE_9', 'PE_10', 'PE_11', 'PE_12', 'PE_13', 'PE_14', 'PE_15'],
        ...]
        """
        if level_name == "PE":
            return [[f"PE_{i}"] for i in range(self.PE_count)]

        if level_name not in self.routers:
            raise ValueError(f"Level '{level_name}' not found.")

        if not self._built:
            self.build()

        routers = self.routers[level_name]
        result = []

        for router in routers:
            reachable_PEs = []
            visited = set()
            queue = [router]

            # BFS downward to find all reachable PEs
            while queue:
                current = queue.pop(0)
                if current in visited:
                    continue
                visited.add(current)

                # Check if this is a PE
                if current.startswith("PE_"):
                    reachable_PEs.append(int(current[3:]))  
                    continue

                # Find all downward connections from this router
                for src, dst, attrs in self.edges:
                    if src == current and attrs.get('type') == 'down':
                        if dst not in visited:
                            queue.append(dst)

            result.append(sorted(reachable_PEs))

        return result 
    def connect_levels(self, lower_level: str, upper_level: str,
                   num_groups: int = 1,
                   children_per_router: int = None,
                   parents_per_router: int = None) -> int:
        """
        Connect two named levels with strict parent/child constraints.
        Supports PE level as lower_level (special case).

        Parameters
        ----------
        lower_level        : name of the lower level ("PE" or router level name)
        upper_level        : name of the upper level (router level name)
        num_groups         : number of symmetric groups/pods to create
        children_per_router: how many lower routers EACH upper router MUST connect to
        parents_per_router : how many upper routers EACH lower router MUST connect to

        Returns
        -------
        0  : Success - all routers exactly follow the constraints
        -1 : Failure - one or more routers don't match constraints

        Special Case: PE Level
        ----------------------
        If lower_level == "PE", then:
        - lower_routers = ["PE_0", "PE_1", ..., "PE_{PE_count-1}"]
        - parents_per_router = number of L1 routers each PE connects to (usually 1)
        - children_per_router = number of PEs each L1 router connects to
        """
        # ── Validate inputs ───────────────────────────────────────────────────
        if upper_level not in self.routers:
            raise ValueError(
                f"Level '{upper_level}' not found. "
                f"Available levels: {list(self.routers.keys())}"
            )
        if num_groups < 1:
            raise ValueError(f"num_groups must be >= 1, got {num_groups}")

        # ── Handle PE level as special case ───────────────────────────────────
        if lower_level == "PE":
            lower_routers = [f"PE_{i}" for i in range(self.PE_count)]
            num_lower = self.PE_count
        else:
            if lower_level not in self.routers:
                raise ValueError(
                    f"Level '{lower_level}' not found. "
                    f"Available levels: {list(self.routers.keys())}"
                )
            lower_routers = self.routers[lower_level]
            num_lower = len(lower_routers)

        upper_routers = self.routers[upper_level]
        num_upper = len(upper_routers)

        # ── Calculate routers per group ───────────────────────────────────────
        lower_per_group = num_lower // num_groups
        upper_per_group = num_upper // num_groups

        # ── Require both parameters ───────────────────────────────────────────
        if children_per_router is None or parents_per_router is None:
            raise ValueError(
                f"Both children_per_router and parents_per_router must be specified."
            )

        # ── Check mathematical consistency ────────────────────────────────────
        total_links_from_lower = num_lower * parents_per_router
        total_links_from_upper = num_upper * children_per_router
        
        if total_links_from_lower != total_links_from_upper:
            return -1

        # ── Validate parameters against available routers ─────────────────────
        if children_per_router > lower_per_group:
            return -1

        if parents_per_router > upper_per_group:
            return -1

        # ── Build edges with intelligent strided distribution ─────────────────
        existing = {(src, dst) for src, dst, _ in self.edges}

        for g in range(num_groups):
            g_lower_start = g * lower_per_group
            g_upper_start = g * upper_per_group
            g_lower_end = (g + 1) * lower_per_group if g < num_groups - 1 else num_lower
            g_upper_end = (g + 1) * upper_per_group if g < num_groups - 1 else num_upper
            
            g_lower = lower_routers[g_lower_start:g_lower_end]
            g_upper = upper_routers[g_upper_start:g_upper_end]

            actual_lower_per_group = len(g_lower)
            actual_upper_per_group = len(g_upper)

            if actual_upper_per_group == 0 or actual_lower_per_group == 0:
                continue

            stride = max(1, actual_upper_per_group // parents_per_router)

            for lower_idx, lower in enumerate(g_lower):
                for k in range(parents_per_router):
                    upper_idx = (lower_idx * stride + k) % actual_upper_per_group
                    upper = g_upper[upper_idx]

                    if (upper, lower) not in existing:
                        existing.add((upper, lower))
                        self.edges.append((upper, lower, {"type": "down"}))

                    if (lower, upper) not in existing:
                        existing.add((lower, upper))
                        self.edges.append((lower, upper, {"type": "up"}))

        # ── VALIDATION: Check each router follows the parent/child rule ───────
        lower_parent_count = {router: 0 for router in lower_routers}
        upper_child_count = {router: 0 for router in upper_routers}
        
        for src, dst, attrs in self.edges:
            if src in upper_routers and dst in lower_routers and attrs.get('type') == 'down':
                upper_child_count[src] += 1
            if src in lower_routers and dst in upper_routers and attrs.get('type') == 'up':
                lower_parent_count[src] += 1
        
        for lower in lower_routers:
            if lower_parent_count[lower] != parents_per_router:
                return -1
        
        for upper in upper_routers:
            if upper_child_count[upper] != children_per_router:
                return -1
        
        return 0
    
    @staticmethod
    def calculate_radix(num_lower: int, num_upper: int, num_groups: int = 1, redundancy_factor: int = 1) -> Tuple[int, int]:
        """
        Calculate valid parents_per_router and children_per_router values.

        Parameters
        ----------
        num_lower          : number of routers in lower level (or PE_count for PE level)
        num_upper          : number of routers in upper level
        num_groups         : number of symmetric groups/pods
        redundancy_factor  : multiply connections by this factor for more redundancy (default=1)

        Returns
        -------
        (parents_per_router, children_per_router) : tuple of valid values
        """
        import math

        # Calculate routers per group
        lower_per_group = num_lower // num_groups
        upper_per_group = num_upper // num_groups
        if num_groups == num_upper:
            redundancy_factor = 1  # Override redundancy factor if it would cause invalid configuration
        # Check divisibility
        if num_lower % num_groups != 0 or num_upper % num_groups != 0:
            raise ValueError(
                f"num_lower ({num_lower}) and num_upper ({num_upper}) must be "
                f"evenly divisible by num_groups ({num_groups})"
            )

        # Special case: num_groups == num_upper means 1 upper router per group
        if num_groups == num_upper:
            if redundancy_factor > 1:
                raise ValueError(
                    f"When num_groups ({num_groups}) == num_upper ({num_upper}), "
                    f"each group has only 1 upper router. "
                    f"redundancy_factor must be 1, got {redundancy_factor}."
                )

        # Find GCD for minimum valid ratio
        gcd = math.gcd(lower_per_group, upper_per_group)

        # Calculate minimum values
        parents_per_router = upper_per_group // gcd
        children_per_router = lower_per_group // gcd

        # Apply redundancy factor
        parents_per_router *= redundancy_factor
        children_per_router *= redundancy_factor

        # Validate against available routers
        if parents_per_router > upper_per_group:
            raise ValueError(
                f"parents_per_router ({parents_per_router}) exceeds "
                f"upper_per_group ({upper_per_group}). "
                f"Reduce redundancy_factor or num_groups."
            )

        if children_per_router > lower_per_group:
            raise ValueError(
                f"children_per_router ({children_per_router}) exceeds "
                f"lower_per_group ({lower_per_group}). "
                f"Reduce redundancy_factor or num_groups."
            )

        return parents_per_router, children_per_router

    def build(self):
        """Construct the NetworkX graph"""
        self.G = nx.DiGraph()
        for level_idx, (level_name, routers) in enumerate(self.routers.items()):
            for r in routers:
                self.G.add_node(r, node_type="router", level=level_idx + 1, level_name=level_name)
        for i in range(self.PE_count):
            self.G.add_node(f"PE_{i}", node_type="PE", level=0)
        for src, dst, attrs in self.edges:
            if src in self.G and dst in self.G:
                self.G.add_edge(src, dst, **attrs)
        self._built = True
        return self

    def get_network_dict(self) -> Dict:
        if not self._built:
            self.build()
        result = {}
        for node in self.G.nodes():
            neighbors = []
            for neighbor in self.G.successors(node):
                weight = self.G[node][neighbor].get('weight', 1)
                neighbors.append((neighbor, weight))
            if neighbors:
                result[node] = neighbors
        return result

    def visualize(self, figsize=(10, 8), title=None, show_degree=False):
        """Visualize with hierarchical layout"""
        if not self._built:
            self.build()
        plt.figure(figsize=figsize)
        pos = {}
        levels = {}
        for node, data in self.G.nodes(data=True):
            lvl = data.get('level', 0)
            if lvl not in levels:
                levels[lvl] = []
            levels[lvl].append(node)

        def sort_key(node):
            if node == "Host":
                return (0, "")
            elif "PE" in node:
                return (2, int(node.split('_')[1]))
            else:
                lvl = self.G.nodes[node].get('level', 0)
                try:
                    router = int(node.split('_')[-1])
                except:
                    router = 0
                return (1, lvl, router)

        for lvl in levels:
            levels[lvl].sort(key=sort_key)

        max_width = max(len(nodes) for nodes in levels.values()) if levels else 1
        for lvl, nodes in levels.items():
            count = len(nodes)
            spacing = max_width / max(count, 1)
            for i, node in enumerate(nodes):
                x = (i - (count - 1) / 2) * spacing * 1.5
                y = lvl
                pos[node] = (x, y)

        colors = []
        for node in self.G.nodes():
            if node == "Host":
                colors.append("#e74c3c")
            elif "PE" in node:
                colors.append("#2ecc71")
            else:
                colors.append("#3498db")

        edge_colors = []
        for u, v, data in self.G.edges(data=True):
            link_type = data.get('type', '')
            if 'shortcut' in link_type:
                edge_colors.append("#e67e22")
            elif 'lateral' in link_type:
                edge_colors.append("#9b59b6")
            else:
                edge_colors.append("#34495e")

        nx.draw_networkx_nodes(self.G, pos, node_color=colors, node_size=800,
                               alpha=0.9, edgecolors='white', linewidths=2)
        nx.draw_networkx_edges(self.G, pos, edge_color=edge_colors, arrows=True,
                               alpha=0.6, width=1.5, arrowstyle='->', arrowsize=20)

        if show_degree:
            labels = {}
            for node in self.G.nodes():
                if node == "Host":
                    labels[node] = "Host"
                elif "PE" in node:
                    labels[node] = node.split('_')[1]
                else:
                    deg = self.G.degree(node)
                    labels[node] = f"{node}\n({deg})"
        else:
            labels = {n: n.replace('_', '\n') if 'R' in n or 'L' in n else n for n in self.G.nodes()}

        nx.draw_networkx_labels(self.G, pos, labels, font_size=7)
        plt.title(title or f"Network: {self.PE_count} leaves, {len(self.level_names)} levels")
        plt.axis('off')
        plt.tight_layout()
        if title:
            plt.savefig(f"{title}.png", dpi=150, bbox_inches='tight')
        plt.close()
###################################################################
from collections import defaultdict
import math
def get_router_ports(network_input):
    """
    Calculate the number of ports for each router in the network.
    Each bidirectional link counts as 1 port.
    
    Args:
        network_input: Network topology (dict or object with get_network_dict())
    
    Returns:
        list: Number of ports for each router, ordered by router name
    """
    
    # 1. Extract Topology
    if hasattr(network_input, 'get_network_dict'):
        tree = network_input.get_network_dict()
    else:
        tree = network_input
    
    # 2. Count connections for each node
    port_count = defaultdict(int)
    seen_links = set()
    
    for node, children in tree.items():
        for child, _ in children:
            # Create a canonical link representation (sorted tuple)
            link = tuple(sorted([node, child]))
            
            # Only count each bidirectional link once
            if link not in seen_links:
                seen_links.add(link)
                # Add 1 port to both ends of the link
                port_count[node] += 1
                port_count[child] += 1
    
    # 3. Sort routers by name and return port counts
    def get_router_key(name):
        # Extract level and number for sorting (e.g., L2_0 -> (2, 0))
        match = re.search(r'L(\d+)_(\d+)', name)
        if match:
            return (int(match.group(1)), int(match.group(2)))
        # For PE nodes
        match = re.search(r'PE_(\d+)', name)
        if match:
            return (float('inf'), int(match.group(1)))
        # For other nodes
        return (float('inf'), float('inf'))
    
    # Filter only router nodes (nodes that match L\d+_\d+ pattern)
    routers = [node for node in port_count.keys() if re.match(r'L\d+_\d+', node)]
    routers_sorted = sorted(routers, key=get_router_key)
    
    # Return list of port counts in order
    return [port_count[router] for router in routers_sorted]
   
def evaluate_network(network_input, traffic_matrix, root_nodes=None, dram_point='L1', force=None):
    if root_nodes is None:
        root_nodes = ['Host']
    elif isinstance(root_nodes, str):
        root_nodes = [root_nodes]
    
    # 1. Extract Topology
    if hasattr(network_input, 'get_network_dict'):
        tree = network_input.get_network_dict()
    else:
        tree = network_input

    # 2. Identify Valid Nodes
    valid_nodes = set(tree.keys()) | {child for children in tree.values() for child, weight in children}

    # 3. Build Hierarchy
    # 3. Build Hierarchy using BFS from all root nodes simultaneously
    parent = {}
    depth = {}
    from collections import deque
    
    queue = deque()
    
    # Initialize all root nodes
    for root_node in root_nodes:
        if root_node in valid_nodes:
            parent[root_node] = None
            depth[root_node] = 0
            queue.append(root_node)
    
    # BFS outward from all roots simultaneously
    visited = set(parent.keys())
    while queue:
        node = queue.popleft()
        for child, _ in tree.get(node, []):
            if child not in visited:
                visited.add(child)
                parent[child] = node
                depth[child] = depth[node] + 1
                queue.append(child)
    
    # Handle any remaining unvisited nodes
    for node in valid_nodes:
        if node not in visited:
            visited.add(node)
            parent[node] = None
            depth[node] = 0
            queue.append(node)
            while queue:
                n = queue.popleft()
                for child, _ in tree.get(n, []):
                    if child not in visited:
                        visited.add(child)
                        parent[child] = n
                        depth[child] = depth[n] + 1
                        queue.append(child)

    # Build children map for downward traversal
    children_map = defaultdict(list)
    for node, par in parent.items():
        if par is not None:
            children_map[par].append(node)

    # 4a. BFS downward from src to dst
    def find_downward_path(src, dst):
        """BFS using actual tree edges, not just parent-based children_map"""
        queue = deque([(src, [src])])
        visited_bfs = set()
        while queue:
            cur, path = queue.popleft()
            if cur == dst:
                return path
            if cur in visited_bfs:
                continue
            visited_bfs.add(cur)
            for child, _ in tree.get(cur, []):
                if child not in visited_bfs:
                    queue.append((child, path + [child]))
        return None

    # 4b. LCA Routing
    def get_lca_path(src, dst):
        if src == dst:
            return [src]
        if src not in depth or dst not in depth:
            return None

        src_ancestors = []
        cur = src
        while cur is not None:
            src_ancestors.append(cur)
            cur = parent.get(cur)
        
        dst_ancestors = []
        cur = dst
        while cur is not None:
            dst_ancestors.append(cur)
            cur = parent.get(cur)

        src_set = set(src_ancestors)
        dst_set = set(dst_ancestors)

        lca = None
        for ancestor in src_ancestors:
            if ancestor in dst_set:
                lca = ancestor
                break
        if lca is None:
            for ancestor in dst_ancestors:
                if ancestor in src_set:
                    lca = ancestor
                    break

        # If LCA not found, fall back to Dijkstra (happens in multi-root shared subtrees)
        if lca is None:
            return get_dijkstra_path(src, dst)
        
        if lca == src:
            return find_downward_path(src, dst)
        
        if lca == dst:
            path_up = []
            cur = src
            while cur != dst:
                path_up.append(cur)
                cur = parent.get(cur)
                if cur is None:
                    return get_dijkstra_path(src, dst)
            path_up.append(dst)
            return path_up

        path_up = []
        cur = src
        while cur != lca:
            path_up.append(cur)
            cur = parent.get(cur)
            if cur is None:
                return get_dijkstra_path(src, dst)
        path_up.append(lca)
        
        path_down = find_downward_path(lca, dst)
        if path_down is None:
            return get_dijkstra_path(src, dst)
        
        return path_up + path_down[1:]

    # 4c. Dijkstra Routing
    def get_dijkstra_path(src, dst):
        if src == dst:
            return [src]
        if src not in valid_nodes or dst not in valid_nodes:
            return None

        adj = defaultdict(list)
        for u in tree:
            for v, w in tree[u]:
                adj[u].append((v, w))
                adj[v].append((u, w))

        heap = [(0, src, [src])]
        visited_d = set()

        while heap:
            cost, node, path = heapq.heappop(heap)
            if node in visited_d:
                continue
            visited_d.add(node)
            if node == dst:
                return path
            for neighbor, weight in adj[node]:
                if neighbor not in visited_d:
                    heapq.heappush(heap, (cost + weight, neighbor, path + [neighbor]))

        return None

    # 5. Select routing method
    def get_path(src, dst):
        if force == 'lca':
            return get_lca_path(src, dst)
        elif force == 'dijkstra':
            return get_dijkstra_path(src, dst)
        if isinstance(src, str) and src.startswith(dram_point):
            return get_dijkstra_path(src, dst)
        return get_lca_path(src, dst)

    # 6. Accumulate Traffic
    link_util = defaultdict(int)
    
    if hasattr(traffic_matrix, 'stack'):
        traffic_matrix = traffic_matrix.T.groupby(level=0).sum().T.groupby(level=0).sum()
        traffic_dict = traffic_matrix.stack().to_dict()
        traffic_dict = {k: v for k, v in traffic_dict.items() if v > 0}
    else:
        traffic_dict = {k: v for k, v in traffic_matrix.items() if v > 0}

    for (src, dst), vol in traffic_dict.items():
        if src not in valid_nodes or dst not in valid_nodes:
            continue
        if vol == 0:
            continue

        path = get_path(src, dst)
        # print(f"Routing from {src} to {dst} with volume {vol}: Path: {path}")
        if not path or len(path) < 2:
            continue

        for i in range(len(path) - 1):
            link = (path[i], path[i + 1])
            link_util[link] += vol

    # 7. Calculate Stats
    all_physical_links = []
    for u in tree:
        for v, _ in tree[u]:
            all_physical_links.append((u, v))

    num_directed_links = len(all_physical_links)

    if num_directed_links == 0:
        return 0.0, 0.0

    total_util = sum(link_util.values())
    m = total_util / num_directed_links

    variance_sum = 0.0
    for link in all_physical_links:
        util = link_util.get(link, 0)
        variance_sum += (util - m) ** 2

    d = math.sqrt(variance_sum / num_directed_links)
    area=sum(get_router_ports(network_input))
    # area=max(get_router_ports(network_input))**2
    return m, d, area
######################
def get_factors(n):
    """
    Returns all factors of n (excluding n itself for upper levels, 
    but including it for flexibility)
    """
    factors = []
    for i in range(1, n + 1):  # Start from 2 to avoid single router levels
        if n % i == 0:
            factors.append(i)
    return factors

def generate_random_solution(net_input):
    while True:
        # Reset network at start of each attempt
        net = NetworkTopology(PE_count=net_input.PE_count)
        PE_count = net.PE_count
        
        num_levels = random.randint(2, 5)
        level_router_counts = []
        
        current_count = PE_count
        for i in range(num_levels):
            factors = get_factors(current_count)
            if i == 0:
                factors = [f for f in factors if f < PE_count and f > 1]
            else:
                factors = [f for f in factors if f <= current_count and f > 1]
            # if i == num_levels - 1:
            #     factors = [f for f in factors if f < current_count]
            if not factors:
                num_levels = i  # Set to actual number of levels created
                break
            selected_routers = random.choice(factors)
            level_router_counts.append(selected_routers)
            current_count = selected_routers
            net.add_level(f"L{i+1}", router_count=selected_routers)
            # if selected_routers == 1:
            #     num_levels = i + 1
            #     break

        try:
            num_groups = len(net.routers['L1'])
            p, c = net.calculate_radix(PE_count, level_router_counts[0], num_groups=num_groups, redundancy_factor=2)
            net.connect_levels("PE", "L1", num_groups=num_groups, parents_per_router=p, children_per_router=c)
        except Exception as e:

            continue  # restart from top

        groups = [num_groups]
        failed = False

        for i in range(1, num_levels):
            max_attempts = 100
            attempt = 0
            result = -1

            while result == -1 and attempt < max_attempts:
                attempt += 1
                try:
                    if i == num_levels - 1:
                        num_groups = 1
                    elif level_router_counts[i] == level_router_counts[i-1]:
                        x = [f for f in get_factors(level_router_counts[i]) if f < level_router_counts[i]]
                        num_groups = random.choice(x) if x else 1
                    else:
                        x = [f for f in get_factors(level_router_counts[i]) if f < level_router_counts[i]]
                        num_groups = random.choice(x) if x else 1

                    p, c = net.calculate_radix(level_router_counts[i-1], level_router_counts[i], num_groups=num_groups, redundancy_factor=2)
                    result = net.connect_levels(f"L{i}", f"L{i+1}", num_groups=num_groups, parents_per_router=p, children_per_router=c)


                except Exception as e:

                    result = -1

            if result == -1:
                failed = True
                break

            groups.append(num_groups)

        if failed:
            continue  # restart from top

        return net, level_router_counts, groups

def perturb(net_input, level_router_counts, groups):
    """
    Perturbs an existing solution. Tries each perturbation until one succeeds.
    Returns: (new_net, new_level_router_counts, new_groups)
    """
    max_attempts = 50
    PE_count = net_input.PE_count

    for _ in range(max_attempts):
        # Deep copy current solution state
        new_counts = level_router_counts.copy()
        num_levels = len(new_counts)

        # Pick a random perturbation
        perturb_type = random.choice(['change_router_count', 'change_groups', 'add_level', 'remove_level'])
        # print(f"Attempting perturbation: {perturb_type}")
        # ── PERTURB 1: Change one level's router count ─────────────────────
        if perturb_type == 'change_router_count':
            
            i = random.randint(0, num_levels - 1)
            below_count = PE_count if i == 0 else new_counts[i - 1]
            above_count = new_counts[i]
            valid = [f for f in get_factors(below_count) if f < below_count and f > 1 and f>=above_count]
                 
            # valid = [f for f in valid if f != new_counts[i]]  # must be different
       
            if not valid:
                continue
            new_counts[i] = random.choice(valid)
            # Cascade: levels above must still be valid factors
            for j in range(i + 1, num_levels):
                valid_above = [f for f in get_factors(new_counts[j - 1]) if f < new_counts[j - 1] and f > 1]
        
                if not valid_above:
                    break
                if new_counts[j] not in valid_above:
                    new_counts[j] = random.choice(valid_above)

        # ── PERTURB 2: Change num_groups for one connection ─────────────────
        elif perturb_type == 'change_groups':
            # Only intermediate connections (not PE->L1, not topmost)
            if num_levels < 3:
                continue
            i = random.randint(1, num_levels - 2)  # intermediate only
            valid = [f for f in get_factors(new_counts[i]) if f < new_counts[i]]
            valid = [f for f in valid if f != groups[i + 1]]  # must be different
            if not valid:
                continue
            new_groups_list = groups.copy()
            new_groups_list[i + 1] = random.choice(valid)
        # ── PERTURB 2: Change num_groups for one connection ─────────────────
        # elif perturb_type == 'change_groups':
        #     if num_levels < 2:
        #         continue
                
        #     # Allow picking index from 0 (PE->L1) up to num_levels - 2
        #     i = random.randint(0, num_levels - 2) 
        #     new_groups_list = groups.copy()
            
        #     if i == 0:
        #         # Level 0 (PE -> L1 connection)
        #         # Groups here must be a factor of PE_count and strictly > 1
        #         valid = [f for f in get_factors(PE_count) if f < PE_count and f > 1]
        #         valid = [f for f in valid if f != new_groups_list[0]]
                
        #         if not valid:
        #             continue
                    
        #         chosen_group = random.choice(valid)
        #         new_groups_list[0] = chosen_group
                
        #         # RULE: Number of L1 routers must match this new group count
        #         new_counts[0] = chosen_group
                
        #         # Cascade Fix: Because L1 routers changed, we MUST validate higher levels
        #         truncate_at = None
        #         for j in range(1, num_levels):
        #             # Ensure cascade never picks 1
        #             valid_above = [f for f in get_factors(new_counts[j - 1]) 
        #                            if f < new_counts[j - 1] and f > 1]
        #             if not valid_above:
        #                 truncate_at = j
        #                 break
        #             if new_counts[j] not in valid_above:
        #                 new_counts[j] = random.choice(valid_above)
                
        #         # Safely drop any top levels if the cascade hit a dead end
        #         if truncate_at is not None:
        #             new_counts = new_counts[:truncate_at]
        #             new_groups_list = new_groups_list[:truncate_at]
                    
        #     else:
        #         # Intermediate connections (Keeping your original logic for i >= 1)
        #         valid = [f for f in get_factors(new_counts[i]) if f < new_counts[i]]
        #         valid = [f for f in valid if f != new_groups_list[i + 1]] 
        #         if not valid:
        #             continue
        #         new_groups_list[i + 1] = random.choice(valid)

        # ── PERTURB 3: Add a level ──────────────────────────────────────────
        elif perturb_type == 'add_level':
            if num_levels >= 5:
                continue
            
            # 1. Create a shuffled list of all possible insertion points
            possible_indices = list(range(num_levels))
            random.shuffle(possible_indices)
            
            level_added = False
            for i in possible_indices:
                below_count = PE_count if i == 0 else new_counts[i - 1]
                above_count = new_counts[i]
                
                # 2. Add 'f != 1' to ensure 1 is excluded from valid options
                valid = [f for f in get_factors(below_count) 
                         if f < below_count and above_count in get_factors(f) and f != 1]
                
                if valid:
                    new_count = random.choice(valid)
                    print(f"Adding level between L{i} and L{i+1} with {new_count} routers")
                    new_counts.insert(i, new_count)
                    num_levels += 1
                    level_added = True
                    break  # Success! Break out of the loop
            
            # If we tried all possible levels and none were valid, skip perturbation
            if not level_added:
                continue

        # ── PERTURB 4: Remove a level ───────────────────────────────────────
        elif perturb_type == 'remove_level':
            if num_levels <= 2:
                continue
            i = random.randint(0, num_levels - 2)  # don't remove topmost
            below_count = PE_count if i == 0 else new_counts[i - 1]
            above_count = new_counts[i + 1]
            # above_count must be a valid factor of below_count
            if above_count not in get_factors(below_count):
                continue
            new_counts.pop(i)
            num_levels -= 1

        # ── Rebuild network with new_counts ─────────────────────────────────
        result = _build_network_from_counts(PE_count, new_counts)
        if result is not None:
            return result

    # If all attempts fail, return original unchanged
    return net_input, level_router_counts, groups

def _build_network_from_counts(PE_count, level_router_counts):
    """
    Builds a full network from PE_count and level_router_counts.
    Returns (net, level_router_counts, groups) or None if invalid.
    """
    num_levels = len(level_router_counts)
    net = NetworkTopology(PE_count=PE_count)

    for i, count in enumerate(level_router_counts):
        net.add_level(f"L{i+1}", router_count=count)

    # PE -> L1
    try:
        num_groups = level_router_counts[0]
        p, c = net.calculate_radix(PE_count, level_router_counts[0],
                                   num_groups=num_groups, redundancy_factor=2)
        res = net.connect_levels("PE", "L1", num_groups=num_groups,
                                 parents_per_router=p, children_per_router=c)
        if res == -1:
            return None
    except Exception:
        return None

    groups = [num_groups]

    # L1 -> L2 -> ... -> Ln
    for i in range(1, num_levels):
        max_attempts = 100
        attempt = 0
        result = -1

        while result == -1 and attempt < max_attempts:
            attempt += 1
            try:
                if i == num_levels - 1:
                    num_groups = 1
                elif level_router_counts[i] == level_router_counts[i - 1]:
                    x = [f for f in get_factors(level_router_counts[i])
                         if f < level_router_counts[i]]
                    num_groups = random.choice(x) if x else 1
                else:
                    x = [f for f in get_factors(level_router_counts[i])
                         if f < level_router_counts[i]]
                    num_groups = random.choice(x) if x else 1

                p, c = net.calculate_radix(level_router_counts[i - 1],
                                           level_router_counts[i],
                                           num_groups=num_groups,
                                           redundancy_factor=2)
                result = net.connect_levels(f"L{i}", f"L{i+1}",
                                            num_groups=num_groups,
                                            parents_per_router=p,
                                            children_per_router=c)
            except Exception:
                result = -1

        if result == -1:
            return None

        groups.append(num_groups)

    return net, level_router_counts, groups


def get_objectives(network_input, PE_count, DRAM_point, A_dim, B_dim, PE_dim, partitions=None):
    top_level = max(network_input.routers.keys(), key=lambda x: int(x[1:]))
    host_names = [f'{top_level}_{i}' for i in range(len(network_input.routers[top_level]))]
    TPs_list = get_all_scenarios_even_or_one(PE_count)
    if partitions is not None:
        TPs_list = TPs_list[:partitions]
    
    results = {}
    mean_utils = []    
    std_utils = []
    
    # Wrap the loop with tqdm for progress bar
    for each in tqdm(TPs_list, desc="Evaluating partitions", unit="partition", ncols=100, leave=False):
        results[tuple(each)] = {}
        num_experts_per_router = PE_count // len(network_input.routers[DRAM_point])
        routers = []
        for i in range(len(each)):
            routers.append([f'{DRAM_point}_{i // num_experts_per_router}'])
        
        traffic_matrix = make_complete_traffic(
            A_dim=A_dim,
            B_dim=B_dim,
            PE_dim=PE_dim,
            TPs=each,
            routers=routers,
            host_names=host_names,
            RESOLUTION=16,
            max_pes=PE_count
        )
        min_nonzero = traffic_matrix[traffic_matrix > 0].min().min()
        # 2. If the matrix isn't entirely zeros, scale it down
        if pd.notna(min_nonzero) and min_nonzero > 0:
            scale_factor = min_nonzero / 5.0
            traffic_matrix = (traffic_matrix / scale_factor).round().astype(int)
        # -------------------------
        
        m, d, a = evaluate_network(network_input, traffic_matrix, host_names, dram_point=DRAM_point)
        mean_utils.append(round(m, 1))
        std_utils.append(round(d, 1))
    
    return np.mean(mean_utils).item(), np.max(std_utils).item()-np.mean(std_utils).item(), a

####################################

# def dominates(obj_a, obj_b):
#     """Returns True if obj_a dominates obj_b (minimization)"""
#     return all(a <= b for a, b in zip(obj_a, obj_b)) and any(a < b for a, b in zip(obj_a, obj_b))
def dominates_strict(obj_a, obj_b):
    """Standard Pareto dominance for AMOSA - minimization"""
    return all(a <= b for a, b in zip(obj_a, obj_b)) and any(a < b for a, b in zip(obj_a, obj_b))

def dominates(p, q, k=None):
    """Used by WFG hypervolume insert()"""
    if k is not None:
        return all(p[i] >= q[i] for i in range(k, len(p)))
    else:
        return dominates_strict(p, q)

# def update_archive(archive, candidate_net, candidate_counts, candidate_groups, candidate_obj, archive_size):
#     """Add candidate to archive if non-dominated, remove dominated solutions, prune if needed"""
#     # Remove solutions dominated by candidate
#     archive = [(n, c, g, o) for n, c, g, o in archive if not dominates(candidate_obj, o)]
    
#     # Add candidate if not dominated by any in archive
#     if not any(dominates(o, candidate_obj) for _, _, _, o in archive):
#         archive.append((candidate_net, candidate_counts, candidate_groups, candidate_obj))
    
#     # Prune archive if too large (remove most crowded solutions)
#     if len(archive) > archive_size:
#         archive = prune_archive(archive, archive_size)
    
#     return archive

def prune_archive(archive, archive_size):
    """Remove most crowded solutions to maintain archive size"""
    while len(archive) > archive_size:
        # Calculate crowding distance for each solution
        min_dist = float('inf')
        min_idx = 0
        for i, (_, _, _, obj_i) in enumerate(archive):
            # Find minimum distance to any other solution in objective space
            for j, (_, _, _, obj_j) in enumerate(archive):
                if i == j:
                    continue
                dist = sum((a - b) ** 2 for a, b in zip(obj_i, obj_j)) ** 0.5
                if dist < min_dist:
                    min_dist = dist
                    min_idx = i
        archive.pop(min_idx)
    return archive

# def amosa_acceptance(current_obj, candidate_obj, archive, T_curr):
#     """
#     AMOSA acceptance criterion.
#     Returns True if candidate should be accepted.
#     """
#     candidate_dominated_by_archive = any(dominates(o, candidate_obj) for _, _, _, o in archive)
#     candidate_dominates_current = dominates(candidate_obj, current_obj)
#     current_dominates_candidate = dominates(current_obj, candidate_obj)

#     # Case 1: candidate dominates current and not dominated by archive -> always accept
#     if candidate_dominates_current and not candidate_dominated_by_archive:
#         return True

#     # Case 2: non-dominating each other and candidate not dominated by archive
#     if not candidate_dominates_current and not current_dominates_candidate and not candidate_dominated_by_archive:
#         delta = sum(abs(a - b) for a, b in zip(candidate_obj, current_obj))
#         prob = math.exp(-delta / T_curr)
#         return random.random() < prob

#     # Case 3: candidate is dominated
#     delta = sum(abs(a - b) for a, b in zip(candidate_obj, current_obj))
#     prob = math.exp(-delta / T_curr)
#     return random.random() < prob


def compute_hv(archive_objs, ref_point=None):
    """Compute hypervolume of archive using pymoo"""
    if len(archive_objs) == 0:
        return 0.0
    F = np.array([list(o) for o in archive_objs])
    if ref_point is None:
        ref_point = F.max(axis=0) * 1.1  # 10% worse than worst in each objective
    metric = Hypervolume(ref_point=ref_point)
    return metric.do(F)

def get_ref_point(archive):
    """Dynamic reference point: 10% worse than worst solution in archive"""
    objs = np.array([list(o) for _, _, _, o in archive])
    return objs.max(axis=0) * 1.1


def update_archive(archive, candidate_net, candidate_counts, candidate_groups, candidate_obj, archive_size):
    
    # Remove solutions strictly dominated by candidate
    archive = [(n, c, g, o) for n, c, g, o in archive 
               if not dominates_strict(candidate_obj, o)]
    
    # Add candidate if not strictly dominated by any in archive
    is_dominated = any(dominates_strict(o, candidate_obj) for _, _, _, o in archive)
    if not is_dominated:
        archive.append((candidate_net, candidate_counts, candidate_groups, candidate_obj))
    


    # Prune
    if len(archive) > archive_size:
        ref_point = get_ref_point(archive)
        archive_objs = [o for _, _, _, o in archive]
        hv_full = compute_hv(archive_objs, ref_point)
        hv_loss = []
        for i in range(len(archive)):
            others = [o for j, o in enumerate(archive_objs) if j != i]
            hv_without_i = compute_hv(others, ref_point) if others else 0.0
            hv_loss.append(hv_full - hv_without_i)
        min_idx = hv_loss.index(min(hv_loss))
        archive.pop(min_idx)

    return archive


def amosa_acceptance(current_obj, candidate_obj, archive, T_curr):
    candidate_dominated_by_archive = any(dominates_strict(o, candidate_obj) for _, _, _, o in archive)
    candidate_dominates_current = dominates_strict(candidate_obj, current_obj)
    current_dominates_candidate = dominates_strict(current_obj, candidate_obj)

    archive_objs = [o for _, _, _, o in archive]
    ref_point = get_ref_point(archive)

    if candidate_dominates_current and not candidate_dominated_by_archive:
        return True

    if not candidate_dominates_current and not current_dominates_candidate and not candidate_dominated_by_archive:
        hv_with_candidate = compute_hv(archive_objs + [candidate_obj], ref_point)
        hv_with_current   = compute_hv(archive_objs + [current_obj], ref_point)
        delta = hv_with_current - hv_with_candidate
        prob = math.exp(-delta / T_curr) if delta > 0 else 1.0
        return random.random() < prob

    hv_with_candidate = compute_hv(archive_objs + [candidate_obj], ref_point)
    hv_with_current   = compute_hv(archive_objs + [current_obj], ref_point)
    delta = hv_with_current - hv_with_candidate
    prob = math.exp(-delta / T_curr) if delta > 0 else 1.0
    return random.random() < prob


def compute_total_hv(archive):
    if len(archive) < 1:
        return 0.0
    archive_objs = [o for _, _, _, o in archive]
    ref_point = get_ref_point(archive)
    return compute_hv(archive_objs, ref_point)



def plot_pareto_front_3d(archive, history_objs):
    """
    Plots a static 3D scatter plot of all explored points and the Pareto archive.
    
    Args:
        archive: List of tuples from AMOSA, e.g., [(_, counts, groups, (m, d, a)), ...]
        history_objs: List of all evaluated objective tuples, e.g., [(m, d, a), ...]
    """
    # 1. Extract Archive (Pareto Front) objectives
    # Assuming archive structure: (solution, counts, groups, (m, d, a))
    a_m = [o[3][0] for o in archive]
    a_d = [o[3][1] for o in archive]
    a_a = [o[3][2] for o in archive]

    # 2. Extract History (All Explored) objectives
    # Assuming history_objs structure: (m, d, a)
    h_m = [obj[0] for obj in history_objs]
    h_d = [obj[1] for obj in history_objs]
    h_a = [obj[2] for obj in history_objs]

    # 3. Setup Matplotlib 3D Figure
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 4. Plot all explored points (Background, semi-transparent)
    ax.scatter(h_m, h_d, h_a, 
               c='blue', 
               alpha=0.4, 
               marker='o', 
               s=20, 
               label='Explored Points')

    # 5. Plot Pareto archive points (Foreground, distinct color)
    ax.scatter(a_m, a_d, a_a, 
               c='red', 
               alpha=1.0, 
               marker='^', 
               s=60, 
               edgecolors='black', 
               label='Pareto Front (Archive)')

    # 6. Formatting
    ax.set_title("AMOSA Design Space Exploration", fontsize=14, pad=20)
    ax.set_xlabel('Mean Utilization (m)', labelpad=10)
    ax.set_ylabel('Std Utilization (d)', labelpad=10)
    ax.set_zlabel('Hardware Cost (a)', labelpad=10)
    
    # Adjust viewing angle if desired (elevation, azimuth)
    ax.view_init(elev=30, azim=120) 

    ax.legend(loc='upper left')
    plt.tight_layout()

    # Save and show
    plt.savefig('pareto_front_static.png', dpi=300)
    print("Saved to pareto_front_static.png")