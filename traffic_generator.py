import pandas as pd
import re, math
from collections import defaultdict
import pprint
#############Traffic Generator###############
def get_PE_traffic(rows, cols, M, precision, host_names=None):
    """
    Generate per-cycle traffic for a single PE array configuration.

    Args:
        rows, cols: PE array dimensions
        M: Number of activations per PE row
        precision: Bits per data packet
        host_names: List of host names (e.g., ['L3_0', 'L3_1']). If None, uses single 'Host'
    """
    # print(f"Generating traffic for PE array {rows}x{cols} with M={M} and precision={precision} bits")
    if host_names is None:
        host_names = ['Host']

    num_hosts = len(host_names)
    h_state = [[0] * cols for _ in range(rows)]
    max_cycles = M + rows + cols + M + rows + 20
    row_inputs = {
        r: [0] * r + [1] * M + [0] * max_cycles
        for r in range(rows)
    }

    t = 0
    consecutive_empty = 0
    total_outputs = 0
    expected_outputs = M * cols
    all_cycle_traffic = []
    t_fully_utilized = -1
    while True:
        cycle_traffic = {}
        next_h_state = [[0] * cols for _ in range(rows)]

        # Horizontal Flow (Activations) - distribute among hosts
        for r in range(rows):
            val = row_inputs[r][t] if t < len(row_inputs[r]) else 0
            if val == 1:
                # Split traffic equally among all hosts
                traffic_per_host = precision / num_hosts
                for host_name in host_names:
                    cycle_traffic[(host_name, f"PE_{r}_0")] = traffic_per_host
                next_h_state[r][0] = 1

            for c in range(cols - 1):
                if h_state[r][c] == 1:
                    cycle_traffic[(f"PE_{r}_{c}", f"PE_{r}_{c+1}")] = precision
                    next_h_state[r][c + 1] = 1

        # Vertical Flow (Partial Sums) - distribute among hosts
        for r in range(rows):
            for c in range(cols):
                if h_state[r][c] == 1:
                    if r < rows - 1:
                        cycle_traffic[(f"PE_{r}_{c}", f"PE_{r+1}_{c}")] = precision
                    else:
                        # Bottom row: final result exits to hosts (split equally)
                        traffic_per_host = precision / num_hosts
                        for host_name in host_names:
                            cycle_traffic[(f"PE_{r}_{c}", host_name)] = traffic_per_host
                        total_outputs += 1

        all_cycle_traffic.append(cycle_traffic.copy())
        h_state = next_h_state
        # --- NEW PIPELINE DETECTION LOGIC ---
        active_pe_count = sum(sum(row) for row in h_state)
        # print(f"Cycle {t}: Active PEs = {active_pe_count}, Total Outputs = {total_outputs}")
        if active_pe_count == rows * cols:
            # print(f"100% Utilization achieved at cycle {t}")
            t_fully_utilized = t
            
        if t == t_fully_utilized:
            # print("Terminating 1 cycle after reaching full utilization.")
            # pprint.pprint(cycle_traffic)
            break
        # ------------------------------------
        # Termination Logic
        if cycle_traffic:
            consecutive_empty = 0
        else:
            consecutive_empty += 1
            if total_outputs == expected_outputs:
                break
            elif consecutive_empty > max_cycles:
                break
        t += 1
    # pprint.pprint(all_cycle_traffic[-1])
    return [all_cycle_traffic[-1]]


def create_traffic_matrices_per_cycle(traffic_list):
    """Convert cycle-by-cycle traffic into dataframe matrices"""
    all_sources = set()
    all_destinations = set()

    for cycle_traffic in traffic_list:
        for (source, dest), traffic in cycle_traffic.items():
            all_sources.add(source)
            all_destinations.add(dest)

    # Separate PE nodes, host nodes (L\d+_\d+), and other nodes
    pe_sources = sorted([node for node in all_sources if node.startswith('PE_')])
    host_sources = sorted([node for node in all_sources if re.match(r'L\d+_\d+', node)],
                         key=lambda x: tuple(map(int, re.findall(r'\d+', x))))
    other_sources = sorted([node for node in all_sources
                           if not node.startswith('PE_') and not re.match(r'L\d+_\d+', node)])

    pe_destinations = sorted([node for node in all_destinations if node.startswith('PE_')])
    host_destinations = sorted([node for node in all_destinations if re.match(r'L\d+_\d+', node)],
                              key=lambda x: tuple(map(int, re.findall(r'\d+', x))))
    other_destinations = sorted([node for node in all_destinations
                                if not node.startswith('PE_') and not re.match(r'L\d+_\d+', node)])

    all_sources_sorted = pe_sources + host_sources + other_sources
    all_destinations_sorted = pe_destinations + host_destinations + other_destinations

    dfs = []
    for cycle_traffic in traffic_list:
        traffic_matrix = pd.DataFrame(0, index=all_sources_sorted, columns=all_destinations_sorted)
        for (source, dest), traffic in cycle_traffic.items():
            traffic_matrix.loc[source, dest] = traffic
        dfs.append(traffic_matrix)

    return dfs


def distribute_dram_traffic(reachable_pes_per_router, router_names, tp_pe_ranges, total_traffic_per_pe):
    """
    Distribute DRAM traffic across routers based on reachability overlap.

    Args:
        reachable_pes_per_router: List of lists, each sublist contains PEs reachable from a router
        router_names: List of router names (e.g., ['L2_0', 'L2_1', 'L2_2', 'L2_3'])
        tp_pe_ranges: List of (start_pe, end_pe) tuples for each TP
        total_traffic_per_pe: Amount of traffic needed per PE for weight loading

    Returns:
        Dictionary: {(router_name, pe_num): traffic_amount}
    """
    traffic_distribution = {}

    for tp_start, tp_end in tp_pe_ranges:
        tp_pes = set(range(tp_start, tp_end + 1))

        # Find which routers can reach each PE in this TP
        pe_to_routers = defaultdict(list)
        for router_idx, reachable_pes in enumerate(reachable_pes_per_router):
            for pe in reachable_pes:
                if pe in tp_pes:
                    pe_to_routers[pe].append(router_idx)

        # Distribute traffic for each PE
        for pe in tp_pes:
            routers_for_this_pe = pe_to_routers.get(pe, [])
            if routers_for_this_pe:
                # Split traffic equally among all routers that can reach this PE
                traffic_per_router = total_traffic_per_pe / len(routers_for_this_pe)
                for router_idx in routers_for_this_pe:
                    router_name = router_names[router_idx]
                    traffic_distribution[(router_name, pe)] = traffic_per_router

    return traffic_distribution

def calculate_traffic_for_TPs(A_dim, B_dim, PE_dim, TPs,
                               routers=None,  # e.g., [['L2_0','L2_1'], ['L2_2','L2_3']]
                               host_names=None,
                               RESOLUTION=16):
    if host_names is None:
        host_names = ['Host']

    sum_dfs = []

    # Calculate PE ranges per expert (TP)
    tp_pe_ranges = []
    current_pe = 0
    for tp_size in TPs:
        tp_pe_ranges.append((current_pe, current_pe + tp_size - 1))
        current_pe += tp_size

    for tp_idx, TP in enumerate(TPs):
        # Setup Dimensions
        A_row_tiles = math.ceil(A_dim[0] / PE_dim[0])

        A_col_tiles = math.ceil(A_dim[1] / PE_dim[1])

        B_col_tiles = math.ceil(B_dim[1] / PE_dim[1])


        sqrt_tp = int(math.sqrt(TP))
        rows = 1
        cols = TP
        for r in range(sqrt_tp, 0, -1):
            if TP % r == 0:
                rows = r
                cols = TP // r
                break

        M = A_col_tiles
        total_tile_ops = A_row_tiles * B_col_tiles * A_col_tiles
        
        parallel_cycles = math.ceil(total_tile_ops / TP)
        # print(f"\nparallel_cycles: {parallel_cycles}")
        # Generate base PE traffic
        cycle_dfs = create_traffic_matrices_per_cycle(
            get_PE_traffic(rows, cols, M, precision=4, host_names=host_names)
        )
   
        sum_df = sum(cycle_dfs, pd.DataFrame(0, index=cycle_dfs[0].index, columns=cycle_dfs[0].columns)).fillna(0)*PE_dim[0]
        # pprint.pprint(sum_df)
        # Add DRAM weight-loading traffic
        
        if routers is not None and tp_idx < len(routers):
            router_group = routers[tp_idx]       # e.g., ['L2_0', 'L2_1'] for this expert
            N = len(router_group)
            tp_start, tp_end = tp_pe_ranges[tp_idx]
            # print(tp_start,tp_end)
            # 1. Calculate Weight (B) tiles
            B_row_tiles = math.ceil(B_dim[0] / PE_dim[0])
            B_col_tiles = math.ceil(B_dim[1] / PE_dim[1])
            total_weight_tiles = B_row_tiles * B_col_tiles
            # 2. Ncycle: Weight tiles per PE (assuming even distribution)
            # Ncycle = total_weight_tiles / TP
            Ncycle = 1
            # 3. Elements per tile
            elements_per_tile = PE_dim[0] * PE_dim[1]
            # 4. Traffic per Router per PE
            # Ncycle * elements_per_tile * Resolution / Routers
            traffic_per_router_per_pe = math.ceil(Ncycle * elements_per_tile*4 / N)
            
            for pe_global in range(tp_start, tp_end + 1):
                pe_local = pe_global - tp_start
                pe_node = f"PE_{pe_local // cols}_{pe_local % cols}"

                for router_name in router_group:
                    if router_name not in sum_df.index:
                        sum_df.loc[router_name] = 0
                    if router_name not in sum_df.columns:
                        sum_df[router_name] = 0
                    if pe_node in sum_df.columns:
                        sum_df.loc[router_name, pe_node] += traffic_per_router_per_pe

        sum_dfs.append(sum_df)

    return sum_dfs

def renumber_and_extend_pes(df, start_num=0, max_pes=16, all_router_nodes=None, all_host_nodes=None):
    """Renumber PEs and extend dataframe to include all routers and hosts"""
    all_nodes = set(df.index) | set(df.columns)

    def get_coords(name):
        parts = re.findall(r'\d+', name)
        return tuple(map(int, parts)) if parts else (float('inf'),)

    def get_router_key(name):
        # Extract level and number for sorting (e.g., L2_0 -> (2, 0))
        match = re.search(r'L(\d+)_(\d+)', name)
        if match:
            return (int(match.group(1)), int(match.group(2)))
        return (float('inf'), float('inf'))

    pe_nodes = sorted([n for n in all_nodes if n.startswith('PE')], key=get_coords)

    # Separate routers from hosts based on provided lists
    router_nodes = []
    host_nodes = []

    for n in all_nodes:
        if re.match(r'L\d+_\d+', n):
            if all_host_nodes and n in all_host_nodes:
                host_nodes.append(n)
            elif all_router_nodes and n in all_router_nodes:
                router_nodes.append(n)
            else:
                # If not explicitly specified, try to infer
                router_nodes.append(n)

    router_nodes = sorted(set(router_nodes), key=get_router_key)
    host_nodes = sorted(set(host_nodes), key=get_router_key)

    # Handle legacy 'Host' node
    if 'Host' in all_nodes:
        host_nodes.append('Host')

    # Renumber PEs
    mapping = {}
    current_num = start_num
    for old_name in pe_nodes:
        mapping[old_name] = f"PE_{current_num}"
        current_num += 1
    last_active_num = current_num - 1 if pe_nodes else -1

    new_df = df.rename(index=mapping, columns=mapping)

    # Build target nodes list
    target_pe_nodes = [f"PE_{i}" for i in range(max_pes)]

    if all_router_nodes is not None:
        target_router_nodes = sorted(all_router_nodes, key=get_router_key)
    else:
        target_router_nodes = router_nodes

    if all_host_nodes is not None:
        target_host_nodes = sorted(all_host_nodes, key=lambda x: get_router_key(x) if re.match(r'L\d+_\d+', x) else (float('inf'), float('inf')))
    else:
        target_host_nodes = host_nodes

    target_nodes = target_pe_nodes + target_router_nodes + target_host_nodes

    extended_df = new_df.reindex(
        index=target_nodes, columns=target_nodes, fill_value=0
    ).fillna(0).astype(int)

    return extended_df, last_active_num


def make_complete_traffic(A_dim, B_dim, PE_dim, TPs,
                          routers=None,        # e.g., [['L2_0','L2_1'], ['L2_2','L2_3']]
                          host_names=None,
                          RESOLUTION=16,
                          max_pes=16):
    if host_names is None:
        host_names = ['Host']

    sum_dfs_list = calculate_traffic_for_TPs(
        A_dim, B_dim, PE_dim, TPs,
        routers=routers,
        host_names=host_names,
        RESOLUTION=RESOLUTION
    )

    # Collect all router nodes across all groups
    global_router_nodes = set()
    if routers is not None:
        for router_group in routers:
            global_router_nodes.update(router_group)
    else:
        for df in sum_dfs_list:
            all_nodes = set(df.index) | set(df.columns)
            for n in all_nodes:
                if re.match(r'L\d+_\d+', n) and (host_names is None or n not in host_names):
                    global_router_nodes.add(n)

    # Renumber and extend each TP's dataframe
    start = 0
    all_dfs = []
    for df in sum_dfs_list:
        new_df, last_used = renumber_and_extend_pes(
            df, start_num=start, max_pes=max_pes,
            all_router_nodes=global_router_nodes,
            all_host_nodes=host_names
        )
        all_dfs.append(new_df)
        start = last_used + 1

    sum_df = sum(all_dfs, pd.DataFrame(0, index=all_dfs[0].index, columns=all_dfs[0].columns)).fillna(0)
    return sum_df
#############################################