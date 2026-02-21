import csv
from network import *
from traffic_generator import *
from TP_config import *
import pprint, csv,tqdm
if __name__ == "__main__":

    PE_count = 32
    Batch=64
    dmodel=2048
    dff=128
    A_dim = [Batch, dmodel]
    B_dim = [dmodel, dff]
    PE_dim = [64, 64]
    
    net = NetworkTopology(PE_count=PE_count)
    
    # BFT##################################
    # net.add_level("L1", router_count=8)
    # net.add_level("L2", router_count=4)
    # net.add_level("L3", router_count=2)
    # num_groups = 8
    # p, c = net.calculate_radix(32, 8, num_groups=num_groups, redundancy_factor=2)
    # result1 = net.connect_levels("PE", "L1", num_groups=num_groups, parents_per_router=p,children_per_router=c)
    # num_groups = 2
    # p, c = net.calculate_radix(8, 4, num_groups=num_groups, redundancy_factor=2)
    # result1 = net.connect_levels("L1", "L2", num_groups=num_groups, parents_per_router=p,children_per_router=c)
    # num_groups = 1
    # p, c = net.calculate_radix(4, 2, num_groups=num_groups, redundancy_factor=2)
    # result1 = net.connect_levels("L2", "L3", num_groups=num_groups, parents_per_router=p,children_per_router=c)
    ####################################### FT###############################
    net.add_level("L1", router_count=8)
    net.add_level("L2", router_count=8)
    net.add_level("L3", router_count=4)
    num_groups = 8
    p, c = net.calculate_radix(32, 8, num_groups=num_groups, redundancy_factor=2)
    result1 = net.connect_levels("PE", "L1", num_groups=num_groups, parents_per_router=p,children_per_router=c)
    num_groups = 4
    p, c = net.calculate_radix(8, 8, num_groups=num_groups, redundancy_factor=2)
    result1 = net.connect_levels("L1", "L2", num_groups=num_groups, parents_per_router=p,children_per_router=c)
    num_groups = 1
    p, c = net.calculate_radix(8, 4, num_groups=num_groups, redundancy_factor=2)
    result1 = net.connect_levels("L2", "L3", num_groups=num_groups, parents_per_router=p,children_per_router=c)
    # ######################Ring
    # level='L1'
    # for i in range(len(net.routers[level])):
    #     net.connect(f'{level}_{i}', f'{level}_{i+1}')
    #     if i == len(net.routers[level]) - 1:
    #         net.connect(f'{level}_{i}', f'{level}_0')
    #######################
    # Build and Visualize
    net.build()
    net.visualize(title=f"Custom Network Topology", figsize=(12, 5))

    TPs_list = get_all_scenarios_even_or_one(PE_count)
    # TPs_list =[[32]]
    DRAM_points = ['L2']
    top_level = max(net.routers.keys(), key=lambda x: int(x[1:]))
    host_names = [f'{top_level}_{i}' for i in range(len(net.routers[top_level]))]

    results = {}  # {each: {dram_point: (m, d)}}

    for each in tqdm.tqdm(TPs_list, desc="Evaluating partitions", unit="partition", ncols=100, leave=False):
        results[tuple(each)] = {}
        for DRAM_point in DRAM_points:
            # pprint.pprint(f"Evaluating for TP scenario: {each}, DRAM point: {DRAM_point}")
            
            num_experts_per_router = PE_count // len(net.routers[DRAM_point])
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
                RESOLUTION=4,
                max_pes=sum(TPs_list[0])
            )
            # pprint.pprint(f"Traffic Matrix for TP scenario: {each}, DRAM point: {DRAM_point}")
            # pprint.pprint(traffic_matrix)
            m, d, a = evaluate_network(net, traffic_matrix, host_names, dram_point=DRAM_point)
            results[tuple(each)][DRAM_point] = (m, d)
            
    print("\n" + "="*40)
    print(" SUMMARY STATISTICS")
    print("="*40)

    for dp in DRAM_points:
        # Extract all means and stds for this DRAM point across all TP scenarios
        means = [tp_results[dp][0] for tp_results in results.values()]
        stds = [tp_results[dp][1] for tp_results in results.values()]
        
        if not means:
            continue

        # Calculate Mean stats
        avg_mean = sum(means) / len(means)
        max_mean = max(means)
        min_mean = min(means)

        # Calculate Std Deviation stats
        avg_std = sum(stds) / len(stds)
        max_std = max(stds)
        min_std = min(stds)

        print(f"DRAM Point: {dp}")
        print("-" * 20)
        print(f"Mean (m)         -> Avg: {avg_mean:.4f} | Max: {max_mean:.4f} | Min: {min_mean:.4f}")
        print(f"Deviation (d)    -> Avg: {avg_std:.4f} | Max: {max_std:.4f} | Min: {min_std:.4f}")
        print(f"Area: {a:.4f}")
        print("="*40)