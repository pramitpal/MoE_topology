from network import *
from traffic_generator import *
from TP_config import *
import pprint, csv


if __name__ == "__main__":

    PE_count = 32
    A_dim = [64, 256]
    B_dim = [256, 64]
    PE_dim = [64, 64]
    DRAM_point = 'L1'
    partitions = 100

    # AMOSA parameters
    T_init = T_curr = 100.0
    T_final = 20
    alpha = 0.96
    Iter_per_temp = 1
    archive_size = 4

    # Generate initial solution
    net = NetworkTopology(PE_count=PE_count)
    net, level_router_counts, groups = generate_random_solution(net)
    current_obj = get_objectives(net, PE_count, DRAM_point, A_dim, B_dim, PE_dim, partitions)
    current_net, current_counts, current_groups = net, level_router_counts, groups
    # Initialize archive
    archive = [(current_net, current_counts, current_groups, current_obj)]
    hv_history = []  # ADD THIS
    print(f"Initial solution: counts={current_counts}, obj={current_obj}")
    best_hv = 0.0  # ADD THIS before the loop
    best_archive = archive.copy()  # ADD THIS
    # AMOSA Main Loop
    while T_curr > T_final:
        for _ in range(Iter_per_temp):
            result = perturb(current_net, current_counts, current_groups)
            if result is None:
                continue
            candidate_net, candidate_counts, candidate_groups = result
            try:
                candidate_obj = get_objectives(candidate_net, PE_count, DRAM_point,
                                               A_dim, B_dim, PE_dim, partitions)
            except Exception as e:
                print(f"  Evaluation failed: {e}")
                continue
            if amosa_acceptance(current_obj, candidate_obj, archive, T_curr):
                current_net = candidate_net
                current_counts = candidate_counts
                current_groups = candidate_groups
                current_obj = candidate_obj
            archive = update_archive(archive, candidate_net, candidate_counts,
                                     candidate_groups, candidate_obj, archive_size)

        # ADD THESE 3 LINES
        hv = compute_total_hv(archive)
        # If HV degraded, revert to best archive
        if hv < best_hv:
            archive = best_archive.copy()
            hv = best_hv
            print(f"  HV degraded, reverting to best archive (HV={best_hv:.4f})")
        else:
            best_hv = hv
            best_archive = archive.copy()

        hv_history.append(hv)

        T_curr *= alpha
        print(f"T={T_curr:.2f} | Archive={len(archive)} | HV={hv:.4f} ")
        # print(f"| Best objs={[f'({float(o[0]):.2f}, {float(o[1]):.2f}, {float(o[2]):.2f})' for _,_,_,o in archive]}")

    # Plot HV at the end

    plt.figure(figsize=(8, 4))
    plt.plot(hv_history, color='blue', linewidth=1.5)
    plt.xlabel('Iterations')
    plt.ylabel('Hypervolume')
    plt.title('Hypervolume Convergence')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('hv_convergence.png', dpi=150)

    # Results
    print("\n=== PARETO FRONT ===")
    plot_pareto_front_interactive(archive)
    for i, (sol_net, sol_counts, sol_groups, sol_obj) in enumerate(archive):
        print(f"Solution {i+1}: counts={sol_counts}, groups={sol_groups}, obj={sol_obj}")
        sol_net.visualize(title=f"solution_{i+1}.png",figsize=(12, 5))