from functools import lru_cache
#############Tensor Parallelism/Expert Parallelism config Generator###############
@lru_cache(None)
def generate_partitions_unrestricted(n, k):
    if k == 0:
        return [[]] if n == 0 else []
    if k > n or n <= 0:
        return []
    if k == n:
        return [[1] * k]
    if k == 1:
        return [[n]]
    results = []
    for p in generate_partitions_unrestricted(n - 1, k - 1):
        results.append([1] + list(p))
    for p in generate_partitions_unrestricted(n - k, k):
        results.append([x + 1 for x in p])
    return results
@lru_cache(None)
def generate_partitions_even_or_one(n, k):
    if k == 0:
        return [[]] if n == 0 else []
    if k > n or n <= 0:
        return []
    if k == 1:
        if n == 1 or n % 2 == 0:
            return [[n]]
        return []
    if k == n:
        return [[1] * n]
    results = []
    for p in generate_partitions_even_or_one(n - 1, k - 1):
        results.append(sorted([1] + list(p), reverse=True))
    if n % 2 == 0:
        for p in generate_partitions_unrestricted(n // 2, k):
            results.append(sorted([x * 2 for x in p], reverse=True))
    unique_results = set(tuple(x) for x in results)
    return [list(x) for x in unique_results]
def get_all_scenarios_even_or_one(total_pes):
    all_scenarios = []
    for e in range(1, total_pes + 1):
        scenarios = generate_partitions_even_or_one(total_pes, e)
        if scenarios:
            scenarios.sort(key=lambda x: (len(x), x), reverse=True)
            all_scenarios.extend(scenarios)
    return all_scenarios
############################################