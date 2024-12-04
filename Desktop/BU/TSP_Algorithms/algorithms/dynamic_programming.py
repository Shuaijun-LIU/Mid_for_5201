import numpy as np


def solve_tsp_dynamic_programming(distance_matrix):
    """
    Solves the TSP using Dynamic Programming (Bellman-Held-Karp algorithm).

    Parameters:
        distance_matrix (ndarray): A 2D array representing distances between cities.

    Returns:
        best_path (list): The ordered list of city indices representing the optimal path.
        best_cost (float): The total cost of the optimal path.
    """
    n = len(distance_matrix)
    all_sets = 1 << n  # 2^n, all subsets of cities
    dp = np.full((all_sets, n), np.inf)  # dp[subset][end_city]
    dp[1][0] = 0  # Start at city 0 with only city 0 in the subset

    for subset in range(all_sets):
        for end in range(n):
            if not (subset & (1 << end)):  # Skip if end is not in subset
                continue

            prev_subset = subset & ~(1 << end)  # Subset without 'end'
            for prev in range(n):
                if not (prev_subset & (1 << prev)):  # Skip if prev is not in prev_subset
                    continue

                dp[subset][end] = min(
                    dp[subset][end],
                    dp[prev_subset][prev] + distance_matrix[prev][end]
                )

    # Find the minimum cost to return to the start city
    full_set = (1 << n) - 1
    best_cost = np.inf
    last_city = -1
    for end in range(1, n):
        cost = dp[full_set][end] + distance_matrix[end][0]
        if cost < best_cost:
            best_cost = cost
            last_city = end

    # Reconstruct the path
    path = [0]
    subset = full_set
    while last_city != 0:
        path.append(last_city)
        next_city = -1
        for prev in range(n):
            if (subset & (1 << prev)) and dp[subset][last_city] == dp[subset & ~(1 << last_city)][prev] + distance_matrix[prev][last_city]:
                next_city = prev
                break
        subset &= ~(1 << last_city)
        last_city = next_city

    path.append(0)  # Return to the start
    path.reverse()

    return path, best_cost
