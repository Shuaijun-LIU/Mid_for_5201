import itertools
import numpy as np
import time


def solve_tsp_brute_force(distance_matrix):

    start_time = time.time()

    num_points = len(distance_matrix)
    all_permutations = itertools.permutations(range(num_points))

    best_cost = float('inf')
    best_path = None

    for perm in all_permutations:
        cost = sum(
            distance_matrix[perm[i], perm[(i + 1) % num_points]]
            for i in range(num_points)
        )
        if cost < best_cost:
            best_cost = cost
            best_path = perm

    end_time = time.time()
    print(f"Brute Force completed in {end_time - start_time:.2f} seconds")
    return list(best_path), best_cost
