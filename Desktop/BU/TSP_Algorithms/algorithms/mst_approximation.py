import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree
from collections import defaultdict


def mst_tsp_approximation(distance_matrix):
    """
    Solves the TSP approximately using the Minimum Spanning Tree (MST) heuristic.

    Parameters:
        distance_matrix (ndarray): A 2D array representing distances between cities.

    Returns:
        approx_path (list): The ordered list of city indices representing the approximate path.
        approx_cost (float): The total cost of the approximate path.
    """
    num_cities = len(distance_matrix)

    # Compute the MST of the distance matrix
    mst = minimum_spanning_tree(distance_matrix).toarray()

    # Convert MST to adjacency list for DFS traversal
    mst_adj_list = defaultdict(list)
    for i in range(num_cities):
        for j in range(num_cities):
            if mst[i, j] > 0:
                mst_adj_list[i].append(j)
                mst_adj_list[j].append(i)

    # Perform DFS to get the preorder traversal of MST
    visited = set()
    approx_path = []

    def dfs(city):
        visited.add(city)
        approx_path.append(city)
        for neighbor in mst_adj_list[city]:
            if neighbor not in visited:
                dfs(neighbor)

    dfs(0)  # Start DFS from city 0

    # Compute the total cost of the approximate TSP path
    approx_cost = 0
    for i in range(len(approx_path) - 1):
        approx_cost += distance_matrix[approx_path[i], approx_path[i + 1]]
    approx_cost += distance_matrix[approx_path[-1], approx_path[0]]  # Return to the start

    approx_path.append(approx_path[0])  # Complete the cycle

    return approx_path, approx_cost
