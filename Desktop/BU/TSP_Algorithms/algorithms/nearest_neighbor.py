import numpy as np


def solve_tsp_nearest_neighbor(distance_matrix, start_city=0):
    """
    Solves the TSP using the Nearest Neighbor heuristic.

    Parameters:
        distance_matrix (ndarray): A 2D array representing distances between cities.
        start_city (int): The starting city index.

    Returns:
        best_path (list): The ordered list of city indices representing the path.
        best_cost (float): The total cost of the path.
    """
    num_cities = len(distance_matrix)
    visited = set()
    path = [start_city]
    current_city = start_city
    visited.add(current_city)
    total_cost = 0

    for _ in range(num_cities - 1):
        # Find the nearest unvisited city
        distances = distance_matrix[current_city]
        nearest_city = np.argmin([distances[i] if i not in visited else np.inf for i in range(num_cities)])
        total_cost += distances[nearest_city]
        visited.add(nearest_city)
        path.append(nearest_city)
        current_city = nearest_city

    # Return to the starting city to complete the loop
    total_cost += distance_matrix[current_city, start_city]
    path.append(start_city)

    return path, total_cost
