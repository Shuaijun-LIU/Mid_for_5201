import numpy as np

def solve_tsp_greedy(distance_matrix):
    """
    Solves the TSP using a Greedy algorithm.

    Parameters:
        distance_matrix (ndarray): A 2D array representing distances between cities.

    Returns:
        best_path (list): The ordered list of city indices representing the greedy path.
        best_cost (float): The total cost of the greedy path.
    """
    num_cities = len(distance_matrix)
    visited = set()
    current_city = 0
    path = [current_city]
    visited.add(current_city)
    total_cost = 0

    while len(visited) < num_cities:
        # Find the nearest unvisited city
        nearest_city = np.argmin([
            distance_matrix[current_city, i] if i not in visited else np.inf
            for i in range(num_cities)
        ])
        total_cost += distance_matrix[current_city, nearest_city]
        visited.add(nearest_city)
        path.append(nearest_city)
        current_city = nearest_city

    # Return to the starting city
    total_cost += distance_matrix[current_city, path[0]]
    path.append(path[0])

    return path, total_cost
