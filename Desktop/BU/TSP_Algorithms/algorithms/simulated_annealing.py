import numpy as np
import random


def calculate_path_cost(path, distance_matrix):
    """
    Calculate the total cost of a given path.

    Parameters:
        path (list): A list representing the order of cities visited.
        distance_matrix (ndarray): A 2D array representing distances between cities.

    Returns:
        float: The total cost of the path.
    """
    return sum(distance_matrix[path[i], path[(i + 1) % len(path)]] for i in range(len(path)))


def solve_tsp_simulated_annealing(distance_matrix, initial_temperature=1000, cooling_rate=0.995, stopping_temperature=1e-3, max_iterations=1000):
    """
    Solves the TSP using Simulated Annealing.

    Parameters:
        distance_matrix (ndarray): A 2D array representing distances between cities.
        initial_temperature (float): The starting temperature for the algorithm.
        cooling_rate (float): The rate at which the temperature decreases.
        stopping_temperature (float): The temperature at which the algorithm stops.
        max_iterations (int): The maximum number of iterations allowed.

    Returns:
        best_path (list): The best path found.
        best_cost (float): The cost of the best path.
    """
    num_cities = len(distance_matrix)
    current_path = list(range(num_cities))
    random.shuffle(current_path)  # Start with a random path
    current_cost = calculate_path_cost(current_path, distance_matrix)
    best_path = current_path[:]
    best_cost = current_cost

    temperature = initial_temperature
    iteration = 0

    while temperature > stopping_temperature and iteration < max_iterations:
        # Create a new path by swapping two random cities
        new_path = current_path[:]
        city1, city2 = random.sample(range(num_cities), 2)
        new_path[city1], new_path[city2] = new_path[city2], new_path[city1]
        new_cost = calculate_path_cost(new_path, distance_matrix)

        # Accept the new path with a certain probability
        if new_cost < current_cost or random.random() < np.exp((current_cost - new_cost) / temperature):
            current_path, current_cost = new_path, new_cost

            # Update the best path if the new path is better
            if new_cost < best_cost:
                best_path, best_cost = new_path[:], new_cost

        # Cool down the temperature
        temperature *= cooling_rate
        iteration += 1

    return best_path + [best_path[0]], best_cost
