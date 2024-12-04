import numpy as np
import time


class AntColonyTSP:
    def __init__(self, distance_matrix, size_pop=50, max_iter=200, alpha=1, beta=2, rho=0.1):
        self.distance_matrix = distance_matrix
        self.size_pop = size_pop
        self.max_iter = max_iter
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.num_points = len(distance_matrix)

        self.prob_matrix_distance = 1 / (distance_matrix + 1e-10 * np.eye(self.num_points))
        self.pheromone_matrix = np.ones_like(distance_matrix)
        self.best_path = None
        self.best_cost = float('inf')
        self.cost_history = []

    def _calculate_path_cost(self, path):
        return sum(
            self.distance_matrix[path[i], path[(i + 1) % len(path)]]
            for i in range(len(path))
        )

    def solve_tsp(self):
        start_time = time.time()

        for _ in range(self.max_iter):
            paths = []
            costs = []
            for _ in range(self.size_pop):
                path = [np.random.choice(range(self.num_points))]  # Random start city
                for _ in range(1, self.num_points):
                    current_city = path[-1]
                    allow_list = list(set(range(self.num_points)) - set(path))
                    probabilities = np.zeros(self.num_points)
                    probabilities[allow_list] = (
                        (self.pheromone_matrix[current_city, allow_list] ** self.alpha)
                        * (self.prob_matrix_distance[current_city, allow_list] ** self.beta)
                    )
                    prob_sum = probabilities.sum()
                    if prob_sum == 0:  # Handle the zero-probability case
                        next_city = np.random.choice(allow_list)
                    else:
                        probabilities /= prob_sum
                        next_city = np.random.choice(range(self.num_points), p=probabilities)
                    path.append(next_city)

                paths.append(path)
                costs.append(self._calculate_path_cost(path))

            # Update best solution
            min_cost_idx = np.argmin(costs)
            if costs[min_cost_idx] < self.best_cost:
                self.best_cost = costs[min_cost_idx]
                self.best_path = paths[min_cost_idx]

            self.cost_history.append(self.best_cost)

            # Update pheromones
            delta_pheromone = np.zeros_like(self.pheromone_matrix)
            for path, cost in zip(paths, costs):
                epsilon = 1e-10  # Small value to prevent divide by zero
                for i in range(self.num_points):
                    n1, n2 = path[i], path[(i + 1) % self.num_points]
                    delta_pheromone[n1, n2] += 1 / (cost + epsilon)
            self.pheromone_matrix = (1 - self.rho) * self.pheromone_matrix + delta_pheromone

        end_time = time.time()
        print(f"Ant Colony Optimization completed in {end_time - start_time:.2f} seconds")
        return self.best_path, self.best_cost, self.cost_history
