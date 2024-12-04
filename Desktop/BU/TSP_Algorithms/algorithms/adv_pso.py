import numpy as np
import random


class ADV_PSO_TSP:
    def __init__(self, func, n_dim, size_pop=100, max_iter=500, w_max=0.9, w_min=0.4, c1=1.8, c2=1.8):
        self.func = func
        self.n_dim = n_dim
        self.size_pop = size_pop
        self.max_iter = max_iter
        self.w_max = w_max
        self.w_min = w_min
        self.c1 = c1
        self.c2 = c2

        self.X = self._initialize_population()
        self.V = np.zeros_like(self.X, dtype=float)
        self.pbest_x = self.X.copy()
        self.pbest_y = np.array([self.func(ind) for ind in self.X])
        self.gbest_x = self.pbest_x[np.argmin(self.pbest_y)]
        self.gbest_y = min(self.pbest_y)
        self.gbest_y_hist = []

    def _initialize_population(self):
        """Randomly initialize the population of particles."""
        population = []
        for _ in range(self.size_pop):
            individual = np.random.permutation(self.n_dim)
            population.append(individual)
        return np.array(population)

    def _adaptive_inertia_weight(self, iteration):
        """Calculate adaptive inertia weight."""
        return self.w_max - (self.w_max - self.w_min) * (iteration / self.max_iter)

    def _fix_path(self, path):
        """Ensure the path includes all cities exactly once."""
        visited = set()
        fixed_path = []
        for city in path:
            if city not in visited:
                fixed_path.append(city)
                visited.add(city)
        missing_cities = set(range(self.n_dim)) - visited
        fixed_path.extend(missing_cities)
        return np.array(fixed_path)

    def _apply_velocity(self):
        """Apply velocity to update the particle's position."""
        for i in range(self.size_pop):
            swap_probability = 1 / (1 + np.exp(-self.V[i]))  # Sigmoid for swap probability
            for j in range(self.n_dim):
                if random.random() < swap_probability[j]:
                    idx = random.randint(0, self.n_dim - 1)
                    self.X[i][j], self.X[i][idx] = self.X[i][idx], self.X[i][j]
            self.X[i] = self._fix_path(self.X[i])  # Ensure path validity

    def _apply_path_perturbation(self):
        """Apply random perturbations with mixed strategies."""
        for i in range(self.size_pop):
            if random.random() < 0.3:  # 30% chance to perturb
                strategy = random.choice(["reverse", "swap", "PMX"])
                idx1, idx2 = np.random.choice(self.n_dim, 2, replace=False)
                if strategy == "reverse":
                    self.X[i][idx1:idx2] = self.X[i][idx1:idx2][::-1]  # Reverse segment
                elif strategy == "swap":
                    self.X[i][idx1], self.X[i][idx2] = self.X[i][idx2], self.X[i][idx1]  # Swap two cities
                elif strategy == "PMX":
                    self._partial_mapping_crossover(i, idx1, idx2)
            self.X[i] = self._fix_path(self.X[i])  # Ensure path validity

    def _partial_mapping_crossover(self, particle_idx, idx1, idx2):
        """Perform PMX crossover on a segment of the path."""
        parent = self.X[particle_idx]
        mapping = {parent[idx1]: parent[idx2], parent[idx2]: parent[idx1]}
        for k in range(idx1, idx2):
            if parent[k] in mapping:
                parent[k] = mapping[parent[k]]

    def _evaluate_population(self):
        """Evaluate the fitness of the population."""
        fitness = np.array([self.func(ind) for ind in self.X])
        for i in range(self.size_pop):
            if fitness[i] < self.pbest_y[i]:
                self.pbest_y[i] = fitness[i]
                self.pbest_x[i] = self.X[i].copy()

        best_idx = np.argmin(self.pbest_y)
        if self.pbest_y[best_idx] < self.gbest_y:
            self.gbest_y = self.pbest_y[best_idx]
            self.gbest_x = self.pbest_x[best_idx]

    def solve_tsp(self):
        """Run the PSO algorithm."""
        for iteration in range(self.max_iter):
            w = self._adaptive_inertia_weight(iteration)
            for i in range(self.size_pop):
                r1, r2 = np.random.rand(self.n_dim), np.random.rand(self.n_dim)
                cognitive = self.c1 * r1 * (self.pbest_x[i] - self.X[i])
                social = self.c2 * r2 * (self.gbest_x - self.X[i])
                self.V[i] = w * self.V[i] + cognitive + social

            self._apply_velocity()
            self._apply_path_perturbation()
            self._evaluate_population()
            self.gbest_y_hist.append(self.gbest_y)

        return list(self.gbest_x) + [self.gbest_x[0]], self.gbest_y
