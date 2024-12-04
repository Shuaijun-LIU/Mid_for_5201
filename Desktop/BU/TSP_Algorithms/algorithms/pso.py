import numpy as np


class PSO_TSP:
    def __init__(self, func, n_dim, size_pop=50, max_iter=200, w=0.8, c1=0.5, c2=0.5):
        """
        PSO for solving TSP.

        Parameters:
            func (function): Objective function to minimize.
            n_dim (int): Number of cities.
            size_pop (int): Number of particles.
            max_iter (int): Maximum number of iterations.
            w (float): Inertia weight.
            c1 (float): Cognitive learning rate.
            c2 (float): Social learning rate.
        """
        self.func = func
        self.n_dim = n_dim
        self.size_pop = size_pop
        self.max_iter = max_iter
        self.w = w
        self.c1 = c1
        self.c2 = c2

        # Initialize particles
        self.X = self._initialize_population()
        self.V = np.zeros_like(self.X)
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

    def _update_velocity(self):
        """Update the velocity of particles."""
        r1 = np.random.rand(self.size_pop, self.n_dim)
        r2 = np.random.rand(self.size_pop, self.n_dim)
        cognitive = self.c1 * r1 * (self.pbest_x - self.X)
        social = self.c2 * r2 * (self.gbest_x - self.X)
        self.V = self.w * self.V + cognitive + social

    def _update_position(self):
        """Update the position of particles."""
        for i in range(self.size_pop):
            # Perform a random swap to update the position
            if np.random.rand() < 0.5:
                idx1, idx2 = np.random.choice(self.n_dim, 2, replace=False)
                self.X[i, [idx1, idx2]] = self.X[i, [idx2, idx1]]

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
        for _ in range(self.max_iter):
            self._update_velocity()
            self._update_position()
            self._evaluate_population()
            self.gbest_y_hist.append(self.gbest_y)

        return list(self.gbest_x) + [self.gbest_x[0]], self.gbest_y
