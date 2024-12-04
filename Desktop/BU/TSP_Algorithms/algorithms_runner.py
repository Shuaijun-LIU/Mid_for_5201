import time
from algorithms.dqn import DQNTSP
from algorithms.pso import PSO_TSP
from algorithms.adv_pso import ADV_PSO_TSP
from algorithms.ant_colony import AntColonyTSP
from algorithms.greedy import solve_tsp_greedy
from algorithms.brute_force import solve_tsp_brute_force
from algorithms.nearest_neighbor import solve_tsp_nearest_neighbor
from algorithms.mst_approximation import mst_tsp_approximation
from algorithms.pointer_networks import solve_tsp_pointer_network
from algorithms.dynamic_programming import solve_tsp_dynamic_programming
from algorithms.simulated_annealing import solve_tsp_simulated_annealing
from utils.plot_utils import save_plot, plot_path, plot_convergence


def run_all_algorithms(coordinates, distance_matrix):
    """
    Run all TSP algorithms and return the results.
    """
    results = {}

    algorithms = [
        ("Ant Colony Optimization", AntColonyTSP(distance_matrix, size_pop=50, max_iter=200).solve_tsp),
        ("Brute Force", lambda: solve_tsp_brute_force(distance_matrix)),
        ("Deep Q-Learning", lambda: DQNTSP(distance_matrix, episodes=300, max_steps=100).solve_tsp()),
        ("Nearest Neighbor", lambda: solve_tsp_nearest_neighbor(distance_matrix)),
        ("MST Approximation", lambda: mst_tsp_approximation(distance_matrix)),
        ("Pointer Networks", lambda: solve_tsp_pointer_network(coordinates, num_epochs=500, batch_size=32, learning_rate=0.001, hidden_dim=128)),
        ("Dynamic Programming", lambda: solve_tsp_dynamic_programming(distance_matrix)),
        ("Simulated Annealing", lambda: solve_tsp_simulated_annealing(distance_matrix, initial_temperature=1000, cooling_rate=0.995, stopping_temperature=1e-3, max_iterations=1000)),
        ("Greedy Algorithm", lambda: solve_tsp_greedy(distance_matrix)),
        ("Particle Swarm Optimization", lambda: run_pso(distance_matrix)),
    ]

    for name, algorithm_func in algorithms:
        run_algorithm(name, algorithm_func, coordinates, distance_matrix, results)

    return results


def run_algorithm(name, algorithm_func, coordinates, distance_matrix, results):
    """
    Run a single algorithm and save its results.
    """
    print(f"\nSolving TSP with {name}...")
    start_time = time.time()
    result = algorithm_func()
    end_time = time.time()

    best_path, best_cost = result[:2]
    results[name] = {
        "Best Path": best_path,
        "Best Cost": best_cost,
        "Execution Time": end_time - start_time,
    }

    print(f"\n{name} Results:")
    print(f"Best Path: {best_path}")
    print(f"Best Cost: {best_cost}")
    print(f"Execution Time: {end_time - start_time:.2f} seconds")

    save_plot(plot_path, coordinates, best_path, filename=f"{name.replace(' ', '_').lower()}_path.png", title=f"{name} Path")
    if len(result) > 2:  # If cost history exists
        save_plot(plot_convergence, result[2], filename=f"{name.replace(' ', '_').lower()}_convergence.png", title=f"{name} Convergence")


def run_pso(distance_matrix):
    """
    Wrapper function for running PSO.
    """
    pso_params = {
        "size_pop": 50,
        "max_iter": 500,
        "w": 0.8,
        "c1": 0.5,
        "c2": 0.5,
    }
    tsp_func = lambda path: sum(distance_matrix[path[i], path[(i + 1) % len(path)]] for i in range(len(path)))
    pso_solver = PSO_TSP(func=tsp_func, n_dim=len(distance_matrix), **pso_params)
    return pso_solver.solve_tsp()
