import os
import matplotlib.pyplot as plt
import numpy as np


def plot_path(coordinates, path, title="TSP Path"):
    """
    Plot the path of a TSP solution.

    Parameters:
        coordinates (list of tuples): List of (x, y) coordinates for the cities.
        path (list of int): The order of cities to visit.
        title (str): Title of the plot.
    """
    path_coordinates = [coordinates[i] for i in path] + [coordinates[path[0]]]
    x, y = zip(*path_coordinates)

    plt.figure(figsize=(10, 6))
    plt.plot(x, y, '-o', label="Path")
    plt.title(title)
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.legend()
    plt.grid(True)


def plot_convergence(history, title="Convergence Curve"):
    """
    Plot the convergence of an algorithm's cost history.

    Parameters:
        history (list of float): The cost history over iterations.
        title (str): Title of the plot.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(history, '-o', label="Cost")
    plt.title(title)
    plt.xlabel("Iterations")
    plt.ylabel("Cost")
    plt.legend()
    plt.grid(True)


def save_plot(plot_function, *args, filename, **kwargs):
    """
    Save a plot to a file in the 'results' directory.

    Parameters:
        plot_function (function): The plotting function to call.
        *args: Positional arguments to pass to the plotting function.
        filename (str): The filename for the saved plot.
        **kwargs: Keyword arguments to pass to the plotting function.
    """
    os.makedirs("results", exist_ok=True)
    filepath = os.path.join("results", filename)

    plot_function(*args, **kwargs)
    plt.savefig(filepath)
    plt.close()
    print(f"Plot saved to {filepath}")
