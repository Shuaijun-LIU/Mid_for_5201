import json
import numpy as np

def generate_complex_tsp_example(num_cities, filename):
    """
    Generates a complex TSP example with num_cities cities.
    Saves the result as a JSON file.
    """
    coordinates = np.random.rand(num_cities, 2).tolist()
    tsp_data = {"coordinates": coordinates}

    with open(filename, "w") as file:
        json.dump(tsp_data, file)

    print(f"Complex TSP example with {num_cities} cities saved to {filename}")

# Example: Generate a TSP example with 50 cities
generate_complex_tsp_example(50, "../data/tsp_example_50.json")
