
from utils.results_utils import save_results
from utils.tsp_utils import load_tsp_data, generate_distance_matrix
from algorithms_runner import run_all_algorithms


def main():
    # Load data
    coordinates = load_tsp_data("data/tsp_example_1.json")
    distance_matrix = generate_distance_matrix(coordinates)

    # Run all algorithms
    results = run_all_algorithms(coordinates, distance_matrix)

    # Save all results to JSON
    save_results(results, filename="tsp_results.json")


if __name__ == "__main__":
    main()
