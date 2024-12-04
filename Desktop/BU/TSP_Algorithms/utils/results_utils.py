import os
import json
import numpy as np


def convert_to_serializable(obj):
    """
    Recursively converts NumPy data types to Python native types.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    return obj


def save_results(results, filename):
    """
    Save results as a JSON file in the results/ directory.
    """
    os.makedirs("results", exist_ok=True)
    filepath = os.path.join("results", filename)
    serializable_results = convert_to_serializable(results)
    with open(filepath, "w") as file:
        json.dump(serializable_results, file, indent=4)
    print(f"Results saved to {filepath}")
