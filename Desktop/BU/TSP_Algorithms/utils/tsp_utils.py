import numpy as np
from scipy.spatial.distance import cdist
import json


def load_tsp_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data['coordinates']


def generate_distance_matrix(coordinates):
    coordinates = np.array(coordinates)
    return cdist(coordinates, coordinates, metric='euclidean')
