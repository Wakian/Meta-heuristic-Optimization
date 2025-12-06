import numpy as np

def load_points(csv_path):
    return np.loadtxt(csv_path, delimiter=",")
