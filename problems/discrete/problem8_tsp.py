import numpy as np
import pandas as pd
import os

def load_points(csv_path="CaixeiroGrupos.csv", N=40):
    """
    Loads the CSV file and returns N random points in 3D.
    Professor says to choose 30 < N < 60.
    """
    df = pd.read_csv(csv_path, header=None)
    df.columns = ["x", "y", "z", "group"]
    df = df.sample(N)  # random subset
    return df[['x','y','z']].to_numpy()


def route_length(route, points):
    """
    Compute total path length of a route (including returning to origin).
    Route is a permutation of indices.
    """

    dist = 0.0
    for i in range(len(route) - 1):
        a = points[route[i]]
        b = points[route[i+1]]
        dist += np.linalg.norm(a - b)

    # Return to the starting point
    a = points[route[-1]]
    b = points[route[0]]
    dist += np.linalg.norm(a - b)

    return dist
