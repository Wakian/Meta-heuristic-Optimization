import csv
import os
import numpy as np

def save_table(name, results):
    """
    Save all results into CSV: x1, x2, f_best
    results is a list of (x_best, f_best) tuples.
    """
    path = f"results/tables/{name}.csv"
    os.makedirs("results/tables", exist_ok=True)

    with open(path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["x1", "x2", "f_best"])

        for x_best, f_best in results:
            writer.writerow([x_best[0], x_best[1], f_best])

    print(f"[saved] {path}")


# ============================================================
# MODE COMPUTATION FOR CONTINUOUS SOLUTIONS
# ============================================================

def compute_mode(results, tol=1e-3):
    """
    Compute mode (most frequent solution) among continuous solutions.

    Parameters:
    -----------
    results : list of (x_best, f_best)
        Each x_best is a numpy array: [x1, x2]

    tol : float
        Tolerance for clustering continuous values.
        Points within tol distance are considered equal.

    Returns:
    --------
    mode_x : np.ndarray
        Coordinates of the mode solution

    mode_f : float
        Function value at the mode point

    count : int
        Number of appearances of the mode
    """

    if len(results) == 0:
        return None, None, 0

    # Extract only coordinates
    points = np.array([x for x, _ in results])

    # We cluster points using tolerance
    used = np.zeros(len(points), dtype=bool)
    clusters = []

    for i in range(len(points)):
        if used[i]:
            continue

        cluster = [i]
        used[i] = True

        for j in range(i + 1, len(points)):
            if np.linalg.norm(points[i] - points[j]) < tol:
                cluster.append(j)
                used[j] = True

        clusters.append(cluster)

    # Find largest cluster
    largest = max(clusters, key=len)
    count = len(largest)

    # Mode representative = mean of cluster
    mode_x = np.mean(points[largest], axis=0)

    # Compute mode f value as mean of f values of cluster
    mode_f = np.mean([results[idx][1] for idx in largest])

    return mode_x, mode_f, count


def save_mode(name, mode_x, mode_f, count):
    """
    Save mode information into CSV.

    name : problem + algorithm name
    mode_x : np.array([x1, x2])
    mode_f : float
    count  : int
    """

    path = f"results/tables/{name}_mode.csv"
    os.makedirs("results/tables", exist_ok=True)

    with open(path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["x1", "x2", "f_best", "count"])
        writer.writerow([mode_x[0], mode_x[1], mode_f, count])

    print(f"[saved mode] {path}")
