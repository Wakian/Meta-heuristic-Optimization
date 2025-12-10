import csv
import os

def save_table(name, results):
    """
    Save results into CSV format.
    results: list of (x_best, f_best)
    """
    path = f"results/tables/{name}.csv"
    os.makedirs("results/tables", exist_ok=True)

    with open(path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["x1", "x2", "f_best"])

        for (x_best, f_best) in results:
            writer.writerow([x_best[0], x_best[1], f_best])

    print(f"[saved] {path}")
