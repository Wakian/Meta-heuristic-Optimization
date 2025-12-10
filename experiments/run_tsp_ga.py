import numpy as np
import time
import os
from problems.discrete.problem8_tsp import load_points, route_length
from algorithms.genetic_algorithm import GeneticTSP
from utils.helpers import compute_mode, save_mode
from utils.plotting import plot_3d_route

def run_tsp_ga():
    print("\n===== Running TSP with Genetic Algorithm =====")

    # 1) Choose number of points (N between 30 and 60)
    N_POINTS = 40
    points = load_points(N=N_POINTS)

    POP_SIZE = 120
    MAX_GEN = 300
    RUNS = 100

    ACCEPTABLE = None  # optional stopping criterion

    generations_needed = []

    best_overall = None
    best_overall_cost = float("inf")
    best_run_index = None

    for run_idx in range(RUNS):

        ga = GeneticTSP(points,
                        pop_size=POP_SIZE,
                        max_gen=MAX_GEN,
                        tournament_k=3,
                        mutation_prob=0.01,
                        elitism=2)

        best_ind, gen = ga.run(acceptable_cost=ACCEPTABLE)
        generations_needed.append(gen)

        # compute cost of this run's best_ind (guard if None)
        if best_ind is not None:
            cost = route_length(best_ind, points)
            # update global best
            if cost < best_overall_cost:
                best_overall_cost = cost
                best_overall = best_ind.copy()
                best_run_index = run_idx

    # SAVE mode of generations
    gens_arr = np.array(generations_needed)
    values = [(np.array([g]), 0) for g in gens_arr]  # hack for mode fn
    mode_g, _, count = compute_mode(values)

    print("\nMode of generations:", int(mode_g[0]))
    print("Frequency:", count)

    save_mode("tsp_ga_generations", mode_g, 0, count)

    # ------------------------
    # Save best route to CSV
    # ------------------------
    if best_overall is not None:
        os.makedirs("results/tables", exist_ok=True)
        csv_path = "results/tables/tsp_best_route.csv"
        with open(csv_path, "w", newline="") as f:
            # columns: order_index, city_index, x, y, z
            import csv
            writer = csv.writer(f)
            writer.writerow(["order", "city_index", "x", "y", "z", "route_length"])
            for order, city_idx in enumerate(best_overall):
                x, y, z = points[city_idx]
                writer.writerow([order, int(city_idx), float(x), float(y), float(z), ""])
            # add route length in a final line
            writer.writerow(["", "", "", "", "", float(best_overall_cost)])
        print(f"[saved] Best route to {csv_path}")
    else:
        print("[warn] No best route found in any run.")

    # ------------------------
    # Plot best route
    # ------------------------
    if best_overall is not None:
        plot_3d_route(points, best_overall, title=f"TSP best route (cost={best_overall_cost:.4f})", filename="tsp_route.png")
    else:
        print("[warn] skipping plotting because no route available")
