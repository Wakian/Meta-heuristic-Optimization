import numpy as np
import os
from algorithms.hill_climbing import HillClimbing
from algorithms.local_random_search import LocalRandomSearch
from algorithms.global_random_search import GlobalRandomSearch

from utils.plotting import plot_3d_surface
from utils.helpers import save_table

# Import continuous problems
from problems.continuous.problem1 import f as f1, domain as d1
from problems.continuous.problem2 import f as f2, domain as d2
from problems.continuous.problem3 import f as f3, domain as d3
from problems.continuous.problem4 import f as f4, domain as d4
from problems.continuous.problem5 import f as f5, domain as d5
from problems.continuous.problem6 import f as f6, domain as d6

from utils.helpers import compute_mode, save_mode


def run_all_continuous():

    problems = [
        ("problem1", f1, d1, "min"),
        ("problem2", f2, d2, "max"),
        ("problem3", f3, d3, "min"),
        ("problem4", f4, d4, "min"),
        ("problem5", f5, d5, "max"),
        ("problem6", f6, d6, "max"),
    ]

    N_RUNS = 100
    MAX_IT = 1000
    PATIENCE = 50

    eps = 0.1    # HC
    sigma = 0.4  # LRS + GRS

    os.makedirs("results/tables", exist_ok=True)
    os.makedirs("results/plots",  exist_ok=True)

    print("\n===== Running Continuous Optimization Experiments =====\n")

    for name, f, domain, mode in problems:
        print(f"\n--- Solving {name} ({mode}imization) ---")

        # Wrap function for max problems
        if mode == "max":
            f_wrapped = lambda x1, x2: -f(x1, x2)
        else:
            f_wrapped = f

        results_hc = []
        results_lrs = []
        results_grs = []

        # ------------------- HILL CLIMBING -------------------
        print("Running Hill Climbing...")
        for _ in range(N_RUNS):
            hc = HillClimbing(f_wrapped, domain, eps=eps, max_it=MAX_IT, patience=PATIENCE)
            x_best, f_best = hc.search()
            if mode == "max":
                f_best = -f_best
            results_hc.append((x_best, f_best))
        mode_x, mode_f, count = compute_mode(results_hc)
        save_mode(f"{name}_hc", mode_x, mode_f, count)

        # ------------------- LOCAL RANDOM SEARCH -------------------
        print("Running Local Random Search...")
        for _ in range(N_RUNS):
            lrs = LocalRandomSearch(f_wrapped, domain, sigma=sigma, max_it=MAX_IT)
            x_best, f_best = lrs.search()
            if mode == "max":
                f_best = -f_best
            results_lrs.append((x_best, f_best))
        mode_x, mode_f, count = compute_mode(results_lrs)
        save_mode(f"{name}_lrs", mode_x, mode_f, count)

        # ------------------- GLOBAL RANDOM SEARCH -------------------
        print("Running Global Random Search...")
        for _ in range(N_RUNS):
            grs = GlobalRandomSearch(f_wrapped, domain, sigma=sigma, max_it=MAX_IT)
            x_best, f_best = grs.search()
            if mode == "max":
                f_best = -f_best
            results_grs.append((x_best, f_best))
        mode_x, mode_f, count = compute_mode(results_grs)
        save_mode(f"{name}_grs", mode_x, mode_f, count)
        
        # ------------------- SAVE TO CSV -------------------
        save_table(f"{name}_hc",  results_hc)
        save_table(f"{name}_lrs", results_lrs)
        save_table(f"{name}_grs", results_grs)

        # ------------------- PLOTS -------------------
        hc_pts  = [[x[0], x[1], f_best] for x, f_best in results_hc]
        lrs_pts = [[x[0], x[1], f_best] for x, f_best in results_lrs]
        grs_pts = [[x[0], x[1], f_best] for x, f_best in results_grs]

        plot_3d_surface(f=f, domain=domain, title=f"{name} - Hill Climbing", best_points=hc_pts, filename=f"{name}_hc.png")
        plot_3d_surface(f=f, domain=domain, title=f"{name} - Local Random Search", best_points=lrs_pts, filename=f"{name}_lrs.png")
        plot_3d_surface(f=f, domain=domain, title=f"{name} - Global Random Search", best_points=grs_pts, filename=f"{name}_grs.png")


        print(f"Completed {name}.\n")
