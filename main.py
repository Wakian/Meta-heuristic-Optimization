from experiments.run_continuous import run_all_continuous
from experiments.run_queens import run_queens_find_all
from experiments.run_tsp_ga import run_tsp_ga


if __name__ == "__main__":
    run_all_continuous()     # Runs all 6 continuous problems with HC, LRS, GRS
    #run_queens_find_all()     # Runs simulated annealing for 8 queens
    #run_tsp_ga()             # Runs GA for TSP
