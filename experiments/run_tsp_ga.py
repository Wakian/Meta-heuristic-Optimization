import numpy as np
import time
from problems.discrete.problem8_tsp import load_points, route_length
from algorithms.genetic_algorithm import GeneticTSP
from utils.helpers import compute_mode, save_mode

def run_tsp_ga():
    print("\n===== Running TSP with Genetic Algorithm =====")

    # 1) Choose number of points (N between 30 and 60)
    N_POINTS = 40
    points = load_points(N=N_POINTS)

    POP_SIZE = 120
    MAX_GEN = 300
    RUNS = 100

    # Acceptable cost (slides) = 10% above best known
    # For didactic purposes you can use:
    ACCEPTABLE = None  # keep None unless you compute best length before

    generations_needed = []

    for _ in range(RUNS):

        ga = GeneticTSP(points,
                        pop_size=POP_SIZE,
                        max_gen=MAX_GEN,
                        tournament_k=3,
                        mutation_prob=0.01,
                        elitism=2)

        best, gen = ga.run(acceptable_cost=ACCEPTABLE)
        generations_needed.append(gen)

    # SAVE mode of generations
    gens_arr = np.array(generations_needed)
    values = [(np.array([g]), 0) for g in gens_arr]  # hack for mode fn
    mode_g, _, count = compute_mode(values)

    print("\nMode of generations:", int(mode_g[0]))
    print("Frequency:", count)

    save_mode("tsp_ga_generations", mode_g, 0, count)
