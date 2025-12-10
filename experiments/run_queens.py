import time
from problems.discrete.problem7 import f, random_solution, neighbor
from algorithms.simulated_annealing import SimulatedAnnealingDiscrete

# ---------------------------------------------------------

def run_queens_sa():
    print("\n===== Running 8-Queens with Simulated Annealing =====")

    sa = SimulatedAnnealingDiscrete(
        f=f,
        neighbor_fn=neighbor,
        T0=10.0,
        alpha=0.99,
        max_it=5000
    )

    x0 = random_solution()
    best_x, best_f = sa.search(x0)

    print("Best solution found:", best_x)
    print("Fitness:", best_f)

    if best_f == 28:
        print("SUCCESS: Valid 8-queen solution found!")
    else:
        print("Did NOT reach the optimal solution.")

    return best_x, best_f


# ---------------------------------------------------------
# OPTIONAL: search for all 92 solutions
# ---------------------------------------------------------

def run_queens_find_all():
    print("\n===== Searching for ALL 92 8-Queen Solutions =====")

    sa = SimulatedAnnealingDiscrete(
        f=f,
        neighbor_fn=neighbor,
        T0=10.0,
        alpha=0.99,
        max_it=5000
    )

    found = set()
    start = time.time()
    attempts = 0

    while len(found) < 92:
        attempts += 1
        x0 = random_solution()
        sol, val = sa.search(x0)

        if val == 28:
            found.add(tuple(sol))

        if attempts % 50 == 0:
            print(f"{len(found)}/92 solutions found...")

    end = time.time()

    print("\n===== Completed =====")
    print("Total unique solutions found:", len(found))
    print("Total attempts:", attempts)
    print("Time:", end - start, "seconds")

    return found
