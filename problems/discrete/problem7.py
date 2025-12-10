import numpy as np

# --------------------------
# 1. Count attacking pairs
# --------------------------

def queen_attacks(x):
    """
    Count how many pairs of queens are attacking each other.
    x is a vector of 8 integers (1–8).
    """

    h = 0
    for i in range(8):
        for j in range(i + 1, 8):
            # same row
            if x[i] == x[j]:
                h += 1
            # diagonal attack
            if abs(x[i] - x[j]) == abs(i - j):
                h += 1
    return h


# --------------------------
# 2. Objective function
# --------------------------

def f(x):
    """
    Objective function:
    f(x) = 28 - h(x)
    MAXIMIZATION.
    """
    return 28 - queen_attacks(x)


# --------------------------
# 3. Random initial solution
# --------------------------

def random_solution():
    """Generate a random queen position (1–8 in each column)."""
    return np.random.randint(1, 9, size=8)


# --------------------------
# 4. Neighbor generation (Option A: MINIMAL MOVEMENT)
# --------------------------

def neighbor(x):
    """
    Minimal movement neighbor:
    pick 1 column, move queen up or down by 1.
    Clip between 1 and 8.
    """
    x_new = x.copy()

    col = np.random.randint(0, 8)
    step = np.random.choice([-1, 1])

    x_new[col] = np.clip(x_new[col] + step, 1, 8)
    return x_new
