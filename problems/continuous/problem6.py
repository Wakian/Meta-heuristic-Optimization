import numpy as np

def f(x1, x2):
    return x1 * np.sin(4 * np.pi * x1) - x2 * np.sin(4 * np.pi * x2 + np.pi) + 1

domain = [(-1, 3), (-1, 3)]
