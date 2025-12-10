import numpy as np

def f(x1, x2):
    return np.exp(-(x1**2 + x2**2)) + 2 * np.exp(-(x1 - 1.7)**2 + (x2 - 1.7)**2)

domain = [(-2, 4), (-2, 5)]
