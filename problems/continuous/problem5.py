import numpy as np

def f(x1, x2):
    return (x1 * np.cos(x1)) / 20 + 2 * np.exp(-(x1**2) - (x2 - 1)**2) + 0.01 * x1 * x2

domain = [(-10, 10), (-10, 10)]
