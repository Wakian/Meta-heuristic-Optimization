import numpy as np

def f(x1, x2):
    return (
        x1**2 - 10 * np.cos(2 * np.pi * x1) + 10 +
        x2**2 - 10 * np.cos(2 * np.pi * x2) + 10
    )

domain = [(-5.12, 5.12), (-5.12, 5.12)]
