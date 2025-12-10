import numpy as np

class SimulatedAnnealingDiscrete:
    """
    General Simulated Annealing for DISCRETE problems.
    Works with:
    - provided objective function f(x)
    - provided neighbor function
    """

    def __init__(self, f, neighbor_fn, T0=10.0, alpha=0.99, max_it=5000):
        self.f = f
        self.neighbor_fn = neighbor_fn
        self.T0 = T0
        self.alpha = alpha
        self.max_it = max_it

    # --------------------------------------------------------------

    def search(self, x0):
        """Run SA starting from initial solution x0."""

        x = np.array(x0, dtype=int)
        fx = self.f(x)

        best_x = x.copy()
        best_f = fx

        T = self.T0

        for _ in range(self.max_it):

            candidate = self.neighbor_fn(x)
            f_candidate = self.f(candidate)

            delta = f_candidate - fx  # MAXIMIZATION

            if delta > 0 or np.random.rand() < np.exp(delta / T):
                x = candidate
                fx = f_candidate

                if fx > best_f:
                    best_f = fx
                    best_x = x.copy()

            T *= self.alpha

            # Stop early if optimum reached
            if best_f == 28:
                break

        return best_x, best_f
