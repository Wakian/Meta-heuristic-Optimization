import numpy as np

class GlobalRandomSearch:
    """
    Global Random Search for continuous optimization.

    Signature:
        GlobalRandomSearch(f, domain, sigma=0.4, max_it=1000)

    Parameters
    ----------
    f : callable
        Objective function f(x1, x2, ...)
    domain : list of tuples
        [(x1_min, x1_max), (x2_min, x2_max), ...]
    sigma : float
        (Optional) used only if you later wish to create 'local candidate around global sample'.
        For plain GRS it's not necessary but kept to match LRS signature.
    max_it : int
        Maximum iterations.
    """

    def __init__(self, f, domain, sigma=0.4, max_it=1000):
        self.f = f
        self.domain = np.array(domain, dtype=float)
        self.sigma = sigma
        self.max_it = max_it
        self.dim = len(domain)

    def _sample_uniform(self):
        """Sample uniformly in the domain box."""
        x = np.empty(self.dim, dtype=float)
        for i in range(self.dim):
            lo, hi = self.domain[i]
            x[i] = np.random.uniform(lo, hi)
        return x

    def _clip(self, x):
        for i in range(self.dim):
            x[i] = np.clip(x[i], self.domain[i][0], self.domain[i][1])
        return x

    def search(self):
        """
        Runs plain Global Random Search:
        - sample uniformly max_it times
        - keep best found
        Returns (best_x, best_f)
        """

        # initial best: sample once
        best_x = self._sample_uniform()
        best_f = self.f(*best_x)

        for _ in range(self.max_it):
            x_cand = self._sample_uniform()
            f_cand = self.f(*x_cand)
            if f_cand < best_f:   # default: minimization
                best_x = x_cand
                best_f = f_cand

        return best_x, best_f
