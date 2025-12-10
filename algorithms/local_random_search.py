import numpy as np

class LocalRandomSearch:
    """
    Local Random Search for continuous optimization.
    
    sigma defines the size of the local neighborhood.
    """

    def __init__(self, f, domain, sigma=0.4, max_it=1000):
        self.f = f
        self.domain = np.array(domain, dtype=float)
        self.sigma = sigma
        self.max_it = max_it
        self.dim = len(domain)

    def _clip(self, x):
        """Keeps x inside the domain box."""
        for i in range(self.dim):
            x[i] = np.clip(x[i], self.domain[i][0], self.domain[i][1])
        return x

    def _neighbor(self, x):
        """Generate local random point within sigma distance."""
        step = np.random.normal(0, self.sigma, size=self.dim)
        candidate = x + step
        return self._clip(candidate)

    def search(self):
        """Standard LRS process."""
        # Start in the lower bound (like hill climbing)
        x = np.array([d[0] for d in self.domain], dtype=float)
        best_x = x.copy()
        best_f = self.f(*best_x)

        for _ in range(self.max_it):
            candidate = self._neighbor(best_x)
            value = self.f(*candidate)

            if value < best_f:  # LRS is MINIMIZATION by default
                best_x = candidate
                best_f = value

        return best_x, best_f
