import numpy as np

class HillClimbing:
    """
    Generic Hill Climbing optimizer for continuous functions with box constraints.

    Parameters
    ----------
    f : callable
        Objective function f(x) where x is a numpy array.
    domain : list of tuples
        Domain of each variable [(min_x1, max_x1), (min_x2, max_x2), ...].
    eps : float
        Neighborhood radius. Candidate points are generated as x_new = x + uniform(-eps, eps).
    max_it : int
        Maximum number of iterations per run (default: 1000).
    patience : int
        Early stopping after 'patience' iterations without improvement.
    """

    def __init__(self, f, domain, eps=0.1, max_it=1000, patience=50):
        self.f = f
        self.domain = np.array(domain, dtype=float)
        self.eps = eps
        self.max_it = max_it
        self.patience = patience
        
        self.dim = len(domain)

    def _clip_to_domain(self, x):
        """Ensure candidate stays within given bounds (box constraints)."""
        for i in range(self.dim):
            x[i] = np.clip(x[i], self.domain[i][0], self.domain[i][1])
        return x

    def _neighbor(self, x):
        """Generate a neighbor within ε distance."""
        noise = np.random.uniform(-self.eps, self.eps, size=self.dim)
        x_new = x + noise
        return self._clip_to_domain(x_new)

    def search(self):
        """
        Executes the Hill Climbing search.

        Returns
        -------
        best_x : np.ndarray
            Best solution found.
        best_f : float
            Objective function value for best_x.
        """

        # Start at lower bound of domain (as required by professor)
        x = np.array([self.domain[i][0] for i in range(self.dim)], dtype=float)
        best_x = x.copy()
        best_f = self.f(*best_x)

        no_improvement_steps = 0

        for it in range(self.max_it):

            x_cand = self._neighbor(best_x)
            f_cand = self.f(*x_cand)

            if f_cand < best_f:       # MINIMIZATION
                best_x = x_cand
                best_f = f_cand
                no_improvement_steps = 0
            else:
                no_improvement_steps += 1

            # Early stopping (patience)
            if no_improvement_steps > self.patience:
                break

        return best_x, best_f
