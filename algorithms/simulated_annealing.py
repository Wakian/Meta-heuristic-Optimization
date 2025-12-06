class SimulatedAnnealing:
    def __init__(self, f, initial_temp, cooling, max_it):
        self.f = f
        self.T = initial_temp
        self.cooling = cooling
        self.max_it = max_it

    def search(self):
        pass  # TODO
