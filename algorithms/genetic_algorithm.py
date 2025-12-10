import numpy as np
from problems.discrete.problem8_tsp import route_length

class GeneticTSP:
    def __init__(self, points, pop_size=100, max_gen=300,
                 tournament_k=3, mutation_prob=0.01, elitism=0):

        self.points = points
        self.pop_size = pop_size
        self.max_gen = max_gen
        self.tournament_k = tournament_k
        self.mutation_prob = mutation_prob
        self.elitism = elitism

        self.N = len(points)  # number of cities

    # -----------------------------------------------------
    # INITIAL POPULATION
    # -----------------------------------------------------

    def generate_population(self):
        pop = []
        for _ in range(self.pop_size):
            ind = np.random.permutation(self.N)
            pop.append(ind)
        return np.array(pop, dtype=int)

    # -----------------------------------------------------
    # FITNESS
    # -----------------------------------------------------

    def fitness(self, individual):
        return 1.0 / route_length(individual, self.points)

    # -----------------------------------------------------
    # TOURNAMENT SELECTION
    # -----------------------------------------------------

    def tournament(self, pop):
        selected = np.random.choice(len(pop), self.tournament_k, replace=False)
        best = None
        best_fit = -np.inf
        for idx in selected:
            f = self.fitness(pop[idx])
            if f > best_fit:
                best = pop[idx]
                best_fit = f
        return best.copy()

    # -----------------------------------------------------
    # 2-POINT ORDERED CROSSOVER (NO REPETITION)
    # -----------------------------------------------------

    def crossover(self, parent1, parent2):
        N = self.N
        a, b = np.sort(np.random.choice(N, 2, replace=False))
        
        child = -np.ones(N, dtype=int)
        child[a:b] = parent1[a:b]

        fill = [gene for gene in parent2 if gene not in child]
        fill_idx = [i for i in range(N) if child[i] == -1]

        for idx, gene in zip(fill_idx, fill):
            child[idx] = gene

        return child

    # -----------------------------------------------------
    # SWAP MUTATION (1% probability)
    # -----------------------------------------------------

    def mutate(self, individual):
        if np.random.rand() < self.mutation_prob:
            i, j = np.random.choice(self.N, 2, replace=False)
            individual[i], individual[j] = individual[j], individual[i]

    # -----------------------------------------------------
    # MAIN LOOP
    # -----------------------------------------------------

    def run(self, acceptable_cost=None):
        """
        acceptable_cost: stopping condition (slide 31/61)
        """
        pop = self.generate_population()
        best_ind = None
        best_fit = -np.inf

        for gen in range(self.max_gen):

            # Evaluate population
            fits = np.array([self.fitness(ind) for ind in pop])
            idx_best = np.argmax(fits)

            if fits[idx_best] > best_fit:
                best_fit = fits[idx_best]
                best_ind = pop[idx_best].copy()

            # STOP if acceptable route length reached
            cost = 1.0 / best_fit
            if acceptable_cost is not None and cost <= acceptable_cost:
                return best_ind, gen

            # ELITISM
            elites = []
            if self.elitism > 0:
                elite_indices = np.argsort(fits)[-self.elitism:]
                elites = [pop[i].copy() for i in elite_indices]

            # GENERATE NEW POP
            new_pop = elites.copy()

            while len(new_pop) < self.pop_size:
                p1 = self.tournament(pop)
                p2 = self.tournament(pop)

                child = self.crossover(p1, p2)
                self.mutate(child)
                new_pop.append(child)

            pop = np.array(new_pop, dtype=int)

        return best_ind, self.max_gen

