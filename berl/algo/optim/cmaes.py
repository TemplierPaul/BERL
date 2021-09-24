import cma
from .es import *

class CMAES(ES):
    def __init__(self, n_genes, config):
        super().__init__(n_genes, config)
        state = n_genes * [0.]
        sigma = 0.2
        self.cma = cma.CMAEvolutionStrategy(state, sigma, {"popsize":config["pop"]})
    
    def populate(self):
        self.genomes = self.cma.ask()

    def update(self):
        # pycma minimizes so we send -1 * fitness to maximize the fitness
        cma_fitness = [-f for f in self.fitnesses] 
        self.cma.tell(self.genomes, cma_fitness)