import numpy as np
from abc import abstractmethod
from .gradient import *

class ES:
    def __init__(self, n_genes, config):
        self.n_genes = n_genes
        self.n_pop = config["pop"]

        self.config = config

        self.rng = np.random.default_rng(self.config["seed"])

        self.genomes = []
        self.fitnesses = []

        self.gen = 0

    def sample_normal(self):
        self.s = self.rng.standard_normal((self.n_genes, self.n_pop))

    def ask(self):
        self.populate()
        self.fitnesses = None
        return self.genomes

    def tell(self, pop, fit):
        assert len(pop) == self.n_pop
        assert len(pop[0]) == self.n_genes
        self.genomes = pop
        self.fitnesses = fit
        self.update()
        self.gen += 1

    @abstractmethod
    def populate(self): # pragma: no cover
        pass

    @abstractmethod
    def update(self): # pragma: no cover
        pass