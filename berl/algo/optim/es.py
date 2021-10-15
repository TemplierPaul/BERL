import numpy as np
from abc import abstractmethod
from .gradient import *

class ES:
    def __init__(self, n_genes, config):
        self.n_genes = n_genes
        self.n_pop = config["pop"]

        self.config = config

        self.rng = np.random.default_rng(self.config["seed"])

        # Current solution (The one that we report).
        self.theta = self.rng.standard_normal(self.n_genes) * 0.05

        self.genomes = []
        self.fitnesses = []

        self.gen = 0

    def __repr__(self):
        return self.__class__.__name__

    def __str__(self):
        return self.__repr__()

    def sample_normal(self):
        self.s = self.rng.standard_normal((self.n_genes, self.n_pop))

    def sample_symmetry(self):
        assert self.n_pop % 2 == 0
        l = self.rng.standard_normal((self.n_genes, int(self.n_pop/2)))
        self.s = np.concatenate([l, -l], axis=1)

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