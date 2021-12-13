import numpy as np
from abc import abstractmethod
from .gradient import *

class ES:
    def __init__(self, n_genes, config):
        self.n_genes = n_genes
        self.n_pop = config["pop"]

        # Number of parents selected
        if config["es_mu_ratio"]>0:
            self.mu = int(self.n_pop / config["es_mu_ratio"])
            config["es_mu"] = self.mu
        else:
            self.mu = config["es_mu"]

        self.config = config

        self.rng = np.random.default_rng(self.config["seed"])

        # Current solution (The one that we report).
        self.theta = self.rng.standard_normal(self.n_genes) * 0.05

        self.noise = None
        self.noise_index = []
        self.fitnesses = []

        self.gen = 0

    def __repr__(self):
        return self.__class__.__name__

    def __str__(self):
        return self.__repr__()

    ### Noise ###

    def get_noise(self, key):
        assert self.noise is not None, "There is no noise matrix"
        if key > 0:
            return self.noise[key:(key+self.n_genes)]
        key = abs(key)
        return -1* self.noise[key:(key+self.n_genes)]

    def r_noise_id(self):
        return self.rng.integers(0, len(self.noise)-self.n_genes)

    def sample_normal(self):
        assert self.noise is not None, "There is no noise matrix"
        self.noise_index = np.array([self.r_noise_id() for _ in range(self.n_pop)])
        return self.noise_index

    def sample_symmetry(self):
        assert self.noise is not None, "There is no noise matrix"
        assert self.n_pop % 2 == 0
        l = np.array([self.r_noise_id() for _ in range(int(self.n_pop/2))])
        self.noise_index = np.concatenate([l, -l], axis=0)
        return self.noise_index

    ### Ask / Tell interface ###
    def ask(self):
        self.populate()
        self.fitnesses = None
        return self.noise_index

    def tell(self, noise_id, fit=None, pop=None):
        assert len(noise_id) == self.n_pop
        self.noise_index = noise_id
        self.fitnesses = fit
        if pop is not None:
            self.update_from_population(pop)
        elif fir is not None:
            self.update()
        else:
            raise ValueError("No population or fitnesses given")
        self.gen += 1

    @abstractmethod
    def populate(self): # pragma: no cover
        pass

    @abstractmethod
    def update(self): # pragma: no cover
        pass

    @abstractmethod
    def update_from_population(self, pop): # pragma: no cover
        pass

    def export(self):
        """
        Export self state as a dict
        """
        d={
            "gen":self.gen,
            "theta":self.theta
        }
        return d

    def load(self, d):
        self.gen = int(d["gen"])
        self.theta = d["theta"]
        return self