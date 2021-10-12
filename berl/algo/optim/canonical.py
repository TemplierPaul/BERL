from .es import *

class Canonical(ES):
    def __init__(self, n_genes, config):
        super().__init__(n_genes, config)

        # Number of parents selected
        self.n_parents = config["es_parents"]

        assert(self.n_parents <= config["pop"])

        self.sigma = config["es_sigma"]
        self.lr = config["es_lr"]
        self.c_sigma_factor = config["es_sigma_factor"]

        # Current solution (The one that we report).
        self.mu = self.rng.random(self.n_genes)
        # Computed update, step in parameter space computed in each iteration.
        self.step = 0

        # Compute weights for weighted mean of the top self.n_parents offsprings
        # (parents for the next generation).
        self.w = np.array([np.log(self.n_parents + 0.5) - np.log(i) for i in range(1, self.n_parents + 1)])
        self.w /= np.sum(self.w)

        # Noise adaptation stuff.
        self.p_sigma = np.zeros(self.n_genes)
        self.u_w = 1 / float(np.sum(np.square(self.w)))
        self.c_sigma = (self.u_w + 2) / (self.n_genes + self.u_w + 5)
        self.c_sigma *= self.c_sigma_factor
        self.const_1 = np.sqrt(self.u_w * self.c_sigma * (2 - self.c_sigma))

        self.s = None

    def populate(self):
        self.sample_normal()
        self.genomes = [self.mu + self.sigma * self.s[:, i] for i in range(self.n_pop)]
        return self    
    
    def back_random(self, genes_after):
        # Compute the s that would have created that genome
        s = (genes_after - self.mu)/self.sigma
        assert (s != s).sum() == 0 # check for NaNs
        return s
    
    def update(self):
        d = self.n_genes
        n = self.n_pop
        
        # self.s = np.array([self.back_random(i) for i in self.genomes]).transpose()

        inv_fitnesses = [- f for f in self.fitnesses]
        idx = np.argsort(inv_fitnesses) # indices from highest fitness to lowest

        step = np.zeros(d)
        
        for i in range(self.n_parents):
            step += self.w[i] * self.s[:, idx[i]]
                
        self.step = self.lr * self.sigma * step
        self.mu += self.step
        
        # Noise adaptation stuff.
        # self.p_sigma = (1 - self.c_sigma) * self.p_sigma + self.const_1 * step
        # self.sigma = self.sigma * np.exp((self.c_sigma / 2) * (np.sum(np.square(self.p_sigma)) / self.n_genes - 1))

    