from .es import *

class SNES(ES):
    def __init__(self, n_genes, config):
        super().__init__(n_genes, config)

        self.mu = self.rng.random(self.n_genes)
        self.sigma = np.ones(self.n_genes) * config["es_sigma"]
        
        n=config["pop"]
        self.u = np.array([max(0, np.log(n/2+1) - np.log(i)) for i in range(1, n+1)])
        self.u = self.u / sum(self.u) - 1/n_genes
        
        self.eta_mu = config["es_eta_mu"]
        self.eta_sigma = (3+np.log(n_genes)) / (5*np.sqrt(n_genes))

        self.s = None

    def back_random(self, genes_after):
        # Compute the s that would have created that genome
        s = (genes_after - self.mu)/self.sigma
        assert (s != s).sum() == 0 # check for NaNs
        return s

    def populate(self):
        self.sample_normal()
        self.genomes = [self.mu + self.sigma * self.s[:, i] for i in range(self.n_pop)]
        return self    
    
    def update(self):
        d =self.n_genes
        n = self.n_pop

        self.s = np.array([self.back_random(i) for i in self.genomes]).transpose()
        
        inv_fitnesses = [- f for f in self.fitnesses]
        idx = np.argsort(inv_fitnesses) # indices from highest fitness to lowest
        
        # Compute gradients
        grad_mu = np.zeros(d)
        grad_sigma = np.zeros(d)
        
        for i in range(n):
            j = idx[i]
            grad_mu += self.u[i] * self.s[:, j]
            grad_sigma += self.u[i] * (self.s[:, j] **2 - 1)
        
        # Update variables
        self.mu += self.eta_mu * self.sigma * grad_mu
        new_sigma = self.sigma * np.exp(self.eta_sigma * grad_sigma)
        # Bound sigma to (0.01, 1000) to avoid NaN
        self.sigma = np.array([min(max(i, 0.01), 1000) for i in new_sigma])

    