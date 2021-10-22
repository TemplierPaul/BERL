from .es import *

class SNES(ES):
    def __init__(self, n_genes, config):
        super().__init__(n_genes, config)

        self.sigma = np.ones(self.n_genes) * config["es_sigma"]
        
        n=config["pop"]
        self.w = np.array([max(0, np.log(n/2+1) - np.log(i)) for i in range(1, n+1)])
        self.w = self.w / sum(self.w) - 1/n
        
        self.eta_mu = config["es_eta_mu"]
        self.eta_sigma = (3+np.log(n_genes)) / (5*np.sqrt(n_genes))

        # self.s = None

    def back_random(self, genes_after):
        # Compute the s that would have created that genome
        s = (genes_after - self.theta)/self.sigma
        assert (s != s).sum() == 0 # check for NaNs
        return s

    def populate(self):
        self.sample_normal()
        # self.genomes = [self.theta + self.sigma * self.s[:, i] for i in range(self.n_pop)]
        return self    
    
    def update(self):
        d =self.n_genes
        n = self.n_pop

        # self.s = np.array([self.back_random(i) for i in self.genomes]).transpose()
        
        inv_fitnesses = [- f for f in self.fitnesses]
        idx = np.argsort(inv_fitnesses) # indices from highest fitness to lowest
        
        # Compute gradients
        grad_theta = np.zeros(d)
        grad_sigma = np.zeros(d)
        
        for i in range(n):
            noise_i = self.noise_index[idx[i]] # Get noise index of ith best fitness
            s = self.get_noise(noise_i) # Get noise 
            grad_theta += self.w[i] * s
            grad_sigma += self.w[i] * (s **2 - 1)
        
        # Update variables
        self.theta += self.eta_mu * self.sigma * grad_theta
        new_sigma = self.sigma * np.exp(self.eta_sigma * grad_sigma)
        # Bound sigma to (0.01, 1000) to avoid NaN
        self.sigma = np.array([min(max(i, 0.01), 1000) for i in new_sigma])

    