from .es import *


def compute_ranks(x):
    """
    Returns ranks in [0, len(x))
    Note: This is different from scipy.stats.rankdata, which returns ranks in [1, len(x)].
    """
    assert x.ndim == 1
    ranks = np.empty(len(x), dtype=int)
    ranks[x.argsort()] = np.arange(len(x))
    return ranks


def compute_centered_ranks(x):
    x = np.array(x)
    y = compute_ranks(x.ravel()).reshape(x.shape).astype(np.float32)
    y /= (x.size - 1)
    y -= .5
    return y


class NewOpenAI(ES):
    def __init__(self, n_genes, config):
        super().__init__(n_genes, config)
        self.n_pop -= self.n_pop % 2  # Make pop size even
        self.sigma = config["es_sigma"]
        self.momentum = config["es_momentum"]
        self.lr = config["es_lr"]
        self.l2coef = config["es_l2coef"]
        self.weight_decay = config["es_wd"]

        self.wd = None
        self.s = None
        self.v = np.zeros(self.n_genes)

    def __repr__(self):
        return f"{self.__class__.__name__})"

    def populate(self):
        self.sample_symmetry()
        # self.genomes = [self.theta + self.sigma * self.s[:, i] for i in range(self.n_pop)]
        return self

    def back_random(self, genes_after):
        # Compute the s that would have created that genome
        s = (genes_after - self.theta)/self.sigma
        assert (s != s).sum() == 0  # check for NaNs
        return s

    def update(self):
        d = self.n_genes
        n = self.n_pop

        self.w = compute_centered_ranks(self.fitnesses)

        self.fitnesses = np.array(self.fitnesses)
        # ranks = np.empty(len(self.fitnesses), dtype=int)
        ranks = self.fitnesses.argsort()
        ranks = ranks / (len(ranks) - 1)  # Normalize

        delta_rank = {}
        for i in range(n):
            noise_i = self.noise_index[i]  # Get noise index
            abs_noise_i = np.abs(noise_i)
            if abs_noise_i not in delta_rank:
                delta_rank[abs_noise_i] = ranks[i]
            else:
                delta_rank[abs_noise_i] -= ranks[i]

        deltas = np.abs(np.array(list(delta_rank.values()))).sum()

        # Compute gradient
        gradient = np.zeros(d)
        for i in range(n):
            noise_i = self.noise_index[i]  # Get noise index
            s = self.get_noise(noise_i)  # Get noise
            gradient += self.w[i] * s

        # Normalize
        gradient /= deltas

        # Weight decay
        self.wd = self.weight_decay * self.theta
        # Update with momentum
        self.v = self.momentum * self.v + self.lr * (gradient - self.wd)
        # Step
        self.theta += self.v

    # def update_from_population(self, pop):
    #     d = self.n_genes
    #     n = self.n_pop

    #     fitnesses = pop.get_fitness()
    #     self.w = compute_centered_ranks(inv_fitnesses)

    #     gradient = np.zeros(d)
    #     for i in range(self.n_pop):
    #         genes = pop.get_indiv(idx[i])
    #         s = self.back_random(genes_after=genes)
    #         gradient += self.w[i] * s

    #     gradient /= self.sigma * self.n_pop  # Normalize
    #     gradient -= self.l2coef * self.theta  # L2 regularization
    #     self.theta += self.gradient_optim.step(gradient)  # Update theta
