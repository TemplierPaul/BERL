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


class OpenAI(ES):
    def __init__(self, n_genes, config):
        super().__init__(n_genes, config)
        self.n_pop -= self.n_pop % 2  # Make pop size even
        self.sigma = config["es_sigma"]
        self.l2coef = config["es_l2coef"]

        # Get gradient optimizer
        grad_name = config["es_gradient_optim"].lower()
        if grad_name == "base":
            self.gradient_optim = GradientOptimizer(
                n_genes=n_genes,
                lr=config["es_lr"]
            )
        elif grad_name == "sgd":
            self.gradient_optim = SGD(
                n_genes=n_genes,
                lr=config["es_lr"],
                momentum=config["es_momentum"]
            )
        elif grad_name == "adam":
            self.gradient_optim = Adam(
                n_genes=n_genes,
                lr=config["es_lr"],
                beta1=config["es_beta1"],
                beta2=config["es_beta2"]
            )
        else:
            raise NotImplementedError(
                f"{grad_name} is not a valid gradient optimizer")

        self.s = None

    def __repr__(self):
        return f"{self.__class__.__name__} ({self.gradient_optim.__class__.__name__})"

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

        gradient = np.zeros(d)
        for i in range(self.n_pop):
            noise_i = self.noise_index[i]  # Get noise index
            s = self.get_noise(noise_i)  # Get noise
            gradient += self.w[i] * s

        gradient /= self.sigma * self.n_pop
        self.theta += self.gradient_optim.step(gradient)

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
