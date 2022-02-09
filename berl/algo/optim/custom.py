from .es import *


class CustomES(ES):
    def __init__(self, n_genes, config):
        raise NotImplementedError
        super().__init__(n_genes, config)
        self.w = None
        if self.config["custom_rank_sum"]:
            self.set_weights()

        self.set_gradient_optim()

    def set_gradient_optim(self):
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

    def set_weights(self):
        n = self.n_pop
        mu = self.mu
        weight_decay = config["custom_wi"].lower()
        if weight_decay == "linear":
            self.w = np.array([max(0, mu-i) for i in range(n)])
        elif weight_decay == "log":
            self.w = np.array([max(0, np.log(mu + 0.5) - np.log(i))
                              for i in range(1, n + 1)])
        else:
            raise NotImplementedError(
                f"{weight_decay} is not a valid weight decay")

        self.w /= sum(self.w)

        if self.config["custom_wi_sum"] == 0:
            self.w -= 1/n
        else:
            assert self.config[
                "custom_wi_sum"] == 1, f"Weights should be summed to 0 or 1, not {self.config['custom_wi_sum']}"

        return self.w

    def get_weights(self):
        if self.config["custom_rank_sum"]:
            return self.w

        n = self.n_pop
        mu = self.mu

        self.w = -np.sort(-self.fitnesses)  # Sort fitnesses from high to low
        mask = np.concatenate([np.ones(mu), np.zeros(n-mu)])
        self.w = mask * self.w  # Keep mu first fitnesses, set the others to 0
        if sum(self.w) <= 1e-5:
            # If the sum is too small, normalize with the max (in abs value)
            # If the max is too small too (eg all values very close to 0), send normalized mask
            self.w = mask / \
                sum(mask) if max(self.w) <= 1e-5 else self.w / max(abs(self.w))
            return self.w

        self.w /= sum(self.w)

        if self.config["custom_wi_sum"] == 0:
            self.w -= 1/n
        else:
            assert self.config[
                "custom_wi_sum"] == 1, f"Weights should be summed to 0 or 1, not {self.config['custom_wi_sum']}"
        return self.w

    def natural_gradient(self):
        # Compute gradient like SNES
        d = self.n_genes
        n = self.n_pop

        inv_fitnesses = [- f for f in self.fitnesses]
        # indices from highest fitness to lowest
        idx = np.argsort(inv_fitnesses)

        self.get_weights()

        # Compute gradients
        grad_theta = np.zeros(d)
        grad_sigma = np.zeros(d)

        if self.config["custom_sigma_update"]:
            for i in range(n):
                # Get noise index of ith best fitness
                noise_i = self.noise_index[idx[i]]
                s = self.get_noise(noise_i)  # Get noise
                grad_theta += self.w[i] * s
                grad_sigma += self.w[i] * (s ** 2 - 1)
        else:
            for i in range(n):
                # Get noise index of ith best fitness
                noise_i = self.noise_index[idx[i]]
                s = self.get_noise(noise_i)  # Get noise
                grad_theta += self.w[i] * s

        return grad_theta, grad_sigma

    def plain_gradient(self):
        pass

    def populate(self):
        if self.config["custom_symmetry"]:
            self.sample_symmetry()
        else:
            self.sample_normal()
        return self
