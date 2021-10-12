# Inspired from https://github.dev/openai/evolution-strategies-starter/blob/master/es_distributed/optimizers.py

import numpy as np

class GradientOptimizer(object):
    def __init__(self, n_genes):
        self.n_genes = n_genes
        self.t = 0

    def step(self, gradient):
        self.t+=1
        return gradient

class SGD(GradientOptimizer):
    def __init__(self, n_genes, lr, momentum=0.9):
        GradientOptimizer.__init__(self, n_genes)
        self.v = np.zeros(self.n_genes, dtype=np.float32)
        self.lr, self.momentum = lr, momentum

    def step(self, gradient):
        self.t += 1
        self.v = self.momentum * self.v + (1. - self.momentum) * gradient
        step = -self.lr * self.v
        return step

class Adam(GradientOptimizer):
    def __init__(self, n_genes, lr, beta1=0.9, beta2=0.999, epsilon=1e-08):
        Optimizer.__init__(self, n_genes)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = np.zeros(self.n_genes, dtype=np.float32)
        self.v = np.zeros(self.n_genes, dtype=np.float32)

    def step(self, gradient):
        self.t += 1
        a = self.lr * np.sqrt(1 - self.beta2 ** self.t) / (1 - self.beta1 ** self.t)
        self.m = self.beta1 * self.m + (1 - self.beta1) * gradient
        self.v = self.beta2 * self.v + (1 - self.beta2) * (gradient * gradient)
        step = -a * self.m / (np.sqrt(self.v) + self.epsilon)
        return step

