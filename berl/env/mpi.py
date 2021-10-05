from mpi4py import MPI
import numpy as np
from .env import *

def get_ids(rank):
    ids = []
    for i in range(n_per_w):
        j = i*w + rank
        if j < n_pop:
            ids.append(j)
    return ids


class Secondary:
    def __init__(self, cfg):
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size() # new: gives number of ranks in self.comm

        self.cfg = cfg

        self.n_pop = None
        self.n_per_w = None
        self.total_n = None
        self.ids = None

        self.genomes = []
        self.fitnesses = {}

        self.keep_running = True
        self.env = None

    def __repr__(self):
        return f"Secondary {self.rank}"

    def __str__(self):
        return self.__repr__()

    def set_sizes(self, n_pop):
        self.n_pop = n_pop
        self.n_per_w = n_pop // self.size + int(n_pop%self.size > 0)
        self.total_n = self.n_per_w * self.size

        self.get_ids()

    def get_ids(self):
        self.ids = []
        for i in range(self.n_per_w):
            j = i * self.size + self.rank
            if j < self.n_pop:
                self.ids.append(j)
        return self.ids 

    def run(self):
        while self.keep_running:
            d = self.comm.bcast(None, root=0)
            if d["stop"]:
                print(f"{self}: Stop signal received")
                self.keep_running = False
                return 
            self.set_sizes(d["pop_size"])
            self.make_env(seed=d["seed"])
            self.genomes = self.comm.scatter(None, root=0)
            self.run_evaluations()

    def run_evaluations(self):
        self.fitnesses = {}
        for i in range(len(self.ids)):
            g = self.genomes[i] # genome
            f = self.evaluate(g)
            self.fitnesses[self.ids[i]] = f

        return self.comm.gather(self.fitnesses, root=0)

    def evaluate(self, genome):
        return self.rank*100 + genome[0]

    def make_env(self, seed=0):
        print("making env")
            

class Primary(Secondary):
    def __init__(self, cfg):
        super().__init__(self)

    def __repr__(self):
        return f"Primary ({self.size})"

    def send_genomes(self, pop, seed=0):
        d = {
            "seed":seed, 
            "pop_size":len(pop), 
            "stop":False
            }

        self.set_sizes(len(pop))
        self.comm.bcast(d, root=0) # Send eval info 

        n_genes = len(pop[0])

        none_pop = [np.full(n_genes, None) for i in range(self.total_n - len(pop))]
        pop = np.array(pop + none_pop) # Fill with None

        split = pop.reshape((self.size, self.n_per_w, n_genes), order='F')

        self.genomes = self.comm.scatter(split, root=0)
        results = self.run_evaluations()

        # Order results into array
        d = {}
        for r in results:
            d = {**d, **r}
        fitnesses = list(OrderedDict(sorted(d.items())).values())

        return fitnesses

    def stop(self):
        d = {
            "stop":True
            }
        # print("Sending stop signal")
        self.comm.bcast(d, root=0) # Send eval info => init_eval()