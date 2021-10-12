from mpi4py import MPI
import numpy as np
from collections import OrderedDict
from ...env.env import *
from .rl_agent import Agent
from .c51_agent import C51Agent

def get_ids(rank):
    ids = []
    for i in range(n_per_w):
        j = i*w + rank
        if j < n_pop:
            ids.append(j)
    return ids


class Secondary:
    def __init__(self, Net, config):
        self.Net = Net
        self.config = config

        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size() # new: gives number of ranks in self.comm

        self.n_pop = None
        self.n_per_w = None
        self.total_n = None
        self.ids = None

        self.genomes = []
        self.fitnesses = {}

        self.keep_running = True
        self.env = None

        self.frames = 0

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
        # try:
        while self.keep_running:
            d = self.comm.bcast(None, root=0)
            if d["stop"]:
                print(f"{self}: Stop signal received")
                self.keep_running = False
                return 
            self.set_sizes(d["pop_size"])
            self.genomes = self.comm.scatter(None, root=0)
            self.run_evaluations()
            self.return_info()
        # except KeyboardInterrupt:
        #     print("Interrupted")

    def run_evaluations(self):
        self.fitnesses = {}
        for i in range(len(self.ids)):
            g = self.genomes[i] # genome
            g = np.float64(g)
            f = self.evaluate(g)
            self.fitnesses[self.ids[i]] = f

        return self.fitnesses
        
    def return_info(self):
        d = {
            "fitnesses":self.fitnesses,
            "frames": self.frames
        }
        return self.comm.gather(d, root=0)

    def evaluate(self, genome, render=False):
        env = make_env(self.config["env"])()
        agent = self.make_agent(genome)
        agent.state.reset()

        try:
            obs = env.reset()
            n_frames = 0
            total_r = 0
            done = False

            while not done and n_frames < self.config["episode_frames"]:
                action = agent.act(obs)
                obs, r, done, _ = env.step(action)

                if self.config["reward_clip"]>0:
                    r = max(min(r, self.config["reward_clip"]), -self.config["reward_clip"])

                if render:
                    env.render()

                total_r += r
                n_frames += 1

        finally:
            env.close()
        self.frames += n_frames
        return total_r

    def make_agent(self, genome=None):
        AgentType = C51Agent if self.config["c51"] else Agent
        i = AgentType(self.Net, self.config)
        if genome is not None:
            i.genes = genome
        return i
        
             

class Primary(Secondary):
    def __init__(self, Net, config):
        super().__init__(Net, config)
        self.total_frames = 0

    def __repr__(self):
        return f"Primary ({self.size})"

    def send_genomes(self, pop, hof=None, seed=0):
        if hof is not None:
            pop = [hof.genes] + pop
        d = {
            "seed":seed, 
            "pop_size":len(pop), 
            "stop":False
            }

        self.set_sizes(len(pop))
        self.comm.bcast(d, root=0) # Send eval info 

        n_genes = len(pop[0])

        none_pop = [np.full(n_genes, np.nan) for i in range(self.total_n - len(pop))]
        pop = np.array(pop + none_pop) # Fill with np.nan

        split = pop.reshape((self.size, self.n_per_w, n_genes), order='F')

        self.genomes = self.comm.scatter(split, root=0)
        self.run_evaluations()
        results = self.return_info()

        # Order results into array
        d = {}
        self.total_frames = 0
        for r in results:
            f = r["fitnesses"]
            d = {**d, **f}
            self.total_frames += r["frames"]
        fitnesses = list(OrderedDict(sorted(d.items())).values())

        if hof is not None:
            hof.fitness = fitnesses.pop(0)

        return fitnesses

    def stop(self):
        d = {
            "stop":True
            }
        # print("Sending stop signal")
        self.comm.bcast(d, root=0) # Send eval info => init_eval()