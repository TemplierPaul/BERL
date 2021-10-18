from mpi4py import MPI
import numpy as np
from collections import OrderedDict
from ...env.env import *
from .rl_agent import Agent, State, FrameStackState
from .c51_agent import C51Agent
import torch

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

        env = make_env(config["env"])
        self.config["obs_shape"] = env.observation_space.shape
        env.close()

        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank() -1
        self.size = self.comm.Get_size() -1

        self.n_pop = None
        self.n_per_w = None
        self.total_n = None
        self.ids = None

        self.genomes = []
        self.fitnesses = {}

        self.keep_running = True
        self.env = None

        self.frames = 0

        self.n_out = None
        self.vb = None

    def __repr__(self):
        return f"Secondary {self.rank}"

    def __str__(self):
        return self.__repr__()

    def set_sizes(self, n_pop):
        if self.size == 0:
            self.n_per_w = n_pop
            self.total_n = n_pop
            self.ids = [range(n_pop)]
            return self.ids 

        self.n_pop = n_pop
        self.n_per_w = n_pop // self.size + int(n_pop%self.size > 0)
        self.total_n = self.n_per_w * self.size

        self.ids = []
        for i in range(self.n_per_w):
            j = i * self.size + self.rank
            if j < self.n_pop:
                self.ids.append(j)
        return self.ids 

    def get_n_out(self):
        model = self.Net(c51=self.config["c51"])
        mod = list(model._modules.values())
        n_out = mod[-1].out_features
        self.config["n_actions"] = int(n_out/51) if self.config["c51"] else n_out

        env = make_env(self.config["env"])
        self.config["obs_shape"] = env.observation_space.shape
        env.close()

    def run(self):
        self.vb = self.comm.bcast(None, root=0)
        # try:
        while self.keep_running:
            d = self.comm.bcast(None, root=0)
            if d["stop"]:
                print(f"{self}: Stop signal received")
                self.keep_running = False
                return 
            self.set_sizes(d["pop_size"])
            self.genomes = self.comm.scatter(None, root=0)
            self.run_evaluations(seed=d["seed"])
            self.return_info()
        # except KeyboardInterrupt:
        #     print("Interrupted")

    def run_evaluations(self, seed=0):
        self.fitnesses = {}
        for i in range(len(self.ids)):
            g = self.genomes[i] # genome
            g = np.float64(g)
            f = self.evaluate(g, seed=seed)
            self.fitnesses[self.ids[i]] = f

        return self.fitnesses
        
    def return_info(self):
        d = {
            "fitnesses":self.fitnesses,
            "frames": self.frames
        }
        return self.comm.gather(d, root=0)

    def evaluate(self, genome, seed=0, render=False, test=False):
        if seed < 0:
            seed = np.random.randint(0, 1000000000)
        env = make_env(self.config["env"], seed=seed)
        agent = self.make_agent(genome)

        # Virtual batch normalization
        agent.model(self.vb)

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

        self.get_vb()
        self.comm.bcast(self.vb, root=0)

    def __repr__(self):
        return f"Primary ({self.size})"

    def get_vb(self):
        if self.n_out is None:
            self.get_n_out()

        env = make_env(self.config["env"])

        # State
        if self.config["stack_frames"] > 1:
            state = FrameStackState(self.config["obs_shape"], self.config["stack_frames"])
        else:
            state = State()

        vb = []
        env.reset()
        while len(vb) < 128:
            # Apply random action and with 1% chance save this state.
            a =  np.random.randint(0, self.config["n_actions"])
            obs, _, done, _ = env.step(a)
            state.update(obs)
            if done:
                env.reset()
            elif np.random.rand() < 0.01:
                vb.append(state.get())

        self.vb = torch.stack(vb).squeeze().double()
        return self.vb

    def send_genomes(self, pop, hof=None, seed=0):
        if self.size == 0:
            return self.evaluate_all(pop, hof=hof, seed=seed)

        d = {
            "seed":seed, 
            "pop_size":len(pop), 
            "stop":False
            }

        self.set_sizes(len(pop))
        self.comm.bcast(d, root=0) # Send eval info 

        # Fill with np.nan
        n_genes = len(pop[0])
        none_pop = [np.full(n_genes, np.nan) for i in range(self.total_n - len(pop))]
        pop = np.array(pop + none_pop) 

        # Split and 
        split = pop.reshape((self.size, self.n_per_w, n_genes), order='F')

        split = [np.nan] + list(split)

        # Send to secondary nodes
        self.genomes = self.comm.scatter(split, root=0)

        if hof is not None:
            g = hof.genes # genome
            g = np.float64(g)
            hof.fitness = self.evaluate(g, seed=seed)
        
        self.fitnesses = {}
        
        results = self.return_info()

        # Order results into array
        d = {}
        self.total_frames = 0
        for r in results:
            f = r["fitnesses"]
            d = {**d, **f}
            self.total_frames += r["frames"]
        fitnesses = list(OrderedDict(sorted(d.items())).values())

        return fitnesses

    def stop(self):
        d = {
            "stop":True
            }
        # print("Sending stop signal")
        self.comm.bcast(d, root=0) # Send eval info => init_eval()

    def eval_elite(self, elite):
        # n = self.size*2
        n = 100 
        pop = [elite for i in range(n)]
        return self.send_genomes(pop, seed=-1)
        
    def evaluate_all(self, pop, hof=None, seed=0):
        f = [self.evaluate(np.float64(g), seed=seed) for g in pop]

        if hof is not None:
            g = hof.genes # genome
            g = np.float64(g)
            hof.fitness = self.evaluate(g, seed=seed)

        return f