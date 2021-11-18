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

        self.theta = None
        self.sigma = None
        self.n_genes = None
        noise_size = (10**(config["noise_size"]))
        self.noise = np.random.RandomState(123).randn(int(noise_size)).astype('float32')

        self.noise_index = None
        self.fitnesses = []

        self.keep_running = True
        self.env = None

        self.frames = 0

        self.n_out = None
        self.vb = None

    def __repr__(self):
        return f"Secondary {self.rank}"

    def __str__(self):
        return self.__repr__()

    def get_noise(self, key):
        if key > 0:
            return self.noise[key:(key+self.n_genes)]
        key = abs(key)
        return -1* self.noise[key:(key+self.n_genes)]

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
            self.n_genes = d["n_genes"]
            self.theta = self.comm.bcast(None, root=0)
            self.sigma = self.comm.bcast(None, root=0)
            self.noise_index = self.comm.scatter(None, root=0)

            self.run_evaluations(seed=d["seed"])
            self.return_info()
        # except KeyboardInterrupt:
        #     print("Interrupted")

    def run_evaluations(self, seed=0):
        self.fitnesses = []
        for i in self.noise_index:
            s = self.get_noise(i)
            g = self.theta + self.sigma * s # Genome
            f = self.evaluate(g, seed=seed)
            self.fitnesses.append(f)

        return self.fitnesses
        
    def return_info(self):
        d = {
            "fitnesses":self.fitnesses,
            "frames": int(self.frames)
        }
        return self.comm.gather(d, root=0)

    def evaluate(self, genome, seed=0, render=False, test=False):
        if seed < 0:
            seed = np.random.randint(0, 1000000000)
        seed=0
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

        self.es = None

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
        vb_rng = np.random.default_rng(seed=123)
        while len(vb) < 128:
            # Apply random action and with 1% chance save this state.
            a =  vb_rng.integers(0, self.config["n_actions"])
            obs, _, done, _ = env.step(a)
            state.update(obs)
            if done:
                env.reset()
            elif vb_rng.random() < 0.01:
                vb.append(state.get())

        self.vb = torch.stack(vb).squeeze().double()
        return self.vb

    def send_genomes(self, noise_id, hof=None, seed=0):
        # print(noise_id)
        assert self.es is not None
        if self.size == 0:
            self.theta = self.es.theta
            self.sigma = self.es.sigma
            self.noise_index = noise_id

        else:
            d = {
                "seed":seed, 
                "n_genes":self.es.n_genes, 
                "stop":False
                }

            self.comm.bcast(d, root=0) # Send eval info 

            # Send theta, sigma
            self.theta = self.comm.bcast(self.es.theta, root=0)
            self.sigma = self.comm.bcast(self.es.sigma, root=0)

            # Split 
            split = np.array_split(noise_id, self.size)
            split = [[]] + split # Add empty list at the beginning
            self.noise_index = self.comm.scatter(split, root=0)

        self.n_genes = self.es.n_genes
        self.run_evaluations(seed=seed)

        if hof is not None:
            g = np.float64(hof.genes) # genome
            hof.fitness = self.evaluate(g, seed=seed)
        
        self.fitnesses = {}
        
        # List of dict {"fitnesses": list, "frames": int}
        results = self.return_info()

        # get all fitnesses into 1 array
        fitnesses = [i for d in results for i in d["fitnesses"]]
        self.total_frames = np.sum([d["frames"] for d in results])
        # print(fitnesses)
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