from mpi4py import MPI
from mpi4py.MPI import ANY_SOURCE
import numpy as np
from ...env.env import make_env
from .rl_agent import Agent, State, FrameStackState
from .c51_agent import C51Agent
import torch
import gym


def flush(msg):
    sys.stdout.write(msg)
    sys.stdout.flush()


SERVER_NODE = 0


class Secondary:
    def __init__(self, Net, config):
        self.Net = Net
        self.config = config

        env = make_env(config["env"])
        self.config["obs_shape"] = env.observation_space.shape
        self.discrete = isinstance(env.action_space, gym.spaces.Discrete)
        env.close()

        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

        self.theta = None
        self.sigma = None
        self.n_genes = None
        noise_size = (10**(config["noise_size"]))
        self.noise = np.random.RandomState(123)\
            .randn(int(noise_size))\
            .astype('float32')

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
        return -1 * self.noise[key:(key+self.n_genes)]

    def get_n_out(self):
        model = self.Net(c51=self.config["c51"])
        mod = list(model._modules.values())
        n_out = mod[-1].out_features
        self.config["n_actions"] = \
            int(n_out/51) if self.config["c51"] else n_out

        env = make_env(self.config["env"])
        self.config["obs_shape"] = env.observation_space.shape
        env.close()

    # def run(self):
    #     self.vb = self.comm.bcast(None, root=0)
    #     # try:
    #     while self.keep_running:
    #         d = self.comm.bcast(None, root=0)
    #         if d["stop"]:
    #             print(f"{self}: Stop signal received")
    #             self.keep_running = False
    #             return
    #         self.n_genes = d["n_genes"]
    #         self.theta = self.comm.bcast(None, root=0)
    #         self.sigma = self.comm.bcast(None, root=0)
    #         self.noise_index = self.comm.scatter(None, root=0)

    #         self.run_evaluations(seed=d["seed"])
    #         self.return_info()
    #     # except KeyboardInterrupt:
    #     #     print("Interrupted")

    def run(self):
        self.vb = self.comm.bcast(None, root=0)

        output = {
            "data": None,
            "rank": self.rank
        }
        # Send a message to the master
        self.comm.send(output, dest=0)
        while True:
            # Wait for a message from the master
            incoming = self.comm.recv(source=0)
            if incoming["stop"]:
                break

            self.n_genes = incoming["data"]["n_genes"]
            self.theta = incoming["data"]["theta"]
            self.sigma = incoming["data"]["sigma"]
            seed = incoming["data"]["seed"]

            if "noise_index" in incoming["data"]:
                self.noise_index = incoming["data"]["noise_index"]

                # Do some work
                s = self.get_noise(self.noise_index)
                # Genome
                g = self.theta + self.sigma * s
            elif "genome" in incoming["data"]:
                g = incoming["data"]["genome"]
            else:
                raise ValueError("No genome or noise index")

            f = self.evaluate(g, seed=seed)

            output["data"] = {
                "fitness": f,
                "index": incoming["data"]["index"],
                "node frames": self.frames
            }
            # Send the result back to the master
            self.comm.send(output, dest=0)

    # def run_evaluations(self, seed=0):
    #     self.fitnesses = []
    #     for i in self.noise_index:
    #         s = self.get_noise(i)
    #         # Genome
    #         g = self.theta + self.sigma * s
    #         f = self.evaluate(g, seed=seed)
    #         self.fitnesses.append(f)

    #     return self.fitnesses

    # def return_info(self):
    #     d = {
    #         "fitnesses": self.fitnesses,
    #         "frames": int(self.frames)
    #     }
    #     return self.comm.gather(d, root=0)

    def evaluate(self, genome, seed=0, render=False, test=False, count_frames=True):
        rewards = []
        for _ in range(self.config["n_evaluations"]):
            r = self.single_evaluate(
                genome, seed=seed, render=render, test=test, count_frames=count_frames)
            rewards.append(r)
        return np.mean(rewards)

    def single_evaluate(self, genome, seed=0, render=False, test=False, count_frames=True):
        if seed < 0:
            seed = np.random.randint(0, 1000000000)

        env = make_env(self.config["env"], seed=seed, render=render)
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
                action = agent.act(
                    obs) if self.discrete else agent.continuous_act(obs)
                obs, r, done, _ = env.step(action)

                if self.config["reward_clip"] > 0:
                    r = max(
                        min(
                            r, self.config["reward_clip"]
                        ), -self.config["reward_clip"]
                    )

                if render:
                    env.render()

                total_r += r
                n_frames += 1

        finally:
            env.close()
        if count_frames:
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
        assert self.rank == SERVER_NODE, f"Server must be rank {SERVER_NODE}"

        # self.frames = 0

        self.get_vb()
        self.comm.bcast(self.vb, root=0)

        self.waitings = []

        self.es = None

        self.node_frames = [0 for _ in range(self.size)]

    @property
    def total_frames(self):
        return sum(self.node_frames)

    def __repr__(self):
        return f"Primary ({self.size})"

    def genome_vb(self, genome):
        agent = self.make_agent(genome)

        # Virtual batch normalization
        agent.model(self.vb)

    def get_vb(self):
        if self.n_out is None:
            self.get_n_out()

        env = make_env(self.config["env"])

        # State
        if self.config["stack_frames"] > 1:
            state = FrameStackState(
                self.config["obs_shape"],
                self.config["stack_frames"]
            )
        else:
            state = State()

        vb = []
        env.reset()
        vb_rng = np.random.default_rng(seed=123)
        vb_size = self.config["vbn"]
        while len(vb) < 128:
            # Apply random action and with 1% chance save this state.
            a = vb_rng.integers(0, self.config["n_actions"])
            obs, _, done, _ = env.step(a)
            state.update(obs)
            if done:
                env.reset()
            elif vb_rng.random() < 0.01:
                vb.append(state.get())

        self.vb = torch.stack(vb).squeeze().double()
        return self.vb

    def send_genomes(self, noise_id, hof=None, seed=-1):
        fit_to_compute = len(noise_id)
        noise_id = noise_id.tolist()
        # print(noise_id)
        assert self.es is not None
        self.theta = self.es.theta
        self.sigma = self.es.sigma
        self.n_genes = self.es.n_genes
        if self.size == 1:
            fitnesses = [None for _ in noise_id]

            self.noise_index = noise_id

            for i in range(len(noise_id)):
                # Do some work
                s = self.get_noise(noise_id[i])
                # Genome
                g = self.theta + self.sigma * s
                fitnesses[i] = self.evaluate(g, seed=seed)

                if hof is not None:
                    g = np.float64(hof.genes)  # genome
                    hof.fitness = self.evaluate(g, seed=seed)

            return fitnesses

        if hof is not None:
            noise_id.append(hof.genes)

        # print(noise_id)
        to_complete = len(noise_id)
        fitnesses = [None for _ in noise_id]

        # Send to waiting clients
        index = 0
        for i in self.waitings:
            # Send new agent to evaluate
            d = {
                "data": {
                    "n_genes": self.n_genes,
                    "index": index,
                    "seed": seed,
                    "sigma": self.sigma,
                    "theta": self.theta
                },
                "stop": False,
            }

            x = noise_id[index]
            if isinstance(x, int):
                d["data"]["noise_index"] = x
            elif isinstance(x, np.ndarray):
                d["data"]["genome"] = x
            else:
                print(x)
                raise ValueError(f"Unknown noise_id type {type(x)}")

            self.comm.send(d, dest=i)
            index += 1
            if index == len(noise_id):
                break

        self.waitings = self.waitings[index:]

        while to_complete > 0:
            msg = self.comm.recv(source=ANY_SOURCE)
            if msg == "stop":
                break

            if msg["data"] is not None:
                agent_index = msg["data"]["index"]
                fitnesses[agent_index] = msg["data"]["fitness"]
                to_complete -= 1
                self.node_frames[msg["rank"]] = msg["data"]["node frames"]

            if index < len(noise_id):
                # Send new agent to evaluate
                d = {
                    "data": {
                        "n_genes": self.n_genes,
                        "index": index,
                        "seed": seed,
                        "sigma": self.sigma,
                        "theta": self.theta
                    },
                    "stop": False,
                }

                x = noise_id[index]
                if isinstance(x, int):
                    d["data"]["noise_index"] = x
                elif isinstance(x, np.ndarray):
                    d["data"]["genome"] = x
                else:
                    print(x)
                    raise ValueError(f"Unknown noise_id type {type(x)}")

                self.comm.send(d, dest=msg["rank"])
                index += 1
            else:
                self.waitings.append(msg["rank"])

        if hof is not None:
            hof.fitness = fitnesses.pop(-1)
            # print("HOF", hof.fitness)

        assert len(
            fitnesses) == fit_to_compute, f"{len(fitnesses)} fitnesses for {fit_to_compute} ids"

        return fitnesses

##############################################################################

        # self.run_evaluations(seed=seed)

        # if hof is not None:
        #     g = np.float64(hof.genes)  # genome
        #     hof.fitness = self.evaluate(g, seed=seed)

        # self.fitnesses = {}

        # # List of dict {"fitnesses": list, "frames": int}
        # results = self.return_info()

        # # get all fitnesses into 1 array
        # fitnesses = [i for d in results for i in d["fitnesses"]]
        # self.total_frames = np.sum([d["frames"] for d in results])
        # # print(fitnesses)
        # return fitnesses

    def stop(self):
        d = {
            "data": None,
            "stop": True,
        }
        for i in range(self.comm.Get_size()):
            if i != SERVER_NODE:
                self.comm.send(d, dest=i)

    def eval_elite(self, elite, seed=-1, n=10):
        pop = [elite for _ in range(n)]
        return self.evaluate_all(pop, seed=seed, count_frames=False)

    def evaluate_all(self, pop, hof=None, seed=0, count_frames=True):
        f = [self.evaluate(np.float64(g), seed=seed,
                           count_frames=count_frames) for g in pop]

        if hof is not None:
            g = hof.genes  # genome
            g = np.float64(g)
            hof.fitness = self.evaluate(
                g, seed=seed, count_frames=count_frames)

        return f
