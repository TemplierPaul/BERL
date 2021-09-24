import torch
import numpy as np
from .agents.rl_agent import *
from ..env.env import *
from ..algo.optim import *
from ..utils import *
from abc import abstractmethod

try: 
    import wandb
    use_wandb = True
except:
    use_wandb = False
    print("No WANDB")

class RL:
    def __init__(self, Net, config, save_path=None):
        self.config = config
        self.Net = Net

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.env = None
        self.rng = np.random.default_rng(self.config["seed"])
        self.get_env_shape()

        self.agents = None
        self.target = None

        self.populate()

        self.logger = Logger()
        self.logger.add_list(
            ["fitness", "total frames", "evaluations", "gradient steps", 
            "score"]
        )

        self.save_path = save_path
        self.wandb_run = None

    def __repr__(self): # pragma: no cover
        s = f"{self.env_name} => RL"
        return s

    def __str__(self): # pragma: no cover
        return self.__repr__()

    def get_env_shape(self):
        self.make_env(n=1)
        self.config["obs_shape"]=self.env.observation_space.shape
        self.config["n_actions"]=self.env.action_space.n
        self.close_env()

    def make_env(self, n=1, seed=0):
        self.env = make_vect_env(env_id=self.config["env_name"], n=n, seed=seed)
        return self.env

    def close_env(self):
        if self.env is not None:
            self.env.close()
            self.env=None
    
    def progress(self):
        # \u03B5 = epsilon
        fit = np.mean(self.logger["fitness"][-100:]) if len(self.logger["fitness"])>0 else "\u2205"
        frames = self.logger.last("total frames") 
        return f"{self.__class__.__name__} | Last 100={fit} | Frames={frames}"

    def make_agent(self, genes=None):
        i = Agent(self.Net, self.config)
        return i

    @abstractmethod
    def populate(self):# pragma: no cover
        pass

    @abstractmethod
    def train(self):# pragma: no cover
        pass

    def get_config(self):
        d = {
            "algo":"neuroevo"
        }
        return {**d, **(self.config)}

    def log(self):
        d = {
                "generations": self.optim.gen,
            }
        l = self.logger.export()
        d = {**d, **l}
        if self.wandb_run is not None: # pragma: no cover
            wandb.log(d)
        return d

    def set_wandb(self, project): # pragma: no cover
        self.wandb_run = wandb.init(
            project=project,
            config=self.get_config()
        )
        print("wandb run:", wandb.run.name)
    
    def close_wandb(self): # pragma: no cover
        self.wandb_run.finish()
        self.wandb_run = None

    @abstractmethod
    def save(self): # pragma: no cover
        pass