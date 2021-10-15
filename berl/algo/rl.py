import warnings
from abc import abstractmethod

import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

from ..algo.optim import *
from ..env.env import *
from ..utils.logger import *
from ..utils.models import *
from ..utils.state import *
from .agents import *

warnings.simplefilter("ignore")

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

        self.rng = np.random.default_rng(self.config["seed"])
        
        self.MPINode = Primary(Net, config)

        self.populate()

        self.logger = Logger()
        self.logger.add_list(
            ["fitness", "total frames", "evaluations", "gradient steps", 
            "score"]
        )

        self.save_path = save_path
        self.wandb_run = None

    def __repr__(self): # pragma: no cover
        s = f"{self.env} => RL"
        return s

    def __str__(self): # pragma: no cover
        return self.__repr__()

    def progress(self):
        # \u03B5 = epsilon
        fit = np.mean(self.logger["fitness"][-100:]) if len(self.logger["fitness"])>0 else "\u2205"
        frames = self.logger.last("total frames") 
        return f"{self.__class__.__name__} | Last 100={fit} | Frames={frames}"

    @abstractmethod
    def populate(self):# pragma: no cover
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
    
    def close_wandb(self): 
        if self.wandb_run is not None: # pragma: no cover
            self.wandb_run.finish()
            self.wandb_run = None

    def close_MPI(self):
        self.MPINode.stop()

    @abstractmethod
    def save(self): # pragma: no cover
        pass

    @abstractmethod
    def step(self):  # pragma: no cover
        pass

    def train(self, steps=None):
        if steps is None:
            steps = self.config["gen"]
        pbar = tqdm(range(steps))  
        try:
            for i in pbar:
                self.step()
                self.log()
                # self.save()
                pbar.set_description(self.progress())
                stop, stop_msg = self.stop()
                if stop:
                    print(stop_msg)
                    break
        except KeyboardInterrupt:
            print("Interrupted")
        finally:
            # self.close_MPI()
            self.close_wandb()

    def stop(self):
        if ( self.logger.last("total frames") or 0 ) >= self.config["max_frames"]:
            return True, "Termination: total frames"
        if ( self.logger.last("evaluations") or 0 ) >= self.config["max_evals"]:
            return True, "Termination: max evaluations"
        return False, ""
    
    def plot(self):
        x = self.logger["total frames"]
        y = self.logger["fitness"]
        plt.plot(x, y)
        plt.title(str(self))
        plt.xlabel("Frames")
        plt.ylabel("Fitness")
        plt.show()
