import errno
import os
import warnings
from abc import abstractmethod
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
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
import json
from dataclasses import dataclass


@dataclass
class Indiv:
    genes: np.ndarray
    fitness: float

def to_units(n):
    l = [1e9, 1e6, 1e3]
    u = ["b", "m", "k"]
    for i in range(len(l)):
        if n >= l[i]:
            return f"{n / l[i]:.1f}{u[i]}"
    return str(n)

class NeuroEvo:
    def __init__(self, Net, config, save_path=None):
        """
        Evolution strategy for neural networks.
        Args:
            Net: Neural network class
            config: dict
            save_path: str
        """
        self.config = config
        self.Net = Net

        self.rng = np.random.default_rng(self.config["seed"])

        self.MPINode = Primary(Net, self.config)
        if self.config["pop_per_cpu"] > 0 and self.MPINode.size > 0:
            self.config["pop"] = self.config["pop_per_cpu"] * self.MPINode.size
            self.MPINode.config = self.config

        self.logger = Logger()
        self.logger.add_list(
            ["fitness", "total frames", "evaluations", "sigma", "score"]
        )

        self.save_path = save_path
        self.wandb_save_name = None
        self.wandb_run = None

        self.noise_index = None
        self.fitness = None
        self.hof = None

        self.optim = None
        self.set_optim(config["optim"])

    ### Print ###
    def __repr__(self): # pragma: no cover
        s = f'{self.config["env"]} => NeuroEvo [{self.optim}]'
        return s

    def __str__(self): # pragma: no cover
        return self.__repr__()

    def progress(self):
        # \u03BB = lambda
        frames = self.logger.last("total frames") 
        fit = np.mean(self.logger["fitness"][-10:]) if len(self.logger["fitness"])>0 else "\u2205"
        return f"{self.config['env']} NeuroEvo [{self.optim}]({self.config['pop']}/{self.MPINode.size}) | Fit:{fit:.2f} | Frames:{to_units(frames)}"     

    ### Setup ###

    def set_optim(self, name):
        d={
            "canonical":Canonical,
            "snes":SNES,
            "cmaes":CMAES,
            "openai": OpenAI,
            "custom": CustomES
        }

        OPTIM = d[name.lower()]
        n_genes = get_genome_size(self.Net, c51=self.config["c51"])
        self.optim = OPTIM(n_genes, self.config)
        self.optim.noise = self.MPINode.noise # share noise with ES
        self.MPINode.es = self.optim # Share ES state with MPI node

    def close_MPI(self):
        self.MPINode.stop()

    def stop(self):
        if ( self.logger.last("total frames") or 0 ) >= self.config["max_frames"]:
            return True, "Termination: total frames"
        if ( self.logger.last("evaluations") or 0 ) >= self.config["max_evals"]:
            return True, "Termination: max evaluations"
        return False, ""

    ### Step ###

    def step(self):
        self.fitness = None
        self.noise_index = self.optim.ask() # Get list of noise indices
        env_seed = int(self.rng.integers(10000000)) # Seed
        self.hof = Indiv(self.optim.theta, 0) # Solution: center of the distribution
        self.fitness = self.MPINode.send_genomes(self.noise_index, hof=self.hof, seed=env_seed) # Evaluate
        # self.get_hof() 
        self.logger("total frames", self.MPINode.total_frames)
        self.logger("fitness", self.hof.fitness)
        self.logger("sigma", np.mean(self.optim.sigma))
        self.optim.tell(self.noise_index, self.fitness) # Optim step

    def train(self, steps=None):
        if steps is None:
            steps = self.config["gen"]
        pbar = tqdm(range(steps))
        try:
            for i in pbar:
                self.step()
                self.log()
                self.save()
                pbar.set_description(self.progress())
                stop, stop_msg = self.stop()
                if stop:
                    print(stop_msg)
                    self.save(self.save_path)
                    break
        except KeyboardInterrupt:
            print("Interrupted")
        finally:
            # self.close_MPI()
            self.close_wandb()

    def render(self, n=1, seed=-1):
        fitnesses = []
        for _ in range(n):
            f = self.MPINode.evaluate(self.hof.genes, render=True, seed=seed)
            print(f)
            fitnesses.append(f)
        print(f"Mean over {n} runs: {np.mean(fitnesses)}")

    def eval_hof(self):
        f = self.MPINode.eval_elite(self.hof.genes)
        print(f"Evaluating elite: {len(f)} evals")
        print("Fitness of elite:", np.mean(f), "\nstd:", np.std(f))
        if self.wandb_run is not None: # pragma: no cover
            wandb.log({
                "final fitness average": np.mean(f),
                "final fitness average": np.std(f)
                })

    # def get_hof(self):
    #     assert self.fitness is not None
    #     best_index = np.argmax(self.fitness)
    #     best_fit = self.fitness[best_index]

    #     if self.hof is None or self.hof.fitness < best_fit:
    #         best_genes = self.noise_index[best_index]
    #         self.hof = Indiv(best_genes, best_fit)
    
    ### Logging ### 
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
        self.wandb_save_name = f'{self.config["env"]}_{self.config["optim"]}_{wandb.run.id}'
    
    def close_wandb(self): 
        if self.wandb_run is not None: # pragma: no cover
            self.wandb_run.finish()
            self.wandb_run = None

    def plot(self):
        x = self.logger["total frames"]
        y = self.logger["fitness"]
        plt.plot(x, y)
        plt.title(str(self))
        plt.xlabel("Frames")
        plt.ylabel("Fitness")
        plt.show()    
    
    ### Save ###
    
    def create_path(self, path):
        try:
            os.makedirs(path)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
    def gen_periodic(self, n):
        """Returns true if a multiple of n (or self.config[n]) gens have been done"""
        if isinstance(n, (int, float)):
            return (self.optim.gen +1) % n == 0
        elif isinstance(n, str):
            return (self.optim.gen +1) % self.config[n] == 0

    def save(self, path=None):
        if self.gen_periodic("save_freq") and path is None:
            path = self.save_path

        if path is not None:
            self.create_path(path)
            # Save config
            config_path = path + "/config.json"
            with open(config_path, 'w') as outfile:
                json.dump(self.config, outfile)

            # Save ES + hof
            save_path = f"{path}/checkpoint_{self.optim.gen}.npz"
            d = self.optim.export()
            d["hof_genes"] = self.hof.genes
            d["hof_fit"] = self.hof.fitness
            np.savez_compressed(save_path, **d)
            
            print(f"Saved at {save_path}")

            # Save Virtual batch
            vb = None
            if self.MPINode is not None:
                vb = self.MPINode.vb
                if vb is not None:
                    vb_path = f"{path}/vb.npz"
                    # vb is a torchg tensor, save it as a numpy array
                    vb = vb.numpy()
                    with open(vb_path, 'wb') as f:
                        np.save(f, vb)
                    print(f"Saved virtual batch at {vb_path}")

            if self.wandb_run is not None:
                wandb_path = f"{path}/{self.wandb_run.id}.npy"
                genome = self.hof.genes
                with open(wandb_path, 'wb') as f:
                    np.save(f, genome)

                artifact = wandb.Artifact(
                    self.wandb_save_name + "_net", 
                    type = self.config["env"],
                    metadata = self.config                    
                    )
                artifact.add_file(wandb_path)
                self.wandb_run.log_artifact(artifact)

                # Add vb artifact
                if vb is not None:
                    artifact = wandb.Artifact(
                        self.wandb_save_name + "_vb", 
                        type = self.config["env"],
                        metadata = self.config                    
                        )
                    artifact.add_file(vb_path)
                    self.wandb_run.log_artifact(artifact)

    def load(self, d):
        self.hof = Indiv(d["hof_genes"], d["hof_fit"])
        self.optim.load(d)


    