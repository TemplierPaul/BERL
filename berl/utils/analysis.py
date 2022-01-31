import torch
import torch.nn as nn
import numpy as np
import wandb
from glob import glob

from ..algo.agents import *
from ..algo.agents.mpi import *
from .models import NETWORKS
from ..env.env import make_env


# Load data from a wandb project
class WandBProject:
    def __init__(self, project, entity):
        self.project = project
        self.entity = entity
        self.api = None
        self.runs = None
        self.refresh()
        
    def refresh(self):
        # Get list of runs
        self.api = wandb.Api()
        self.runs = list(self.api.runs(f"{self.entity}/{self.project}"))
        
    def get_runs(self, filters):
        runs = [
            r
            for r in self.runs
            if all(r.config[k] == v for k, v in filters.items())
        ]

        return runs
    
    def load(self, run, version="latest"):
        if isinstance(run, int):
            run = self.runs[run]
        env_name = run.config["env"]
        algo = run.config["optim"]
        artifact_name = f"{env_name}_{algo}_{run.name}"
        artifact_path = f"{self.entity}/{self.project}/{artifact_name}:{version}"
        resume_run = wandb.init(project=self.project, entity=self.entity, id=run.id, resume="must")
        artifact = resume_run.use_artifact(artifact_path, type=env_name)
        artifact_dir = artifact.download()
        cfg = artifact.metadata
        path = glob(artifact_dir+"/*")[0]
        print(f"Loading Artifact from {path}")
        with open(path, 'rb') as f:
            genome = np.load(f)
            
        return genome, resume_run


# Evaluate an agent on different environments
class Evaluator(Primary):
    def __init__(self):
        self.agent = None
        self.config = None
        self.Net = None
        self.n_out = None

    def __repr__(self):
        return f"Evaluator"

    def __str__(self):
        return self.__repr__()
        
    def load(self, config, genome):
        self.config = config
        self.Net = NETWORKS[config["net"].lower()](config["env"])
        self.agent = self.make_agent(genome)
         # Virtual batch normalization
        self.get_vb()
        self.agent.model(self.vb)
        return self
        
    def evaluate(self, env, render=False):
        self.agent.state.reset()
        try:
            obs = env.reset()
            n_frames = 0
            total_r = 0
            done = False

            while not done and n_frames < self.config["episode_frames"]:
                action = self.agent.act(obs)
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
        return total_r