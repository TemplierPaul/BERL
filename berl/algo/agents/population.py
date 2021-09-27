from .rl_agent import *
from .c51_agent import *
from glob import glob
import os
import errno
import json

class Population:
    def __init__(self, Net, config):
        self.Net = Net
        self.config = config
        
        self.agents = []

    @property
    def genomes(self):
        return [i.genes for i in self.agents]

    @genomes.setter
    def genomes(self, new_genomes):
        self.agents = [self.make_agent(g) for g in new_genomes]

    @property
    def fitness(self):
        return [i.fitness for i in self.agents]

    @fitness.setter
    def fitness(self, fit):
        n = len(self.agents)
        assert len(fit)==n
        # print(fit)
        for i in range(n):
            self.agents[i].fitness = fit[i]
        return self

    def __repr__(self): # pragma: no cover
        s = f"Pop: {len(self)}"
        return s

    def __str__(self): # pragma: no cover
        return self.__repr__()

    def __len__(self):
        return len(self.agents)

    def __getitem__(self, key):
        return self.agents[key]

    def make_agent(self, genes=None):
        AgentType = C51Agent if self.config["c51"] else Agent
        i = AgentType(self.Net, self.config)
        if genes is not None:
            i.genes = genes
        return i

    def act(self, obs, running):
        # Make agents still running act from obs
        a = []
        for i in range(len(self.agents)):
            if running[i]:
                a.append(self.agents[i].act(obs[i]))
            else:
                a.append(0)
        return np.array(a)

    def reset(self):
        for i in self.agents:
            i.state.reset()

    def create_path(self, path):
        try:
            os.makedirs(path)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

    def save_models(self, path):
        self.create_path(path)
        
        # Config
        config_path = path + "/config.json"
        with open(config_path, 'w') as outfile:
            json.dump(self.config, outfile)

        # Models
        for k in range(len(self)):
            model_path = path + f"/model_{k}.pth"
            model = self[k].model
            torch.save(model.state_dict(), model_path)

    def load_models(self, path):
        cfg = glob(path + "/config.json")
        assert len(cfg) == 1, f"{len(cfg)} config files found in path {path}"

        models = glob(path + "/*.pth")
        assert len(models) > 0, f"No models found in path {path}"

        # Config
        with open(cfg[0], 'r') as f:
            self.config = json.load(f)

        #  Models
        self.agents = []
        for model_path in models:
            i = self.make_agent().make_network()
            i.model.load_state_dict(torch.load(model_path))
            self.agents.append(i)

        return self
    
    def has(self, elt):
        return elt in self.agents

    def get_best(self):
        best = self[0]
        assert best.fitness is not None

        for i in self.agents:
            if i.fitness > best.fitness:
                best = i
        return best
