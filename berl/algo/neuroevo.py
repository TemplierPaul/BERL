from .rl import *
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

class NeuroEvo(RL):
    def __init__(self, Net, config, save_path=None):
        super().__init__(Net, config, save_path)

        if self.config["pop_per_cpu"] > 0:
            self.config["pop"] = self.config["pop_per_cpu"] * self.MPINode.size
            self.MPINode.config = self.config

        self.genomes = None
        self.fitness = None
        self.hof = None

        self.optim = None
        self.set_optim(config["optim"])

        self.logger.add("sigma")

    def __repr__(self): # pragma: no cover
        s = f'{self.config["env"]} => NeuroEvo [{self.optim}]'
        return s

    def __str__(self): # pragma: no cover
        return self.__repr__()

    def progress(self):
        # \u03BB = lambda
        frames = self.logger.last("total frames") 
        fit = np.mean(self.logger["fitness"][-10:]) if len(self.logger["fitness"])>0 else "\u2205"
        return f"NeuroEvo [{self.optim}]({self.config['pop']}/{self.MPINode.size}) | Fit:{fit:.2f} | Frames:{to_units(frames)}"     


    def set_optim(self, name):
        d={
            "canonical":Canonical,
            "snes":SNES,
            "cmaes":CMAES,
            "openai": OpenAI
        }

        OPTIM = d[name.lower()]
        n_genes = get_genome_size(self.Net, c51=self.config["c51"])
        self.optim = OPTIM(n_genes, self.config)
    
    def get_hof(self):
        assert self.fitness is not None
        best_index = np.argmax(self.fitness)
        best_fit = self.fitness[best_index]

        if self.hof is None or self.hof.fitness < best_fit:
            best_genes = self.genomes[best_index]
            self.hof = Indiv(best_genes, best_fit)
        
    
    def step(self):
        self.genomes = None
        self.fitness = None
        self.genomes = self.optim.ask() # Get genomes
        env_seed = int(self.rng.integers(10000000))
        self.hof = Indiv(self.optim.theta, 0)
        self.fitness = self.MPINode.send_genomes(self.genomes, hof=self.hof, seed=env_seed)
        self.logger("total frames", self.MPINode.total_frames)
        # self.get_hof() 
        self.logger("fitness", self.hof.fitness)
        self.logger("sigma", np.mean(self.optim.sigma))
        self.optim.tell(self.genomes, self.fitness) # Optim step

    def gen_periodic(self, n):
        """Returns true if a multiple of n (or self.config[n]) gens have been done"""
        if isinstance(n, (int, float)):
            return (self.optim.gen +1) % n == 0
        elif isinstance(n, str):
            return (self.optim.gen +1) % self.config[n] == 0

    def save(self):
        if self.gen_periodic("eval_freq") and self.save_path is not None:
            self.agents.save_models(self.save_path)

    def render(self, n=1):
        fitnesses = []
        for _ in range(n):
            f = self.MPINode.evaluate(self.hof.genes, render=True, seed=-1)
            print(f)
            fitnesses.append(f)
        print(f"Mean over {n} runs: {np.mean(fitnesses)}")

    def eval_hof(self):
        f = self.MPINode.eval_elite(self.hof.genes)
        print(f"Evaluating elite: {len(f)} evals")
        print("Fitness of elite:", np.mean(f), "\nstd:", np.std(f))

