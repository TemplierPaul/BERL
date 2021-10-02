from .rl import *
from .agents import *

class NeuroEvo(RL):
    def __init__(self, Net, config, save_path=None):
        super().__init__(Net, config, save_path)

        self.agents = Population(Net, config)
        self.hof = None

        self.optim = None
        self.set_optim(config["optim"])   

    def __repr__(self): # pragma: no cover
        s = f'{self.env} => NeuroEvo ({self.optim.__class__.__name__})'
        return s

    def __str__(self): # pragma: no cover
        return self.__repr__()

    def progress(self):
        # \u03BB = lambda
        return f"NeuroEvo | Max={self.hof.fitness}"     


    def set_optim(self, name):
        d={
            "canonical":Canonical,
            "snes":SNES,
            "cmaes":CMAES
        }

        OPTIM = d[name.lower()]
        n_genes = get_genome_size(self.Net, c51=self.config["c51"])
        self.optim = OPTIM(n_genes, self.config)

    def evaluate(self, pop=None, seed=0, clip=False):
        if pop is None:
            pop = self.agents
        try:
            n=len(pop)
            self.make_env(n=n)

            pop.reset()

            obs = self.env.reset()

            total_r = np.zeros(n)
            total_discounted_r = np.zeros(n)
            running = np.ones(n)
            
            n_frames = 0
            run_frames = 0

            while any(running) and n_frames < self.config["episode_frames"]:
                obs = torch.tensor(obs).unsqueeze(1)

                actions = pop.act(obs, running)
                self.env.step_async(actions)
                obs, r, done, _ = self.env.step_wait()

                if clip:
                    r = [max(min(i, self.config["reward_clip"]), -self.config["reward_clip"]) for i in r]

                n_frames += 1
                run_frames += sum(running)

                total_r += r*running
                # total_discounted_r += running * gamma * r
                running *= 1 - done.astype(int)

                # gamma *= self.gamma

            pop.fitness = total_r
        finally:
            self.close_env()

        # Count total frames
        prev_frames = self.logger.last("total frames") or 0
        self.logger("total frames", prev_frames + n_frames)
        return self
    
    def get_hof(self):
        best = self.agents.get_best()
        if self.hof is None or self.hof.fitness < best.fitness:
            self.hof = best
                
    def step(self):
        self.agents.genomes = self.optim.ask() # Get genomes
        env_seed = int(self.rng.integers(10000000))
        for subpop in self.agents.split(n_workers=self.config["n_workers"]):
            self.evaluate(
                pop=subpop,
                seed=env_seed,
                clip=self.config["reward_clip"]            
                ) # Evaluate pop
        self.get_hof() 
        self.optim.tell(self.agents.genomes, self.agents.fitness) # Optim step

    def gen_periodic(self, n):
        """Returns true if a multiple of n (or self.config[n]) gens have been done"""
        if isinstance(n, (int, float)):
            return (self.optim.gen +1) % n == 0
        elif isinstance(n, str):
            return (self.optim.gen +1) % self.config[n] == 0

    def save(self):
        if self.gen_periodic("eval_freq") and self.save_path is not None:
            self.agents.save_models(self.save_path)

    def run(self, indiv=None, render=False):
        if indiv is None:
            indiv = self.hof
        self.close_env()
        env = self.make_env(n=1)
        try:
            obs = env.reset() 
            total_r = 0
            t = 0
            while True:
                # obs = torch.unsqueeze(torch.tensor(obs), 1)

                results = [self.eval_indiv(indiv, s[0], True)]
                actions = np.array(results)
                self.env.step_async(actions)
                obs, r, done, _ = self.env.step_wait()

                total_r += r
                env.render()
                t += 1
                if done or t == max_frames:
                    break
            print(f"Stopped after {t} steps")
        finally:
            env.close()
        return total_r