from .rl import *
from .agents import *

class NeuroEvo(RL):
    def __init__(self, Net, config, save_path=None):
        super().__init__(Net, config, save_path)

        self.agents = Population(Net, config)
        self.hof = None

        self.optim = None
        self.set_optim(config["optim"])        


    def set_optim(self, name):
        d={
            "canonical":Canonical,
            "snes":SNES,
            "cmaes":CMAES
        }

        OPTIM = d[name.lower()]
        n_genes = get_genome_size(self.Net)
        self.optim = OPTIM(n_genes, self.config)

    def evaluate(self, max_frames=np.inf, seed=0, clip=False):
        try:
            n=len(self.agents)
            self.make_env(n=n)

            self.agents.reset()

            obs = self.env.reset()
            
            total_r = np.zeros(n)
            total_discounted_r = np.zeros(n)
            running = np.ones(n)
            
            n_frames = 0
            run_frames = 0
            # gamma = 1

            while any(running) or n_frames<=max_frames:
                x = torch.tensor(obs).unsqueeze(1)

                actions = self.agents.act(obs, running)
                self.env.step_async(actions)
                next_obs, r, done, _ = self.env.step_wait()

                if clip:
                    r = [max(min(i, self.config["reward_clip"]), -self.config["reward_clip"]) for i in r]

                n_frames += 1
                run_frames += sum(running)

                total_r += r*running
                # total_discounted_r += running * gamma * r
                running *= 1 - done.astype(int)

                # gamma *= self.gamma

            self.agents.fitness = total_r
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
        self.evaluate(
            max_frames=self.config["max_frames"],
            seed=env_seed,
            clip=self.config["reward_clip"]            
            ) # Evaluate pop
        self.get_hof() 
        self.optim.tell(self.agents.genomes, self.agents.fitness) # Optim step
        self.log()
        self.save

    def train(self, episodes):
        raise NotImplementedError

    def gen_periodic(self, n):
        """Returns true if a multiple of n (or self.config[n]) gens have been done"""
        if isinstance(n, (int, float)):
            return (self.optim.gen +1) % n == 0
        elif isinstance(n, str):
            return (self.optim.gen +1) % self.config[n] == 0

    def save(self):
        if self.gen_periodic("eval_freq"):
            self.agents.save_models(self.save_path)