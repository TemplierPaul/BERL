import numpy as np
import gym
from gym import spaces
from minatar import Environment

# Minatar wrapper
# Inspired from https://github.com/qlan3/gym-games/blob/master/gym_minatar/envs/base.py

MINATAR_ENVS = [
    "min-asterix",
    "min-breakout",
    "min-freeway",
    "min-seaquest",
    "min-space_invaders"
]


class MinatarEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, game_name, **kwargs):
        self.game_name = game_name.replace("min-", "")
        self.display_time = 50
        self.init(**kwargs)

    def init(self, **kwargs):
        self.game = Environment(env_name=self.game_name, **kwargs)
        self.action_set = self.game.env.action_map
        self.action_space = spaces.Discrete(self.game.num_actions())
        self.observation_space = spaces.Box(
            0, 255, shape=self.game.state_shape(), dtype=np.uint8)

    def step(self, action):
        reward, done = self.game.act(action)
        return (self.game.state(), reward, done, {})

    def reset(self):
        self.game.reset()
        return self.game.state()

    def seed(self, seed=None):
        self.game = Environment(env_name=self.game_name, random_seed=seed)
        return seed

    def render(self, mode='human'):  # pragma: no cover
        if mode == 'rgb_array':
            return self.game.state()
        elif mode == 'human':
            self.game.display_state(self.display_time)

    def close(self):
        if self.game.visualized:  # pragma: no cover
            self.game.close_display()
        return 0

    def get_action_meanings(self):
        return ["NOOP"]
