from .rl import *

class DQN(RL):
    def __init__(self, Net, config):
        super().__init__(Net, config)
        self.epsilon = 1

        observation_dtype=np.float64
        if len(self.obs_shape) > 2 and self.env_name not in MINATAR_ENVS:
            observation_dtype=np.uint8

        self.replay_buffer = circular_replay_buffer.OutOfGraphReplayBuffer(
            observation_shape=self.obs_shape,
            observation_dtype=observation_dtype,
            batch_size=self.batch_size,
            stack_size=self.stack_frames,
            replay_capacity=self.buffer_size,
            gamma=self.gamma,
            update_horizon=self.update_horizon)

    def populate(self):
        self.agents = [self.make_agent() for i in range(self.config["pop"])]

    def train(self, episodes):
        raise NotImplementedError