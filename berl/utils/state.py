import torch
import numpy as np
import matplotlib.pyplot as plt

class State():
    def __init__(self):
        self.state = None 
        
    def update(self, obs):
        self.state = obs
        
    def reset(self):
        self.state=None
        
    def __repr__(self): # pragma: no cover
        if self.state is not None:
            return f'State {self.state.shape} | {self.state.dtype}'
        else:
            return "State: Reset"
        
    def __str__(self): # pragma: no cover
        return self.__repr__()

    def get(self, as_tensor=True):
        if self.state is None:
            return None
        if as_tensor:
            return torch.tensor(self.state).double().unsqueeze(0)
        else:
            return self.state


class FrameStackState(State):
    def __init__(self, obs_shape, n_frames=4):
        self.n_frames = n_frames
        self.state = None
        self.shape = [n_frames] + list(obs_shape)
        self.reset()

    def __repr__(self): # pragma: no cover
        if self.state is not None:
            return f'Stacked state ({self.n_frames}) {self.state.shape} | {self.state.dtype}'

        else:
            return "Stacked state ({self.n_frames}): Reset"

    def update(self, obs):
        self.state = np.roll(self.state, -1, axis=0)
        self.state[-1] = obs
        
    def reset(self):
        self.state = np.zeros(self.shape)
        
    def get(self, as_tensor=True):
        if self.state is None:
            return None
        if as_tensor:
            x = torch.from_numpy(self.state).to(torch.uint8)
            x = torch.swapaxes(x, 0, 1)
            return x / 255
        else:
            return self.state

    def plot(self): # pragma: no cover
        plt.figure(figsize=(10, 10))
        for i in range(4):
            plt.subplot(2, 2, i+1)
            plt.imshow(self.state[i, 0, :, :], cmap="gray")
        plt.show()
