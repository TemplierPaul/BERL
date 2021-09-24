import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import numpy as np
import gym

from collections import namedtuple

from ..env import MinatarEnv, CartPoleSwingUp, CustomMountainCarEnv

def get_genome_size(Net, c51=False):
    model = Net(c51=c51)
    with torch.no_grad():
        params = model.parameters()
        vec = torch.nn.utils.parameters_to_vector(params)
    return len(vec.cpu().numpy())

# Flat net for Atari RAM and gym envs

class FFNet(nn.Module):
    def __init__(self, n_in, h_size, n_out):
        super().__init__()
        self.fc1 = nn.Linear(n_in, h_size)
        self.fc2 = nn.Linear(h_size, h_size)
        self.fc3 = nn.Linear(h_size, n_out)

        self.n_out=n_out

    def forward(self, x, debug=False):
        # x = torch.Tensor(x)
        if debug: 
            print("0:", (x != x).sum(), x.max())
            print(x)
        x = self.fc1(x)
        if debug: 
            print("1:", (x != x).sum(), x.max())
            params = self.fc1.parameters()
            params = torch.nn.utils.parameters_to_vector(params)
            nans=(params != params).sum()
            print("NaNs: ", nans)
            print(x)
        x = F.relu(x)
        # if debug: 
        #     print("1 ReLU:", (x != x).sum(), x.max())
        #     print(x)
        x = self.fc2(x)
        if debug: 
            print("2:", (x != x).sum(), x.max())
            params = self.fc2.parameters()
            params = torch.nn.utils.parameters_to_vector(params)
            nans=(params != params).sum()
            print("NaNs: ", nans)
            print(x)
        x = F.relu(x)
        # if debug: 
        #     print("2 ReLU:", (x != x).sum(), x.max())
        #     print(x)
        x = self.fc3(x)
        if debug: 
            print("3:", (x != x).sum(), x.max())
            params = self.fc3.parameters()
            params = torch.nn.utils.parameters_to_vector(params)
            nans=(params != params).sum()
            print("NaNs: ", nans)
            print(x)
        return x
    
def gym_flat_net(env_name, h_size=64):
    if env_name.lower() == "swingup": 
        env = CartPoleSwingUp()
    elif env_name.lower() == "custommc": 
        env = CustomMountainCarEnv()
    else:
        env=gym.make(env_name)
    n_out = env.action_space.n
    n_in = env.observation_space.shape[0]
    env.close()	
    def wrapped(c51=False):
        if c51:
            return FFNet(n_in, h_size, n_out*51)
        else:
            return FFNet(n_in, h_size, n_out)
    return wrapped

## Atari image

class ConvNet(nn.Module):
    def __init__(self, h_size, n_out, stacks=4):
        super().__init__()
        self.conv1 = nn.Conv2d(stacks, 32, 8, stride=4, padding=0)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2, padding=0)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1, padding=0)
        self.conv_output_size = 3136
        self.fc1 = nn.Linear(self.conv_output_size, h_size)
        self.fc2 = nn.Linear(h_size, n_out)

        self.n_out=n_out

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, self.conv_output_size)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def gym_conv(env_name, h_size=512):
    env=gym.make(env_name)
    n_out = env.action_space.n
    env.close()
    def wrapped(c51=False):
        if c51:
            return ConvNet(h_size, n_out*51)
        else:
            return ConvNet(h_size, n_out)
    return wrapped

## Atari: Data efficient network for Atari Image

class DataEfficientConvNet(nn.Module):
    def __init__(self, h_size, n_out, stacks=4):
        super().__init__()
        self.conv1 = nn.Conv2d(stacks, 32, 5, stride=5, padding=0)
        self.conv2 = nn.Conv2d(32, 64, 5, stride=5, padding=0)
        self.conv_output_size = 576
        self.fc1 = nn.Linear(self.conv_output_size, h_size)
        self.fc2 = nn.Linear(h_size, n_out)

        self.n_out=n_out

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, self.conv_output_size)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def gym_conv_efficient(env_name, h_size=256):
    env=gym.make(env_name)
    n_out = env.action_space.n
    env.close()
    def wrapped(c51=False):
        if c51:
            return DataEfficientConvNet(h_size, n_out*51)
        else:
            return DataEfficientConvNet(h_size, n_out)
    return wrapped

## Minatar

class MinatarNet(nn.Module):
    def __init__(self, in_channels, num_actions):

        super(MinatarNet, self).__init__()

        # One hidden 2D convolution layer:
        #   in_channels: variable
        #   out_channels: 16
        #   kernel_size: 3 of a 3x3 filter matrix
        #   stride: 1
        self.conv = nn.Conv2d(in_channels, 16, kernel_size=3, stride=1)

        # Final fully connected hidden layer:
        #   the number of linear unit depends on the output of the conv
        #   the output consist 128 rectified units
        def size_linear_unit(size, kernel_size=3, stride=1):
            return (size - (kernel_size - 1) - 1) // stride + 1
        num_linear_units = size_linear_unit(10) * size_linear_unit(10) * 16
        self.fc_hidden = nn.Linear(in_features=num_linear_units, out_features=128)

        # Output layer:
        self.output = nn.Linear(in_features=128, out_features=num_actions)
        self.n_out=num_actions

    # As per implementation instructions according to pytorch, the forward function should be overwritten by all
    # subclasses
    def forward(self, x):
        # Rectified output from the first conv layer
        x = F.relu(self.conv(x))

        # Rectified output from the final hidden layer
        x = F.relu(self.fc_hidden(x.view(x.size(0), -1)))

        # Returns the output from the fully-connected linear layer
        return self.output(x)

def min_conv(env_name):
    env = MinatarEnv(env_name)
    num_actions= env.action_space.n
    in_channels = env.game.state_shape()[2]
    def wrapped(c51=False):
        if c51:
            return MinatarNet(in_channels, num_actions*51)
        else:
            return MinatarNet(in_channels, num_actions)
    return wrapped