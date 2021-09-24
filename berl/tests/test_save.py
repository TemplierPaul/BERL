from berl import *
import berl
import numpy as np
import types
import torch
import os

cfg = {
    "pop":8,
    "seed":0,
    "SGD":"Adam",
    "lr": 0.001,
    "V_min":-1,
    "V_max":1,
    "atoms":51,
    "c51":False
}

def test_save():
    cfg['stack_frames']=1

    game = "CartPole-v1"
    Net = gym_flat_net(game)

    pop = Population(Net, cfg)

    path = "./saves/test"
    pop.save(path)

    models = glob(path + "/*.pth")
    assert len(models) > 0, f"No models found in path {path}"

    cfg = glob(path + "/config.json")
    assert len(cfg) == 1, f"{len(cfg)} config files found in path {path}"

    new_pop = Population(Net, cfg)
    new_pop.load_models(path)

    assert 
