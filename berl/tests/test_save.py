from berl import *
import berl
import numpy as np
import types
import torch
import os


def test_save():
    cfg = {
        "pop": 8,
        "seed": 0,
        "SGD": "Adam",
        "lr": 0.001,
        "V_min": -1,
        "V_max": 1,
        "atoms": 51,
        "c51": False,
        'stack_frames': 1
    }

    game = "CartPole-v1"
    Net = gym_flat_net(game)

    # Make pop
    pop = Population(Net, cfg)
    a = Agent(Net, cfg).make_network()
    n_genes = len(a.genes)

    n_pop = 10
    genomes = [np.random.random(n_genes) for i in range(n_pop)]
    pop.genomes = genomes

    path = "./saves/test"
    pop.save_models(path)

    models = glob(path + "/*.pth")
    assert len(models) > 0, f"No models found in path {path}"

    cfg = glob(path + "/config.json")
    assert len(cfg) == 1, f"{len(cfg)} config files found in path {path}"

    new_pop = Population(Net, cfg)
    new_pop.load_models(path)

    assert len(new_pop) == len(pop)
    for new_g in new_pop:
        assert new_g in pop
