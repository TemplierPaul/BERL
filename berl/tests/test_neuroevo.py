from berl import *
import berl
import numpy as np
import types
import torch

cfg = {
    "pop": 8,
    "SGD": "Adam",
    "lr": 0.001,
    "V_min": -1,
    "V_max": 1,
    "atoms": 51,
    "c51": False,
    "seed": 0,
    "optim": "snes",
    "env_name": "CartPole-v1",
    "stack_frames":1,
    "reward_clip":0,
    "max_frames":20
}

def test_neuroevo():
    game = "CartPole-v1"
    Net = gym_flat_net(game)
    es = NeuroEvo(Net, cfg)

    assert len(es.agents) == 0
    es.agents.genomes = es.optim.ask()
    assert len(es.agents) == cfg["pop"]

    es.evaluate(10, seed=0, clip=False)

    assert all(i.fitness is not None for i in es.agents)
    assert es.logger.last("total frames") > 0
    assert es.env is None

    es.evaluate(10, seed=0, clip=True)
    assert es.env is None

    es = NeuroEvo(Net, cfg)
    for _ in range(2):
        es.step()

    assert len(es.agents) == cfg["pop"]
    # Test if serializable
    json.dumps(es.get_config())
    