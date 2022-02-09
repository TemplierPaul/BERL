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
    "env": "CartPole-v1",
    "stack_frames": 1,
    "reward_clip": 0,
    "max_frames": 20000,
    "max_evals": 20000,
    "max_gen": 20000,
    "eval_frames": 20,
    "eval_freq": 1,
    "episode_frames": 20,
    "es_sigma": 0.01,
    "es_lr": 1,
    "es_sigma_factor": 1,
    "es_eta_mu": 1
}


def test_neuroevo():
    game = "CartPole-v1"
    Net = gym_flat_net(game)
    es = NeuroEvo(Net, cfg)

    assert len(es.optim.theta) == 4610

    es.populate()

    es.evaluate(seed=0, clip=False)

    assert all(i.fitness is not None for i in es.agents)
    assert es.logger.last("total frames") > 0
    assert es.env is None

    es.evaluate(seed=0, clip=True)
    assert es.env is None

    es = NeuroEvo(Net, cfg)
    es.train(2)

    assert len(es.agents) == cfg["pop"]
    # Test if serializable
    json.dumps(es.get_config())

    # es.eval_hof(render=False)
