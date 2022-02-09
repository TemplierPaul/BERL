from berl import *
import berl
import numpy as np


def single_test(game, obs_shape):
    env = make_env(game)
    assert env.observation_space.shape == obs_shape
    obs = env.reset()
    assert isinstance(obs, (list, np.ndarray))
    assert obs.shape == obs_shape

    action = 0
    assert isinstance(action, int)
    obs, r, d, _ = env.step(action)
    assert isinstance(obs, (list, np.ndarray))
    assert obs.shape == obs_shape
    assert isinstance(r, (int, float))
    assert isinstance(d, (bool, np.bool_))

    env.close()


def test_gym():
    game = "CartPole-v1"
    single_test(game, (4, ))

    game = "custommc"
    single_test(game, (2, ))

    game = "swingup"
    single_test(game, (4, ))


def test_minatar():
    game = "min-breakout"
    single_test(game, (4, 10, 10))


def test_atari():
    game = "Pong-v0"
    single_test(game, (1, 84, 84))


def test_brax():
    raise NotImplementedError
