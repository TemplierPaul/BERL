from berl import *
import berl
import numpy as np

def single_test(game, obs_shape):
    env_func = make_env(game)
    assert str(type(env_func)) == "<class 'function'>"

    env = env_func()
    assert env.observation_space.shape == obs_shape
    obs = env.reset()
    assert isinstance(obs, (list, np.ndarray)) 
    assert obs.shape==obs_shape
    
    action = 0
    assert isinstance(action, int)
    obs, r, d, _ = env.step(action)
    assert isinstance(obs, (list, np.ndarray)) 
    assert obs.shape==obs_shape
    assert isinstance(r, (int, float)) 
    assert isinstance(d, (bool, np.bool_)) 

    env.close()

def vect_test(game, obs_shape):
    n=3
    env = make_vect_env(game, n=n)

    if len(obs_shape) > 2:
        assert type(env) ==  Fixed_VecTransposeImage
    else:
        assert type(env) ==  NoReset_SubprocVecEnv

    assert env.observation_space.shape == obs_shape
    obs = env.reset()
    assert isinstance(obs, (list, np.ndarray)) 
    assert isinstance(obs[0], (list, np.ndarray)) 
    assert len(obs) == n
    assert obs[0].shape==obs_shape
    
    # Action
    action = [0 for _ in range(n)]
    assert isinstance(obs, (list, np.ndarray)) 
    assert len(action) == n

    obs, r, d, _ = env.step(action)

    # Obs
    assert isinstance(obs, (list, np.ndarray)) 
    assert isinstance(obs[0], (list, np.ndarray)) 
    assert len(obs) == n
    assert obs[0].shape==obs_shape

    # Reward
    assert isinstance(r, (list, np.ndarray)) 
    assert len(r) == n
    assert isinstance(r[0], (int, float, np.float64, np.int64)) 
    # Done
    assert isinstance(d, (list, np.ndarray)) 
    assert len(d) == n
    assert isinstance(d[0], (bool, np.bool_)) 

    env.close()


def test_gym():
    game = "CartPole-v1"

    # 1 env
    single_test(game, (4, ))

    # Vectorized env
    vect_test(game, (4, ))

    game = "custommc"

    # 1 env
    single_test(game, (2, ))

    # Vectorized env
    vect_test(game, (2, ))

    game = "swingup"

    # 1 env
    single_test(game, (4, ))

    # Vectorized env
    vect_test(game, (4, ))


def test_minatar():
    game = "min-breakout"

    # 1 env
    single_test(game, (10, 10, 4))

    # CxHxW for Pytorch image processing

    # Vectorized env
    vect_test(game, (4, 10, 10))


def test_atari():
    game = "Pong-v0"

    # 1 env
    single_test(game, (84, 84, 1))

    # CxHxW for Pytorch image processing

    # Vectorized env
    vect_test(game, (1, 84, 84))

def test_brax():
    raise NotImplementedError
