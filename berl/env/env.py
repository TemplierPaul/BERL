from .atari import *
from .brax import *
from .gym import *
from .minatar import *
from .procgen import *

def is_procgen(name):
    return name.split("-")[0] in procgen.env.ENV_NAMES

def make_env(env_id, seed=None, render=False):
    """
    Utility function for multiprocessed env.
    
    :param env_id: (str) the environment ID
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """        

    #TODO Add Brax make_env

    if env_id.lower() == "swingup":
        env = CartPoleSwingUp()
        if seed is not None: env.seed(seed)
        return env

    elif env_id.lower() == "custommc":
        env = CustomMountainCarEnv()
        if seed is not None: env.seed(seed)
        return env

    # Minatar
    elif env_id in MINATAR_ENVS:
        env=MinatarEnv(env_id)
        if seed is not None: env.seed(seed)
        # Re-order channels, from HxWxC to CxHxW for Pytorch
        env = TorchTransposeWrapper(env)
        return env

    # Procgen envs
    elif is_procgen(env_id):
        env = make_procgen_env(env_id, seed=seed, render=render)
        return env
        
    # Atari envs
    else:
        env = gym.make(env_id)
        n_in = env.observation_space.shape

        # Atari pixel: 3D input
        if len(n_in)==3:
            # Atari 2600 preprocessings
            env.close()
            l = env_id.split("-")
            new_name = f"{l[0]}NoFrameskip-{l[1]}"

            env = gym.make(new_name)
            env = wrap_canonical(env)
        if seed is not None: env.seed(seed)
        return env

  
