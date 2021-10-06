from .atari import *
from .brax import *
from .gym import *
from .minatar import *
from .vect_env import *

def make_env(env_id, seed=None):
    """
    Utility function for multiprocessed env.
    
    :param env_id: (str) the environment ID
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def gym_init():
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
        
    def swingup_init():
        env = CartPoleSwingUp()
        if seed is not None: env.seed(seed)
        return env

    def customMC_init():
        env = CustomMountainCarEnv()
        if seed is not None: env.seed(seed)
        return env

    def minatar_init():
        env=MinatarEnv(env_id)
        if seed is not None: env.seed(seed)
        return env

    #TODO Add Brax make_env

    if env_id.lower() == "swingup":
        return swingup_init
    elif env_id.lower() == "custommc":
        return customMC_init
    elif env_id in MINATAR_ENVS:
        return minatar_init
    else:
        return gym_init

# def make_vect_env(env_id, n=1, seed=0):
#     env = NoReset_SubprocVecEnv([make_env(env_id, seed) for i in range(n)], start_method='fork')
#     if len(env.observation_space.shape) > 2:
#         # For Torch convolution: 
#         env = Fixed_VecTransposeImage(env)
#     return env