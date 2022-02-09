from .atari import *
import procgen


def make_procgen_env(env_id, seed, render=False):
    try:
        env_type = env_id.split("-")[1]
        env_name = env_id.split("-")[0]
    except:
        env_type = "easy"
        env_name = env_id

    simple_graphics = False
    if env_type == "simple":
        env_type = "easy"
        simple_graphics = True

    if seed is None:
        seed = 0

    print("Level", seed)

    env = procgen.gym_registration.make_env(
        env_name=env_name,
        distribution_mode=env_type,
        rand_seed=seed,
        start_level=seed,
        use_monochrome_assets=simple_graphics,
        restrict_themes=simple_graphics,
        use_backgrounds=not simple_graphics,
        render=render
    )
    env = TorchTransposeWrapper(env)
    return env
