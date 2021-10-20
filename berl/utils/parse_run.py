from .. import *
import os
import sys
from datetime import datetime
import yaml

import matplotlib.pyplot as plt

try: 
    import wandb
    use_wandb = True
except:
    use_wandb = False
    print("No WANDB")

import argparse
import numpy as np

import pathlib

path = str(pathlib.Path(__file__).parent.resolve())
path = path.split("/BERL/")[0]
presets_folder = f"{path}/BERL/berl/presets"
# print("Presets folder:", presets_folder)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description='Get DQNES arguments')

parser.add_argument('--preset', 
    help='Parameters preset: overrides defaults, but is overridden command args',
    nargs='+',
    default=None)


ALGOS = {
    "neuroevo":NeuroEvo
}

NETWORKS = {
    "flat":gym_flat_net,
    "conv":gym_conv,
    "efficientconv": gym_conv_efficient,
    "canonical":gym_canonical,
    "min":min_conv
}

def load_preset(args):
    # Load default values
    default_file = open(f"{presets_folder}/_default.yaml", 'r')
    default = yaml.load(default_file)

    # Loat presets
    if args.preset is not None:
        preset_list = args.preset
        for p in preset_list:
            yaml_file = open(f"{presets_folder}/{p}.yaml", 'r')
            yaml_content = yaml.load(yaml_file)
            default = {**default, **yaml_content}

    # Override default and presets with non-None command values
    for k, v in args.__dict__.items():
        if v is not None:
            default[k] = v

    args.__dict__ = default
    return args

def set_xp(args):  
    if isinstance(args, str):
        args, unknown = parser.parse_known_args(args.split())

    args = load_preset(args)

    algo = ALGOS[args.algo.lower()]
    net = NETWORKS[args.net.lower()](args.env)

    now = str(datetime.now()).replace(":", "-").replace(" ", "_")

    if args.no_save:
        path=None
    else:
        folder = f"{args.env}_{args.algo}_{args.optim}_{now}"
        path = f"{args.save_path}/saves/{args.env}/{args.algo}/{args.optim}/{folder}"

    pb = algo(
        Net=net,
        config=args.__dict__, 
        save_path=path)

    return pb

def load_xp(path, gen=None):
    # Config
    config_path = path + "/config.json"
    cfg = glob(config_path)
    assert len(cfg) == 1, f"{len(cfg)} config files found in path {path}"
    args = argparse.ArgumentParser().parse_args("")

    with open(cfg[0], 'r') as f:
        args.__dict__ = json.load(f)
    args.preset=[]
    pb = set_xp(args)

    save_path = f"{path}/checkpoint_*.npz"
    checkpoints = glob(save_path)
    checkpoints.sort()
    assert len(checkpoints) > 0, f"No checkpoints found in path {path}"
    save_path = checkpoints[-1] if gen is None else f"./{path}/checkpoint_{gen}.npz"
    f = np.load(save_path)
    d = {k:f[k] for k in f.files}

    pb.load(d)
    return pb


def run_xp(args):
    pb = set_xp(args)
    print(pb)

    if use_wandb and args.wandb is not None and args.wandb != "None":
        print("Using wandb")
        pb.set_wandb(args.wandb)
        pb.train(args.gen)
        wandb.finish()

    else:
        print("Not using wandb")
        pb.train(args.gen)
    # try:
    #     pb.MPINode.stop()
    # except:
    #     pass
    return pb

# Get default config
args, unknown = parser.parse_known_args()
args = load_preset(args)
cfg = args.__dict__
# Add each argument in the yaml file to the parser
for k, v in cfg.items():
    if isinstance(v, str):
        parser.add_argument(f'--{k}', type=str)
    elif isinstance(v, bool):
        parser.add_argument(f'--{k}', default=None, action='store_true')
    elif isinstance(v, int):
        parser.add_argument(f'--{k}', type=int)
    elif isinstance(v, float):
        parser.add_argument(f'--{k}', type=float)
    elif isinstance(v, list):
        assert k =="preset"
    else:
        raise NotImplementedError(f"Argument error: {k} ({v})")
