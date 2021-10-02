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

parser.add_argument('--env')
parser.add_argument('--optim')
parser.add_argument('--algo')
parser.add_argument('--net')
parser.add_argument('--gen', type=int)
parser.add_argument('--pop', type=int)
parser.add_argument('--seed', type=int)
parser.add_argument('--max_frames', type=int)
parser.add_argument('--max_evals', type=int)
parser.add_argument('--frames', type=int, dest="episode_frames")
parser.add_argument('--stack', type=int, dest="stack_frames")
parser.add_argument('--reward_clip', type=int)

parser.add_argument('--plot', default=None, action="store_true")
parser.add_argument('--wandb')
parser.add_argument('--no_save', default=None, action="store_true")
parser.add_argument('--save_freq', type=int)
parser.add_argument('--eval_freq', type=int)
parser.add_argument('--save_path')

parser.add_argument('--c51')
parser.add_argument('--V_min', type=int)
parser.add_argument('--V_max', type=int)
parser.add_argument('--atoms', type=int)

algos = {
    "neuroevo":NeuroEvo
}

networks = {
    "flat":gym_flat_net,
    "conv":gym_conv,
    "efficientconv": gym_conv_efficient,
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

    algo = algos[args.algo.lower()]
    net = networks[args.net.lower()](args.env)

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