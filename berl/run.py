# import os
# import sys
# from datetime import datetime
# import yaml

# from dqnes import *
# import matplotlib.pyplot as plt


# try: 
#     import wandb
#     use_wandb = True
# except:
#     use_wandb = False
#     print("No WANDB")

# import argparse
# import numpy as np

# # python DQNES/run_xp.py --optim=snes --algo=dqnes --c51 --net=flat --env=swingup --frames=300 --gen=100 --seed=0 --pop=8  --nu=10 --nu-type=early --nu-stochastic=True --plot

# import pathlib
# path = pathlib.Path(__file__).parent.resolve()
# presets_folder = f"{path}/presets"

# def str2bool(v):
#     if isinstance(v, bool):
#         return v
#     if v.lower() in ('yes', 'true', 't', 'y', '1'):
#         return True
#     elif v.lower() in ('no', 'false', 'f', 'n', '0'):
#         return False
#     else:
#         raise argparse.ArgumentTypeError('Boolean value expected.')

# parser = argparse.ArgumentParser(description='Get DQNES arguments')

# parser.add_argument('--preset', 
#     help='Parameters preset: overrides defaults, but is overridden command args',
#     nargs='+',
#     default=None)

# parser.add_argument('--optim', help='optimizer')
# parser.add_argument('--algo', help='algorithm')
# parser.add_argument('--net', help='network')
# parser.add_argument('--env', help='environment')
# parser.add_argument('--gen', help='generations to run', type=int)
# parser.add_argument('--c51', help='C51', action="store_true")
# parser.add_argument('--wandb', help='WeightsAndBiases project name')
# parser.add_argument('--seed', help='Seed', type=int)
# parser.add_argument('--frames', help='Max frames', type=int)
# parser.add_argument('--pop', help='Population size', type=int)
# parser.add_argument('--lr', help='Learning rate', type=float, dest="learning_rate")
# parser.add_argument('--SGD', help='SQG optimizer (Adam / RMSprop)')
# parser.add_argument('--epsilon', help='DQN epsilon decay: number of steps', type=int)
# parser.add_argument('--stack', help='Stack frames', type=int, dest="stack_frames")
# parser.add_argument('--reward_clip', help='Clip rewards, 0 to not clip', type=int)
# parser.add_argument('--buffer', help='Replay buffer size', type=int, dest="buffer_size")
# parser.add_argument('--no-save', help='No save', action="store_true", dest="no_store")
# parser.add_argument('--save_path', help='Save directory')
# parser.add_argument('--save_freq', help='Number of gen between 2 saves', type=int)
# parser.add_argument('--eval_freq', help='Number of gen between 2 evaluations', type=int)

# # DQNES
# parser.add_argument('--nu', help='nu max', type=int, dest="nu_max")
# parser.add_argument('--nu-type', help='Nu policy type', dest="nu_type")
# parser.add_argument('--nu-stochastic', help='Stochastic number of steps', type=str2bool, dest="nu_stochastic")

# parser.add_argument('--plot', help='Plot max fitness', action="store_true", dest="plot")

# optims = {
#     "snes":SNES,
#     "canonical":Canonical,
#     "elites":ElitES,
#     "elite":ElitES
# }

# algos = {
#     "dqnes":DQNES,
#     # "per":PER_DQNES,
#     "neuroevo":NeuroEvo
# }

# networks = {
#     "flat":gym_flat_net,
#     "conv":gym_conv,
#     "efficientconv": gym_conv_efficient,
#     "min":min_conv
# }

# def load_preset(args):
#     # Load default values
#     default_file = open(f"{presets_folder}/_default.yaml", 'r')
#     default = yaml.load(default_file)

#     # Loat presets
#     if args.preset is not None:
#         preset_list = args.preset
#         for p in preset_list:
#             yaml_file = open(f"{presets_folder}/{p}.yaml", 'r')
#             yaml_content = yaml.load(yaml_file)
#             default = {**default, **yaml_content}

#     # Override default and presets with non-None command values
#     for k, v in args.__dict__.items():
#         if v is not None:
#             default[k] = v

#     args.__dict__ = default
#     return args


# def set_xp(args):  # sourcery skip: extract-method
#     print(args, "\n")
#     args = load_preset(args)

#     if args.algo.lower() == "dqn":
#         net = networks[args.net.lower()]
#         now = str(datetime.now()).replace(":", "-").replace(" ", "_")

#         if args.no_store:
#             path=None
#         else:
#             folder = f"{args.env}_dqn_{now}"
#             path = f"./saves/{args.env}/dqn/{folder}"

#         print(args)

#         args.epsilon_decay = 1/args.epsilon

#         pb = DQN(net(args.env), args.env, c51=args.c51, 
#             config=args.__dict__,
#             seed=args.seed, save_path=path,
#             plot=args.plot)

#     else:
#         optim = optims[args.optim.lower()]
#         algo = algos[args.algo.lower()]
#         net = networks[args.net.lower()]

#         now = str(datetime.now()).replace(":", "-").replace(" ", "_")

#         if args.no_store:
#             path=None
#         else:
#             folder = f"{args.env}_{args.algo}_{args.optim}_{now}"
#             path = f"./saves/{args.env}/{args.algo}/{args.optim}/{folder}"

#         print(args)

#         pb = algo(optim, net(args.env), args.env, c51=args.c51, direction="max", 
#             config=args.__dict__,
#             optim_args={"n":args.pop},
#             seed=args.seed, save_path=path)
#     print(pb)
#     return pb

# def run_xp(args):
#     pb = set_xp(args)

#     if use_wandb and args.wandb is not None and args.wandb != "None":
#         print("Using wandb")
#         pb.set_wandb(wandb, args.wandb)
#         pb.train(args.gen, max_frames=args.frames)
#         wandb.finish()

#     else:
#         print("Not using wandb")
#         pb.train(args.gen, max_frames=args.frames)

#     if args.plot:
#         pb.plot("linear", size=4, data="max")
#         pb.plot("linear", size=4, data="overall")
#         if  args.algo.lower() == "dqnes":
#             y = pb.gradient_impact
#             x = pb.optim.evaluations
#             plt.plot(x, y)
#         input()

# def parse_and_run(s):
#     args, unknown = parser.parse_known_args(s.split())
#     run_xp(args)