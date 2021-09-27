from berl import *


def test_parse():
    s = "--algo=neuroevo --gen=100 --env=CartPole-v1 --c51=True "
    args, unknown = parser.parse_known_args(s.split())

    assert args.c51 
    assert args.env == "CartPole-v1"
    assert args.algo == "neuroevo"
    assert args.gen == 100

def test_preset():
    s = ""
    args, unknown = parser.parse_known_args(s.split())
    print("BEFORE", args)
    args = load_preset(args)

    print("\n\nAFTER", args)

    assert args.c51 
    assert args.env == "CartPole-v1"
    assert args.algo == "neuroevo"
    assert args.gen == 100

    s = "--preset atari"
    args, unknown = parser.parse_known_args(s.split())
    print("BEFORE", args)
    args = load_preset(args)

    print("\n\nAFTER", args)

    assert args.c51 
    assert args.env == "Breakout-v0"
    assert args.algo == "neuroevo"
    assert args.episode_frames == 27000

def test_set_xp():
    s = "--preset atari"
    args, unknown = parser.parse_known_args(s.split())
    pb = set_xp(args)

    assert isinstance(pb, NeuroEvo)

    assert pb.config["c51"] == True
    assert pb.config["env"] == "Breakout-v0"
    assert pb.config["algo"] == "neuroevo"
    
def test_run():
    configs = [
        "--gen=10", # Cartpole
        "--preset minatar --gen=10 --frames=300", # Minatar
        "--preset atari --gen=3 --frames=300", # Atari
        "--preset atari --gen=3 --frames=300 --net=conv" # Atari, big network
    ]
    for s in configs:
        pb = set_xp(s)
        assert isinstance(pb, NeuroEvo)
        pb.train()