from berl import *

cfg = {
    "pop":8,
    "seed":0,
    "SGD":"Adam",
    "lr": 0.001,
    "V_min":-1,
    "V_max":1,
    "atoms":51,
    "c51":False,
    "env_name":"CartPole-v1"
}

def run_dqn_test(Net, config):
    dqn = DQN(Net, config)

def test_dqn():
    raise NotImplementedError
#     run_dqn_test(cfg)

#     cfg["c51"]=True
#     run_dqn_test(cfg)
    
