from berl import *
import berl
import numpy as np
import types
import torch

cfg = {
    "pop":8,
    "seed":0,
    "SGD":"Adam",
    "lr": 0.001,
    "V_min":-1,
    "V_max":1,
    "atoms":51,
    "c51":False
}

def agent_test(agent_type):
    cfg['stack_frames']=1

    game = "CartPole-v1"
    Net = gym_flat_net(game)
    a = agent_type(Net, cfg)
    assert isinstance(a, Agent)
    assert isinstance(a.Net, types.FunctionType)
    assert a.fitness is None
    assert isinstance(a.state, State)
    assert a.model is None
    a.make_network()
    assert a.model is not None

    assert isinstance(a.model, FFNet)

    a.state.reset()
    assert a.state.get() is None

    env = make_vect_env(game, n=1)
    obs = env.reset()[0]

    # State without stacking gives obs
    a.state.update(obs)
    s = a.state.get()
    assert isinstance(s, torch.Tensor)
    assert (s == torch.tensor(obs).double()).all()

    a.state.reset()
    assert a.state.get() is None

    obs = env.reset()
    action = a.act(obs)
    assert isinstance(action, (int, float))

    env.close()

    g = a.genes
    g_copy = np.copy(g)
    assert isinstance(g, np.ndarray)
    a.genes = g
    assert (a.genes == g_copy).all

def agent_test_min(agent_type):
    cfg['stack_frames']=1

    game = "min-asterix"
    Net = min_conv(game)
    a = agent_type(Net, cfg)
    assert isinstance(a, Agent)
    assert isinstance(a.Net, types.FunctionType)
    assert a.fitness is None
    assert isinstance(a.state, State)
    assert a.model is None
    a.make_network()
    assert a.model is not None

    assert isinstance(a.model, MinatarNet)

    a.state.reset()
    assert a.state.get() is None

    env = make_vect_env(game, n=1)
    obs = env.reset()[0]

    # State without stacking gives obs
    a.state.update(obs)
    s = a.state.get()
    assert isinstance(s, torch.Tensor)
    assert (s == torch.tensor(obs).double()).all()

    a.state.reset()
    assert a.state.get() is None

    obs = env.reset()
    action = a.act(obs)
    assert isinstance(action, (int, float))

    env.close()

    g = a.genes
    g_copy = np.copy(g)
    assert isinstance(g, np.ndarray)
    a.genes = g
    assert (a.genes == g_copy).all

def agent_test_stack(agent_type, efficient=False):
    cfg['stack_frames']=4

    game = "Pong-v0"

    env = make_vect_env(game, n=1)
    obs = env.reset()[0]

    cfg["obs_shape"]=env.observation_space.shape
    cfg["n_actions"]=env.action_space.n

    Net = gym_conv_efficient(game) if efficient else gym_conv(game)

    a = agent_type(Net, cfg)
    assert isinstance(a, Agent)
    assert isinstance(a.Net, types.FunctionType)
    assert a.fitness is None
    assert isinstance(a.state, FrameStackState)
    assert len(a.state.state) == 4

    assert a.model is None
    a.make_network()
    assert a.model is not None

    if efficient:
        assert isinstance(a.model, DataEfficientConvNet)
    else:
        assert isinstance(a.model, ConvNet)

    a.state.reset()
    assert a.state.get().sum() == 0

    # State with stacking gives obs
    a.state.update(obs)
    s = a.state.get()
    assert isinstance(s, torch.Tensor)
    # assert (s == torch.tensor(obs).double()).all()

    a.state.reset()
    assert a.state.get().sum() == 0

    obs = env.reset()
    action = a.act(obs)
    assert isinstance(action, (int, float))

    env.close()

def test_agent():
    agent_test(Agent)
    agent_test_min(Agent)
    agent_test_stack(Agent, efficient=False)
    agent_test_stack(Agent, efficient=True)

def test_c51_agent():
    agent_test(C51Agent)
    agent_test_min(C51Agent)
    agent_test_stack(C51Agent, efficient=False)
    agent_test_stack(C51Agent, efficient=True)


def test_population():
    cfg['stack_frames']=1

    game = "CartPole-v1"
    Net = gym_flat_net(game)

    pop = Population(Net, cfg)

    assert isinstance(len(pop), int)
    assert len(pop) == 0

    a=Agent(Net, cfg).make_network()
    n_genes = len(a.genes)

    n_pop=10
    genomes = [np.random.random(n_genes) for i in range(n_pop)]
    assert len(genomes) == n_pop
    pop.genomes = genomes

    assert len(pop.agents) == n_pop
    assert len(pop) == n_pop
    assert isinstance(pop[3], Agent)
    assert all(i.fitness is None for i in pop.agents)

    new_g = pop.genomes
    assert all((new_g[i] == genomes[i]).all() for i in range(n_pop))

    pop.fitness = [i for i in range(n_pop)]

    assert all(pop[i].fitness == i for i in range(n_pop))
    new_f = pop.fitness
    assert all(new_f[i] == i for i in range(n_pop))


    pop.reset()
    assert all(i.state.get() is None for i in pop)

    running = np.array([True for i in range(n_pop)])
    running[3] = False

    env = make_vect_env(game, n=n_pop)
    obs = env.reset()

    actions = pop.act(obs, running)
    assert isinstance(actions, (list, np.ndarray))
    assert all(pop[i].state.get() is not None or i==3 for i in range(n_pop))
    assert actions[3] == 0
    
    env.step_async(actions)
    next_obs, r, done, _ = env.step_wait()
    
    env.close()

    i = 0
    for l in pop.split(4):
        i += 1
        assert isinstance(l, Population)
        assert len(l) == 4 or i == 3
        l.fitness = [1 for _ in l]
    assert i == 3
    assert all(i.fitness == 1 for i in pop)