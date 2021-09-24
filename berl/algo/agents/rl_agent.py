from ...utils.state import * 
import torch.nn.functional as F


class Agent:
    def __init__(self, Net, config):
        self.config = config
        self.Net = Net 

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = None
        self.target = None

        self.fitness = None

        # State
        if self.config["stack_frames"] > 1:
            self.state = FrameStackState(self.config["obs_shape"], self.config["stack_frames"])
        else:
            self.state = State()

        self.criterion = torch.nn.MSELoss()
        

    def __repr__(self): # pragma: no cover
        return f"Agent {self.model} > fitness={self.fitness}" 
        
    def __str__(self): # pragma: no cover
        return self.__repr__()

    def make_network(self):
        self.model = self.Net(c51=False).to(self.device).double()
        return self

    @property
    def genes(self):
        if self.model is None:
            return None
        with torch.no_grad():
            params = self.model.parameters()
            vec = torch.nn.utils.parameters_to_vector(params)
        return vec.cpu().double().numpy()

    @genes.setter
    def genes(self, params):
        if self.model is None:
            self.make_network()
        assert len(params) == len(self.genes), "Genome size does not fit the network size"
        if np.isnan(params).any():
            raise
        a = torch.tensor(params, device=self.device)
        torch.nn.utils.vector_to_parameters(a, self.model.parameters())
        self.model = self.model.to(self.device).double()
        self.set_optim()
        self.fitness = None
        return self

    def act(self, obs):
        self.state.update(obs)
        with torch.no_grad():
            x = self.state.get().to(self.device).double()
            actions = self.model(x).cpu().detach().numpy()
        return int(np.argmax(actions))

    def set_optim(self):
        if self.model is None:
            self.make_network()
        if self.config["SGD"].lower()=="adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config["lr"])
        else:
            self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=self.config["lr"])
        return self

    def dqn_loss(self, target, X, A, R, Y, D):
        state = X.to(self.device)
        actions = A.to(torch.long).to(self.device)
        r = R.to(self.device).unsqueeze(1)
        next_s = Y.to(self.device)
        done = D.to(self.device).unsqueeze(1)

        # for i in [state, actions, r, next_s, done]:
        #     print(i.shape)
        
        current_Q = self.model(state).to(self.device).gather(1, actions.unsqueeze(1)) # Q(s, a) with a the action selected in A
        # print("Current Q", current_Q.shape)

        if self.double_dqn: # Double DQN
            next_actions = self.model(next_s).argmax(1).unsqueeze(1)
            # print("next actions", next_actions.shape)

            next_Q = target(next_s).gather(1, next_actions)
            # print("next_Q", next_Q.shape)

        else: # Simple DQN
            next_Q = target(next_s).max(1)[0].unsqueeze(1) # max Q(s', a')
            # print("next_Q", next_Q.shape)
        
        next_Q = next_Q.cpu().detach().to(self.device) 

        # Update
        target_Q =  r + (1-done) * (self.gamma ** self.update_horizon) * next_Q # R + gamma * (1-D) * max Q(s', a') 
        target_Q = target_Q.detach()
        # print("target_Q", target_Q.shape)

        loss = self.criterion(current_Q, target_Q)
        return loss

    def backprop(self, loss):
        # loss = self.dqn_loss(*transitions)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # self.update_target()
        # self.genes = self.get_params()
        return self