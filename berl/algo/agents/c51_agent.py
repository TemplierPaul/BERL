import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

from .rl_agent import Agent


class C51Agent(Agent):
    def __init__(self, Net, config):
        super().__init__(Net, config)

        self.atoms = config["atoms"]
        self.V_min = config["V_min"]
        self.V_max = config["V_max"]
        self.delta_z = float(self.V_max - self.V_min) / (self.atoms - 1)
        self.support = torch.linspace(
            self.V_min, self.V_max, self.atoms
        ).to(device=self.device)
        self.criterion = torch.nn.CrossEntropyLoss()

    def make_network(self):
        self.model = self.Net(c51=True).to(self.device).double()

    def act(self, obs):
        self.state.update(obs)
        with torch.no_grad():
            # x = torch.from_numpy(state.astype(float))
            x = self.state.get().to(self.device).double()
            q = self.model(x).cpu().detach()
            n_out = int(self.model.n_out/self.atoms)
            q = q.view(-1, n_out, self.atoms)
            # Probabilities with action over second dimension
            q = F.softmax(q, dim=2).to(device=self.device)

            return (q * self.support).sum(2).argmax(1).item()

    def forward_probas(self, model, state, log=False):
        # sourcery skip: assign-if-exp
        x = state.to(self.device).double()
        if torch.isnan(x).any():
            raise
        q = model(x).cpu()
        n_out = int(model.n_out/self.atoms)
        q = q.view(-1, n_out, self.atoms)
        if log:  # Use log softmax for numerical stability
            # Log probabilities with action over second dimension
            q = F.log_softmax(q, dim=2)
        else:
            # Probabilities with action over second dimension
            q = F.softmax(q, dim=2)
        return q.to(device=self.device)

    def dqn_loss(self, target, X, A, R, Y, D):
        # Reshape
        R, D = R.reshape(-1, 1), D.reshape(-1, 1)
        # Calculate current state probabilities
        # (online network noise already sampled)
        # Log probabilities log p(s_t, ·; θonline)
        log_ps = self.forward_probas(self.model, X, log=True)

        # assert type(A) == None
        # log p(s_t, a_t; θonline)
        log_ps_a = log_ps[range(self.batch_size), A.long()]

        with torch.no_grad():
            # Calculate nth next state probabilities
            # Probabilities p(s_t+n, ·; θonline)
            pns = self.forward_probas(self.model, Y)
            # Distribution d_t+n = (z, p(s_t+n, ·; θonline))
            dns = self.support.expand_as(pns) * pns
            # Perform argmax action selection using online network:
            # argmax_a[(z, p(s_t+n, a; θonline))]
            argmax_indices_ns = dns.sum(2).argmax(1)
            # Probabilities p(s_t+n, ·; θonline)
            pns = self.forward_probas(target, Y)
            # Double-Q probabilities
            # p(s_t+n, argmax_a[(z, p(s_t+n, a; θonline))]; θtarget)
            pns_a = pns[range(self.batch_size), argmax_indices_ns]

            # Compute Tz (Bellman operator T applied to z)
            # Tz = R^n + (γ^n)z (accounting for terminal states)
            Tz = R + (1-D) * (self.gamma ** self.update_horizon) * self.support
            # Clamp between supported values
            Tz = Tz.clamp(min=self.V_min, max=self.V_max)
            # Compute L2 projection of Tz onto fixed support z
            b = (Tz - self.V_min) / self.delta_z  # b = (Tz - V_min) / Δz
            lower, u = b.floor().to(torch.int64), b.ceil().to(torch.int64)
            # Fix disappearing probability mass when lower = b = u (b is int)
            lower[(u > 0) * (lower == u)] -= 1
            u[(lower < (self.atoms - 1)) * (lower == u)] += 1

            # Distribute probability of Tz
            m = X.new_zeros(self.batch_size, self.atoms)
            offset = torch.linspace(0,
                                    ((self.batch_size - 1) * self.atoms),
                                    self.batch_size
                                    ).unsqueeze(1).expand(
                                        self.batch_size,
                                        self.atoms
            ).to(A)
            # m_l = m_l + p(s_t+n, a*)(u - b)
            m.view(-1).index_add_(0, (lower + offset).view(-1),
                                  (pns_a * (u.double() - b)).view(-1))
            # m_u = m_u + p(s_t+n, a*)(b - l)
            m.view(-1).index_add_(0, (u + offset).view(-1),
                                  (pns_a * (b - lower.double())).view(-1))

        # Cross-entropy loss (minimises DKL(m||p(s_t, a_t)))
        loss = -torch.sum(m * log_ps_a, 1)
        # loss.requires_grad = True
        return loss.mean()

    def backprop(self, loss):
        self.model.zero_grad()
        clip_grad_norm_(self.model.parameters(), self.norm_clip)
        loss.backward()
        self.optimizer.step()
        self.genes = self.get_params()
        return self
