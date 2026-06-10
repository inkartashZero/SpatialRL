import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy
import random
from collections import deque

from .base import BaseAgent

class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden, action_scale, action_bias):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),  nn.ReLU(),
            nn.Linear(hidden, action_dim), nn.Tanh(),
        )
        self.register_buffer("scale", torch.FloatTensor(action_scale))
        self.register_buffer("bias",  torch.FloatTensor(action_bias))

    def forward(self, x):
        return self.net(x) * self.scale + self.bias

class Critic(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )
    def forward(self, obs, act):
        return self.net(torch.cat([obs, act], dim=-1))

class DDPGAgent(BaseAgent):
    def __init__(self, obs_dim, n_actions, config, action_space=None):
        super().__init__(obs_dim, n_actions, config)
        hidden      = config.get("hidden_size", 256)
        self.gamma  = config.get("gamma", 0.99)
        self.tau    = config.get("tau", 0.005)
        self.expl_noise = config.get("expl_noise", 0.1)
        self.batch_size = config.get("batch_size", 256)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        max_vel = config.get("max_vel", 10.0)
        self._low  = np.array([-max_vel, -1.0], dtype=np.float32)
        self._high = np.array([ max_vel,  1.0], dtype=np.float32)
        action_dim = len(self._low)
        scale = (self._high - self._low) / 2.0
        bias  = (self._high + self._low) / 2.0

        self.actor = Actor(obs_dim, action_dim, hidden, scale, bias).to(self.device)
        self.actor_target = deepcopy(self.actor)
        self.critic = Critic(obs_dim, action_dim, hidden).to(self.device)
        self.critic_target = deepcopy(self.critic)

        self.opt_a = optim.Adam(self.actor.parameters(), lr=config.get("lr_actor", 3e-4))
        self.opt_c = optim.Adam(self.critic.parameters(), lr=config.get("lr_critic", 3e-4))
        
        self.buffer = deque(maxlen=config.get("buffer_capacity", 50000))
        self.learn_start = config.get("learn_start", 1000)

    def select_action(self, obs, explore=True):
        with torch.no_grad():
            t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            act = self.actor(t).cpu().numpy()[0]
        if explore:
            noise = np.random.normal(0, self.expl_noise, size=act.shape) * (self._high - self._low) / 2.0
            act = np.clip(act + noise, self._low, self._high)
        return act.astype(np.float32)

    def update(self, obs, action, reward, next_obs, done):
        self.buffer.append((obs, action, reward, next_obs, float(done)))
        if len(self.buffer) < self.learn_start: return {}

        batch = random.sample(self.buffer, self.batch_size)
        o, a, r, no, d = map(lambda x: torch.FloatTensor(np.array(x)).to(self.device), zip(*batch))
        r, d = r.unsqueeze(1), d.unsqueeze(1)

        # Critic Update
        with torch.no_grad():
            next_a = self.actor_target(no)
            target_q = r + self.gamma * (1 - d) * self.critic_target(no, next_a)
        current_q = self.critic(o, a)
        c_loss = nn.MSELoss()(current_q, target_q)
        self.opt_c.zero_grad(); c_loss.backward(); self.opt_c.step()

        # Actor Update
        a_loss = -self.critic(o, self.actor(o)).mean()
        self.opt_a.zero_grad(); a_loss.backward(); self.opt_a.step()

        # Soft Update Targets
        for p, pt in zip(self.actor.parameters(), self.actor_target.parameters()):
            pt.data.copy_(self.tau * p.data + (1 - self.tau) * pt.data)
        for p, pt in zip(self.critic.parameters(), self.critic_target.parameters()):
            pt.data.copy_(self.tau * p.data + (1 - self.tau) * pt.data)

        return {"critic_loss": c_loss.item(), "actor_loss": a_loss.item()}