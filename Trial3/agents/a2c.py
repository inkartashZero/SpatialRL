"""
A2C — Advantage Actor-Critic (Continuous Synchronous)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal

from .base import BaseAgent


class A2CNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden):
        super().__init__()
        # Decoupled backbone for continuous control stability
        self.actor_net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),  nn.ReLU()
        )
        self.mean_head = nn.Linear(hidden, action_dim)
        self.log_std_head = nn.Parameter(torch.zeros(action_dim)) 

        self.critic_net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),  nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward_actor(self, obs):
        h = self.actor_net(obs)
        mean = torch.tanh(self.mean_head(h))  
        std = self.log_std_head.exp().expand_as(mean)
        return Normal(mean, std)

    def forward_critic(self, obs):
        return self.critic_net(obs)


class A2CAgent(BaseAgent):
    # Using your exact requested signature here
    def __init__(self, obs_dim, n_actions, config, action_space=None):
        super().__init__(obs_dim, n_actions, config)
        
        hidden_size = config.get("hidden_size", 256)
        self.lr = config.get("lr", 3e-4)
        self.gamma = config.get("gamma", 0.99)
        self.entropy_coef = config.get("entropy_coef", 0.01)
        self.rollout_steps = config.get("rollout_steps", 5) 
        
        self.device = torch.device("cuda" if torch.cuda.is_available() and config.get("device") == "auto" else "cpu")
        
        self.net = A2CNetwork(obs_dim, n_actions, hidden_size).to(self.device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)

        # Dynamically handle action space limits
        if action_space is not None:
            self._low = action_space.low
            self._high = action_space.high
        else:
            self._low = np.full(n_actions, -1.0)
            self._high = np.full(n_actions, 1.0)
            
        self.scale = (self._high - self._low) / 2.0
        self.bias = (self._high + self._low) / 2.0

        # Rollout storage
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []

    def select_action(self, obs, explore=True):
        obs_t = torch.FloatTensor(obs).to(self.device)
        with torch.no_grad():
            dist = self.net.forward_actor(obs_t)
            if explore:
                action = dist.sample()
            else:
                action = dist.mean
        
        action_np = action.cpu().numpy()
        scaled_action = action_np * self.scale + self.bias
        return scaled_action

    def update(self, obs, action, reward, next_obs, done):
        unscaled_action = (action - self.bias) / self.scale
        
        self.states.append(obs)
        self.actions.append(unscaled_action)
        self.rewards.append(reward)
        self.next_states.append(next_obs)
        self.dones.append(float(done))

        if len(self.states) >= self.rollout_steps or done:
            metrics = self._train_step()
            self._clear_buffer()
            return metrics
            
        return {}

    def _train_step(self):
        o = torch.FloatTensor(np.array(self.states)).to(self.device)
        a = torch.FloatTensor(np.array(self.actions)).to(self.device)
        r = torch.FloatTensor(np.array(self.rewards)).to(self.device)
        no = torch.FloatTensor(np.array(self.next_states)).to(self.device)
        d = torch.FloatTensor(np.array(self.dones)).to(self.device)

        values = self.net.forward_critic(o).squeeze(-1)
        
        with torch.no_grad():
            next_values = self.net.forward_critic(no).squeeze(-1)
            
        returns = torch.zeros_like(r).to(self.device)
        running_return = next_values[-1] * (1 - d[-1]) if not d[-1] else 0.0
        
        for t in reversed(range(len(r))):
            running_return = r[t] + self.gamma * running_return * (1 - d[t])
            returns[t] = running_return

        advantages = returns - values

        dist = self.net.forward_actor(o)
        log_probs = dist.log_prob(a).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)

        actor_loss = -(log_probs * advantages.detach()).mean()
        critic_loss = nn.MSELoss()(values, returns)
        entropy_loss = -entropy.mean()

        total_loss = actor_loss + 0.5 * critic_loss + self.entropy_coef * entropy_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
        self.optimizer.step()

        return {
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "entropy": entropy.mean().item()
        }

    def _clear_buffer(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.next_states.clear()
        self.dones.clear()