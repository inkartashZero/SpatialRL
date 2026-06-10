import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

from .base import BaseAgent

class VPGNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden, use_critic):
        super().__init__()
        self.use_critic = use_critic
        # Actor
        self.actor_net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU()
        )
        self.mean_head = nn.Linear(hidden, action_dim)
        self.log_std_head = nn.Parameter(torch.zeros(action_dim)) # Learnable variance
        
        # Critic (Baseline)
        if use_critic:
            self.critic_net = nn.Sequential(
                nn.Linear(obs_dim, hidden), nn.ReLU(),
                nn.Linear(hidden, hidden), nn.ReLU(),
                nn.Linear(hidden, 1)
            )

    def forward_actor(self, obs):
        h = self.actor_net(obs)
        mean = torch.tanh(self.mean_head(h)) # squashed to [-1, 1], env can scale it
        std = self.log_std_head.exp().expand_as(mean)
        return Normal(mean, std)

class VPGAgent(BaseAgent):
    def __init__(self, obs_dim, n_actions, config, action_space=None):
        super().__init__(obs_dim, n_actions, config)
        hidden = config.get("hidden_size", 256)
        self.gamma = config.get("gamma", 0.99)
        self.use_critic = config.get("use_critic", True) # False = REINFORCE
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.action_dim = n_actions if action_space is None else len(action_space.low)
        
        # Max bounds (we handle scaling manually since VPG doesn't use squashed targets easily)
        self.max_vel = config.get("max_vel", 10.0)
        self.scale = torch.FloatTensor([self.max_vel, 1.0]).to(self.device)

        self.net = VPGNetwork(obs_dim, self.action_dim, hidden, self.use_critic).to(self.device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=config.get("lr", 3e-4))
        
        # Rollout buffer
        self.trajectory = []

    def select_action(self, obs, explore=True):
        with torch.no_grad():
            t = torch.FloatTensor(obs).to(self.device)
            dist = self.net.forward_actor(t)
            act = dist.sample() if explore else dist.mean
            # Scale to environment bounds
            scaled_act = act * self.scale
        return scaled_act.cpu().numpy().astype(np.float32)

    def update(self, obs, action, reward, next_obs, done):
        # Unscale action for log_prob calculation later
        unscaled_action = action / self.scale.cpu().numpy()
        self.trajectory.append((obs, unscaled_action, reward))
        
        # VPG/REINFORCE only learns at the end of the trajectory
        if not done: return {}

        o, a, r = map(np.array, zip(*self.trajectory))
        self.trajectory.clear()

        # Calculate discounted returns-to-go
        returns = np.zeros_like(r, dtype=np.float32)
        G = 0
        for t in reversed(range(len(r))):
            G = r[t] + self.gamma * G
            returns[t] = G
            
        o_t = torch.FloatTensor(o).to(self.device)
        a_t = torch.FloatTensor(a).to(self.device)
        ret_t = torch.FloatTensor(returns).unsqueeze(1).to(self.device)

        # Baseline (Critic) calculation
        if self.use_critic:
            values = self.net.critic_net(o_t)
            advantages = ret_t - values.detach()
            critic_loss = nn.MSELoss()(values, ret_t)
        else:
            advantages = ret_t # Pure REINFORCE
            critic_loss = torch.tensor(0.0)

        # Actor loss
        dist = self.net.forward_actor(o_t)
        log_probs = dist.log_prob(a_t).sum(axis=-1, keepdim=True)
        actor_loss = -(log_probs * advantages).mean()

        total_loss = actor_loss + critic_loss
        self.optimizer.zero_grad(); total_loss.backward(); self.optimizer.step()

        return {"actor_loss": actor_loss.item(), "critic_loss": critic_loss.item() if self.use_critic else float('nan')}