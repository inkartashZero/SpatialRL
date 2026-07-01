"""
TD3 — Twin Delayed Deep Deterministic Policy Gradient
======================================================
Handles continuous action spaces natively.
Actor outputs [velocity, lick] directly.
The lick output is thresholded at 0 in the environment, so the actor
can learn to suppress licking by pushing the lick logit negative.

Paper: Fujimoto et al. 2018 (https://arxiv.org/abs/1802.09477)

config keys
-----------
hidden_size      : MLP units per layer        (default 256)
lr_actor         : actor Adam lr              (default 3e-4)
lr_critic        : critic Adam lr             (default 3e-4)
gamma            : discount                   (default 0.99)
tau              : soft update rate           (default 0.005)
policy_noise     : target policy smoothing σ  (default 0.2)
noise_clip       : target noise clip          (default 0.5)
policy_delay     : actor update frequency     (default 2)
expl_noise       : exploration noise σ        (default 0.3)
buffer_capacity  :                            (default 50_000)
batch_size       :                            (default 256)
learn_start      : min buffer before training (default 1_000)
device           : 'auto'|'cpu'|'cuda'
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy
from collections import deque
import random

from .base import BaseAgent


# ── Networks ─────────────────────────────────────────────────────────────────

class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden, action_scale, action_bias):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),  nn.ReLU(),
            nn.Linear(hidden, action_dim), nn.Tanh(),
        )
        # rescale tanh output to action bounds
        self.register_buffer("scale", torch.FloatTensor(action_scale))
        self.register_buffer("bias",  torch.FloatTensor(action_bias))

    def forward(self, x):
        return self.net(x) * self.scale + self.bias


class Critic(nn.Module):
    """Twin Q-networks."""
    def __init__(self, obs_dim, action_dim, hidden):
        super().__init__()
        inp = obs_dim + action_dim
        self.q1 = nn.Sequential(
            nn.Linear(inp, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self.q2 = nn.Sequential(
            nn.Linear(inp, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=-1)
        return self.q1(x), self.q2(x)

    def q1_only(self, obs, act):
        return self.q1(torch.cat([obs, act], dim=-1))


# ── Replay Buffer ─────────────────────────────────────────────────────────────

class ReplayBuffer:
    def __init__(self, capacity):
        self.buf = deque(maxlen=capacity)

    def push(self, *transition):
        self.buf.append(transition)

    def sample(self, batch_size):
        batch = random.sample(self.buf, batch_size)
        obs, act, rew, nobs, done = zip(*batch)
        return (
            np.array(obs,  dtype=np.float32),
            np.array(act,  dtype=np.float32),
            np.array(rew,  dtype=np.float32).reshape(-1, 1),
            np.array(nobs, dtype=np.float32),
            np.array(done, dtype=np.float32).reshape(-1, 1),
        )

    def __len__(self): return len(self.buf)


# ── Agent ─────────────────────────────────────────────────────────────────────

class TD3Agent(BaseAgent):

    def __init__(self, obs_dim, n_actions, config, action_space=None):
        super().__init__(obs_dim, n_actions, config)

        hidden        = config.get("hidden_size", 256)
        lr_a          = config.get("lr_actor",  3e-4)
        lr_c          = config.get("lr_critic", 3e-4)
        self.gamma    = config.get("gamma",     0.99)
        self.tau      = config.get("tau",       0.005)
        self.pol_noise= config.get("policy_noise", 0.2)
        self.noise_clip=config.get("noise_clip",   0.5)
        self.pol_delay= config.get("policy_delay",  2)
        self.expl_noise=config.get("expl_noise",   0.3)
        self.batch_size=config.get("batch_size",  256)
        self.learn_start=config.get("learn_start",1_000)
        dev            = config.get("device", "auto")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") \
                      if dev == "auto" else torch.device(dev)

        # action bounds from space if provided
        if action_space is not None:
            self._action_low  = action_space.low.astype(np.float32)
            self._action_high = action_space.high.astype(np.float32)
        else:
            max_vel = config.get("max_vel", 10.0)
            self._action_low  = np.array([-max_vel, -1.0], dtype=np.float32)
            self._action_high = np.array([ max_vel,  1.0], dtype=np.float32)

        action_dim   = len(self._action_low)
        action_scale = (self._action_high - self._action_low) / 2.0
        action_bias  = (self._action_high + self._action_low) / 2.0

        self.actor        = Actor(obs_dim, action_dim, hidden, action_scale, action_bias).to(self.device)
        self.actor_target = deepcopy(self.actor)
        self.critic       = Critic(obs_dim, action_dim, hidden).to(self.device)
        self.critic_target= deepcopy(self.critic)

        self.opt_a = optim.Adam(self.actor.parameters(),  lr=lr_a)
        self.opt_c = optim.Adam(self.critic.parameters(), lr=lr_c)

        self.buffer = ReplayBuffer(config.get("buffer_capacity", 50_000))
        self._step  = 0

        # noise scale (can anneal)
        self._action_scale_t = torch.FloatTensor(action_scale).to(self.device)

    def select_action(self, obs: np.ndarray, explore: bool = True) -> np.ndarray:
        with torch.no_grad():
            t   = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            act = self.actor(t).cpu().numpy()[0]

        if explore:
            noise = np.random.normal(0, self.expl_noise, size=act.shape) \
                    * (self._action_high - self._action_low) / 2.0
            act   = np.clip(act + noise, self._action_low, self._action_high)

        return act.astype(np.float32)

    def update(self, obs, action, reward, next_obs, done) -> dict:
        self.buffer.push(obs, action, reward, next_obs, float(done))
        self._step += 1

        if len(self.buffer) < self.learn_start:
            return {}

        obs_b, act_b, rew_b, nobs_b, done_b = self.buffer.sample(self.batch_size)
        obs_t  = torch.FloatTensor(obs_b).to(self.device)
        act_t  = torch.FloatTensor(act_b).to(self.device)
        rew_t  = torch.FloatTensor(rew_b).to(self.device)
        nobs_t = torch.FloatTensor(nobs_b).to(self.device)
        done_t = torch.FloatTensor(done_b).to(self.device)

        with torch.no_grad():
            noise = (torch.randn_like(act_t) * self.pol_noise).clamp(
                -self.noise_clip, self.noise_clip)
            noise = noise * self._action_scale_t   # scale noise to action range
            next_act = (self.actor_target(nobs_t) + noise).clamp(
                torch.FloatTensor(self._action_low).to(self.device),
                torch.FloatTensor(self._action_high).to(self.device),
            )
            q1_t, q2_t = self.critic_target(nobs_t, next_act)
            q_target    = rew_t + self.gamma * (1 - done_t) * torch.min(q1_t, q2_t)

        q1, q2    = self.critic(obs_t, act_t)
        critic_loss = nn.MSELoss()(q1, q_target) + nn.MSELoss()(q2, q_target)

        self.opt_c.zero_grad(); critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 10.0)
        self.opt_c.step()

        actor_loss_val = float("nan")
        if self._step % self.pol_delay == 0:
            actor_loss = -self.critic.q1_only(obs_t, self.actor(obs_t)).mean()
            self.opt_a.zero_grad(); actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0)
            self.opt_a.step()
            actor_loss_val = actor_loss.item()

            # soft updates
            for p, pt in zip(self.actor.parameters(),  self.actor_target.parameters()):
                pt.data.copy_(self.tau * p.data + (1 - self.tau) * pt.data)
            for p, pt in zip(self.critic.parameters(), self.critic_target.parameters()):
                pt.data.copy_(self.tau * p.data + (1 - self.tau) * pt.data)

        return {"critic_loss": critic_loss.item(), "actor_loss": actor_loss_val}

    def save(self, path):
        torch.save({
            "actor" : self.actor.state_dict(),
            "critic": self.critic.state_dict(),
        }, path)

    def load(self, path):
        d = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(d["actor"])
        self.actor_target = deepcopy(self.actor)
        self.critic.load_state_dict(d["critic"])
        self.critic_target = deepcopy(self.critic)
