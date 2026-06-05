"""
Deep Q-Network (DQN) Agent
===========================
Uses a small MLP Q-network with experience replay and a target network.
Works directly with the raw 7-dim observation (no discretisation needed).

Requirements: torch
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from .base import BaseAgent


# ── Network ──────────────────────────────────────────────────────────────────

class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)


# ── Replay Buffer ─────────────────────────────────────────────────────────────

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buf = deque(maxlen=capacity)

    def push(self, obs, action, reward, next_obs, done):
        self.buf.append((obs, action, reward, next_obs, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buf, batch_size)
        obs, act, rew, nobs, done = zip(*batch)
        return (
            np.array(obs,  dtype=np.float32),
            np.array(act,  dtype=np.int64),
            np.array(rew,  dtype=np.float32),
            np.array(nobs, dtype=np.float32),
            np.array(done, dtype=np.float32),
        )

    def __len__(self): return len(self.buf)


# ── Agent ─────────────────────────────────────────────────────────────────────

class DQNAgent(BaseAgent):
    """
    config keys
    -----------
    hidden_size      : MLP hidden units          (default 64)
    lr               : Adam learning rate        (default 1e-3)
    gamma            : discount                  (default 0.99)
    epsilon          : initial epsilon           (default 1.0)
    epsilon_min      : floor epsilon             (default 0.05)
    epsilon_decay    : per-step multiplicative   (default 0.9995)
    buffer_capacity  : replay buffer size        (default 10_000)
    batch_size       : training batch            (default 64)
    target_update    : steps between target sync (default 200)
    learn_start      : min buffer size to start  (default 200)
    device           : 'cpu' | 'cuda' | 'auto'  (default 'auto')
    """

    def __init__(self, obs_dim: int, n_actions: int, config: dict):
        super().__init__(obs_dim, n_actions, config)

        hidden           = config.get("hidden_size", 64)
        lr               = config.get("lr", 1e-3)
        self.gamma       = config.get("gamma", 0.99)
        self.epsilon     = config.get("epsilon", 1.0)
        self.epsilon_min = config.get("epsilon_min", 0.05)
        self.eps_decay   = config.get("epsilon_decay", 0.9995)
        self.batch_size  = config.get("batch_size", 64)
        self.target_upd  = config.get("target_update", 200)
        self.learn_start = config.get("learn_start", 200)
        device_str       = config.get("device", "auto")

        if device_str == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device_str)

        self.q_net      = QNetwork(obs_dim, n_actions, hidden).to(self.device)
        self.target_net = QNetwork(obs_dim, n_actions, hidden).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.buffer    = ReplayBuffer(config.get("buffer_capacity", 10_000))
        self._step     = 0

    def select_action(self, obs: np.ndarray, explore: bool = True) -> int:
        if explore and np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        with torch.no_grad():
            t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            return int(self.q_net(t).argmax(dim=1).item())

    def update(self, obs, action, reward, next_obs, done) -> dict:
        self.buffer.push(obs, action, reward, next_obs, done)
        self._step += 1

        if len(self.buffer) < self.learn_start:
            return {}

        obs_b, act_b, rew_b, nobs_b, done_b = self.buffer.sample(self.batch_size)

        obs_t  = torch.FloatTensor(obs_b).to(self.device)
        act_t  = torch.LongTensor(act_b).to(self.device)
        rew_t  = torch.FloatTensor(rew_b).to(self.device)
        nobs_t = torch.FloatTensor(nobs_b).to(self.device)
        done_t = torch.FloatTensor(done_b).to(self.device)

        current_q = self.q_net(obs_t).gather(1, act_t.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q = self.target_net(nobs_t).max(1)[0]
            target = rew_t + self.gamma * next_q * (1 - done_t)

        loss = nn.MSELoss()(current_q, target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.optimizer.step()

        # update target network
        if self._step % self.target_upd == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        # decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.eps_decay)

        return {"loss": loss.item(), "epsilon": self.epsilon}

    def save(self, path: str):
        torch.save({
            "q_net"    : self.q_net.state_dict(),
            "target"   : self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon"  : self.epsilon,
            "step"     : self._step,
        }, path)

    def load(self, path: str):
        d = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(d["q_net"])
        self.target_net.load_state_dict(d["target"])
        self.optimizer.load_state_dict(d["optimizer"])
        self.epsilon = d["epsilon"]
        self._step   = d["step"]
