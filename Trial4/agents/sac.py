import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from copy import deepcopy
from collections import deque
import random
import math

from .base import BaseAgent

LOG_STD_MIN, LOG_STD_MAX = -20, 2

class GaussianActor(nn.Module):
    scale: torch.Tensor
    bias: torch.Tensor
    def __init__(self, obs_dim, action_dim, hidden, action_scale, action_bias):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),  nn.ReLU(),
        )
        self.mean_head    = nn.Linear(hidden, action_dim)
        self.log_std_head = nn.Linear(hidden, action_dim)
        self.register_buffer("scale", torch.FloatTensor(action_scale))
        self.register_buffer("bias",  torch.FloatTensor(action_bias))

    def forward(self, obs):
        h       = self.net(obs)
        mean    = self.mean_head(h)
        log_std = self.log_std_head(h).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(self, obs):
        mean, log_std = self(obs)
        std  = log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        x_t  = dist.rsample()                            # reparameterisation
        y_t  = torch.tanh(x_t)
        act  = y_t * self.scale + self.bias

        # log-prob with tanh squashing correction
        log_prob = dist.log_prob(x_t) \
                   - torch.log(self.scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        mean_act = torch.tanh(mean) * self.scale + self.bias
        return act, log_prob, mean_act

class TwinCritic(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden):
        super().__init__()
        inp = obs_dim + action_dim
        def _mlp():
            return nn.Sequential(
                nn.Linear(inp, hidden), nn.ReLU(),
                nn.Linear(hidden, hidden), nn.ReLU(),
                nn.Linear(hidden, 1),
            )
        self.q1 = _mlp(); self.q2 = _mlp()

    def forward(self, o, a):
        x = torch.cat([o, a], -1)
        return self.q1(x), self.q2(x)

class ReplayBuffer:
    def __init__(self, cap):
        self.buf = deque(maxlen=cap)
    def push(self, *t): self.buf.append(t)
    def sample(self, n):
        b = random.sample(self.buf, n)
        o, a, r, no, d = zip(*b)
        return (np.array(o,  np.float32), np.array(a, np.float32),
                np.array(r,  np.float32).reshape(-1,1),
                np.array(no, np.float32), np.array(d, np.float32).reshape(-1,1))
    def __len__(self): return len(self.buf)

class SACAgent(BaseAgent):
    def __init__(self, obs_dim, n_actions, config, action_space=None):
        super().__init__(obs_dim, n_actions, config)

        hidden       = config.get("hidden_size", 256)
        lr           = config.get("lr",          3e-4)
        self.gamma   = config.get("gamma",       0.99)
        self.tau     = config.get("tau",         0.005)
        self.alpha   = config.get("alpha",       0.2)
        auto_ent     = config.get("auto_entropy",True)
        self.batch   = config.get("batch_size",  256)
        self.learn_s = config.get("learn_start", 1_000)
        dev          = config.get("device","auto")

        self.device  = torch.device("cuda" if torch.cuda.is_available() else "cpu") \
                       if dev == "auto" else torch.device(dev)

        if action_space is not None:
            low  = action_space.low.astype(np.float32)
            high = action_space.high.astype(np.float32)
        else:
            mv = config.get("max_vel", 10.0)
            low  = np.array([-mv, -1.0], np.float32)
            high = np.array([ mv,  1.0], np.float32)

        action_dim   = len(low)
        action_scale = (high - low) / 2.0
        action_bias  = (high + low) / 2.0

        self.actor  = GaussianActor(obs_dim, action_dim, hidden, action_scale, action_bias).to(self.device)
        self.critic = TwinCritic(obs_dim, action_dim, hidden).to(self.device)
        self.critic_target = deepcopy(self.critic)

        self.opt_a = optim.Adam(self.actor.parameters(),  lr=lr)
        self.opt_c = optim.Adam(self.critic.parameters(), lr=lr)

        self._auto_entropy = auto_ent
        target_ent = config.get("target_entropy", -float(action_dim))
        self.target_entropy = target_ent
        if auto_ent:
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.opt_alpha  = optim.Adam([self.log_alpha], lr=lr)

        self.buffer = ReplayBuffer(config.get("buffer_capacity", 50_000))
        self._step  = 0

    def select_action(self, obs, explore=True):
        with torch.no_grad():
            t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            if explore:
                act, _, _ = self.actor.sample(t)
            else:
                _, _, act = self.actor.sample(t)
        return act.cpu().numpy()[0].astype(np.float32)

    def update(self, obs, action, reward, next_obs, done):
        self.buffer.push(obs, action, reward, next_obs, float(done))
        self._step += 1
        if len(self.buffer) < self.learn_s:
            return {}

        ob, ac, rw, no, dn = self.buffer.sample(self.batch)
        ob_t = torch.FloatTensor(ob).to(self.device)
        ac_t = torch.FloatTensor(ac).to(self.device)
        rw_t = torch.FloatTensor(rw).to(self.device)
        no_t = torch.FloatTensor(no).to(self.device)
        dn_t = torch.FloatTensor(dn).to(self.device)

        alpha = self.log_alpha.exp().detach() if self._auto_entropy else self.alpha

        with torch.no_grad():
            na, nl, _ = self.actor.sample(no_t)
            q1_t, q2_t = self.critic_target(no_t, na)
            q_target   = rw_t + self.gamma*(1-dn_t)*(torch.min(q1_t,q2_t) - alpha*nl)
        
        q1, q2     = self.critic(ob_t, ac_t)
        c_loss     = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)
        self.opt_c.zero_grad(); c_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 10.); self.opt_c.step()

        na2, nl2, _ = self.actor.sample(ob_t)
        q1_a, q2_a  = self.critic(ob_t, na2)
        a_loss      = (alpha * nl2 - torch.min(q1_a, q2_a)).mean()
        self.opt_a.zero_grad(); a_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 10.); self.opt_a.step()

        alpha_loss_val = float("nan")
        if self._auto_entropy:
            al = -(self.log_alpha * (nl2.detach() + self.target_entropy)).mean()
            self.opt_alpha.zero_grad(); al.backward(); self.opt_alpha.step()
            alpha_loss_val = al.item()
            self.alpha = self.log_alpha.exp().item()

        for p, pt in zip(self.critic.parameters(), self.critic_target.parameters()):
            pt.data.copy_(self.tau*p.data + (1-self.tau)*pt.data)

        return {"critic_loss": c_loss.item(), "actor_loss": a_loss.item(),
                "alpha": self.alpha, "alpha_loss": alpha_loss_val}

    def save(self, path):
        torch.save({"actor": self.actor.state_dict(),
                    "critic": self.critic.state_dict()}, path)

    def load(self, path):
        d = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(d["actor"])
        self.critic.load_state_dict(d["critic"])
        self.critic_target = deepcopy(self.critic)