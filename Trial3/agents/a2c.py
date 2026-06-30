import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from copy import deepcopy

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
        x_t  = dist.sample()                            
        y_t  = torch.tanh(x_t)
        act  = y_t * self.scale + self.bias

        mean_act = torch.tanh(mean) * self.scale + self.bias
        # Return scaled action, raw unsquashed sample (x_t), and scaled mean
        return act, x_t, mean_act

    def evaluate(self, obs, x_t):
        mean, log_std = self(obs)
        std  = log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        y_t  = torch.tanh(x_t)

        # log-prob with tanh squashing correction for stable gradients
        log_prob = dist.log_prob(x_t) \
                   - torch.log(self.scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        
        entropy = dist.entropy().sum(dim=-1, keepdim=True)
        return log_prob, entropy

class ValueCritic(nn.Module):
    def __init__(self, obs_dim, hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, o):
        return self.net(o)

class RolloutBuffer:
    def __init__(self):
        self.buf = []
    def push(self, *t): 
        self.buf.append(t)
    def get(self):
        o, a, r, no, d = zip(*self.buf)
        return (np.array(o,  np.float32), np.array(a, np.float32),
                np.array(r,  np.float32).reshape(-1,1),
                np.array(no, np.float32), np.array(d, np.float32).reshape(-1,1))
    def clear(self): 
        self.buf.clear()
    def __len__(self): 
        return len(self.buf)

class A2CAgent(BaseAgent):
    def __init__(self, obs_dim, n_actions, config, action_space=None):
        super().__init__(obs_dim, n_actions, config)

        hidden           = config.get("hidden_size", 256)
        lr               = config.get("lr",          3e-4)
        self.gamma       = config.get("gamma",       0.99)
        self.entropy_coef = config.get("entropy_coef", 0.01)
        self.rollout_steps = config.get("rollout_steps", 5)
        dev              = config.get("device","auto")

        self.device      = torch.device("cuda" if torch.cuda.is_available() else "cpu") \
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
        self.critic = ValueCritic(obs_dim, hidden).to(self.device)

        self.opt_a = optim.Adam(self.actor.parameters(),  lr=lr)
        self.opt_c = optim.Adam(self.critic.parameters(), lr=lr)

        self.buffer = RolloutBuffer()
        self._temp_raw_action = None

    def select_action(self, obs, explore=True):
        with torch.no_grad():
            t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            if explore:
                act, x_t, _ = self.actor.sample(t)
                self._temp_raw_action = x_t.cpu().numpy()[0]
                return act.cpu().numpy()[0].astype(np.float32)
            else:
                _, _, mean_act = self.actor.sample(t)
                # Fallback storage for raw mean to prevent crash if update is called
                mean, _ = self.actor(t)
                self._temp_raw_action = mean.cpu().numpy()[0]
                return mean_act.cpu().numpy()[0].astype(np.float32)

    def update(self, obs, action, reward, next_obs, done):
        # Store the raw, unsquashed action (x_t) for correct log_prob evaluation
        self.buffer.push(obs, self._temp_raw_action, reward, next_obs, float(done))
        
        if len(self.buffer) >= self.rollout_steps or done:
            metrics = self._train_step()
            self.buffer.clear()
            return metrics
            
        return {}

    def _train_step(self):
        ob, ac, rw, no, dn = self.buffer.get()
        ob_t = torch.FloatTensor(ob).to(self.device)
        ac_t = torch.FloatTensor(ac).to(self.device)
        rw_t = torch.FloatTensor(rw).to(self.device)
        no_t = torch.FloatTensor(no).to(self.device)
        dn_t = torch.FloatTensor(dn).to(self.device)

        values = self.critic(ob_t)
        
        with torch.no_grad():
            next_values = self.critic(no_t)
            
        returns = torch.zeros_like(rw_t).to(self.device)
        running_return = next_values[-1] * (1 - dn_t[-1]) if not dn_t[-1] else 0.0
        
        for t in reversed(range(len(rw_t))):
            running_return = rw_t[t] + self.gamma * running_return * (1 - dn_t[t])
            returns[t] = running_return

        advantages = returns - values

        log_prob, entropy = self.actor.evaluate(ob_t, ac_t)

        a_loss = -(log_prob * advantages.detach()).mean() - (self.entropy_coef * entropy.mean())
        c_loss = F.mse_loss(values, returns)

        self.opt_a.zero_grad(); a_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 10.); self.opt_a.step()

        self.opt_c.zero_grad(); c_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 10.); self.opt_c.step()

        return {
            "actor_loss": a_loss.item(),
            "critic_loss": c_loss.item(),
            "entropy": entropy.mean().item()
        }

    def save(self, path):
        torch.save({"actor": self.actor.state_dict(),
                    "critic": self.critic.state_dict()}, path)

    def load(self, path):
        d = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(d["actor"])
        self.critic.load_state_dict(d["critic"])