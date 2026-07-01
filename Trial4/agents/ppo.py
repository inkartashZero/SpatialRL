"""
PPO — Proximal Policy Optimization (Continuous)
===============================================
On-policy actor-critic algorithm with clipped objective.
Highly stable and robust against buffer dilution in sparse reward tasks.

config keys
-----------
hidden_size   : MLP units per layer           (default 256)
lr            : Shared Adam lr                (default 3e-4)
gamma         : Discount factor               (default 0.99)
lam           : GAE lambda                    (default 0.95)
clip_ratio    : PPO epsilon clip              (default 0.2)
target_kl     : Early stopping KL divergence  (default 0.015)
ppo_epochs    : Gradient steps per update     (default 10)
rollout_steps : Steps collected before update (default 2048)
batch_size    : Mini-batch size for epochs    (default 64)
device        : 'auto'|'cpu'|'cuda'           (default 'auto')
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal

from .base import BaseAgent


class PPOBuffer:
    """Stores on-policy rollouts for PPO updates."""
    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val, logp):
        assert self.ptr < self.max_size
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0):
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)

        # GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = self._discount_cumsum(deltas, self.gamma * self.lam)
        self.ret_buf[path_slice] = self._discount_cumsum(rews, self.gamma)[:-1]
        self.path_start_idx = self.ptr

    def get(self):
        assert self.ptr == self.max_size
        self.ptr, self.path_start_idx = 0, 0
        adv_mean, adv_std = np.mean(self.adv_buf), np.std(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / (adv_std + 1e-8)
        return dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf,
                    adv=self.adv_buf, logp=self.logp_buf)

    @staticmethod
    def _discount_cumsum(x, discount):
        import scipy.signal
        return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class PPOActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden, action_scale, action_bias):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, act_dim)
        )
        # Learnable standalone log_std is standard for continuous PPO
        self.log_std = nn.Parameter(-0.5 * torch.ones(act_dim))
        
        self.register_buffer("scale", torch.FloatTensor(action_scale))
        self.register_buffer("bias",  torch.FloatTensor(action_bias))

    def forward(self, obs, act=None):
        mu = self.net(obs)
        std = torch.exp(self.log_std)
        pi = Normal(mu, std)
        
        if act is None:
            # PPO samples unscaled actions from the distribution
            unscaled_act = pi.sample()
            logp_a = pi.log_prob(unscaled_act).sum(axis=-1)
            # Scale for the environment
            scaled_act = torch.tanh(unscaled_act) * self.scale + self.bias
            return scaled_act, unscaled_act, logp_a
        else:
            logp_a = pi.log_prob(act).sum(axis=-1)
            entropy = pi.entropy().sum(axis=-1)
            return logp_a, entropy


class PPOCritic(nn.Module):
    def __init__(self, obs_dim, hidden):
        super().__init__()
        self.v_net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1)
        )

    def forward(self, obs):
        return self.v_net(obs).squeeze(-1)


class PPOAgent(BaseAgent):
    def __init__(self, obs_dim, n_actions, config, action_space=None):
        super().__init__(obs_dim, n_actions, config)
        
        hidden           = config.get("hidden_size", 256)
        lr               = config.get("lr", 3e-4)
        self.gamma       = config.get("gamma", 0.99)
        self.lam         = config.get("lam", 0.95)
        self.clip_ratio  = config.get("clip_ratio", 0.2)
        self.target_kl   = config.get("target_kl", 0.015)
        self.ppo_epochs  = config.get("ppo_epochs", 10)
        self.batch_size  = config.get("batch_size", 64)
        self.rollout_steps = config.get("rollout_steps", 2048)
        dev              = config.get("device", "auto")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") \
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

        self.actor = PPOActor(obs_dim, action_dim, hidden, action_scale, action_bias).to(self.device)
        self.critic = PPOCritic(obs_dim, hidden).to(self.device)
        
        self.optimizer = optim.Adam([
            {'params': self.actor.parameters(), 'lr': lr},
            {'params': self.critic.parameters(), 'lr': lr}
        ])
        
        self.buffer = PPOBuffer(obs_dim, action_dim, self.rollout_steps, self.gamma, self.lam)
        self._temp_transition = {}
        self._step = 0

    def select_action(self, obs, explore=True):
        with torch.no_grad():
            o_t = torch.FloatTensor(obs).to(self.device)
            scaled_act, unscaled_act, logp = self.actor(o_t)
            val = self.critic(o_t)
            
            # Store components needed for the on-policy buffer update
            self._temp_transition = {
                'unscaled_act': unscaled_act.cpu().numpy(),
                'val': val.cpu().numpy(),
                'logp': logp.cpu().numpy()
            }
            
            if not explore:
                # Deterministic mean for evaluation
                mu = self.actor.net(o_t)
                return (torch.tanh(mu) * self.actor.scale + self.actor.bias).cpu().numpy()
                
            return scaled_act.cpu().numpy()

    def update(self, obs, action, reward, next_obs, done):
        # We store the *unscaled* action because that's what the Normal distribution log_prob expects
        self.buffer.store(obs, self._temp_transition['unscaled_act'], reward, 
                          self._temp_transition['val'], self._temp_transition['logp'])
        
        self._step += 1
        
        # If the episode ends, bootstrap value. If truncated, bootstrap with critic.
        if done or self._step % self.rollout_steps == 0:
            if done:
                last_val = 0.0
            else:
                with torch.no_grad():
                    o_t = torch.FloatTensor(next_obs).to(self.device)
                    last_val = self.critic(o_t).cpu().numpy()
            
            self.buffer.finish_path(last_val)

        # Execute PPO update only when buffer is full
        if self._step % self.rollout_steps == 0:
            return self._optimize_ppo()
        return {}

    def _optimize_ppo(self):
        data = self.buffer.get()
        obs = torch.FloatTensor(data['obs']).to(self.device)
        act = torch.FloatTensor(data['act']).to(self.device)
        ret = torch.FloatTensor(data['ret']).to(self.device)
        adv = torch.FloatTensor(data['adv']).to(self.device)
        logp_old = torch.FloatTensor(data['logp']).to(self.device)

        dataset = torch.utils.data.TensorDataset(obs, act, ret, adv, logp_old)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        metrics = {'pi_loss': [], 'v_loss': [], 'kl': []}

        for i in range(self.ppo_epochs):
            for b_obs, b_act, b_ret, b_adv, b_logp_old in loader:
                # Calculate new log probabilities and entropy
                logp, entropy = self.actor(b_obs, b_act)
                val = self.critic(b_obs)

                # Policy Loss with Clipping
                ratio = torch.exp(logp - b_logp_old)
                clip_adv = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * b_adv
                pi_loss = -(torch.min(ratio * b_adv, clip_adv)).mean()

                # Value Loss
                v_loss = nn.MSELoss()(val, b_ret)

                # Total Loss
                loss = pi_loss + 0.5 * v_loss - 0.01 * entropy.mean()

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.optimizer.step()

                # Calculate approx KL divergence for early stopping
                approx_kl = (b_logp_old - logp).mean().item()
                metrics['pi_loss'].append(pi_loss.item())
                metrics['v_loss'].append(v_loss.item())
                metrics['kl'].append(approx_kl)
            
            if np.mean(metrics['kl']) > 1.5 * self.target_kl:
                break # Early stopping

        return {
            "actor_loss": np.mean(metrics['pi_loss']),
            "critic_loss": np.mean(metrics['v_loss']),
            "kl_divergence": np.mean(metrics['kl'])
        }

    def save(self, path):
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
        }, path)

    def load(self, path):
        d = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(d["actor"])
        self.critic.load_state_dict(d["critic"])