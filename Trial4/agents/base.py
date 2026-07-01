# agents/base.py
class BaseAgent:
    def __init__(self, obs_dim, n_actions, config):
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.config = config

    def select_action(self, obs, explore=True):
        raise NotImplementedError

    def update(self, obs, action, reward, next_obs, done):
        raise NotImplementedError

    def on_episode_end(self):
        return {}
        
    def save(self, path):
        pass

    def load(self, path):
        pass