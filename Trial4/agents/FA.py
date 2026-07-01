import numpy as np
from itertools import product
import gymnasium as gym

# ── 1. Discrete Action Wrapper ───────────────────────────────────────────────

class DiscreteActionWrapper(gym.ActionWrapper):
    """Wraps the continuous track to accept discrete integers."""
    def __init__(self, env, n_vel_bins=5):
        super().__init__(env)
        
        # Create discrete bins: e.g., velocity in [-10, -5, 0, 5, 10]
        vel_space = np.linspace(-env.max_vel, env.max_vel, n_vel_bins)
        lick_space = [-1.0, 1.0] # No lick, Lick
        
        # Create Cartesian product of all possible actions
        self.action_map = list(product(vel_space, lick_space))
        self.action_space = gym.spaces.Discrete(len(self.action_map))

    def action(self, action):
        # Map integer to continuous [velocity, lick] array
        return np.array(self.action_map[action], dtype=np.float32)

# ── 2. Feature Extractor (RBF) ────────────────────────────────────────────────

class RBFExtractor:
    def __init__(self, obs_dim, n_components=100, gamma=2.0):
        self.n_components = n_components
        self.gamma = gamma
        # Randomly sample centers across the normalised observation space [0, 1]
        self.centers = np.random.uniform(0.0, 1.0, (n_components, obs_dim))

    def get_features(self, obs):
        # Compute RBF: exp(-gamma * ||x - c||^2)
        diff = obs - self.centers
        sq_dist = np.sum(diff**2, axis=1)
        features = np.exp(-self.gamma * sq_dist)
        # Add bias term
        return np.append(features, 1.0)

# ── 3. Base Linear FA Agent ───────────────────────────────────────────────────

class LinearFA_Base:
    def __init__(self, obs_dim, n_actions, config):
        self.n_actions = n_actions
        self.lr = config.get("lr", 0.01)
        self.gamma = config.get("gamma", 0.99)
        self.epsilon = config.get("epsilon", 0.1)
        
        self.rbf = RBFExtractor(obs_dim, n_components=config.get("n_features", 100))
        self.feature_dim = self.rbf.n_components + 1
        
        # Initialize weight matrix [n_actions, feature_dim]
        self.weights = np.zeros((self.n_actions, self.feature_dim))
        
    def get_q_values(self, features):
        return self.weights.dot(features)
        
    def select_action(self, obs, explore=True):
        if explore and np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        
        features = self.rbf.get_features(obs)
        q_vals = self.get_q_values(features)
        return np.argmax(q_vals)

    def on_episode_end(self):
        return {}

# ── 4. Q-Learning with Linear FA ─────────────────────────────────────────────

class QLearningFA(LinearFA_Base):
    def update(self, obs, action, reward, next_obs, done):
        phi = self.rbf.get_features(obs)
        phi_next = self.rbf.get_features(next_obs)
        
        q_val = self.weights[action].dot(phi)
        
        if done:
            target = reward
        else:
            q_next = np.max(self.get_q_values(phi_next))
            target = reward + self.gamma * q_next
            
        error = target - q_val
        self.weights[action] += self.lr * error * phi
        
        return {"td_error": error}

# ── 5. SARSA with Linear FA ──────────────────────────────────────────────────

class SarsaFA(LinearFA_Base):
    def __init__(self, obs_dim, n_actions, config):
        super().__init__(obs_dim, n_actions, config)
        self.next_action = None # Keep track of action for next step
        
    def update(self, obs, action, reward, next_obs, done):
        phi = self.rbf.get_features(obs)
        q_val = self.weights[action].dot(phi)
        
        if done:
            target = reward
            self.next_action = None
        else:
            # On-policy: select next action using current policy
            self.next_action = self.select_action(next_obs, explore=True)
            phi_next = self.rbf.get_features(next_obs)
            q_next = self.weights[self.next_action].dot(phi_next)
            target = reward + self.gamma * q_next
            
        error = target - q_val
        self.weights[action] += self.lr * error * phi
        
        return {"td_error": error}

# ── 6. SARSA(λ) / TD(λ) with Linear FA ───────────────────────────────────────

class SarsaLambdaFA(LinearFA_Base):
    def __init__(self, obs_dim, n_actions, config):
        super().__init__(obs_dim, n_actions, config)
        self.lambd = config.get("lambda", 0.9)
        self.eligibility_traces = np.zeros_like(self.weights)
        self.next_action = None
        
    def update(self, obs, action, reward, next_obs, done):
        phi = self.rbf.get_features(obs)
        q_val = self.weights[action].dot(phi)
        
        # Accumulating traces
        self.eligibility_traces *= (self.gamma * self.lambd)
        self.eligibility_traces[action] += phi
        
        if done:
            target = reward
            self.next_action = None
        else:
            self.next_action = self.select_action(next_obs, explore=True)
            phi_next = self.rbf.get_features(next_obs)
            q_next = self.weights[self.next_action].dot(phi_next)
            target = reward + self.gamma * q_next
            
        error = target - q_val
        self.weights += self.lr * error * self.eligibility_traces
        
        # Reset traces if done
        if done:
            self.eligibility_traces.fill(0)
            
        return {"td_error": error}