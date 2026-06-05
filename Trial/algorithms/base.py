"""
Shared tabular agent utilities (epsilon-greedy policy over Q).
"""

from __future__ import annotations

import numpy as np

from .encoding import N_ACTIONS, action_to_env, encode_state, n_states


class TabularAgent:
    """Base class for model-free tabular control on LinearTrackEnv."""

    name: str = "tabular"

    def __init__(
        self,
        n_cells: int,
        *,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        lambda_: float = 0.0,
        seed: int | None = None,
    ):
        self.n_cells = n_cells
        self.n_states = n_states(n_cells)
        self.n_actions = N_ACTIONS
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.lambda_ = lambda_

        self.rng = np.random.default_rng(seed)
        self.Q = np.zeros((self.n_states, self.n_actions), dtype=np.float64)
        self._episode = 0

    def state(self, env) -> int:
        return encode_state(env)

    def select_action(self, state: int, explore: bool = True) -> int:
        if explore and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))
        q = self.Q[state]
        best = np.flatnonzero(q == q.max())
        return int(self.rng.choice(best))

    def greedy_action(self, state: int) -> int:
        q = self.Q[state]
        best = np.flatnonzero(q == q.max())
        return int(self.rng.choice(best))

    def env_action(self, action_idx: int) -> dict:
        return action_to_env(action_idx)

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def begin_episode(self) -> None:
        """Reset per-episode quantities (e.g. eligibility traces)."""
        pass

    def end_episode(self) -> None:
        self._episode += 1
        self.decay_epsilon()

    def update(self, *args, **kwargs) -> float | None:
        raise NotImplementedError
