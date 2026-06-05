"""
SARSA — on-policy TD control.

With lambda_=0 this is one-step SARSA; with lambda_>0 it is SARSA(λ)
using accumulating eligibility traces (episodic).
"""

from __future__ import annotations

import numpy as np

from .base import TabularAgent


class SARSA(TabularAgent):
    """On-policy SARSA / SARSA(λ)."""

    name = "sarsa"

    def __init__(self, n_cells: int, *, lambda_: float = 0.0, **kwargs):
        super().__init__(n_cells, lambda_=lambda_, **kwargs)
        self._e = np.zeros_like(self.Q)

    def begin_episode(self) -> None:
        self._e.fill(0.0)

    def update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        next_action: int,
        terminated: bool,
    ) -> float:
        next_q = 0.0 if terminated else self.Q[next_state, next_action]
        td_error = reward + self.gamma * next_q - self.Q[state, action]

        if self.lambda_ == 0.0:
            self.Q[state, action] += self.alpha * td_error
            return td_error

        self._e[state, action] += 1.0
        self.Q += self.alpha * td_error * self._e
        self._e *= self.gamma * self.lambda_
        return td_error
