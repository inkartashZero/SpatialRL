"""
TD(λ) control — on-policy and off-policy variants.

OnPolicyTDLambda  : SARSA(λ)  (same update as SARSA with traces)
OffPolicyTDLambda : Watkins Q(λ)  (greedy bootstrap, traces cut on exploratory next action)
"""

from __future__ import annotations

import numpy as np

from .base import TabularAgent
from .sarsa import SARSA


class OnPolicyTDLambda(SARSA):
    """
    On-policy TD(λ) control via SARSA(λ).

    Parameters
    ----------
    lambda_ : trace decay in [0, 1]; 0 → SARSA(0), 1 → Monte Carlo-like credit.
    """

    name = "td_lambda_on"


class OffPolicyTDLambda(TabularAgent):
    """
    Off-policy TD(λ) control via Watkins Q(λ).

    Bootstrap target uses max_a Q(s', a). Eligibility traces are cleared
    when the next action is not greedy w.r.t. Q(s', ·).
    """

    name = "td_lambda_off"

    def __init__(self, n_cells: int, *, lambda_: float = 0.7, **kwargs):
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
        if terminated:
            next_q = 0.0
            greedy_next = True
        else:
            next_q = float(self.Q[next_state].max())
            greedy_next = next_action == self.greedy_action(next_state)

        td_error = reward + self.gamma * next_q - self.Q[state, action]

        if self.lambda_ == 0.0:
            self.Q[state, action] += self.alpha * td_error
            return td_error

        self._e[state, action] += 1.0
        self.Q += self.alpha * td_error * self._e
        self._e *= self.gamma * self.lambda_

        if not greedy_next:
            self._e.fill(0.0)

        return td_error
