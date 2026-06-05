"""
Q-Learning — off-policy one-step TD control (lambda_=0 only).

For bootstrapping with traces and lambda_>0, use OffPolicyTDLambda (Watkins Q(λ)).
"""

from __future__ import annotations

from .base import TabularAgent


class QLearning(TabularAgent):
    """Off-policy Q-learning (TD(0) control)."""

    name = "q_learning"

    def __init__(self, n_cells: int, **kwargs):
        if kwargs.get("lambda_", 0.0) != 0.0:
            raise ValueError(
                "Q-learning is one-step off-policy (lambda=0). "
                "Use OffPolicyTDLambda for lambda > 0."
            )
        kwargs["lambda_"] = 0.0
        super().__init__(n_cells, **kwargs)

    def update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        terminated: bool,
    ) -> float:
        next_q = 0.0 if terminated else float(self.Q[next_state].max())
        td_error = reward + self.gamma * next_q - self.Q[state, action]
        self.Q[state, action] += self.alpha * td_error
        return td_error
