"""
Behavioural metrics for LinearTrack rollouts.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class EpisodeRecord:
    positions: list[int] = field(default_factory=list)
    licked: list[bool] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    goal_reached_steps: list[int] = field(default_factory=list)
    rewarded_lick_steps: list[int] = field(default_factory=list)

    def append_step(
        self,
        pos: int,
        licked: bool,
        reward: float,
        *,
        goal_reached: bool,
        rewarded_lick: bool,
    ) -> None:
        """Record one env step (post-step position and lick flag)."""
        t = len(self.licked)
        self.positions.append(pos)
        self.licked.append(licked)
        self.rewards.append(reward)
        if goal_reached:
            self.goal_reached_steps.append(t)
        if rewarded_lick:
            self.rewarded_lick_steps.append(t)


@dataclass
class EpisodeMetrics:
    inter_terminal_steps: int | None
    lick_latencies: list[int]
    lick_frequency_by_pos: np.ndarray
    total_licks: int
    goal_reached: bool


def inter_terminal_steps(record: EpisodeRecord, n_cells: int) -> int | None:
    """Steps from episode start (left end) to first arrival at right terminal."""
    right = n_cells - 1
    for t, p in enumerate(record.positions):
        if p == right:
            return t + 1  # 1-indexed step count from start
    return None


def lick_latencies(record: EpisodeRecord) -> list[int]:
    """
    Steps between a reward-associated lick and the next lick.

    Anchor = lick on the goal-reward step, or (if none) the goal step itself.
    """
    lick_steps = [t for t, l in enumerate(record.licked) if l]
    anchors = list(record.rewarded_lick_steps)
    if not anchors and record.goal_reached_steps:
        anchors = list(record.goal_reached_steps)

    latencies: list[int] = []
    for anchor in anchors:
        later = [s for s in lick_steps if s > anchor]
        if later:
            latencies.append(later[0] - anchor)
    return latencies


def lick_frequency_by_position(record: EpisodeRecord, n_cells: int) -> np.ndarray:
    """
    Licks per position, counted within each contiguous stay before moving away.
    Uses post-step positions (one per lick flag).
    """
    freq = np.zeros(n_cells, dtype=np.int32)
    n = len(record.licked)
    if n == 0:
        return freq

    i = 0
    while i < n:
        pos = record.positions[i]
        j = i
        while j < n and record.positions[j] == pos:
            j += 1
        freq[pos] += sum(1 for k in range(i, j) if record.licked[k])
        i = j
    return freq


def compute_episode_metrics(record: EpisodeRecord, n_cells: int) -> EpisodeMetrics:
    return EpisodeMetrics(
        inter_terminal_steps=inter_terminal_steps(record, n_cells),
        lick_latencies=lick_latencies(record),
        lick_frequency_by_pos=lick_frequency_by_position(record, n_cells),
        total_licks=sum(record.licked),
        goal_reached=bool(record.goal_reached_steps),
    )


def aggregate_lick_frequency(records: list[EpisodeRecord], n_cells: int) -> np.ndarray:
    total = np.zeros(n_cells, dtype=np.float64)
    for rec in records:
        total += lick_frequency_by_position(rec, n_cells)
    return total
