"""
Training and evaluation rollouts for tabular agents on LinearTrackEnv.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from linear_track import LinearTrackEnv
from metrics import EpisodeMetrics, EpisodeRecord, compute_episode_metrics

from .base import TabularAgent
from .encoding import encode_state
from .q_learning import QLearning
from .sarsa import SARSA
from .td_lambda import OffPolicyTDLambda, OnPolicyTDLambda


REGISTRY: dict[str, type[TabularAgent]] = {
    "sarsa": SARSA,
    "q_learning": QLearning,
    "td_lambda_on": OnPolicyTDLambda,
    "td_lambda_off": OffPolicyTDLambda,
}


@dataclass
class EpisodeStats:
    episode: int
    total_reward: float
    steps: int
    success: bool
    epsilon: float
    inter_terminal_steps: int | None = None
    mean_lick_latency: float | None = None


@dataclass
class RolloutResult:
    stats: EpisodeStats
    record: EpisodeRecord
    metrics: EpisodeMetrics
    trajectory: list[int]


def run_episode(
    env: LinearTrackEnv,
    agent: TabularAgent,
    explore: bool = True,
    *,
    record: EpisodeRecord | None = None,
) -> RolloutResult:
    agent.begin_episode()
    env.reset()
    state = encode_state(env)
    action = agent.select_action(state, explore=explore)

    if record is None:
        record = EpisodeRecord()
    start_pos = env._pos

    total_reward = 0.0
    steps = 0
    terminated = truncated = False
    info: dict = {}

    while not (terminated or truncated):
        obs, reward, terminated, truncated, info = env.step(agent.env_action(action))
        total_reward += reward
        steps += 1

        record.append_step(
            info["pos"],
            info["licked"],
            reward,
            goal_reached=bool(info.get("goal_reached")),
            rewarded_lick=bool(info.get("rewarded_lick")),
        )

        next_state = encode_state(env)

        if terminated or truncated:
            if isinstance(agent, QLearning):
                agent.update(state, action, reward, next_state, terminated)
            else:
                agent.update(state, action, reward, next_state, 0, terminated)
        else:
            next_action = agent.select_action(next_state, explore=explore)
            if isinstance(agent, QLearning):
                agent.update(state, action, reward, next_state, terminated)
            else:
                agent.update(
                    state, action, reward, next_state, next_action, terminated
                )
            state, action = next_state, next_action

    agent.end_episode()
    metrics = compute_episode_metrics(record, env.n_cells)
    success = bool(env._goal_reward_given)

    latencies = metrics.lick_latencies
    mean_lat = float(np.mean(latencies)) if latencies else None

    stats = EpisodeStats(
        episode=agent._episode,
        total_reward=total_reward,
        steps=steps,
        success=success,
        epsilon=agent.epsilon,
        inter_terminal_steps=metrics.inter_terminal_steps,
        mean_lick_latency=mean_lat,
    )
    return RolloutResult(
        stats=stats,
        record=record,
        metrics=metrics,
        trajectory=[start_pos] + list(record.positions),
    )


def train(
    agent: TabularAgent,
    env: LinearTrackEnv,
    n_episodes: int,
    *,
    log_every: int = 100,
    csv_path: str | Path | None = None,
) -> list[EpisodeStats]:
    history: list[EpisodeStats] = []
    writer = None
    csv_file = None
    fields = [
        "episode",
        "total_reward",
        "steps",
        "success",
        "epsilon",
        "inter_terminal_steps",
        "mean_lick_latency",
    ]

    if csv_path is not None:
        csv_path = Path(csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        csv_file = open(csv_path, "w", newline="")
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writeheader()

    for ep in range(n_episodes):
        result = run_episode(env, agent, explore=True)
        history.append(result.stats)

        if writer is not None:
            s = result.stats
            writer.writerow(
                {
                    "episode": s.episode,
                    "total_reward": s.total_reward,
                    "steps": s.steps,
                    "success": int(s.success),
                    "epsilon": s.epsilon,
                    "inter_terminal_steps": s.inter_terminal_steps or "",
                    "mean_lick_latency": s.mean_lick_latency or "",
                }
            )

        if log_every and (ep + 1) % log_every == 0:
            recent = history[-log_every:]
            sr = 100.0 * np.mean([s.success for s in recent])
            rew = np.mean([s.total_reward for s in recent])
            print(
                f"  ep {ep + 1:5d}/{n_episodes}  "
                f"reward={rew:+.3f}  success={sr:5.1f}%  eps={agent.epsilon:.3f}"
            )

    if csv_file is not None:
        csv_file.close()
        print(f"  log -> {csv_path}")

    return history


def evaluate_rollout(
    env: LinearTrackEnv,
    agent: TabularAgent,
    *,
    explore: bool = False,
    seed: int | None = None,
) -> RolloutResult:
    if seed is not None:
        env.reset(seed=seed)
    else:
        env.reset()
    return run_episode(env, agent, explore=explore, record=EpisodeRecord())


def build_agent(name: str, n_cells: int, **kwargs) -> TabularAgent:
    if name not in REGISTRY:
        raise KeyError(f"Unknown agent '{name}'. Choose from: {list(REGISTRY)}")
    return REGISTRY[name](n_cells, **kwargs)
