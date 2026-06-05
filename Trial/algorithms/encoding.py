"""
State / action encoding for LinearTrackEnv (tabular methods).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from linear_track import LinearTrackEnv

# Tabular indices -> env dict {move: -1|0|+1, lick: 0|1}
ACTION_TABLE: list[dict] = [
    {"move": -1, "lick": 0},
    {"move": 0, "lick": 0},
    {"move": 1, "lick": 0},
    {"move": 0, "lick": 1},
]

N_ACTIONS = len(ACTION_TABLE)
ACTION_NAMES = ("back(-1)", "wait(0)", "fwd(+1)", "lick")


def action_to_env(action_idx: int) -> dict:
    return ACTION_TABLE[int(action_idx)].copy()


def action_to_dict(action_idx: int) -> dict:
    return action_to_env(action_idx)


def encode_state(env: LinearTrackEnv) -> int:
    pos = int(env._pos)
    facing = 0 if env._facing < 0 else 1
    vL = int(env._visited_left)
    vR = int(env._visited_right)
    return pos * 8 + facing * 4 + vL * 2 + vR


def n_states(n_cells: int) -> int:
    return n_cells * 8


def decode_state(state_idx: int, n_cells: int) -> tuple[int, int, int, int]:
    vR = state_idx % 2
    state_idx //= 2
    vL = state_idx % 2
    state_idx //= 2
    facing = state_idx % 2
    pos = state_idx // 2
    return pos, facing, vL, vR
