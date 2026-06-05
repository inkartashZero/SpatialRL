"""
LinearTrack Environment
=======================
A 1D track where:
  - Positions 0..N-1
  - Left half:  tactile=0 (smooth), colour=BLACK  [0, 0, 0]
  - Right half: tactile=1 (rough),  colour=YELLOW [1, 1, 0]  (or WHITE [1,1,1])
  - Terminal cells (pos=0 and pos=N-1): colour=RED [1, 0, 0]

Actions (2-part, rodent-style):
  action = {"move": -1|0|+1, "lick": 0|1}
  - move : step relative to facing (+1 = ahead, -1 = backward, 0 = stay)
  - lick : 1 = lick in place (move forced to 0; no displacement)

Reward structure (left → right goal):
  Start at the left terminal (pos=0). Reach the right terminal (pos=N-1)
  after having been at the left end → REWARD (+10 by default), episode ends.

Observation:
  obs = [
    position_normalised,   # float in [0, 1]
    tactile,               # 0 (smooth/left) or 1 (rough/right)
    colour_r,              # red channel
    colour_g,              # green channel
    colour_b,              # blue channel
    visited_left_flag,     # binary: has agent been at the left terminal this episode
    visited_right_flag,    # binary: has agent reached the right terminal this episode
  ]
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


# ── colour palette ──────────────────────────────────────────────────────────
COLOUR_BLACK  = np.array([0.0, 0.0, 0.0], dtype=np.float32)
COLOUR_YELLOW = np.array([1.0, 1.0, 0.0], dtype=np.float32)
COLOUR_RED    = np.array([1.0, 0.0, 0.0], dtype=np.float32)

TACTILE_SMOOTH = 0.0   # left half
TACTILE_ROUGH  = 1.0   # right half


# Flat tabular index -> (move, lick) for agents using Discrete(4)
_FLAT_ACTIONS = ((-1, 0), (0, 0), (1, 0), (0, 1))


class LinearTrackEnv(gym.Env):
    """
    Parameters
    ----------
    length       : total track length (must be even, >= 4)
    max_steps     : episode time-limit
    step_penalty    : small negative reward every step (encourages efficiency)
    goal_reward     : reward for completing left → right (+10 by default)
    wrong_end_penalty : unused for directed goal (kept for API compatibility)
    terminate_on_goal : if True, end episode when goal reward is delivered
    render_mode       : 'ansi' | 'rgb_array' | None
    """

    metadata = {"render_modes": ["ansi", "rgb_array"]}

    def __init__(
        self,
        length: float = 50,
        max_steps: int = 200,
        step_penalty: float = -0.01,
        goal_reward: float = 10.0,
        wrong_end_penalty: float = 0.0,
        terminate_on_goal: bool = False,
        render_mode: str | None = None,
    ):
        super().__init__()
        assert length >= 4 and length % 2 == 0, "length must be even and >= 4"

        self.length           = length
        self.max_steps         = max_steps
        self.step_penalty      = step_penalty
        self.goal_reward       = goal_reward
        self.wrong_end_penalty = wrong_end_penalty
        self.terminate_on_goal = terminate_on_goal
        self.render_mode       = render_mode

        self.mid = length // 2          # first index of right half

        # move: 0=-1, 1=0, 2=+1 (relative to facing); lick: 0/1
        self.action_space = spaces.Dict({
            "move": spaces.Discrete(3),
            "lick": spaces.Discrete(2),
        })
        self._MOVE_VALUES = (-1, 0, 1)

        # Observation: [pos_norm, tactile, r, g, b, visited_left, visited_right]
        self.observation_space = spaces.Box(
            low   = np.zeros(7, dtype=np.float32),
            high  = np.ones(7,  dtype=np.float32),
            dtype = np.float32,
        )

        self._pos            = 0
        self._facing         = 1    # +1 toward higher indices, -1 toward lower
        self._visited_left   = False
        self._visited_right  = False
        self._steps          = 0
        self._licks              = 0
        self._goal_reward_given  = False
        self._trajectory         = []

    # ── helpers ─────────────────────────────────────────────────────────────

    def _get_colour(self, pos: int) -> np.ndarray:
        if pos == 0 or pos == self.length - 1:
            return COLOUR_RED.copy()
        if pos < self.mid:
            return COLOUR_BLACK.copy()
        return COLOUR_YELLOW.copy()

    def _get_tactile(self, pos: int) -> float:
        return TACTILE_ROUGH if pos >= self.mid else TACTILE_SMOOTH

    def _build_obs(self) -> np.ndarray:
        pos = self._pos
        colour  = self._get_colour(pos)
        tactile = self._get_tactile(pos)
        obs = np.array([
            pos / (self.length - 1),   # normalised position
            tactile,
            colour[0], colour[1], colour[2],
            float(self._visited_left),
            float(self._visited_right),
        ], dtype=np.float32)
        return obs

    def _is_terminal_left(self)  -> bool: return self._pos == 0
    def _is_terminal_right(self) -> bool: return self._pos == self.length - 1

    def _move_to_index(self, move: int) -> int:
        """Map move in {-1,0,+1} or index {0,1,2} to action_space move index."""
        if move in (-1, 0, 1):
            return move + 1
        if move in (0, 1, 2):
            return move
        raise ValueError(f"Invalid move {move}")

    def _parse_action(self, action) -> tuple[int, bool]:
        """Return (move_relative, lick_bool). move in {-1, 0, +1} vs facing."""
        if isinstance(action, dict):
            if "move" in action and "lick" in action:
                lick = bool(int(action["lick"]))
                if lick:
                    return 0, True
                raw = action["move"]
                if isinstance(raw, (int, np.integer)) and raw in (-1, 0, 1):
                    return int(raw), False
                move_idx = int(raw)
                return self._MOVE_VALUES[move_idx], False
            # legacy flat tabular index
            if "action" in action:
                idx = int(action["action"])
                move, lick_i = _FLAT_ACTIONS[idx]
                return move, bool(lick_i)

        arr = np.asarray(action, dtype=np.int64).ravel()
        if len(arr) == 1:
            move, lick_i = _FLAT_ACTIONS[int(arr[0])]
            return move, bool(lick_i)
        if len(arr) == 2:
            move_idx, lick = int(arr[0]), bool(arr[1])
            if lick:
                return 0, True
            return self._MOVE_VALUES[move_idx], False

        raise ValueError(f"Invalid action {action}")

    # ── gym interface ────────────────────────────────────────────────────────

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        # start at left end, facing right (goal: traverse to right terminal)
        self._pos           = 0
        self._facing        = 1
        self._visited_left  = True
        self._visited_right      = False
        self._goal_reward_given  = False
        self._steps              = 0
        self._licks              = 0
        self._trajectory         = [self._pos]
        return self._build_obs(), {}

    def step(self, action):
        move, lick = self._parse_action(action)
        if isinstance(action, dict) and "move" in action and "lick" in action:
            lick_i = int(action["lick"])
            move_i = self._move_to_index(int(action["move"]))
            assert self.action_space.contains({"move": move_i, "lick": lick_i}), (
                f"Invalid action {action}"
            )

        if move != 0:
            next_pos = self._pos + move * self._facing
            if 0 <= next_pos < self.length:
                self._pos = next_pos

        if lick:
            self._licks += 1

        self._steps += 1
        self._trajectory.append(self._pos)

        reward    = self.step_penalty
        terminated = False

        if self._is_terminal_left():
            self._visited_left = True
        goal_reached = False
        got_goal_reward = False
        if self._is_terminal_right():
            self._visited_right = True
            if self._visited_left and not self._goal_reward_given:
                reward += self.goal_reward
                self._goal_reward_given = True
                goal_reached = True
                got_goal_reward = True
                if self.terminate_on_goal:
                    terminated = True
        rewarded_lick = lick and got_goal_reward

        truncated = (self._steps >= self.max_steps)

        info = {
            "pos"            : self._pos,
            "facing"         : self._facing,
            "visited_left"   : self._visited_left,
            "visited_right"  : self._visited_right,
            "goal_reached"   : goal_reached,
            "rewarded_lick"  : rewarded_lick,
            "move"           : move,
            "action"         : action,
            "licked"         : lick,
            "licks"          : self._licks,
            "trajectory"     : list(self._trajectory),
            "steps"          : self._steps,
        }

        if self.render_mode == "ansi":
            self.render()

        return self._build_obs(), reward, terminated, truncated, info

    # ── rendering ────────────────────────────────────────────────────────────

    def render(self):
        if self.render_mode == "ansi":
            track = []
            for i in range(self.length):
                if i == self._pos:
                    track.append("A")          # Agent
                elif i == 0 or i == self.length - 1:
                    track.append("R")          # Red terminal
                elif i < self.mid:
                    track.append(".")          # Black / smooth
                else:
                    track.append("#")          # Yellow / rough
            facing = "→" if self._facing > 0 else "←"
            print(f"[{''.join(track)}]  pos={self._pos:3d} {facing}  "
                  f"vL={int(self._visited_left)} vR={int(self._visited_right)}  "
                  f"licks={self._licks}  step={self._steps}")
            return None

        if self.render_mode == "rgb_array":
            # Return H×W×3 image of the track (one row high per cell, scaled up)
            cell_h, cell_w = 40, 20
            img = np.zeros((cell_h, self.length * cell_w, 3), dtype=np.uint8)
            for i in range(self.length):
                col = (self._get_colour(i) * 255).astype(np.uint8)
                img[:, i*cell_w:(i+1)*cell_w] = col
            # draw agent as a white square
            ax = self._pos * cell_w
            img[10:30, ax+5:ax+15] = [255, 255, 255]
            return img

    # ── utility ──────────────────────────────────────────────────────────────

    def get_track_colours(self) -> np.ndarray:
        """Returns (length, 3) array of RGB colours for visualisation."""
        return np.stack([self._get_colour(i) for i in range(self.length)])

    def get_track_tactile(self) -> np.ndarray:
        """Returns (length,) array of tactile values."""
        return np.array([self._get_tactile(i) for i in range(self.length)])
