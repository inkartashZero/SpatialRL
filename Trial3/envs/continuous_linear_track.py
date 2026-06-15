"""
ContinuousLinearTrackEnv (Streamlined Cognitive Version)
========================================================
State space (continuous, 6-dim):
  [position_norm, velocity_norm, is_terminal, tactile, licked_left_flag, licked_right_flag]

Features:
  • is_terminal : 1.0 if inside any lick port zone, 0.0 otherwise.
  • tactile     : 0.0 if on the left half of the track body, 1.0 on the right half.
  • water_reward: Flat +1.0 (Temporal discounting is handled inherently by the agent's Gamma).
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

class ContinuousLinearTrackEnv(gym.Env):
    metadata = {"render_modes": ["ansi", "rgb_array"]}

    def __init__(
        self,
        track_length      : float = 120.0,
        max_vel           : float = 10.0,
        terminal_width    : float = 1.0,
        dt                : float = 1.0,
        max_steps         : int   = 500,
        water_reward      : float = 1.0,
        step_penalty      : float = -0.005,
        lick_penalty      : float = -0.05,
        wrong_lick_penalty: float = 0.0,
        render_mode       : str | None = None,
    ):
        super().__init__()

        self.L               = float(track_length)
        self.max_vel         = float(max_vel)
        self.terminal_width  = float(terminal_width)
        self.dt              = float(dt)
        self.max_steps       = max_steps
        self.water_reward    = water_reward
        self.step_penalty    = step_penalty
        self.lick_penalty    = lick_penalty
        self.wrong_lick_pen  = wrong_lick_penalty
        self.render_mode     = render_mode
        self.phase           = "mapping"
        
        self.action_space = spaces.Box(
            low  = np.array([-max_vel, -1.0], dtype=np.float32),
            high = np.array([ max_vel,  1.0], dtype=np.float32),
            dtype= np.float32,
        )

        # 6-Dimensional Streamlined Observation Space
        self.observation_space = spaces.Box(
            low  = np.array([0.0, -1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            high = np.ones(6, dtype=np.float32),
            dtype= np.float32,
        )

        self._pos          = 0.0
        self._vel          = 0.0
        self._licked_left  = False
        self._licked_right = False
        self._steps        = 0
        self._trajectory   : list[float] = []
        self._lick_events  : list[tuple[float, str]] = []  

    def set_phase(self, phase: str):
        assert phase in ["mapping", "remapping"], f"Unknown phase: {phase}"
        self.phase = phase

    def _in_left_zone(self)  -> bool:
        return self._pos <= self.terminal_width

    def _in_right_zone(self) -> bool:
        return self._pos >= self.L - self.terminal_width

    def _get_tactile(self) -> float:
        return 0.0 if self._pos <= self.L / 2.0 else 1.0

    def _build_obs(self) -> np.ndarray:
        pos_norm    = np.array([np.clip(self._pos / self.L, 0.0, 1.0)], dtype=np.float32)
        vel_norm    = np.array([np.clip(self._vel / self.max_vel, -1.0, 1.0)], dtype=np.float32)
        is_terminal = np.array([1.0 if (self._in_left_zone() or self._in_right_zone()) else 0.0], dtype=np.float32)
        tactile     = np.array([self._get_tactile()], dtype=np.float32)
        flags       = np.array([float(self._licked_left), float(self._licked_right)], dtype=np.float32)
        
        return np.concatenate([pos_norm, vel_norm, is_terminal, tactile, flags])

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        lo = self.L / 3
        hi = 2 * self.L / 3
        self._pos          = float(self.np_random.uniform(lo, hi))
        self._vel          = 0.0
        self._licked_left  = False
        self._licked_right = False
        self._steps        = 0
        self._trajectory   = [self._pos]
        self._lick_events  = []
        
        self._step_left_first = None
        self._step_right_after_left = None
        self._step_right_first = None
        self._step_left_after_right = None
        self._n_successes = 0

        return self._build_obs(), {}

    def step(self, action: np.ndarray):
        action  = np.asarray(action, dtype=np.float32)
        vel_cmd = float(np.clip(action[0], -self.max_vel, self.max_vel))
        do_lick = bool(action[1] > 0.0)

        self._vel  = vel_cmd
        self._pos  = float(np.clip(self._pos + vel_cmd * self.dt, 0.0, self.L))
        self._steps += 1
        self._trajectory.append(self._pos)

        reward     = self.step_penalty
        terminated = False
        lick_zone  = None
        intra_trial_duration_this_step = None

        if do_lick:
            reward += self.lick_penalty   

            if self._in_left_zone():
                lick_zone = "left"
                self._lick_events.append((self._steps, "left"))
                
                if not self._licked_left:
                    if self.phase == "remapping":
                        self._licked_left = True
                        self._step_left_first = self._steps
                    elif self.phase == "mapping":
                        if self._licked_right:
                            self._licked_left = True
                            self._step_left_after_right = self._steps
                            
                            intra_trial_duration_this_step = self._step_left_after_right - self._step_right_first
                            reward += self.water_reward
                            self._n_successes += 1
                            
                            self._licked_left  = False
                            self._licked_right = False
                            self._step_right_first = None

            elif self._in_right_zone():
                lick_zone = "right"
                self._lick_events.append((self._steps, "right"))
                
                if not self._licked_right:
                    if self.phase == "remapping" and self._licked_left:
                        self._licked_right = True
                        self._step_right_after_left = self._steps
                        
                        intra_trial_duration_this_step = self._step_right_after_left - self._step_left_first
                        reward += self.water_reward
                        self._n_successes += 1
                        
                        self._licked_left  = False
                        self._licked_right = False
                        self._step_left_first = None
                    elif self.phase == "mapping":
                        self._licked_right = True
                        self._step_right_first = self._steps

            else:
                lick_zone = "body"
                self._lick_events.append((self._steps, "body"))
                reward += self.wrong_lick_pen   

        truncated = self._steps >= self.max_steps

        info = {
            "pos"              : self._pos,
            "vel"              : self._vel,
            "licked_left"      : self._licked_left,
            "licked_right"     : self._licked_right,
            "lick_zone"        : lick_zone,
            "steps"            : self._steps,
            "n_successes"      : self._n_successes,
            "intra_trial_duration" : intra_trial_duration_this_step,  
            "in_left_zone"     : self._in_left_zone(),
            "in_right_zone"    : self._in_right_zone(),
            "trajectory"       : list(self._trajectory),
            "lick_events"      : list(self._lick_events),
        }

        return self._build_obs(), float(reward), terminated, truncated, info

    @property
    def obs_labels(self):
        return ["pos_norm", "vel_norm", "is_terminal", "tactile", "licked_L", "licked_R"]