"""
ContinuousLinearTrackEnv
========================
A continuous 1D shuttle task inspired by rodent lick-port experiments.

Track layout (default length=120):
  ─────────────────────────────────────────────────────────────
  [LEFT LICK PORT]          uniform black           [RIGHT LICK PORT]
  pos=0  (RED)              pos ∈ (0, L)            pos=L  (YELLOW)
  ─────────────────────────────────────────────────────────────

State space  (continuous, 6-dim):
  [position_norm, velocity_norm, colour_r, colour_g, colour_b,
   licked_left_flag, licked_right_flag]

  • Track body: uniform BLACK  [0, 0, 0]
  • Left terminal:  RED        [1, 0, 0]   (lick-port A)
  • Right terminal: YELLOW     [1, 1, 0]   (lick-port B)
  • Terminal zone half-width configurable (default 3 units)

Action space (continuous, 2-dim):
  [velocity, lick]
  • velocity ∈ [-max_vel, +max_vel]   (default max_vel=10)
  • lick     ∈ [-1, +1]  → treated as binary: lick if > 0

Reward structure (ONE-DIRECTIONAL: LEFT → RIGHT ONLY):
  • step_penalty       : -0.005  per step (keeps agent moving)
  • lick_penalty       : -0.05   per lick action anywhere
  • water_reward       : +1.0    on correct sequence (every time LEFT then RIGHT)
  • wrong_lick_penalty : -0.02   licking at wrong port before
                                  visiting left first (optional, off by default)

Behavioral Metrics Tracked:
  • n_successes            : count of completed LEFT→RIGHT sequences per episode
  • intra_trial_duration   : steps between first left lick and subsequent right lick

Rationale (Ng et al. 1999; Engelhard et al. 2019):
  lick_penalty / water_reward ≈ 5 %  — in the safe shaping window
  step_penalty × max_steps   ≈ 0.5   — well below water_reward=1.0
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


# ── colour constants (float RGB) ──────────────────────────────────────────────
C_BLACK  = np.array([0.0, 0.0, 0.0], dtype=np.float32)
C_RED    = np.array([1.0, 0.0, 0.0], dtype=np.float32)   # left port
C_YELLOW = np.array([1.0, 1.0, 0.0], dtype=np.float32)   # right port


class ContinuousLinearTrackEnv(gym.Env):
    """
    Parameters
    ----------
    track_length      : physical length of track              (default 120)
    max_vel           : maximum speed per step                (default 10)
    terminal_width    : half-width of each lick-port zone     (default 3)
    dt                : timestep (position += velocity * dt)  (default 1.0)
    max_steps         : episode time-limit                    (default 500)
    water_reward      : reward for completing shuttle         (default 1.0)
    step_penalty      : per-step cost                        (default -0.005)
    lick_penalty      : cost of a lick action                 (default -0.05)
    wrong_lick_penalty: extra cost for licking wrong port     (default 0.0)
    render_mode       : 'ansi' | 'rgb_array' | None
    """

    metadata = {"render_modes": ["ansi", "rgb_array"]}

    def __init__(
        self,
        track_length      : float = 120.0,
        max_vel           : float = 10.0,
        terminal_width    : float = 1.0,
        dt                : float = 1.0,
        max_steps         : int   = 500,
        water_reward      : float = 10.0,
        step_penalty      : float = -0.005,
        lick_penalty      : float = -0.005,
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

        # ── action space: [velocity, lick] ──────────────────────────────────
        self.action_space = spaces.Box(
            low  = np.array([-max_vel, -1.0], dtype=np.float32),
            high = np.array([ max_vel,  1.0], dtype=np.float32),
            dtype= np.float32,
        )

        # ── observation: [pos_norm, vel_norm, r, g, b, lick_L, lick_R] ──────
        self.observation_space = spaces.Box(
            low  = np.zeros(3, dtype=np.float32),
            high = np.ones(3,  dtype=np.float32),
            dtype= np.float32,
        )

        # internal state
        self._pos          = 0.0
        self._vel          = 0.0
        self._licked_left  = False
        self._licked_right = False
        self._shuttle_completed = False  # track if shuttle completed (for one-time reward)
        self._steps        = 0
        self._trajectory   : list[float] = []
        self._lick_events  : list[tuple[float, str]] = []  # (pos, 'left'|'right'|'body')

    # ── terminal zone checks ──────────────────────────────────────────────────
    def set_phase(self, phase: str):
        """Sets the behavioral phase: 'mapping' or 'remapping'."""
        assert phase in ["mapping", "remapping"], f"Unknown phase: {phase}"
        self.phase = phase

    def _in_left_zone(self)  -> bool:
        return self._pos <= self.terminal_width

    def _in_right_zone(self) -> bool:
        return self._pos >= self.L - self.terminal_width

    # ── colour at current position ────────────────────────────────────────────

    def _get_colour(self, pos: float | None = None) -> np.ndarray:
        p = self._pos if pos is None else pos
        if p <= self.terminal_width:
            return C_RED.copy()
        if p >= self.L - self.terminal_width:
            return C_YELLOW.copy()
        return C_BLACK.copy()

    # ── observation builder ───────────────────────────────────────────────────

    def _build_obs(self) -> np.ndarray:
        return np.array([
            np.clip(self._pos / self.L, 0.0, 1.0),                    # position
            float(self._licked_left),
            float(self._licked_right),
        ], dtype=np.float32)

    # ── reset ─────────────────────────────────────────────────────────────────

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        # start in middle third, zero velocity
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
        self._n_successes = 0

        return self._build_obs(), {}

    # ── step ──────────────────────────────────────────────────────────────────

    def step(self, action: np.ndarray):
        action  = np.asarray(action, dtype=np.float32)
        vel_cmd = float(np.clip(action[0], -self.max_vel, self.max_vel))
        do_lick = bool(action[1] > 0.0)

        # physics
        self._vel  = vel_cmd
        self._pos  = float(np.clip(self._pos + vel_cmd * self.dt, 0.0, self.L))
        self._steps += 1
        self._trajectory.append(self._pos)

        reward     = self.step_penalty
        terminated = False
        lick_zone  = None
        intra_trial_duration_this_step = None

        # ── lick logic ────────────────────────────────────────────────────────
        if do_lick:
            reward += self.lick_penalty   # universal lick cost

            if self._in_left_zone():
                
                lick_zone = "left"
                self._lick_events.append((self._steps, "left"))
                
                if not self._licked_left:
                    # first lick to left port
                    if self.phase == "mapping":
                        self._licked_left = True
                        self._step_left_first = self._steps
                    elif self.phase == "remapping":
                        if self._licked_right:
                            # first lick to left port after right (completes right→left)
                            self._licked_left = True
                            self._step_left_after_right = self._steps
                            # Calculate duration before resetting
                            intra_trial_duration_this_step = self._step_left_after_right - self._step_right_first
                            reward     += self.water_reward
                            self._n_successes += 1
                            # Reset flags to allow multiple right→left sequences
                            self._licked_left = False
                            self._licked_right = False
                            self._step_left_first = None
                            self._step_right_after_left = None
            elif self._in_right_zone():
                lick_zone = "right"
                self._lick_events.append((self._steps, "right"))
                
                if  not self._licked_right:
                    if self.phase == "mapping" and self._licked_left:
                        # first lick to right port after left (completes left→right)
                        self._licked_right = True
                        self._step_right_after_left = self._steps
                        # Calculate duration before resetting
                        intra_trial_duration_this_step = self._step_right_after_left - self._step_left_first
                        reward     += self.water_reward
                        self._n_successes += 1
                        # Reset flags to allow multiple left→right sequences
                        self._licked_left = False
                        self._licked_right = False
                        self._step_left_first = None
                        self._step_right_after_left = None
                    elif self.phase == "remapping":
                        self._licked_right = True
                        self._step_right_first = self._steps


            else:
                lick_zone = "body"
                self._lick_events.append((self._steps, "body"))
                reward += self.wrong_lick_pen   # licking on the track body

        truncated = self._steps >= self.max_steps

        info = {
            "pos"              : self._pos,
            "vel"              : self._vel,
            "licked_left"      : self._licked_left,
            "licked_right"     : self._licked_right,
            "lick_zone"        : lick_zone,
            "steps"            : self._steps,
            "n_successes"      : self._n_successes,
            "intra_trial_duration" : intra_trial_duration_this_step,  # steps between left and right licks
            "in_left_zone"     : self._in_left_zone(),
            "in_right_zone"    : self._in_right_zone(),
            "trajectory"       : list(self._trajectory),
            "lick_events"      : list(self._lick_events),
        }

        if self.render_mode == "ansi":
            self._render_ansi()

        return self._build_obs(), float(reward), terminated, truncated, info

    # ── render ────────────────────────────────────────────────────────────────

    def _render_ansi(self):
        width  = 60
        frac   = self._pos / self.L
        cursor = int(frac * (width - 1))
        bar    = list("─" * width)

        # mark zones
        zone_w = int(self.terminal_width / self.L * width)
        for i in range(zone_w):
            bar[i] = "L"
            bar[width - 1 - i] = "R"

        bar[cursor] = "A"
        flags = f"lL={int(self._licked_left)} lR={int(self._licked_right)}"
        print(f"|{''.join(bar)}| pos={self._pos:6.1f} vel={self._vel:+5.1f} {flags} s={self._steps}")

    def render(self):
        if self.render_mode == "ansi":
            self._render_ansi()
            return None
        if self.render_mode == "rgb_array":
            W, H = 600, 60
            img  = np.zeros((H, W, 3), dtype=np.uint8)

            # colour each pixel column
            for col in range(W):
                pos = col / W * self.L
                c = self._get_colour(pos) * 255
                img[:, col] = c.astype(np.uint8)

            # draw agent
            ax = int(self._pos / self.L * (W - 1))
            ax = np.clip(ax, 5, W - 6)
            img[15:45, ax-5:ax+5] = [0, 255, 200]   # teal agent blob
            return img

    # ── utility ───────────────────────────────────────────────────────────────

    def get_track_render(self, n_cells: int = 120) -> np.ndarray:
        """Returns (n_cells, 3) colour array for plotting."""
        positions = np.linspace(0, self.L, n_cells)
        return np.stack([self._get_colour(p) for p in positions])

    @property
    def obs_labels(self):
        return ["pos_norm", "licked_L", "licked_R"]
