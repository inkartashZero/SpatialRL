"""
Visualisation
=============
All plotting functions for the LinearTrack spatial RL project.

Functions
---------
plot_track(env)                  → static image of the track + sensory info
plot_learning_curve(csv_path)    → reward / success-rate vs episode
plot_comparison(results_dir)     → multi-agent comparison (mean ± std across seeds)
plot_trajectory(env, positions)  → heatmap of agent positions along the track
plot_q_values(agent, env)        → heatmap of Q-values for tabular agents
animate_episode(env, agent)      → animated episode rollout (saves to GIF)
"""

from __future__ import annotations

import csv
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap
from matplotlib import animation

from linear_track import LinearTrackEnv
from metrics import EpisodeRecord, aggregate_lick_frequency


# ── colour constants ──────────────────────────────────────────────────────────
AGENT_COLOUR  = "#00FFCC"
TRACK_PALETTE = {
    "black"  : "#111111",
    "yellow" : "#F5C518",
    "red"    : "#E63946",
}
AGENT_COLOURS = {
    "q_learning"    : "#4CC9F0",
    "sarsa"         : "#F72585",
    "td_lambda_on"  : "#3A86FF",
    "td_lambda_off" : "#7209B7",
    "dqn"           : "#7209B7",
    "reinforce"     : "#3A86FF",
}


# ── helpers ───────────────────────────────────────────────────────────────────

def _smooth(x: np.ndarray, w: int = 20) -> np.ndarray:
    if len(x) < w:
        return x
    return np.convolve(x, np.ones(w) / w, mode="valid")


def _load_csv(path: str | Path) -> dict[str, np.ndarray]:
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    if not rows:
        return {}
    keys = rows[0].keys()
    out = {}
    for k in keys:
        vals = []
        for r in rows:
            v = r[k]
            vals.append(np.nan if v == "" or v is None else float(v))
        out[k] = np.array(vals, dtype=float)
    return out


# ── 1. Track visualisation ────────────────────────────────────────────────────

def plot_track(env: LinearTrackEnv, save_path: str | None = None):
    """
    Renders the track as a colour bar + tactile profile + reward diagram.
    """
    n  = env.n_cells
    colours  = env.get_track_colours()    # (n, 3) float RGB
    tactile  = env.get_track_tactile()    # (n,) float

    fig, axes = plt.subplots(3, 1, figsize=(14, 5),
                             gridspec_kw={"height_ratios": [2, 1, 1]},
                             facecolor="#0D0D0D")
    fig.suptitle("LinearTrack — Sensory Layout", color="white",
                 fontsize=14, fontweight="bold", y=1.01)

    # ── colour bar ────────────────────────────────────────────────────────────
    ax = axes[0]
    ax.set_facecolor("#0D0D0D")
    for i in range(n):
        col = tuple(colours[i])
        rect = mpatches.FancyBboxPatch(
            (i, 0), 1, 1, boxstyle="square,pad=0",
            linewidth=0, facecolor=col,
        )
        ax.add_patch(rect)
        # label terminals
        if i == 0 or i == n - 1:
            ax.text(i + 0.5, 0.5, "T", ha="center", va="center",
                    fontsize=9, color="white", fontweight="bold")
        elif i == env.mid:
            ax.axvline(i, color="#AAAAAA", lw=1.2, linestyle="--")

    ax.set_xlim(0, n)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xticks(range(n))
    ax.tick_params(axis="x", colors="white", labelsize=7)
    ax.set_title("Floor Colour  (black=smooth half | yellow=rough half | red=terminal)",
                 color="#AAAAAA", fontsize=9)

    # ── tactile bar ───────────────────────────────────────────────────────────
    ax2 = axes[1]
    ax2.set_facecolor("#0D0D0D")
    ax2.bar(range(n), tactile, color=[
        TRACK_PALETTE["yellow"] if t > 0.5 else TRACK_PALETTE["black"]
        for t in tactile
    ], width=1.0, edgecolor="none")
    ax2.set_xlim(0, n); ax2.set_ylim(-0.1, 1.3)
    ax2.set_yticks([0, 1]); ax2.set_yticklabels(["smooth", "rough"], color="white", fontsize=8)
    ax2.tick_params(axis="x", colors="white", labelsize=7)
    ax2.set_title("Tactile Signal", color="#AAAAAA", fontsize=9)
    ax2.spines["bottom"].set_color("#333333")
    ax2.spines["left"].set_color("#333333")
    for sp in ["top", "right"]:
        ax2.spines[sp].set_visible(False)

    # ── reward diagram ────────────────────────────────────────────────────────
    ax3 = axes[2]
    ax3.set_facecolor("#0D0D0D")
    reward_profile = np.zeros(n)
    reward_profile[0]   = 1.0
    reward_profile[n-1] = 1.0
    ax3.bar(range(n), reward_profile,
            color=[TRACK_PALETTE["red"] if r > 0 else "#1A1A1A" for r in reward_profile],
            width=1.0)
    ax3.set_xlim(0, n); ax3.set_ylim(-0.1, 1.3)
    ax3.set_yticks([0, 1]); ax3.set_yticklabels(["0", "+1"], color="white", fontsize=8)
    ax3.tick_params(axis="x", colors="white", labelsize=7)
    ax3.set_title("Terminals (left start | right goal +10)", color="#AAAAAA", fontsize=9)
    for sp in ["top", "right"]:
        ax3.spines[sp].set_visible(False)
    ax3.spines["bottom"].set_color("#333333")
    ax3.spines["left"].set_color("#333333")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#0D0D0D")
        print(f"  Track plot -> {save_path}")
        plt.close(fig)
    return fig


# ── 2. Learning curve ─────────────────────────────────────────────────────────

def plot_learning_curve(
    csv_path: str | Path,
    smooth_window: int = 50,
    save_path: str | None = None,
):
    """Reward + success-rate over training for a single run."""
    data = _load_csv(csv_path)
    episodes = data["episode"]
    rewards  = data["total_reward"]
    success  = data["success"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), facecolor="#0D0D0D",
                                    sharex=True)
    fig.suptitle(Path(csv_path).stem, color="white", fontsize=11)

    for ax in (ax1, ax2):
        ax.set_facecolor("#111111")
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)
        ax.spines["bottom"].set_color("#333")
        ax.spines["left"].set_color("#333")
        ax.tick_params(colors="white")

    # reward
    ax1.plot(episodes, rewards, alpha=0.25, color="#4CC9F0", linewidth=0.8)
    if len(rewards) >= smooth_window:
        s = _smooth(rewards, smooth_window)
        ax1.plot(episodes[smooth_window-1:], s, color="#4CC9F0", linewidth=2)
    ax1.set_ylabel("Episode Reward", color="white")
    ax1.axhline(0, color="#555", linestyle="--", linewidth=0.8)

    # success rate
    sr_smooth = _smooth(success, smooth_window) * 100
    ax2.plot(episodes, success * 100, alpha=0.15, color="#F72585", linewidth=0.6)
    if len(success) >= smooth_window:
        ax2.plot(episodes[smooth_window-1:], sr_smooth, color="#F72585", linewidth=2)
    ax2.set_ylabel("Success Rate (%)", color="white")
    ax2.set_xlabel("Episode", color="white")
    ax2.set_ylim(-5, 105)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#0D0D0D")
        print(f"  Learning curve -> {save_path}")
        plt.close(fig)
    return fig


# ── 3. Multi-agent comparison ─────────────────────────────────────────────────

def plot_comparison(
    results_dir : str | Path = "results",
    smooth_window: int = 50,
    metric       : str = "success",   # or 'total_reward'
    save_path    : str | None = None,
):
    """
    Aggregates all CSVs in results_dir by agent name (across seeds)
    and plots mean ± std.
    """
    results_dir = Path(results_dir)
    agent_data: dict[str, list[np.ndarray]] = defaultdict(list)

    for csv_file in sorted(results_dir.glob("*_train.csv")):
        agent = csv_file.stem.split("_seed")[0]
        data = _load_csv(csv_file)
        if metric in data and len(data[metric]) > 0:
            agent_data[agent].append(data[metric])

    if not agent_data:
        print(f"No CSV files found in {results_dir}")
        return

    fig, ax = plt.subplots(figsize=(13, 6), facecolor="#0D0D0D")
    ax.set_facecolor("#111111")
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color("#333")
    ax.spines["left"].set_color("#333")
    ax.tick_params(colors="white")

    for agent, runs in agent_data.items():
        min_len = min(len(r) for r in runs)
        arr = np.stack([r[:min_len] for r in runs])
        w = min(smooth_window, max(5, min_len // 10))
        mean = _smooth(arr.mean(axis=0), w)
        std = _smooth(arr.std(axis=0), w)
        ep = np.arange(w - 1, min_len)
        colour = AGENT_COLOURS.get(agent, "#AAAAAA")
        scale = 100 if metric == "success" else 1

        ax.plot(ep, mean * scale, color=colour, linewidth=2, label=agent)
        if len(runs) > 1:
            ax.fill_between(
                ep,
                (mean - std) * scale,
                (mean + std) * scale,
                color=colour,
                alpha=0.15,
            )

    ylabel = "Success Rate (%)" if metric == "success" else "Episode Reward"
    ax.set_xlabel("Episode", color="white")
    ax.set_ylabel(ylabel, color="white")
    ax.set_title("Agent Comparison — LinearTrack", color="white", fontsize=13)
    ax.legend(facecolor="#1A1A1A", edgecolor="#444", labelcolor="white")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#0D0D0D")
        print(f"  Comparison plot -> {save_path}")
        plt.close(fig)
    return fig


# ── 4. Trajectory heatmap ─────────────────────────────────────────────────────

def plot_trajectory(
    env       : LinearTrackEnv,
    positions : Sequence[int],
    title     : str = "Agent Trajectory Heatmap",
    save_path : str | None = None,
):
    """
    Visualises visit frequency along the track + the actual trajectory
    over time as a 2D image (position × time).
    """
    n    = env.n_cells
    traj = np.array(positions)
    T    = len(traj)

    fig  = plt.figure(figsize=(14, 6), facecolor="#0D0D0D")
    gs   = gridspec.GridSpec(2, 1, height_ratios=[2, 1], hspace=0.4)

    # ── 2D pos×time ───────────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor("#0D0D0D")
    img = np.zeros((n, T))
    for t, p in enumerate(traj):
        img[p, t] = 1.0
    ax1.imshow(img, aspect="auto", origin="lower",
               cmap="inferno", interpolation="nearest")
    ax1.set_ylabel("Track Position", color="white")
    ax1.set_xlabel("Timestep", color="white")
    ax1.set_title(title, color="white")
    ax1.tick_params(colors="white")
    # mark terminals
    ax1.axhline(0, color=TRACK_PALETTE["red"], linewidth=1.5, alpha=0.7)
    ax1.axhline(n-1, color=TRACK_PALETTE["red"], linewidth=1.5, alpha=0.7)
    ax1.axhline(env.mid - 0.5, color="#AAAAAA", linewidth=1, linestyle="--", alpha=0.5)

    # ── visit frequency bar ───────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1])
    ax2.set_facecolor("#111111")
    counts = np.bincount(traj, minlength=n)
    colours_bar = []
    for i in range(n):
        if i == 0 or i == n - 1:
            colours_bar.append(TRACK_PALETTE["red"])
        elif i >= env.mid:
            colours_bar.append(TRACK_PALETTE["yellow"])
        else:
            colours_bar.append("#555555")
    ax2.bar(range(n), counts, color=colours_bar, width=1.0)
    ax2.set_xlabel("Track Position", color="white")
    ax2.set_ylabel("Visit Count", color="white")
    ax2.tick_params(colors="white")
    for sp in ["top", "right"]:
        ax2.spines[sp].set_visible(False)
    ax2.spines["bottom"].set_color("#333")
    ax2.spines["left"].set_color("#333")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#0D0D0D")
        print(f"  Trajectory plot -> {save_path}")
    plt.close(fig)
    return fig


# ── 5. Q-value heatmap ────────────────────────────────────────────────────────

def plot_q_values(agent, env: LinearTrackEnv, save_path: str | None = None):
    """Heatmap of Q(s,a) for tabular agents (states with vL=1, vR=0, facing right)."""
    if not hasattr(agent, "Q"):
        print("plot_q_values: agent has no Q table (tabular agents only)")
        return

    from algorithms.encoding import ACTION_NAMES, decode_state, n_states

    n = env.n_cells
    n_act = agent.Q.shape[1]
    # states: at each position, facing=1 (right), visited left, not yet right
    q_by_pos = np.zeros((n, n_act))
    for s in range(n_states(n)):
        pos, facing, vL, vR = decode_state(s, n)
        if facing == 1 and vL == 1 and vR == 0:
            q_by_pos[pos] = agent.Q[s]

    fig, ax = plt.subplots(figsize=(10, 5), facecolor="#0D0D0D")
    ax.set_facecolor("#111111")
    im = ax.imshow(
        q_by_pos.T,
        aspect="auto",
        origin="lower",
        cmap="RdYlGn",
        interpolation="nearest",
    )
    ax.set_title("Q-values (facing right, vL=1, vR=0)", color="white")
    ax.set_xlabel("Position", color="white")
    ax.set_ylabel("Action", color="white")
    ax.set_yticks(range(n_act))
    ax.set_yticklabels(list(ACTION_NAMES)[:n_act], color="white")
    ax.tick_params(colors="white")
    plt.colorbar(im, ax=ax)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#0D0D0D")
        print(f"  Q-value plot -> {save_path}")
    plt.close(fig)
    return fig


# ── 6. Animated episode ───────────────────────────────────────────────────────

def animate_episode(
    env   : LinearTrackEnv,
    agent,
    max_steps : int  = 200,
    save_path : str  = "episode.gif",
    fps       : int  = 10,
    explore   : bool = False,
):
    """
    Rolls out one episode and saves an animated GIF of the track.
    """
    n = env.n_cells

    obs, _   = env.reset()
    frames   = []
    done     = False
    step     = 0
    total_r  = 0.0
    traj     = [int(round(obs[0] * (n-1)))]

    from algorithms.encoding import encode_state

    state = encode_state(env)
    while not done and step < max_steps:
        action = agent.select_action(state, explore=explore)
        obs, r, terminated, truncated, info = env.step(agent.env_action(action))
        state = encode_state(env)
        done    = terminated or truncated
        total_r += r
        step    += 1
        traj.append(info["pos"])
        frames.append((list(traj), info["visited_left"], info["visited_right"]))

    # build animation
    fig, ax = plt.subplots(figsize=(14, 2.5), facecolor="#0D0D0D")
    ax.set_facecolor("#0D0D0D")
    ax.set_xlim(0, n); ax.set_ylim(0, 1)
    ax.axis("off")
    title_obj = ax.set_title("", color="white", fontsize=10)

    patches = []
    for i in range(n):
        col = env._get_colour(i)
        p = mpatches.FancyBboxPatch(
            (i + 0.05, 0.1), 0.9, 0.8,
            boxstyle="round,pad=0.03",
            facecolor=col, linewidth=0,
        )
        ax.add_patch(p)
        patches.append(p)

    agent_patch = mpatches.Circle((traj[0] + 0.5, 0.5), 0.32,
                                   color=AGENT_COLOUR, zorder=10)
    ax.add_patch(agent_patch)

    def _update(frame_idx):
        traj_f, vL, vR = frames[frame_idx]
        pos = traj_f[-1]
        agent_patch.center = (pos + 0.5, 0.5)
        title_obj.set_text(
            f"Step {frame_idx+1}  |  pos={pos}  "
            f"vL={int(vL)} vR={int(vR)}  |  reward≈{total_r:.2f}"
        )
        return agent_patch, title_obj

    ani = animation.FuncAnimation(
        fig, _update, frames=len(frames), interval=1000//fps, blit=True,
    )
    ani.save(save_path, writer="pillow", fps=fps)
    plt.close(fig)
    print(f"  Animation saved -> {save_path}")
    return save_path


# ── 7. Lick frequency heatmap (position × time) ────────────────────────────────

def plot_lick_frequency_heatmap(
    env: LinearTrackEnv,
    record: EpisodeRecord,
    title: str = "Lick frequency by position (per stay)",
    save_path: str | None = None,
):
    """Heatmap: track position vs time; colour = lick at that step."""
    from metrics import lick_frequency_by_position

    n = env.n_cells
    T = len(record.positions)
    img = np.zeros((n, T))
    for t, (p, l) in enumerate(zip(record.positions, record.licked)):
        if l:
            img[p, t] = 1.0

    freq = lick_frequency_by_position(record, n)

    fig = plt.figure(figsize=(14, 7), facecolor="#0D0D0D")
    gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1], hspace=0.35)

    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor("#0D0D0D")
    ax1.imshow(img, aspect="auto", origin="lower", cmap="magma", vmin=0, vmax=1)
    ax1.set_ylabel("Position", color="white")
    ax1.set_xlabel("Timestep", color="white")
    ax1.set_title(f"{title} — events over time", color="white")
    ax1.tick_params(colors="white")
    ax1.axhline(0, color=TRACK_PALETTE["red"], lw=1.2, alpha=0.7)
    ax1.axhline(n - 1, color=TRACK_PALETTE["red"], lw=1.2, alpha=0.7)

    ax2 = fig.add_subplot(gs[1])
    ax2.set_facecolor("#111111")
    colours = []
    for i in range(n):
        if i == 0 or i == n - 1:
            colours.append(TRACK_PALETTE["red"])
        elif i >= env.mid:
            colours.append(TRACK_PALETTE["yellow"])
        else:
            colours.append("#555555")
    ax2.bar(range(n), freq, color=colours, width=1.0)
    ax2.set_xlabel("Track position", color="white")
    ax2.set_ylabel("Licks per stay", color="white")
    ax2.set_title("Lick count before leaving each position", color="#AAAAAA", fontsize=9)
    ax2.tick_params(colors="white")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#0D0D0D")
        print(f"  Lick frequency plot -> {save_path}")
    plt.close(fig)
    return fig


# ── 8. Inter-terminal passage time ───────────────────────────────────────────

def plot_inter_terminal_steps(
    inter_terminal_history: Sequence[int | None],
    title: str = "Inter-terminal passage (steps)",
    save_path: str | None = None,
):
    """Plot distribution / trend of steps from left to right terminal."""
    vals = [v for v in inter_terminal_history if v is not None]
    if not vals:
        print("plot_inter_terminal_steps: no successful traversals to plot")
        return None

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), facecolor="#0D0D0D")
    for ax in axes:
        ax.set_facecolor("#111111")
        ax.tick_params(colors="white")
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)

    axes[0].hist(vals, bins=min(20, max(5, len(set(vals)))), color="#4CC9F0", edgecolor="#0D0D0D")
    axes[0].set_xlabel("Steps (left → right)", color="white")
    axes[0].set_ylabel("Count", color="white")
    axes[0].set_title("Histogram", color="white")

    axes[1].plot(range(len(vals)), vals, color="#F72585", marker="o", markersize=3)
    axes[1].axhline(np.mean(vals), color="#AAAAAA", linestyle="--", label=f"mean={np.mean(vals):.1f}")
    axes[1].set_xlabel("Evaluation episode", color="white")
    axes[1].set_ylabel("Steps", color="white")
    axes[1].set_title(title, color="white")
    axes[1].legend(facecolor="#1A1A1A", labelcolor="white")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#0D0D0D")
        print(f"  Inter-terminal plot -> {save_path}")
    plt.close(fig)
    return fig
