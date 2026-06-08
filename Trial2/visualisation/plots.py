"""
visualisation/continuous_plots.py
===================================
Plots specific to the ContinuousLinearTrackEnv:
  - plot_continuous_track()       — colour + zone diagram
  - plot_lick_analysis()          — lick events over position
  - plot_velocity_profile()       — velocity trace for one episode
  - plot_continuous_learning()    — reward/success/lick-count curves
  - plot_reward_shaping_diagram() — visual guide to reward components
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.collections import LineCollection

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from envs import continuous_linear_track


def _load_csv(path):
    import csv
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            rows.append(row)
    if not rows:
        return {}
    return {k: np.array([r[k] for r in rows], dtype=float) for k in rows[0]}


def _smooth(x, w=30):
    if len(x) < w:
        return x
    return np.convolve(x, np.ones(w)/w, mode="valid")


# ── 1. Track layout ────────────────────────────────────────────────────────────

def plot_continuous_track(
    env: continuous_linear_track.ContinuousLinearTrackEnv,
    run_name: str | None = None,
    save_path: str | None = None,
):
    """
    Shows the track as a gradient colour bar, marks zone boundaries,
    and annotates the reward structure.
    
    Args:
        env: The ContinuousLinearTrackEnv environment
        run_name: Experiment run name for organizing plots folder
        save_path: Custom save path (if None, uses plots/{run_name}/ structure)
    """
    from pathlib import Path
    
    # Auto-generate save path if run_name provided
    if save_path is None and run_name:
        plots_dir = Path(__file__).resolve().parents[1] / "results" / "plots" / run_name
        plots_dir.mkdir(parents=True, exist_ok=True)
        save_path = str(plots_dir / "track_layout.png")
    L  = env.L
    tw = env.terminal_width
    W  = 800

    fig, axes = plt.subplots(2, 1, figsize=(14, 4),
                              gridspec_kw={"height_ratios": [3, 1]},
                              facecolor="#0D0D0D")
    fig.suptitle("ContinuousLinearTrack — Sensory Layout",
                 color="white", fontsize=13, fontweight="bold")

    # ── colour bar ────────────────────────────────────────────────────────────
    ax = axes[0]
    ax.set_facecolor("#0D0D0D")

    # build pixel columns
    img = np.zeros((40, W, 3))
    for col in range(W):
        pos = col / W * L
        if pos <= tw:
            img[:, col] = [0.85, 0.1, 0.1]    # red
        elif pos >= L - tw:
            img[:, col] = [0.95, 0.85, 0.0]   # yellow
        else:
            img[:, col] = [0.07, 0.07, 0.07]  # near-black

    ax.imshow(img, aspect="auto", extent=[0, L, 0, 1])
    ax.set_xlim(0, L); ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.tick_params(axis="x", colors="white")
    ax.set_xlabel("Position (cm)", color="white")

    # annotations
    for x, label, col in [
        (tw/2,     "LEFT\nLICK PORT\n(RED)",   "#FF6B6B"),
        (L/2,      "Track Body\n(BLACK)",       "#888888"),
        (L-tw/2,   "RIGHT\nLICK PORT\n(YELLOW)","#F5C518"),
    ]:
        ax.text(x, 0.5, label, ha="center", va="center",
                color=col, fontsize=8, fontweight="bold")
    ax.axvline(L/2, color="#4CC9F0", lw=2.0, linestyle=":", alpha=0.8, label="Tactile Boundary")
    ax.axvline(tw,   color="white", lw=1.2, linestyle="--", alpha=0.5)
    ax.axvline(L-tw, color="white", lw=1.2, linestyle="--", alpha=0.5)

    # ── reward diagram ────────────────────────────────────────────────────────
    ax2 = axes[1]
    ax2.set_facecolor("#111111")
    xs  = np.linspace(0, L, 1000)
    rew = np.zeros_like(xs)
    for i, x in enumerate(xs):
        if x <= tw or x >= L - tw:
            rew[i] = 1.0   # potential water reward zone

    ax2.fill_between(xs, rew, color="#00FFAA", alpha=0.35)
    ax2.axhline(0, color="#444", lw=0.8)

    # step penalty line
    ax2.axhline(-0.005 * 100, color="#F72585", lw=1.5, linestyle=":",
                label=f"step_penalty×100 = {env.step_penalty*100:.2f}")
    ax2.axhline(env.lick_penalty, color="#FF9500", lw=1.5, linestyle="--",
                label=f"lick_penalty = {env.lick_penalty:.3f}")
    ax2.axhline(env.water_reward, color="#00FFAA", lw=1.5, linestyle="-",
                label=f"water_reward = {env.water_reward:.2f}")

    ax2.set_xlim(0, L)
    ax2.set_ylim(-0.15, 1.2)
    ax2.set_yticks([]); ax2.tick_params(axis="x", colors="white")
    ax2.set_xlabel("Position (cm)", color="white")
    leg = ax2.legend(facecolor="#1A1A1A", edgecolor="#333",
                     labelcolor="white", fontsize=8, loc="upper center",
                     ncol=3)
    for sp in ["top", "right"]:
        ax2.spines[sp].set_visible(False)
    ax2.spines["bottom"].set_color("#333"); ax2.spines["left"].set_visible(False)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#0D0D0D")
        print(f"  Continuous track plot → {save_path}")
    plt.show()
    return fig


# ── 2. Lick analysis ──────────────────────────────────────────────────────────

def plot_lick_analysis(
    lick_events: list[tuple[float, str]],
    track_length: float = 120.0,
    terminal_width: float = 3.0,
    title: str = "Lick Event Distribution",
    run_name: str | None = None,
    save_path: str | None = None,
):
    """
    Scatter of lick positions coloured by zone (left/right/body).
    Shows how well the agent has learned to lick only at the ports.
    
    Args:
        lick_events: List of (position, zone) tuples
        track_length: Length of track
        terminal_width: Width of terminal zones
        title: Plot title
        run_name: Experiment run name for organizing plots folder
        save_path: Custom save path (if None, uses plots/{run_name}/ structure)
    """
    from pathlib import Path
    
    # Auto-generate save path if run_name provided
    if save_path is None and run_name:
        plots_dir = Path(__file__).resolve().parents[1] / "results" / "plots" / run_name
        plots_dir.mkdir(parents=True, exist_ok=True)
        save_path = str(plots_dir / "lick_analysis.png")
    if not lick_events:
        print("No lick events to plot.")
        return

    positions = np.array([e[0] for e in lick_events])
    zones     = [e[1] for e in lick_events]

    colour_map = {"left": "#FF4444", "right": "#F5C518", "body": "#888888"}
    colours    = [colour_map[z] for z in zones]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), facecolor="#0D0D0D")

    # scatter over episode index
    ax1.set_facecolor("#111111")
    ax1.scatter(range(len(positions)), positions, c=colours, s=8, alpha=0.6)
    ax1.axhline(terminal_width,             color="#FF4444",  lw=1, linestyle="--", alpha=0.6)
    ax1.axhline(track_length - terminal_width, color="#F5C518", lw=1, linestyle="--", alpha=0.6)
    ax1.set_xlabel("Lick index",  color="white")
    ax1.set_ylabel("Position",    color="white")
    ax1.set_title(title,          color="white")
    ax1.tick_params(colors="white")
    for sp in ["top", "right"]: ax1.spines[sp].set_visible(False)
    ax1.spines["bottom"].set_color("#333"); ax1.spines["left"].set_color("#333")

    # histogram
    ax2.set_facecolor("#111111")
    bins = np.linspace(0, track_length, 40)
    for zone, col in colour_map.items():
        pos_z = [p for p, z in lick_events if z == zone]
        if pos_z:
            ax2.hist(pos_z, bins=bins, color=col, alpha=0.7, label=zone)
    ax2.set_xlabel("Position", color="white")
    ax2.set_ylabel("Lick count", color="white")
    ax2.set_title("Lick Histogram by Zone", color="white")
    ax2.tick_params(colors="white")
    ax2.legend(facecolor="#1A1A1A", labelcolor="white", fontsize=8)
    for sp in ["top", "right"]: ax2.spines[sp].set_visible(False)
    ax2.spines["bottom"].set_color("#333"); ax2.spines["left"].set_color("#333")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#0D0D0D")
        print(f"  Lick analysis → {save_path}")
    plt.show()
    return fig


# ── 3. Velocity profile ───────────────────────────────────────────────────────

def plot_velocity_profile(
    trajectory : list[float],
    lick_events: list[tuple[float, str]] | None = None,
    dt         : float = 1.0,
    title      : str   = "Position & Velocity Profile",
    run_name   : str | None = None,
    save_path  : str | None = None,
):
    """
    Position and velocity trace for a single episode.
    
    Args:
        trajectory: List of positions over time
        lick_events: Optional list of (position, zone) tuples
        dt: Time step duration
        title: Plot title
        run_name: Experiment run name for organizing plots folder
        save_path: Custom save path (if None, uses plots/{run_name}/ structure)
    """
    from pathlib import Path
    
    # Auto-generate save path if run_name provided
    if save_path is None and run_name:
        plots_dir = Path(__file__).resolve().parents[1] / "results" / "plots" / run_name
        plots_dir.mkdir(parents=True, exist_ok=True)
        save_path = str(plots_dir / "velocity_profile.png")
    traj = np.array(trajectory)
    vel  = np.diff(traj) / dt
    t    = np.arange(len(traj))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 6), facecolor="#0D0D0D", sharex=True)

    for ax in (ax1, ax2):
        ax.set_facecolor("#111111")
        for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
        ax.spines["bottom"].set_color("#333"); ax.spines["left"].set_color("#333")
        ax.tick_params(colors="white")

    # colour position trace by zone
    ax1.plot(t, traj, color="#4CC9F0", lw=1.5, alpha=0.9)
    if lick_events:
        lick_pos   = [e[0] for e in lick_events]
        lick_col   = ["#FF4444" if e[1]=="left" else
                      "#F5C518" if e[1]=="right" else "#888" for e in lick_events]
        # approximate time indices
        for lp, lc in zip(lick_pos, lick_col):
            idxs = np.where(np.abs(traj - lp) < 1.0)[0]
            if len(idxs):
                ax1.axvline(idxs[0], color=lc, lw=0.8, alpha=0.5)

    ax1.set_ylabel("Position (cm)", color="white")
    ax1.set_title(title, color="white")

    ax2.plot(t[1:], vel, color="#F72585", lw=1.2)
    ax2.axhline(0, color="#555", lw=0.8, linestyle="--")
    ax2.set_ylabel("Velocity (cm/step)", color="white")
    ax2.set_xlabel("Step", color="white")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#0D0D0D")
        print(f"  Velocity profile → {save_path}")
    plt.show()
    return fig


# ── 4. Continuous learning curves ────────────────────────────────────────────

def plot_continuous_learning(
    csv_path     : str,
    smooth_window: int = 50,
    run_name     : str | None = None,
    save_path    : str | None = None,
):
    """
    Plot training curves: episode reward, success rate, licks, successes, and intra-trial duration.
    
    Args:
        csv_path: Path to the CSV file containing episode metrics
        smooth_window: Window size for smoothing curves
        run_name: Experiment run name for organizing plots folder
        save_path: Custom save path (if None, uses plots/{run_name}/ structure)
    """
    from pathlib import Path
    
    data = _load_csv(csv_path)
    ep   = data["episode"]
    
    # Auto-generate save path if run_name provided
    if save_path is None and run_name:
        plots_dir = Path(__file__).resolve().parents[1] / "results" / "plots" / run_name
        plots_dir.mkdir(parents=True, exist_ok=True)
        save_path = str(plots_dir / "learning_curves.png")

    fig, axes = plt.subplots(4, 1, figsize=(12, 11), facecolor="#0D0D0D", sharex=True)
    fig.suptitle(Path(csv_path).stem if not run_name else run_name, 
                 color="white", fontsize=11)

    metrics = [
        ("total_reward", "Episode Reward",          "#4CC9F0"),
        ("success",      "Success Rate (%)",        "#F72585"),
        ("n_licks",      "Licks per Episode",       "#F5C518"),
        ("intra_trial_duration", "Intra-Trial Duration (steps)", "#00FFAA"),
    ]

    for ax, (key, ylabel, col) in zip(axes, metrics):
        ax.set_facecolor("#111111")
        for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
        ax.spines["bottom"].set_color("#333"); ax.spines["left"].set_color("#333")
        ax.tick_params(colors="white")

        # Handle missing key gracefully
        if key not in data:
            ax.text(0.5, 0.5, f"'{key}' not in data", ha="center", va="center",
                   color="#FF6666", transform=ax.transAxes)
            ax.set_ylabel(ylabel, color="white")
            continue

        vals = data[key] * (100 if key == "success" else 1)
        
        # Filter out NaN for intra-trial duration
        if key == "intra_trial_duration":
            valid_mask = ~np.isnan(vals)
            valid_ep = ep[valid_mask]
            valid_vals = vals[valid_mask]
            if len(valid_vals) > 0:
                ax.plot(valid_ep, valid_vals, alpha=0.2, color=col, lw=0.7)
                if len(valid_vals) >= smooth_window:
                    s = _smooth(valid_vals, smooth_window)
                    ax.plot(valid_ep[smooth_window-1:], s, color=col, lw=2)
        else:
            ax.plot(ep, vals, alpha=0.2, color=col, lw=0.7)
            if len(vals) >= smooth_window:
                s = _smooth(vals, smooth_window)
                ax.plot(ep[smooth_window-1:], s, color=col, lw=2)
        
        ax.set_ylabel(ylabel, color="white")

    axes[-1].set_xlabel("Episode", color="white")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#0D0D0D")
        print(f"  Continuous learning curves → {save_path}")
    plt.show()
    return fig


# ── 5. Reward shaping diagram ─────────────────────────────────────────────────

def plot_reward_shaping_diagram(
    water_reward  : float = 1.0,
    lick_penalty  : float = -0.05,
    step_penalty  : float = -0.005,
    max_steps     : int   = 500,
    run_name      : str | None = None,
    save_path     : str | None = None,
):
    """
    Visualises the ratio of reward components to guide hyperparameter choice.
    Shows the safe shaping window (Ng et al. 1999).
    
    Args:
        water_reward: Reward for correct sequence
        lick_penalty: Penalty for licking
        step_penalty: Penalty per step
        max_steps: Maximum steps per episode
        run_name: Experiment run name for organizing plots folder
        save_path: Custom save path (if None, uses plots/{run_name}/ structure)
    """
    from pathlib import Path
    
    # Auto-generate save path if run_name provided
    if save_path is None and run_name:
        plots_dir = Path(__file__).resolve().parents[1] / "results" / "plots" / run_name
        plots_dir.mkdir(parents=True, exist_ok=True)
        save_path = str(plots_dir / "reward_shaping.png")
    fig, ax = plt.subplots(figsize=(10, 5), facecolor="#0D0D0D")
    ax.set_facecolor("#111111")
    for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color("#333"); ax.spines["left"].set_color("#333")
    ax.tick_params(colors="white")

    labels   = ["Water\nReward", "Lick\nPenalty", "Max Step\nCost", "Lick/Water\nRatio (%)"]
    values   = [water_reward, abs(lick_penalty), abs(step_penalty)*max_steps,
                abs(lick_penalty)/water_reward*100]
    colours  = ["#00FFAA", "#FF9500", "#F72585", "#4CC9F0"]
    x        = np.arange(len(labels))

    bars = ax.bar(x[:3], values[:3], color=colours[:3], width=0.5, zorder=3)
    ax.set_xticks(x[:3]); ax.set_xticklabels(labels[:3], color="white")
    ax.set_ylabel("Magnitude", color="white")
    ax.set_title("Reward Component Magnitudes", color="white", fontsize=12)
    ax.grid(axis="y", color="#333", zorder=0)

    # safe zone annotation
    safe_lo = water_reward * 0.01
    safe_hi = water_reward * 0.10
    ax.axhspan(safe_lo, safe_hi, color="#4CC9F0", alpha=0.12,
               label=f"Safe lick_penalty window: {safe_lo:.3f}–{safe_hi:.3f}")
    ax.axhline(abs(lick_penalty), color="#FF9500", lw=2, linestyle="--",
               label=f"Current lick_penalty = {lick_penalty:.3f}  "
                     f"({abs(lick_penalty)/water_reward*100:.1f}% of water reward)")
    ax.legend(facecolor="#1A1A1A", labelcolor="white", fontsize=9, loc="upper right")

    # ratio text
    ratio = abs(lick_penalty) / water_reward * 100
    status = "✓ IN safe window" if 1 <= ratio <= 10 else "⚠ OUTSIDE safe window"
    ax.text(0.5, 0.92, f"lick/water = {ratio:.1f}%  {status}",
            transform=ax.transAxes, ha="center", color="white",
            fontsize=10, fontweight="bold",
            bbox=dict(boxstyle="round", fc="#1A2A1A" if ratio <= 10 else "#2A1A1A",
                      ec="#444", alpha=0.9))

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#0D0D0D")
        print(f"  Reward shaping diagram → {save_path}")
    plt.show()
    return fig


if __name__ == "__main__":
    # run_name = "example_run"
    # csv_path = f"results/{run_name}.csv"

    # # Plots auto-save to: results/plots/{run_name}/learning_curves.png
    # plot_continuous_learning(csv_path, run_name=run_name)


    # All functions support this pattern
    run_name ="Track_Plot"
    dummy_env = continuous_linear_track.ContinuousLinearTrackEnv()
    plot_continuous_track(dummy_env, run_name=run_name)
    
    # 2. Test the shaping diagram
    plot_reward_shaping_diagram(run_name=run_name)