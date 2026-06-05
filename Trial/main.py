"""
main.py — train tabular RL agents on LinearTrackEnv and plot behaviour metrics.

Usage (from project root):
  python main.py --agent q_learning --episodes 1000
  python main.py --agent sarsa --lambda 0.7 --episodes 2000 --eval-episodes 20
  python main.py --agent td_lambda_off --episodes 500 --no-train  # eval only
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from linear_track import LinearTrackEnv
from algorithms import REGISTRY, build_agent, evaluate_rollout, train
from metrics import aggregate_lick_frequency
from plots import (
    plot_inter_terminal_steps,
    plot_learning_curve,
    plot_lick_frequency_heatmap,
    plot_q_values,
    plot_track,
    plot_trajectory,
)


def parse_args():
    p = argparse.ArgumentParser(description="LinearTrack RL experiments")
    p.add_argument(
        "--agent",
        choices=list(REGISTRY),
        default="q_learning",
        help="Algorithm to train / evaluate",
    )
    p.add_argument("--episodes", type=int, default=1000, help="Training episodes")
    p.add_argument("--eval-episodes", type=int, default=10, help="Greedy eval rollouts after train")
    p.add_argument("--n-cells", type=int, default=20)
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lambda", dest="lambda_", type=float, default=0.0)
    p.add_argument("--epsilon", type=float, default=1.0)
    p.add_argument("--epsilon-min", type=float, default=0.05)
    p.add_argument("--epsilon-decay", type=float, default=0.995)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--results-dir", type=str, default="results")
    p.add_argument("--no-train", action="store_true", help="Skip training (eval/plots only)")
    p.add_argument("--no-plots", action="store_true", help="Skip saving figures")
    p.add_argument(
        "--terminate-on-goal",
        action="store_true",
        help="End episode immediately when goal reward is delivered (training)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.results_dir)
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{args.agent}_seed{args.seed}"

    print("=" * 60)
    print("  LinearTrack experiment")
    print("=" * 60)

    # Training env: optional early stop at goal
    env_train = LinearTrackEnv(
        n_cells=args.n_cells,
        max_steps=args.max_steps,
        terminate_on_goal=args.terminate_on_goal,
    )
    # Eval env: continue after goal so lick latency / post-reward licks are measurable
    env_eval = LinearTrackEnv(
        n_cells=args.n_cells,
        max_steps=args.max_steps,
        terminate_on_goal=False,
    )

    agent = build_agent(
        args.agent,
        args.n_cells,
        alpha=args.alpha,
        gamma=args.gamma,
        lambda_=args.lambda_,
        epsilon=args.epsilon,
        epsilon_min=args.epsilon_min,
        epsilon_decay=args.epsilon_decay,
        seed=args.seed,
    )

    print(f"\n  Agent        : {args.agent}")
    print(f"  Actions      : move in {{-1,0,+1}} (vs facing) + lick {{0,1}}")
    print(f"  Lambda       : {agent.lambda_}")
    print(f"  State count  : {agent.n_states}")

    if not args.no_plots:
        plot_track(env_eval, save_path=str(out / "track_layout.png"))

    csv_path = out / f"{tag}_train.csv"

    if not args.no_train:
        print(f"\n[train] {args.episodes} episodes …")
        env_train.reset(seed=args.seed)
        train(
            agent,
            env_train,
            args.episodes,
            log_every=args.log_every,
            csv_path=csv_path,
        )
        if not args.no_plots and csv_path.exists():
            plot_learning_curve(csv_path, save_path=str(out / f"{tag}_learning_curve.png"))

    # Greedy evaluation + behaviour metrics
    print(f"\n[eval] {args.eval_episodes} greedy episodes …")
    agent.epsilon = 0.0
    inter_terminal_hist: list[int | None] = []
    eval_records = []

    last_result = None
    for i in range(args.eval_episodes):
        result = evaluate_rollout(
            env_eval, agent, explore=False, seed=args.seed + 10_000 + i
        )
        last_result = result
        inter_terminal_hist.append(result.metrics.inter_terminal_steps)
        eval_records.append(result.record)
        m = result.metrics
        lat_str = (
            f"{m.lick_latencies}" if m.lick_latencies else "n/a (no post-reward lick)"
        )
        print(
            f"  eval {i + 1:2d}: steps={result.stats.steps}  "
            f"inter_terminal={m.inter_terminal_steps}  "
            f"licks={m.total_licks}  lick_latency={lat_str}"
        )

    if args.no_plots or last_result is None:
        print("\nDone.")
        return

    print("\n[plots] saving figures …")
    plot_trajectory(
        env_eval,
        last_result.trajectory,
        title=f"{args.agent} — greedy trajectory (last eval)",
        save_path=str(out / f"{tag}_trajectory.png"),
    )

    plot_lick_frequency_heatmap(
        env_eval,
        last_result.record,
        title=f"{args.agent} — lick map",
        save_path=str(out / f"{tag}_lick_frequency.png"),
    )

    agg_freq = aggregate_lick_frequency(eval_records, args.n_cells)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 3), facecolor="#0D0D0D")
    ax.set_facecolor("#111111")
    ax.bar(range(args.n_cells), agg_freq, color="#F72585", width=1.0)
    ax.set_xlabel("Position", color="white")
    ax.set_ylabel("Total licks (all eval eps)", color="white")
    ax.set_title(f"{args.agent} — aggregated lick frequency", color="white")
    ax.tick_params(colors="white")
    plt.tight_layout()
    agg_path = out / f"{tag}_lick_freq_agg.png"
    plt.savefig(agg_path, dpi=150, bbox_inches="tight", facecolor="#0D0D0D")
    plt.close(fig)
    print(f"  Aggregated lick freq -> {agg_path}")

    plot_inter_terminal_steps(
        inter_terminal_hist,
        title=f"{args.agent} — left to right passage",
        save_path=str(out / f"{tag}_inter_terminal.png"),
    )

    plot_q_values(agent, env_eval, save_path=str(out / f"{tag}_q_values.png"))

    print(f"\nDone. Outputs in {out.resolve()}/")


if __name__ == "__main__":
    main()
