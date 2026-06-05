import argparse, csv, json, os, time
from pathlib import Path
from typing import Any

import numpy as np
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from envs import continuous_linear_track
from agents import CONTINUOUS_REGISTRY
import agents.FA as FA  # Assuming FA.py is in the agents/ folder

# 1. Register the FA agents dynamically
CONTINUOUS_REGISTRY["q_fa"] = FA.QLearningFA
CONTINUOUS_REGISTRY["sarsa_fa"] = FA.SarsaFA
CONTINUOUS_REGISTRY["sarsa_lambda_fa"] = FA.SarsaLambdaFA

DEFAULT_CONFIGS = {
    "td3": dict(
        hidden_size=256, lr_actor=3e-4, lr_critic=3e-4, gamma=0.99,
        tau=0.005, policy_noise=0.2, noise_clip=0.5, policy_delay=2,
        expl_noise=0.3, buffer_capacity=50_000, batch_size=256,
        learn_start=1_000, device="auto",
    ),
    "sac": dict(
        hidden_size=256, lr=3e-4, gamma=0.99, tau=0.005,
        alpha=0.2, auto_entropy=True,
        buffer_capacity=50_000, batch_size=256,
        learn_start=1_000, device="auto",
    ),
    # 2. Add default configurations for the new FA agents
    "q_fa": dict(lr=0.01, gamma=0.99, epsilon=0.1, n_features=200),
    "sarsa_fa": dict(lr=0.01, gamma=0.99, epsilon=0.1, n_features=200),
    "sarsa_lambda_fa": dict(lr=0.01, gamma=0.99, epsilon=0.1, lambda_=0, n_features=200),
}

def run_continuous_experiment(
    agent_name       : str   = "td3",
    n_episodes       : int   = 3000,
    track_length     : float = 120.0,
    max_vel          : float = 10.0,
    terminal_width   : float = 3.0,
    max_steps        : int   = 500,
    water_reward     : float = 1.0,
    step_penalty     : float = -0.005,
    lick_penalty     : float = -0.05,
    wrong_lick_penalty: float = 0.0,
    seed             : int   = 42,
    results_dir      : str   = "results",
    extra_config     : dict | None = None,
    verbose          : bool  = True,
    log_every        : int   = 200,
) -> dict[str, Any]:

    agent_name = agent_name.lower()
    assert agent_name in CONTINUOUS_REGISTRY, \
        f"Unknown agent '{agent_name}'. Choose from {list(CONTINUOUS_REGISTRY)}"

    np.random.seed(seed)

    env = continuous_linear_track.ContinuousLinearTrackEnv(
        track_length       = track_length,
        max_vel            = max_vel,
        terminal_width     = terminal_width,
        max_steps          = max_steps,
        water_reward       = water_reward,
        step_penalty       = step_penalty,
        lick_penalty       = lick_penalty,
        wrong_lick_penalty = wrong_lick_penalty,
    )

    # 3. Dynamic Environment Wrapping for FA Agents
    is_discrete_fa = agent_name in ["q_fa", "sarsa_fa", "sarsa_lambda_fa"]
    
    if is_discrete_fa:
        env = FA.DiscreteActionWrapper(env, n_vel_bins=5)
        # Add type: ignore to tell Pylance we know what we are doing
        n_actions = env.action_space.n # type: ignore
    else:
        n_actions = 2 

    # Tell Pylance the shape tuple definitely exists
    assert env.observation_space.shape is not None
    obs_dim = env.observation_space.shape[0]
    config  = {**DEFAULT_CONFIGS[agent_name]}
    config["max_vel"] = max_vel
    if extra_config:
        config.update(extra_config)

    # 4. Instantiate the Agent based on type
    if is_discrete_fa:
        agent = CONTINUOUS_REGISTRY[agent_name](obs_dim, n_actions, config)
    else:
        agent = CONTINUOUS_REGISTRY[agent_name](
            obs_dim, n_actions, config, action_space=env.action_space
        )

    Path(results_dir).mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_name  = f"{agent_name}_cont_seed{seed}_{timestamp}"
    csv_path  = Path(results_dir) / f"{run_name}.csv"
    json_path = Path(results_dir) / f"{run_name}_manifest.json"
    ckpt_path = Path(results_dir) / f"{run_name}_checkpoint"

    fieldnames = [
        "episode", "total_reward", "steps", "success",
        "licked_left", "licked_right", "n_licks",
        "critic_loss", "actor_loss", "alpha",
    ]
    csv_rows : list[dict] = []
    start_t   = time.time()
    successes  = 0

    for ep in range(1, n_episodes + 1):
        obs, _   = env.reset(seed=seed + ep)
        ep_reward = 0.0
        ep_metrics: dict = {}
        n_licks   = 0
        done      = False
        info      = {} # INITIALIZE THIS HERE to fix the unbound warnings

        while not done:
            action = agent.select_action(obs, explore=True)
            nobs, r, term, trunc, info = env.step(action)
            done = term or trunc

            if info.get("lick_zone") is not None:
                n_licks += 1

            m = agent.update(obs, action, r, nobs, done)
            if m: 
                ep_metrics.update(m)

            obs        = nobs
            # Cast r to float to fix the operator issue
            ep_reward += float(r)

        end_m = agent.on_episode_end()
        if end_m:
            ep_metrics.update(end_m)

        success = info["licked_left"] and info["licked_right"]
        if success:
            successes += 1

        row = {
            "episode"     : ep,
            "total_reward": round(ep_reward, 4),
            "steps"       : info["steps"],
            "success"     : int(success),
            "licked_left" : int(info["licked_left"]),
            "licked_right": int(info["licked_right"]),
            "n_licks"     : n_licks,
            "critic_loss" : round(ep_metrics.get("critic_loss", float("nan")), 6),
            "actor_loss"  : round(ep_metrics.get("actor_loss",  float("nan")), 6),
            "alpha"       : round(ep_metrics.get("alpha",       float("nan")), 6),
        }
        csv_rows.append(row)

        if verbose and ep % log_every == 0:
            recent = csv_rows[-log_every:]
            avg_r  = np.mean([r["total_reward"] for r in recent])
            sr     = np.mean([r["success"]      for r in recent]) * 100
            lk     = np.mean([r["n_licks"]      for r in recent])
            print(f"  Ep {ep:5d}/{n_episodes} | "
                  f"avg_R={avg_r:+.3f} | SR={sr:.1f}% | "
                  f"avg_licks={lk:.1f}")

    elapsed = time.time() - start_t

    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader(); w.writerows(csv_rows)

    try:
        agent.save(str(ckpt_path))
    except Exception as e:
        print(f"  [warn] checkpoint: {e}")

    manifest = {
        "run_name"          : run_name,
        "agent"             : agent_name,
        "env"               : "ContinuousLinearTrack",
        "n_episodes"        : n_episodes,
        "track_length"      : track_length,
        "max_vel"           : max_vel,
        "terminal_width"    : terminal_width,
        "max_steps"         : max_steps,
        "water_reward"      : water_reward,
        "step_penalty"      : step_penalty,
        "lick_penalty"      : lick_penalty,
        "wrong_lick_penalty": wrong_lick_penalty,
        "seed"              : seed,
        "config"            : config,
        "final_success_rate": round(successes / n_episodes, 4),
        "last_100_success"  : round(np.mean([r["success"] for r in csv_rows[-100:]]), 4),
        "elapsed_seconds"   : round(elapsed, 2),
        "csv_path"          : str(csv_path),
        "ckpt_path"         : str(ckpt_path),
    }
    with open(json_path, "w") as f:
        json.dump(manifest, f, indent=2)

    if verbose:
        print(f"\n✓ Done in {elapsed:.1f}s | "
              f"SR: {manifest['final_success_rate']*100:.1f}% | "
              f"last-100: {manifest['last_100_success']*100:.1f}%")
        print(f"  CSV -> {csv_path}")

    return manifest

def _parse():
    p = argparse.ArgumentParser()
    p.add_argument("--agent",          default="td3", choices=list(CONTINUOUS_REGISTRY))
    p.add_argument("--episodes",       type=int,   default=3000)
    p.add_argument("--track_length",   type=float, default=120.0)
    p.add_argument("--max_vel",        type=float, default=10.0)
    p.add_argument("--terminal_width", type=float, default=3.0)
    p.add_argument("--max_steps",      type=int,   default=500)
    p.add_argument("--step_penalty",   type=float, default=-0.005)
    p.add_argument("--lick_penalty",   type=float, default=-0.05)
    p.add_argument("--seed",           type=int,   default=42)
    p.add_argument("--results_dir",    default="results")
    p.add_argument("--log_every",      type=int,   default=200)
    return p.parse_args()

if __name__ == "__main__":
    args = _parse()
    print(f"\n=== ContinuousLinearTrack | {args.agent} | {args.episodes} eps ===\n")
    run_continuous_experiment(
        agent_name    = args.agent,
        n_episodes    = args.episodes,
        track_length  = args.track_length,
        max_vel       = args.max_vel,
        terminal_width= args.terminal_width,
        max_steps     = args.max_steps,
        step_penalty  = args.step_penalty,
        lick_penalty  = args.lick_penalty,
        seed          = args.seed,
        results_dir   = args.results_dir,
        log_every     = args.log_every,
    )