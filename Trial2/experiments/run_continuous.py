import argparse, csv, json, os, time
from pathlib import Path
from typing import Any

import numpy as np
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from envs import continuous_linear_track
from agents import CONTINUOUS_REGISTRY
import agents.FA as FA  # Assuming FA.py is in the algorithms/ folder

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
        n_actions = env.action_space.n # type: ignore
    else:
        n_actions = 2 

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

    # REMOVED: success, licked_left, licked_right
    fieldnames = [
        "episode", "total_reward", "steps", "n_licks", "n_successes",
        "intra_trial_duration",
        "critic_loss", "actor_loss", "alpha",
    ]
    
    csv_rows : list[dict] = []
    start_t   = time.time()
    total_episodes_with_success  = 0

    for ep in range(1, n_episodes + 1):
        obs, _   = env.reset(seed=seed + ep)
        ep_reward = 0.0
        ep_metrics: dict = {}
        n_licks   = 0
        done      = False
        info      = {} 
        ep_durations = []

        while not done:
            action = agent.select_action(obs, explore=True)
            nobs, r, term, trunc, info = env.step(action)
            done = term or trunc

            if info.get("lick_zone") is not None:
                n_licks += 1

            if info.get("intra_trial_duration") is not None:
                ep_durations.append(info["intra_trial_duration"])

            m = agent.update(obs, action, r, nobs, done)
            if m: 
                ep_metrics.update(m)

            obs        = nobs
            ep_reward += float(r)

        end_m = agent.on_episode_end()
        if end_m:
            ep_metrics.update(end_m)

        # Track if the episode had AT LEAST one success for the final manifest
        if info.get("n_successes", 0) > 0:
            total_episodes_with_success += 1

        # Extract the duration and force it to NaN if it is None
        safe_dur = float(np.mean(ep_durations)) if ep_durations else float("nan")

        row = {
            "episode"               : ep,
            "total_reward"          : round(ep_reward, 4),
            "steps"                 : info["steps"],
            "n_licks"               : n_licks,
            "n_successes"           : info.get("n_successes", 0),
            "intra_trial_duration"  : safe_dur,
            "critic_loss"           : round(ep_metrics.get("critic_loss", float("nan")), 6),
            "actor_loss"            : round(ep_metrics.get("actor_loss",  float("nan")), 6),
            "alpha"                 : round(ep_metrics.get("alpha",       float("nan")), 6),
        }
        csv_rows.append(row)

        if verbose and ep % log_every == 0:
            recent = csv_rows[-log_every:]
            avg_r  = np.mean([r["total_reward"] for r in recent])
            lk     = np.mean([r["n_licks"]      for r in recent])
            avg_succ = np.mean([r["n_successes"] for r in recent])
            
            valid_durations = [
                r["intra_trial_duration"] for r in recent 
                if isinstance(r["intra_trial_duration"], (int, float)) and not np.isnan(r["intra_trial_duration"])
            ]
            avg_dur = np.mean(valid_durations) if valid_durations else float("nan")
            print(f"  Ep {ep:5d}/{n_episodes} | "
                  f"avg_R={avg_r:+.3f} | "
                  f"avg_succ={avg_succ:.2f} | avg_dur={avg_dur:.1f} steps")

    elapsed = time.time() - start_t

    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader(); w.writerows(csv_rows)

    try:
        agent.save(str(ckpt_path))
    except Exception as e:
        print(f"  [warn] checkpoint: {e}")

    # FIXED: Replaced "success" check with "n_successes > 0"
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
        "final_success_rate": round(total_episodes_with_success / n_episodes, 4),
        "last_100_success"  : round(np.mean([1 if r["n_successes"] > 0 else 0 for r in csv_rows[-100:]]), 4),
        "elapsed_seconds"   : round(elapsed, 2),
        "csv_path"          : str(csv_path),
        "ckpt_path"         : str(ckpt_path),
    }
    with open(json_path, "w") as f:
        json.dump(manifest, f, indent=2)

    if verbose:
        print(f"\n✓ Done in {elapsed:.1f}s | "
              f"last-100 SR: {manifest['last_100_success']*100:.1f}%")
        print(f"  CSV -> {csv_path}")

    return manifest

def _parse():
    p = argparse.ArgumentParser()
    p.add_argument("--agent",          default="sac", choices=list(CONTINUOUS_REGISTRY))
    p.add_argument("--episodes",       type=int,   default=3000)
    p.add_argument("--track_length",   type=float, default=120.0)
    p.add_argument("--max_vel",        type=float, default=10.0)
    p.add_argument("--terminal_width", type=float, default=1.0)
    p.add_argument("--max_steps",      type=int,   default=500)
    p.add_argument("--step_penalty",   type=float, default=-0.005)
    p.add_argument("--lick_penalty",   type=float, default=-0.005) # Defaulted to 0 for you
    p.add_argument("--water_reward",   type=float, default=100.0) # Added parser for reward
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
        water_reward  = args.water_reward,
        step_penalty  = args.step_penalty,
        lick_penalty  = args.lick_penalty,
        seed          = args.seed,
        results_dir   = args.results_dir,
        log_every     = args.log_every,
    )