import argparse, csv, json, os, time
from pathlib import Path
from typing import Any

import numpy as np
import sys
import torch 

# Ensure these imports match your project structure
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from envs import continuous_linear_track
from agents import CONTINUOUS_REGISTRY
import agents.FA as FA  
from stepwise_Telemetry import TelemetryLogger

DEFAULT_CONFIGS = {
    "td3": dict(
        hidden_size=256, lr_actor=3e-4, lr_critic=3e-4, gamma=0.99,
        tau=0.005, policy_noise=0.2, noise_clip=0.5, policy_delay=2,
        expl_noise=0.3, buffer_capacity=500, batch_size=256,
        learn_start=1_000, device="auto",
    ),
    "sac": dict(
        hidden_size=256, lr=3e-4, gamma=0.97, tau=0.005,
        alpha=0.2, auto_entropy=True,
        buffer_capacity=500, batch_size=256,
        learn_start=1_000, device="auto",
    ),
    "q_fa": dict(lr=0.01, gamma=0.99, epsilon=0.1, n_features=200),
    "sarsa_fa": dict(lr=0.01, gamma=0.99, epsilon=0.1, n_features=200),
    "sarsa_lambda_fa": dict(lr=0.01, gamma=0.99, epsilon=0.1, lambda_=0, n_features=200),
    "ddpg": dict(hidden_size=256, lr_actor=3e-4, lr_critic=3e-4, gamma=0.99, tau=0.005, expl_noise=10, batch_size=256, learn_start=1000),
    "vpg": dict(hidden_size=256, lr=3e-4, gamma=0.90, use_critic=True), 
    "ppo": dict(
        hidden_size=256, lr=3e-4, gamma=0.99, lam=0.95,
        clip_ratio=0.2, target_kl=0.015, ppo_epochs=10, 
        rollout_steps=2048, batch_size=64, device="auto",
    ),
    "a2c": dict(
        hidden_size=256, 
        lr=3e-4, 
        gamma=0.99,
        entropy_coef=0.01, 
        rollout_steps=5, 
        device="auto"
    ),
}

def run_continuous_experiment(
    agent_name       : str   = "ppo",
    n_episodes       : int   = 3000,
    track_length     : float = 120.0,
    max_vel          : float = 20.0,
    terminal_width   : float = 3.0,
    max_steps        : int   = 500,
    water_reward     : float = 1.0,
    step_penalty     : float = -0.005,
    lick_penalty     : float = -0.05,
    poke_penalty     : float = -0.03,
    wrong_lick_penalty: float = 0.0,
    seed             : int   = 42,
    results_dir      : str   = "results",
    extra_config     : dict | None = None,
    verbose          : bool  = True,
    log_every        : int   = 50,
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

    n_actions = 2 

    assert env.observation_space.shape is not None
    obs_dim = env.observation_space.shape[0]
    config  = {**DEFAULT_CONFIGS[agent_name]}
    config["max_vel"] = max_vel
    if extra_config:
        config.update(extra_config)

    agent = CONTINUOUS_REGISTRY[agent_name](
        obs_dim, n_actions, config, action_space=env.action_space
    )

    Path(results_dir).mkdir(parents=True, exist_ok=True)
    Path("checkpoints").mkdir(parents=True, exist_ok=True)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_name  = f"{agent_name}_remapping_seed{seed}_{timestamp}"
    
    # 1. Update this line to include the run_name
    checkpoint_dir = Path("checkpoints") / run_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    csv_path  = Path(results_dir) / f"{run_name}.csv"
    json_path = Path(results_dir) / f"{run_name}_manifest.json"
    ckpt_path = checkpoint_dir / f"{run_name}_checkpoint.pt"

    fieldnames = [
        "episode", "phase", "total_reward", "steps", "success",
        "n_licks","n_pokes", "n_successes", "intra_trial_duration",
        "critic_loss", "actor_loss", "alpha",
    ]
    csv_rows : list[dict] = []
    start_t   = time.time()
    total_episodes_with_success = 0

    halfway_point = n_episodes // 3
    telemetry_dir = f"./results/{run_name}/telemetry_logs"
    telemetry = TelemetryLogger(M=halfway_point, R=n_episodes - halfway_point, save_dir=telemetry_dir)
    
    for ep in range(1, n_episodes + 1):
        if ep <= halfway_point:
            current_phase = "mapping"
            phase_idx = 1.0
        else:
            current_phase = "remapping"
            phase_idx = 2.0
            
        env.unwrapped.set_phase(current_phase)
        obs, _   = env.reset(seed=seed + ep)
        
        ep_reward = 0.0
        ep_metrics: dict = {}
        n_licks   = 0
        n_pokes   = 0
        done      = False
        info      = {}
        ep_durations = []

        is_target_ep = ep in telemetry.target_episodes

        while not done:
            # --- 1. Action Selection (Safe Native Call) ---
            # ALWAYS use the agent's built-in method to get the valid environment action
            action = agent.select_action(obs, explore=True)
            value_est = 0.0
            policy_std = 0.0

            # --- 2. Telemetry Extraction ---
            if is_target_ep:
                with torch.no_grad():
                    obs_tensor = torch.tensor(obs, dtype=torch.float32).to(agent.device)
                    if obs_tensor.dim() == 1:
                        obs_tensor = obs_tensor.unsqueeze(0)
                        
                    # Safely extract Value (Handles both PPO V(s) and SAC Q(s,a))
                    if hasattr(agent, 'critic'):
                        try:
                            # Try PPO/VPG style: V(s)
                            v_out = agent.critic(obs_tensor)
                            # Handle twin-critics (tuple) or single critics
                            value_est = v_out[0].item() if isinstance(v_out, tuple) else v_out.mean().item()
                        except Exception:
                            try:
                                # Try SAC/TD3 style: Q(s, a)
                                a_tensor = torch.tensor(action, dtype=torch.float32).to(agent.device).unsqueeze(0)
                                q_out = agent.critic(obs_tensor, a_tensor)
                                value_est = q_out[0].item() if isinstance(q_out, tuple) else q_out.mean().item()
                            except Exception:
                                value_est = 0.0
                    
                    # Safely extract Variance
                    if hasattr(agent, 'actor'):
                        try:
                            actor_out = agent.actor(obs_tensor)
                            if hasattr(actor_out, 'stddev'):
                                policy_std = actor_out.stddev.mean().item()
                            elif isinstance(actor_out, tuple):
                                # If tuple, the second element is usually log_std or std
                                std_tensor = actor_out[1]
                                # Heuristic: if mean is negative, it's likely log_std, so exponentiate it
                                policy_std = torch.exp(std_tensor).mean().item() if torch.mean(std_tensor) < 0 else std_tensor.mean().item()
                        except Exception:
                            policy_std = 0.0

            # --- 3. Environment Step ---
            nobs, r, trunc, info = env.step(action)
            done =  trunc

            # --- 4. Telemetry Logging ---
            if is_target_ep:
                telemetry.log_step(
                    step=info.get("steps", 0),
                    pos=info.get("pos", 0.0),
                    vel=info.get("vel", 0.0),
                    action_vel=action[0],   
                    action_lick=action[1],  
                    value_est=value_est,
                    policy_std=policy_std,
                    reward=float(r),
                    behaviour=info.get("behaviour", "move"), # Move, Lick, or Poke
                )

            # --- 5. Standard Metrics ---

            if info.get("lick_zone") is not None:
                n_licks += 1
            if info.get("nose_poke") is not None: 
                n_pokes += 1
            if info.get("intra_trial_duration") is not None:
                ep_durations.append(info["intra_trial_duration"])

            # --- 5. Agent Update ---
            m = agent.update(obs, action, r, nobs, done)
            if m: 
                ep_metrics.update(m)

            obs = nobs
            ep_reward += float(r)

        # --- END OF WHILE LOOP ---
        
        # Save telemetry buffer for this specific episode
        telemetry.save_episode(ep)

        end_m = agent.on_episode_end()
        if end_m:
            ep_metrics.update(end_m)

        has_success = int(info.get("n_successes", 0) > 0)
        if has_success:
            total_episodes_with_success += 1

        safe_dur = float(np.mean(ep_durations)) if ep_durations else float("nan")

        row = {
            "episode"               : ep,
            "phase"                 : phase_idx,
            "total_reward"          : round(ep_reward, 4),
            "steps"                 : info.get("steps", 0),
            "success"               : has_success,
            "n_licks"               : n_licks,
            "n_pokes"               : n_pokes,
            "n_successes"           : info.get("n_successes", 0),
            "intra_trial_duration"  : safe_dur,
            "critic_loss"           : round(ep_metrics.get("critic_loss", float("nan")), 6),
            "actor_loss"            : round(ep_metrics.get("actor_loss",  float("nan")), 6),
            "alpha"                 : round(ep_metrics.get("alpha",       float("nan")), 6),
        }
        csv_rows.append(row)

        if ep % 500 == 0:
            torch.save(agent, checkpoint_dir / f"model_ep{ep}.pt")

        if verbose and ep % log_every == 0:
            recent = csv_rows[-log_every:]
            avg_r  = np.mean([r["total_reward"] for r in recent])
            avg_sr = np.mean([r["success"]      for r in recent]) * 100
            lk     = np.mean([r["n_licks"]      for r in recent])
            avg_succ = np.mean([r["n_successes"] for r in recent])
            
            valid_durations = [
                r["intra_trial_duration"] for r in recent 
                if not np.isnan(r["intra_trial_duration"])
            ]
            avg_dur = np.mean(valid_durations) if valid_durations else float("nan")
            
            print(f"  Ep {ep:5d}/{n_episodes} ({current_phase.upper()}) | "
                  f"avg_R={avg_r:+.3f} | SR={avg_sr:.1f}% | "
                  f"avg_succ={avg_succ:.2f} | avg_dur={avg_dur:.1f} steps")

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
        "final_success_rate": round(total_episodes_with_success / n_episodes, 4),
        "last_100_success"  : round(np.mean([r["success"] for r in csv_rows[-100:]]), 4),
        "elapsed_seconds"   : round(elapsed, 2),
        "csv_path"          : str(csv_path),
        "ckpt_path"         : str(ckpt_path),
    }
    with open(json_path, "w") as f:
        json.dump(manifest, f, indent=2)

    return manifest

def _parse():
    p = argparse.ArgumentParser()
    p.add_argument("--agent",          default="ppo", choices=list(CONTINUOUS_REGISTRY))
    p.add_argument("--episodes",       type=int,   default=3000)
    p.add_argument("--track_length",   type=float, default=120.0)
    p.add_argument("--max_vel",        type=float, default=20.0) 
    p.add_argument("--terminal_width", type=float, default=3.0)
    p.add_argument("--max_steps",      type=int,   default=500)
    p.add_argument("--step_penalty",   type=float, default=-0.005)
    p.add_argument("--lick_penalty",   type=float, default=-0.05) 
    p.add_argument("--water_reward",   type=float, default=10.0) 
    p.add_argument("--seed",           type=int,   default=42)
    p.add_argument("--results_dir",    default="results")
    p.add_argument("--log_every",      type=int,   default=50)   
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