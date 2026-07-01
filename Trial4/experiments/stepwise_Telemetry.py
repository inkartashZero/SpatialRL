import pandas as pd
import numpy as np
import os

class TelemetryLogger:
    def __init__(self, M, R, save_dir="telemetry_logs"):
        """
        M: Total episodes in mapping phase (e.g., 1500)
        R: Total episodes in remapping phase (e.g., 1500)
        """
        self.target_episodes = [
            5,                  # Initial exploration
            M // 2,             # Middle of mapping
            M,                  # End of mapping (Peak entrenchment)
            M + (R // 2),       # Middle of remapping
            M + R               # Final episode
        ]
        self.current_ep_data = []
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        
        print(f"Telemetry active for milestone episodes: {self.target_episodes}")

    def log_step(self, step, pos, vel, action_vel, action_lick, value_est, policy_std, reward):
        """Appends a single timestep's data to the current episode buffer."""
        self.current_ep_data.append({
            "Step": step,
            "Position": round(pos, 3),
            "Velocity": round(vel, 3),
            "Action_Vel_Cmd": round(action_vel, 3),
            "Action_Lick_Cmd": round(action_lick, 3),
            "Critic_Value": round(value_est, 4),
            "Policy_StdDev": round(policy_std, 4),
            "Reward": reward
        })

    def save_episode(self, episode):
        """Dumps the buffer to a CSV if it's a target episode, then clears the buffer."""
        if episode in self.target_episodes and len(self.current_ep_data) > 0:
            df = pd.DataFrame(self.current_ep_data)
            filepath = os.path.join(self.save_dir, f"telemetry_ep_{episode}.csv")
            df.to_csv(filepath, index=False)
            print(f" -> Saved high-frequency telemetry: {filepath}")
            
        # Always clear the buffer at the end of an episode to prevent memory leaks
        self.current_ep_data = []
