import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse

def plot_telemetry_profiles(csv_path, track_length=120.0, terminal_width=3.0):
    """
    Reads a telemetry CSV and plots Velocity, Lick Commands, and Value Estimates
    against the physical position on the track.
    """
    # Auto-append .csv if missing
    if not csv_path.endswith('.csv'):
        csv_path += '.csv'
        
    if not os.path.exists(csv_path):
        print(f"Error: Could not find file at {csv_path}")
        return

    # Load the telemetry data
    df = pd.DataFrame(pd.read_csv(csv_path))
    
    # Sort by step to ensure chronological drawing if the agent backtracks
    df = df.sort_values(by="Step")
    
    # Extract episode number from filename for the title
    ep_num = os.path.basename(csv_path).replace('telemetry_ep_', '').replace('.csv', '')

    # Set up a 3-panel figure
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.suptitle(f'Agent Behavioral Profile - Episode {ep_num}', fontsize=16, fontweight='bold')

    # Common visual settings
    left_zone = terminal_width
    right_zone = track_length - terminal_width

    for ax in axes:
        # Draw the lick port zones
        ax.axvspan(0, left_zone, color='blue', alpha=0.1, label='Left Port Zone' if ax == axes[0] else "")
        ax.axvspan(right_zone, track_length, color='orange', alpha=0.1, label='Right Port Zone' if ax == axes[0] else "")
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.set_xlim(0, track_length)

    # --- Panel 1: Velocity Profile ---
    ax1 = axes[0]
    ax1.plot(df["Position"], df["Velocity"], label="Actual Velocity", color='black', linewidth=2)
    ax1.plot(df["Position"], df["Action_Vel_Cmd"], label="Network Command (Raw)", color='red', linestyle=':', alpha=0.7)
    ax1.set_ylabel('Velocity')
    ax1.set_title('Locomotion Momentum')
    ax1.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=4)

    # --- Panel 2: Lick Profile ---
    ax2 = axes[1]
    ax2.plot(df["Position"], df["Action_Lick_Cmd"], color='purple', linewidth=2)
    ax2.axhline(0, color='black', linewidth=1) # Lick threshold (typically > 0 triggers a lick)
    ax2.fill_between(df["Position"], 0, df["Action_Lick_Cmd"], where=(df["Action_Lick_Cmd"] > 0), color='purple', alpha=0.3)
    ax2.set_ylabel('Lick Action Value')
    ax2.set_title('Lick Port Interaction (Values > 0 trigger lick)')

    # --- Panel 3: Value Function ---
    ax3 = axes[2]
    ax3.plot(df["Position"], df["Critic_Value"], color='green', linewidth=2)
    ax3.fill_between(df["Position"], 0, df["Critic_Value"], color='green', alpha=0.2)
    ax3.set_ylabel('Predicted Value $V(s)$')
    ax3.set_xlabel('Track Position')
    ax3.set_title('Internal Reward Prediction Landscape')

    plt.tight_layout()
    
    # Save the plot next to the CSV
    save_dir = os.path.dirname(csv_path)
    save_filename = f"profile_plot_ep_{ep_num}.png"
    save_path = os.path.join(save_dir, save_filename)
    
    plt.savefig(save_path, dpi=300)
    print(f"-> Successfully saved profile plot to: {save_path}")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot high-frequency RL telemetry.")
    parser.add_argument("csv_path", type=str, help="Path to the telemetry_ep_X.csv file")
    parser.add_argument("--track_length", type=float, default=120.0)
    args = parser.parse_args()

    plot_telemetry_profiles(args.csv_path, args.track_length)