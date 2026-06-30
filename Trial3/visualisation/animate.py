import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def animate_behavior_trajectory(telemetry_csv_path, track_length=120.0, terminal_width=1.0):
    # Load step-by-step tracking data
    df = pd.read_csv(telemetry_csv_path)
    
    positions = df['positions'].values
    behaviors = df['behaviors'].values
    rewards = df['rewards'].values
    
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.set_xlim(-5, track_length + 5)
    ax.set_ylim(-1, 1)
    ax.set_yticks([])  # 1D linear track representation
    ax.set_title("Mice Tracking Visualizer: Spatial Action Architecture", fontsize=12, fontweight='bold')
    
    # Draw track infrastructure and lick zones
    ax.plot([0, track_length], [0, 0], color='black', lw=3, zorder=1)
    ax.axvspan(0, terminal_width, color='blue', alpha=0.2, label="Left Port Zone")
    ax.axvspan(track_length - terminal_width, track_length, color='blue', alpha=0.2, label="Right Port Zone")
    
    # Animated dynamic objects
    agent_dot, = ax.plot([], [], 'o', markersize=10, zorder=3)
    text_annotation = ax.text(track_length / 2, 0.5, '', ha='center', va='center', fontsize=11, fontweight='bold')
    
    def init():
        agent_dot.set_data([], [])
        text_annotation.set_text('')
        return agent_dot, text_annotation

    def update(frame):
        x = positions[frame]
        act = behaviors[frame]
        r = rewards[frame]
        
        agent_dot.set_data([x], [0])
        
        # Color coding state matching your cost logic
        if act == "lick":
            # Green if it triggered water reward, red if it was an empty lick
            color = 'limegreen' if r > 0 else 'crimson'
            text_annotation.set_text('★ FULL LICK EXECUTION ★' if r > 0 else '⚠️ INCORRECT LICK')
            agent_dot.set_markersize(16)
        elif act == "nose_poke":
            color = 'darkorange'
            text_annotation.set_text('🔍 Probing: Nose Poke')
            agent_dot.set_markersize(12)
        else:
            color = 'slategray'
            text_annotation.set_text('')
            agent_dot.set_markersize(8)
            
        agent_dot.set_color(color)
        return agent_dot, text_annotation

    ani = animation.FuncAnimation(
        fig, update, frames=len(df), init_func=init, 
        blit=True, interval=40
    )
    
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2)
    plt.tight_layout()
    plt.show()

# Example execution entry point
if __name__ == "__main__":
    animate_behavior_trajectory("your_output_rollout.csv")
    