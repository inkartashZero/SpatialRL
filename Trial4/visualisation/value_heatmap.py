import numpy as np
import matplotlib.pyplot as plt
import torch
import glob
import re

# Adjust these to match your environment's exact parameters
TRACK_LENGTH = 120.0
MAX_VEL = 10.0
POS_RESOLUTION = 120  # Number of points to sample across the track
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_test_state_batch(licked_L=False, licked_R=False, phase="mapping"):
    """
    Builds a batch of states sweeping across the track positions.
    Assumes a 7-dim state: [pos_norm, vel_norm, is_terminal, tactile, licked_L, licked_R, phase_bit]
    """
    states = []
    for pos in np.linspace(0, TRACK_LENGTH, POS_RESOLUTION):
        pos_norm = pos / TRACK_LENGTH
        vel_norm = 0.0 # Evaluate value while stationary
        is_terminal = 1.0 if (pos <= 1.0 or pos >= TRACK_LENGTH - 1.0) else 0.0
        tactile = 0.0 if pos <= TRACK_LENGTH / 2.0 else 1.0
        phase_bit = 0.0 if phase == "mapping" else 1.0
        
        state = [pos_norm, vel_norm, is_terminal, tactile, float(licked_L), float(licked_R), phase_bit]
        states.append(state)
        
    return torch.tensor(states, dtype=torch.float32).to(DEVICE)

def extract_episode_num(filename):
    """Extracts episode number from a checkpoint filename like 'model_ep1500.pt'"""
    match = re.search(r'ep(\d+)', filename)
    return int(match.group(1)) if match else -1

def plot_temporal_v_heatmap(checkpoint_dir, save_path="v_heatmap.png"):
    """
    Loads model checkpoints, evaluates V(s) across the track, and plots a temporal heatmap.
    """
    checkpoints = glob.glob(f"{checkpoint_dir}/*.pt")
    checkpoints.sort(key=extract_episode_num)
    
    if not checkpoints:
        print("No checkpoints found!")
        return

    episodes = []
    value_matrix = []

    # Choose the context you want to evaluate. 
    # E.g., Left port is flagged, testing value of moving right.
    test_states = build_test_state_batch(licked_L=True, licked_R=False, phase="mapping")

    for ckpt_path in checkpoints:
        ep_num = extract_episode_num(ckpt_path)
        episodes.append(ep_num)
        
        # Load your specific agent/critic architecture
        # agent = torch.load(ckpt_path)
        # critic = agent.critic 
        
        # NOTE: Replace 'mock_critic' with your actual loaded critic network
        # with torch.no_grad():
        #     values = critic(test_states).cpu().numpy().flatten()
        
        # --- DUMMY DATA FOR SCRIPT TESTING ---
        values = np.zeros(POS_RESOLUTION) 
        value_matrix.append(values)

    value_matrix = np.array(value_matrix)

    # Plotting
    plt.figure(figsize=(12, 8))
    # origin='lower' puts Episode 0 at the bottom
    im = plt.imshow(value_matrix, aspect='auto', origin='lower', cmap='viridis',
                    extent=[0, TRACK_LENGTH, min(episodes), max(episodes)])
    
    plt.colorbar(im, label='Predicted Value V(s)')
    plt.axhline(y=1500, color='red', linestyle='--', linewidth=2, label='Phase Shift (Remapping)')
    
    plt.title('Temporal Evolution of V(s) [Context: Licked Left]')
    plt.xlabel('Track Position')
    plt.ylabel('Training Episode')
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(save_path)
    print(f"Heatmap saved to {save_path}")

if __name__ == "__main__":
    # Point this to the folder where your PyTorch weights (.pt) are saved during training
    plot_temporal_v_heatmap("./checkpoints_sac")
    # # For Q(s, a_lick = 1)
    # action_lick = torch.tensor([[0.0, 1.0]] * POS_RESOLUTION, dtype=torch.float32).to(DEVICE)
    # values_lick = critic(test_states, action_lick).cpu().numpy().flatten()
        
    #     # For Q(s, a_lick = 0)
    # action_no_lick = torch.tensor([[10.0, -1.0]] * POS_RESOLUTION, dtype=torch.float32).to(DEVICE)
    # values_no_lick = critic(test_states, action_no_lick).cpu().numpy().flatten()