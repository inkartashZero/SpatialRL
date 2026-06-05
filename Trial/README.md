# Spatial RL — LinearTrack

A modular codebase for testing classical and deep RL algorithms on a
1D **shuttle task** inspired by rodent linear-track experiments.

---

## Environment: `LinearTrackEnv`

```
[R] [.] [.] [.] [.] [.] [.] [.] [.] [.] | [#] [#] [#] [#] [#] [#] [#] [#] [#] [R]
 0   1   2   3   4   5   6   7   8   9     10  11  12  13  14  15  16  17  18  19
```

| Symbol | Colour  | Tactile | Meaning              |
|--------|---------|---------|----------------------|
| `R`    | RED     | —       | Terminal (reward end) |
| `.`    | BLACK   | smooth  | Left half            |
| `#`    | YELLOW  | rough   | Right half           |

**Shuttle task**: the agent must visit **both** terminals (in any order)
within a single episode to earn `+1`. A small per-step penalty (`-0.01`)
encourages efficiency.

### Observation (7-dim)

| Index | Meaning              | Range  |
|-------|----------------------|--------|
| 0     | normalised position  | [0, 1] |
| 1     | tactile              | {0, 1} |
| 2–4   | RGB colour           | [0, 1] |
| 5     | visited left flag    | {0, 1} |
| 6     | visited right flag   | {0, 1} |

---

## Agents

| Name          | Class             | Type           |
|---------------|-------------------|----------------|
| `q_learning`  | `QLearningAgent`  | Tabular         |
| `sarsa`       | `SARSAAgent`      | Tabular (on-policy) |
| `dqn`         | `DQNAgent`        | Deep (MLP + replay) |
| `reinforce`   | `REINFORCEAgent`  | Policy gradient |

---

## Quick Start

```bash
pip install -r requirements.txt

# Run demo (Q-Learning + DQN, 500 eps each)
python demo.py

# Train a single agent
python -m experiments.run --agent dqn --episodes 2000

# Compare all agents across 3 seeds
python -m experiments.compare --agents q_learning sarsa dqn reinforce \
                               --seeds 0 1 2 --episodes 2000
```

---

## Results & Visualisation

All CSVs and manifests land in `results/`.

```python
from visualisation import plot_learning_curve, plot_comparison, plot_track

# Single run
plot_learning_curve("results/dqn_seed42_....csv")

# Multi-agent comparison (reads all CSVs in results/)
plot_comparison("results/")

# Track layout
from envs import LinearTrackEnv
plot_track(LinearTrackEnv())
```

---

## File Structure

```
spatial_rl/
├── envs/
│   ├── __init__.py
│   └── linear_track.py        ← environment
├── agents/
│   ├── __init__.py
│   ├── base.py                ← BaseAgent ABC
│   ├── q_learning.py          ← tabular Q-Learning
│   ├── sarsa.py               ← tabular SARSA
│   ├── dqn.py                 ← Deep Q-Network
│   └── reinforce.py           ← REINFORCE policy gradient
├── experiments/
│   ├── __init__.py
│   ├── run.py                 ← single experiment runner + CLI
│   └── compare.py             ← multi-agent comparison + CLI
├── visualisation/
│   ├── __init__.py
│   └── plots.py               ← all plotting functions
├── results/                   ← auto-created, CSVs + manifests land here
├── demo.py                    ← quick sanity-check script
└── requirements.txt
```

---

## Extending

**New agent**: subclass `BaseAgent` in `agents/`, implement
`select_action` and `update`, then register in `agents/__init__.py`'s `REGISTRY`.

**New environment variant**: copy `linear_track.py`, adjust `_get_colour`
/ `_get_tactile` / `step` reward logic.

**Hyperparameter sweep**: import `run_experiment` and loop over your grid;
each call writes its own CSV.
