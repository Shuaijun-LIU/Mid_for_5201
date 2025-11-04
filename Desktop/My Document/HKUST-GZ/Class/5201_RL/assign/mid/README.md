# Q-Learning Assignment

This is the Mid-term Assignment for IOTA5201 (L01) - RL for Intelligent Decision Making in Cyber-Physical Systems, implements Q-learning algorithm for a grid world navigation task with 6 states (A through F).

## Quick Start

To run both experiments:

```bash
python main.py
```

This will execute experiment 1 (F -> C) and experiment 2 (A -> C) sequentially, generating all results and visualizations.

## Individual Experiments

To run experiments separately:

```bash
# Experiment 1: F -> C
python run_experiment1.py

# Experiment 2: A -> C  
python run_experiment2.py
```

## Multi-Parameter Experiments

To test different hyperparameter combinations:

```bash
# Run parameter grid for experiment 1
python run_experiment1.py --grid

# Run parameter grid for experiment 2
python run_experiment2.py --grid

# Run parameter grids for both experiments
python main.py --grid
```

The default parameter grid tests:
- Alpha (learning rate): [0.05, 0.1, 0.2]
- Gamma (discount factor): [0.8, 0.9, 0.95]

Results for each combination will be saved in separate directories under `results/`.

## Output Files

Each experiment generates the following files in `results/exp1/` or `results/exp2/`:

- `convergence_plot.png` - Convergence curves for Q-table, rewards, and steps
- `qtable_heatmap.png` - Heatmap visualization of the final Q-table
- `final_policy.png` - Visual representation of the optimal policy and path
- `*_qtable_final.npy` - Final Q-table as NumPy array
- `*_training_data.pkl` - Complete training data (pickle format)
- `*_rewards_steps.txt` - Episode rewards and steps (text format)

## Requirements

See `requirements.txt` for required packages. Install with:

```bash
pip install -r requirements.txt
```

## Project Structure

- `environment.py` - Grid world environment implementation
- `q_learning.py` - Q-learning algorithm and training functions
- `visualization.py` - Plotting functions for convergence and policy
- `data_utils.py` - Data saving and loading utilities
- `experiment_utils.py` - Shared utility functions
- `run_experiment1.py` - Experiment 1 script (F -> C)
- `run_experiment2.py` - Experiment 2 script (A -> C)
- `main.py` - Script to run both experiments

## Features

- Q-learning algorithm with epsilon-greedy exploration
- Early stopping when Q-table converges
- Fixed random seed for reproducibility (seed=42)
- Multiple visualization types (convergence plots, heatmaps, policy graphs)
- Support for parameter grid experiments
- Automatic result saving in multiple formats

## Default Hyperparameters

- Learning rate (alpha): 0.1
- Discount factor (gamma): 0.9
- Epsilon start: 0.9
- Epsilon min: 0.1
- Episodes: 1000
- Early stopping: enabled (patience=100, threshold=1e-4)

