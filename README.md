# Q-Learning Assignment

This is the Mid-term Assignment for IOTA5201 (L01) - RL for Intelligent Decision Making in Cyber-Physical Systems. The project implements Q-learning algorithm for two different environments:

- **Environment 1**: Abstract room-based environment with 6 states (A through F) *(Required)*
- **Environment 2**: Grid-based environment with configurable map sizes and complex obstacles *(Extra)*

## Assignment Requirements (Required Tasks)

The following tasks are required by the assignment:

1. **Q-Learning Implementation**: Python implementation of Q-learning algorithm
2. **Environment Setup**: Define environment with states and actions
3. **Training**: Train Q-learning with suitable number of episodes
4. **Convergence Plot**: Generate and analyze convergence plots
5. **State Configuration Changes**: 
   - Experiment 1: F → C (initial state F, target state C)
   - Experiment 2: A → C (initial state A, target state C)
6. **Report**: Include Q-table values, convergence plots, and analysis

### Running Required Experiments

```bash
# Environment 1 - Required experiments
python main_env1.py --single    # Run both required experiments (F->C and A->C)

# Or run individually:
python run_experiment1.py       # Experiment 1: F -> C
python run_experiment2.py       # Experiment 2: A -> C
```

These will generate the required outputs:
- Convergence plots for both configurations
- Q-table values (saved as `.npy` files)
- Training data for analysis

## Extra Features (Beyond Requirements)

The following features extend beyond the assignment requirements:

### Additional Environment
- **Environment 2**: Grid-based maze navigation with regions, walls, and obstacles
  - More realistic visualization with cell-based navigation
  - Complex obstacle layouts
  - Multiple map sizes (20×20, 25×25, 30×30)

### Parameter Grid Search *(Extra)*
- Test multiple combinations of alpha (learning rate) and gamma (discount factor)
- Default grid: α ∈ [0.05, 0.1, 0.2], γ ∈ [0.8, 0.9, 0.95]
- Automatic analysis and visualization of parameter effects

### Parameter Analysis Tools *(Extra)*
- Comprehensive parameter sensitivity analysis
- Cross-grid comparison visualizations (for Environment 2)
- Statistical analysis and ranking of parameter combinations
- Automatic report generation

### Optimized Q-Learning Implementation *(Extra)*
- Sparse Q-table implementation for large state spaces (memory efficient)
- Progress tracking with `tqdm` progress bars
- Enhanced early stopping mechanism
- Episode length limits to prevent infinite loops
- Reduced computational overhead

### Advanced Visualizations *(Extra)*
- Q-table statistics and heatmaps
- Policy visualization with pathfinding
- Unified statistics plots for large Q-tables
- Cross-grid performance comparisons

## Quick Start

### Required Experiments (Environment 1)

```bash
# Run both required experiments (F->C and A->C) with default parameters
python main_env1.py --single
```

### Extra Features

**Environment 1 - Parameter Grid Search:**
```bash
# Run all experiments with parameter grid
python main_env1.py

# Run parameter grid only
python main_env1.py --grid
```

**Environment 2 - Grid-based Environment:**
```bash
# Run all experiments with all map sizes and parameter combinations
python main_env2.py

# Run single experiment with default parameters (20×20 map)
python main_env2.py --single
```

### Individual Experiments

**Required (Environment 1):**
```bash
python run_experiment1.py        # Experiment 1: F -> C (Required)
python run_experiment2.py        # Experiment 2: A -> C (Required)
```

**Extra (Environment 2):**
```bash
python run_experiment1_env2.py  # Experiment 1: F -> C (Extra)
python run_experiment2_env2.py  # Experiment 2: A -> C (Extra)
```

## Parameter Grid Search *(Extra)*

### Default Parameter Grid

The default parameter combinations tested:
- **Alpha (learning rate)**: [0.05, 0.1, 0.2]
- **Gamma (discount factor)**: [0.8, 0.9, 0.95]
- **Total combinations**: 9 per experiment

### Running Parameter Grid

```bash
# Environment 1
python run_experiment1.py --grid
python run_experiment2.py --grid

# Environment 2 (with specific map size)
python run_experiment1_env2.py --grid  # Uses default 20×20
```

For Environment 2, running `python main_env2.py` will automatically test all map sizes (20×20, 25×25, 30×30) with all parameter combinations.

## Parameter Analysis *(Extra)*

After running parameter grid experiments, analyze the results:

```bash
python analyze_parameters.py
```

This generates:
- Parameter comparison charts for each experiment
- Heatmaps showing parameter sensitivity
- Cross-grid comparisons (for Environment 2)
- Markdown reports with rankings and statistics

### Output Analysis Files

- `*_parameter_analysis.png` - Comprehensive parameter analysis dashboard
- `*_parameter_analysis.md` - Detailed markdown report
- `*_cross_grid_comparison.png` - Comparison across different map sizes (Environment 2)

## Output Files

### Experiment Results

Each experiment generates files in `results/{exp_name}/`:

**Visualizations:**
- `convergence_plot*.png` - Convergence curves for rewards and steps
- `qtable_statistics*.png` - Q-table statistics and heatmaps
- `final_policy*.png` - Optimal policy visualization with path

**Data Files:**
- `*_training_data.pkl` - Complete training data (pickle format)
- `*_qtable_final.npy` - Final Q-table as NumPy array
- `*_qtable_initial.npy` - Initial Q-table (all zeros)
- `*_rewards_steps.txt` - Episode rewards and steps (text format)

**File Naming Convention:**
- Single experiment: `convergence_plot.png`, `final_policy.png`
- Parameter grid: `convergence_plot_grid20x20_alpha0_1_gamma0_9.png`
- Training data: `env2_exp1_grid20x20_alpha0_1_gamma0_9_training_data.pkl`

## Project Structure

```
.
├── main_env1.py              # Main script for Environment 1
├── main_env2.py              # Main script for Environment 2
├── environment.py            # Environment 1 implementation
├── environment2.py            # Environment 2 implementation (grid-based)
├── q_learning.py             # Basic Q-learning implementation
├── q_learning_optimized.py   # Optimized Q-learning (sparse Q-table, tqdm)
├── visualization.py           # Plotting functions
├── data_utils.py             # Data saving and loading utilities
├── experiment_utils.py        # Shared utility functions
├── analyze_parameters.py     # Parameter analysis and visualization
│
├── run_experiment1.py        # Environment 1 - Experiment 1 (F -> C)
├── run_experiment2.py        # Environment 1 - Experiment 2 (A -> C)
├── run_experiment1_env2.py  # Environment 2 - Experiment 1 (F -> C)
├── run_experiment2_env2.py  # Environment 2 - Experiment 2 (A -> C)
│
└── results/                   # Output directory
    ├── env1_exp1/            # Environment 1 - Experiment 1 results
    ├── env1_exp2/            # Environment 1 - Experiment 2 results
    ├── env2_exp1/            # Environment 2 - Experiment 1 results
    └── env2_exp2/            # Environment 2 - Experiment 2 results
```

## Default Hyperparameters

### Environment 1
- Learning rate (alpha): 0.1
- Discount factor (gamma): 0.9
- Epsilon start: 0.9
- Epsilon min: 0.1
- Episodes: 1000
- Early stopping: enabled (patience=100, threshold=1e-4)

### Environment 2
- Learning rate (alpha): 0.1
- Discount factor (gamma): 0.9
- Epsilon start: 0.9
- Epsilon min: 0.1
- Episodes: Scaled by map size (1500 base for 20×20, scales with grid size)
- Early stopping: enabled (patience=100, threshold=1e-4)
- Max steps per episode: min(1000, n_states × 2)

## Environment Details

### Environment 1: Abstract Room-based *(Required)*
- 6 states (rooms A-F)
- Direct state transitions
- Simple reward structure
- Used for required assignment tasks

### Environment 2: Grid-based *(Extra)*
- Configurable grid size (default: 20×20)
- 6 regions (A-F) with complex layouts
- Walls and obstacles between regions
- Cell-based navigation with 4 actions (up, down, left, right)
- More complex pathfinding requirements
- Additional challenge for analysis

**Map Sizes (Extra):**
- **20×20**: Base map size (400 states)
- **25×25**: Medium complexity (625 states)
- **30×30**: Large complexity (900 states)

## Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

Key dependencies:
- `numpy` - Numerical computations
- `matplotlib` - Plotting and visualization
- `tqdm` - Progress bars
- `scipy` (optional) - Additional utilities

## Reproducibility

All experiments use a fixed random seed (42) for reproducibility. The seed is set in:
- `main_env1.py`
- `main_env2.py`
- Individual experiment scripts

## Performance Optimizations *(Extra)*

The optimized Q-learning implementation (`q_learning_optimized.py`) includes:
- **Sparse Q-table**: Uses `defaultdict` for memory efficiency with large state spaces
- **Episode length limits**: Prevents infinite loops in early exploration
- **Optimized early stopping**: Checks convergence less frequently
- **Progress tracking**: Visual progress bars during training
- **Reduced overhead**: Minimized unnecessary copying operations

## Analysis Features *(Extra)*

### Parameter Analysis Dashboard
- Convergence curves for rewards and steps
- Heatmaps for final rewards, steps, and convergence
- Parameter effect plots (alpha/gamma sensitivity)
- Ranked results table
- Summary statistics

### Cross-Grid Comparison (Environment 2)
- Average performance by grid size
- Parameter sensitivity across map sizes
- Convergence speed comparison
- Visual comparison of all three map sizes

## Notes

- For large Q-tables (≥100 states), the visualization automatically switches to statistical summaries instead of full heatmaps
- Convergence episode calculation uses improved stability checks
- If convergence episode shows no variation, the plot automatically switches to reward stability metric
- All file names include grid size information to prevent overwriting when running multiple map sizes

## Troubleshooting

**Long training times:**
- Reduce map size in Environment 2
- Use fewer episodes
- Adjust early stopping parameters

**Memory issues:**
- The optimized Q-learning uses sparse Q-tables automatically
- Reduce grid size if needed

**Missing files:**
- Ensure output directories exist (created automatically)
- Check file naming includes grid size and parameters for parameter grid runs
