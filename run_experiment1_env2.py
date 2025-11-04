"""
Experiment 1 (Environment 2): Original configuration
Grid-based environment with cells.
Initial region: F
Target region: C
"""

from environment2 import GridWorld2
from q_learning_optimized import train_q_learning, set_random_seed
from visualization import plot_convergence, plot_q_table_heatmap, plot_grid_policy
from data_utils import save_training_data
from experiment_utils import print_experiment_stats
import numpy as np
import os


def print_q_table_summary_env2(env, q_table):
    """Print Q-table summary for grid environment."""
    print("\n" + "=" * 60)
    print("Final Q-table Summary (sample states):")
    print("=" * 60)
    
    # Sample some states from each region
    for region_idx in range(6):
        region_name = env.regions[region_idx]
        region_id = region_idx + 2
        region_cells = np.argwhere(env.grid == region_id)
        
        if len(region_cells) > 0:
            # Sample a few states from this region
            sample_indices = np.linspace(0, len(region_cells)-1, 
                                       min(3, len(region_cells)), dtype=int)
            for idx in sample_indices:
                pos = tuple(region_cells[idx])
                state = env._pos_to_state(pos)
                valid_actions = env.get_valid_actions(state)
                
                if valid_actions:
                    q_values = [(action, q_table[state][action]) for action in valid_actions]
                    q_values.sort(key=lambda x: x[1], reverse=True)
                    
                    best_action, best_q = q_values[0]
                    best_pos = env._state_to_pos(best_action)
                    print(f"Region {region_name}, State {state} (pos {pos}): "
                          f"Best action -> {best_action} (pos {best_pos}), Q={best_q:.2f}")
    
    print("\n" + "=" * 60)


def main(alpha=0.1, gamma=0.9, exp_suffix="", grid_size=(20, 20), file_suffix=""):
    """
    Run experiment 1 with Environment 2: F -> C
    
    Args:
        alpha: learning rate
        gamma: discount factor
        exp_suffix: suffix for experiment directory name
        grid_size: grid dimensions for Environment 2 (default: 20x20)
        file_suffix: suffix for individual file names (for parameter grid)
    """
    # set random seed for reproducibility
    set_random_seed(42)
    
    print("=" * 60)
    print("Experiment 1 (Environment 2): Original Configuration")
    print("=" * 60)
    print("Initial region: F")
    print("Target region: C")
    print(f"Grid size: {grid_size}")
    print()
    
    # setup experiment
    initial_region = 'F'
    target_region = 'C'
    # Include grid size in experiment name
    grid_str = f"grid{grid_size[0]}x{grid_size[1]}"
    exp_name = f"env2_exp1{exp_suffix}"
    save_dir = f"results/{exp_name}"
    
    # create output directory
    os.makedirs(save_dir, exist_ok=True)
    
    # create environment (20x20 grid, 400 states)
    env = GridWorld2(grid_size=grid_size, 
                     initial_region=initial_region, 
                     target_region=target_region)
    
    # hyperparameters (adjusted for larger state space)
    # Scale episodes based on grid size
    base_episodes = 1500
    grid_factor = (grid_size[0] * grid_size[1]) / 400  # 400 is base 20x20
    n_episodes = int(base_episodes * grid_factor)
    epsilon_start = 0.9
    epsilon_min = 0.1
    log_interval = 100  # More frequent logging for progress tracking
    
    print(f"Hyperparameters:")
    print(f"  Episodes: {n_episodes}")
    print(f"  Learning rate (alpha): {alpha}")
    print(f"  Discount factor (gamma): {gamma}")
    print(f"  Epsilon start: {epsilon_start}, min: {epsilon_min}")
    print(f"  Total states: {env.n_states}")
    print()
    
    # train
    episode_rewards, episode_steps, q_table_history, early_stopped, initial_q_table = train_q_learning(
        env, n_episodes=n_episodes, alpha=alpha, gamma=gamma,
        epsilon_start=epsilon_start, epsilon_min=epsilon_min,
        log_interval=log_interval
    )
    
    # get final Q-table
    final_q_table = q_table_history[-1][1]
    
    # print statistics
    print_experiment_stats(episode_rewards, episode_steps)
    
    # print Q-table summary (sample)
    print_q_table_summary_env2(env, final_q_table)
    
    # visualize
    # Add file suffix to visualization files: include grid size and parameters
    if file_suffix:
        viz_suffix = f"_{grid_str}{file_suffix}"
    else:
        viz_suffix = f"_{grid_str}"
    
    print("\nGenerating convergence plot...")
    plot_convergence(
        episode_rewards, episode_steps, q_table_history,
        save_path=f'{save_dir}/convergence_plot{viz_suffix}.png',
        show_plot=False,
        title_suffix=f" - Experiment 1 (F -> C) - Environment 2 - {grid_str}"
    )
    
    print("Generating Q-table statistics (unified heatmap and comparison)...")
    # For large Q-tables, both heatmap and comparison are saved to the same unified file
    plot_q_table_heatmap(
        final_q_table, state_names=None,
        save_path=f'{save_dir}/qtable_statistics{viz_suffix}.png',
        show_plot=False,
        title=f" - Experiment 1 (F -> C) - Environment 2 - {grid_str}",
        initial_q_table=initial_q_table  # Pass initial Q-table for comparison
    )
    
    print("Generating grid policy visualization...")
    plot_grid_policy(
        env, final_q_table, initial_region, target_region,
        save_path=f'{save_dir}/final_policy{viz_suffix}.png',
        show_plot=False,
        title=f" - Experiment 1 (F -> C) - Environment 2 - {grid_str}"
    )
    
    # save data
    print("\nSaving training data...")
    # Include grid size and parameters in training data filename
    training_name = f"{exp_name}_{grid_str}{file_suffix}" if file_suffix else f"{exp_name}_{grid_str}"
    save_training_data(
        episode_rewards, episode_steps, q_table_history, final_q_table,
        save_dir=save_dir, experiment_name=training_name, initial_q_table=initial_q_table
    )
    
    print("\n" + "=" * 60)
    print("Experiment 1 (Environment 2) completed!")
    if early_stopped:
        print(f"Training stopped early at episode {len(episode_rewards)}")
    print(f"Results saved to: {save_dir}/")
    print("=" * 60)
    
    return exp_name, save_dir


def run_parameter_grid(alpha_list=[0.05, 0.1, 0.2], gamma_list=[0.8, 0.9, 0.95], 
                       grid_size=(20, 20)):
    """
    Run experiments with multiple parameter combinations.
    
    Args:
        alpha_list: list of learning rates to test
        gamma_list: list of discount factors to test
        grid_size: grid dimensions for Environment 2
    """
    # set random seed for reproducibility
    set_random_seed(42)
    
    grid_str = f"grid{grid_size[0]}x{grid_size[1]}"
    print("=" * 70)
    print(f"Multi-Parameter Experiment: Experiment 1 (F -> C) - {grid_str}")
    print("=" * 70)
    print(f"Testing {len(alpha_list)} alpha values × {len(gamma_list)} gamma values")
    print(f"Total combinations: {len(alpha_list) * len(gamma_list)}")
    print()
    
    results = []
    
    for alpha in alpha_list:
        for gamma in gamma_list:
            # create suffix for file names (not directory)
            file_suffix = f"_alpha{alpha}_gamma{gamma}"
            file_suffix = file_suffix.replace(".", "_")  # replace dots for file names
            
            print("\n" + "=" * 70)
            print(f"Running: alpha={alpha}, gamma={gamma}, grid={grid_size}")
            print("=" * 70)
            
            try:
                exp_name, save_dir = main(alpha=alpha, gamma=gamma, grid_size=grid_size, 
                                         file_suffix=file_suffix)
                results.append({
                    'alpha': alpha,
                    'gamma': gamma,
                    'grid_size': grid_size,
                    'exp_name': exp_name,
                    'save_dir': save_dir,
                    'success': True
                })
            except Exception as e:
                print(f"Error running alpha={alpha}, gamma={gamma}, grid={grid_size}: {e}")
                results.append({
                    'alpha': alpha,
                    'gamma': gamma,
                    'grid_size': grid_size,
                    'success': False,
                    'error': str(e)
                })
    
    # print summary
    print("\n" + "=" * 70)
    print("Multi-Parameter Experiment Summary")
    print("=" * 70)
    print(f"Total experiments: {len(results)}")
    print(f"Successful: {sum(1 for r in results if r['success'])}")
    print(f"Failed: {sum(1 for r in results if not r['success'])}")
    print()
    
    return results


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--grid":
        # run with default parameter grid
        alpha_list = [0.05, 0.1, 0.2]
        gamma_list = [0.8, 0.9, 0.95]
        run_parameter_grid(alpha_list, gamma_list, grid_size=(20, 20))
    else:
        # run single experiment with default parameters
        main()

