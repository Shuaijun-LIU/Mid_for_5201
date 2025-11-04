"""
Experiment 2 (Environment 2): Changed configuration
Grid-based environment with cells.
Initial region: A
Target region: C
"""

from environment2 import GridWorld2
from q_learning import train_q_learning, set_random_seed
from visualization import plot_convergence, plot_q_table_heatmap, plot_grid_policy
from data_utils import save_training_data
from experiment_utils import print_experiment_stats
import numpy as np


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


def main(alpha=0.1, gamma=0.9, exp_suffix="", grid_size=(15, 15)):
    """
    Run experiment 2 with Environment 2: A -> C
    
    Args:
        alpha: learning rate
        gamma: discount factor
        exp_suffix: suffix for experiment directory name
        grid_size: grid dimensions for Environment 2
    """
    # set random seed for reproducibility
    set_random_seed(42)
    
    print("=" * 60)
    print("Experiment 2 (Environment 2): Changed Configuration")
    print("=" * 60)
    print("Initial region: A")
    print("Target region: C")
    print(f"Grid size: {grid_size}")
    print()
    
    # setup experiment
    initial_region = 'A'
    target_region = 'C'
    exp_name = f"env2_exp2{exp_suffix}"
    save_dir = f"results/{exp_name}"
    
    # create environment (use smaller grid for faster training)
    if grid_size == (20, 20):
        grid_size = (12, 12)  # Reduce to 12x12 for reasonable training time
    env = GridWorld2(grid_size=grid_size, 
                     initial_region=initial_region, 
                     target_region=target_region)
    
    # hyperparameters (adjusted for larger state space)
    n_episodes = 2000  # More episodes for grid environment
    epsilon_start = 0.9
    epsilon_min = 0.1
    log_interval = 200  # Less frequent logging for larger state space
    
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
    print("\nGenerating convergence plot...")
    plot_convergence(
        episode_rewards, episode_steps, q_table_history,
        save_path=f'{save_dir}/convergence_plot.png',
        show_plot=False,
        title_suffix=" - Experiment 2 (A -> C) - Environment 2"
    )
    
    print("Generating Q-table heatmap...")
    plot_q_table_heatmap(
        final_q_table, state_names=None,
        save_path=f'{save_dir}/qtable_heatmap.png',
        show_plot=False,
        title=" - Experiment 2 (A -> C) - Environment 2"
    )
    
    print("Generating grid policy visualization...")
    plot_grid_policy(
        env, final_q_table, initial_region, target_region,
        save_path=f'{save_dir}/final_policy.png',
        show_plot=False,
        title=" - Experiment 2 (A -> C) - Environment 2"
    )
    
    print("Generating Q-table comparison...")
    from visualization import plot_q_table_comparison
    plot_q_table_comparison(
        initial_q_table, final_q_table, state_names=None,
        save_path=f'{save_dir}/qtable_comparison.png',
        show_plot=False,
        title=" - Experiment 2 (A -> C) - Environment 2"
    )
    
    # save data
    print("\nSaving training data...")
    save_training_data(
        episode_rewards, episode_steps, q_table_history, final_q_table,
        save_dir=save_dir, experiment_name=exp_name, initial_q_table=initial_q_table
    )
    
    print("\n" + "=" * 60)
    print("Experiment 2 (Environment 2) completed!")
    if early_stopped:
        print(f"Training stopped early at episode {len(episode_rewards)}")
    print(f"Results saved to: {save_dir}/")
    print("=" * 60)
    
    return exp_name, save_dir


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--grid":
        # run with default parameter grid
        alpha_list = [0.05, 0.1, 0.2]
        gamma_list = [0.8, 0.9, 0.95]
        # Note: parameter grid not implemented for env2 yet
        print("Parameter grid not implemented for Environment 2 yet")
    else:
        # run single experiment with default parameters
        main()

