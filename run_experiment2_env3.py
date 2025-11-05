"""
Experiment 2 (Environment 3): Changed configuration
Maze-based environment with generated mazes.
Initial region: A
Target region: C
"""

from environment3 import GridWorld3
from q_learning_optimized import train_q_learning, set_random_seed
from visualization import plot_convergence, plot_q_table_heatmap
from visualization_maze import plot_maze_policy
from data_utils import save_training_data
from experiment_utils import print_experiment_stats
import numpy as np
import os


def print_q_table_summary_env3(env, q_table):
    """Print Q-table summary for maze environment."""
    print("\n" + "=" * 60)
    print("Final Q-table Summary (sample states):")
    print("=" * 60)
    
    # Sample some states from different areas of the maze
    sample_states = [
        0,  # top-left
        env.grid_cols // 2,  # top-middle
        env.grid_cols - 1,  # top-right
        (env.grid_rows // 2) * env.grid_cols,  # middle-left
        (env.grid_rows // 2) * env.grid_cols + env.grid_cols // 2,  # center
        (env.grid_rows - 1) * env.grid_cols,  # bottom-left
        env.n_states - 1  # bottom-right
    ]
    
    for state in sample_states:
        if state < env.n_states:
            valid_actions = env.get_valid_actions(state)
            
            if valid_actions:
                q_values = [(action, q_table[state][action]) for action in valid_actions]
                q_values.sort(key=lambda x: x[1], reverse=True)
                
                best_action, best_q = q_values[0]
                pos = env._state_to_pos(state)
                best_pos = env._state_to_pos(best_action)
                state_name = env.get_state_name(state)
                print(f"State {state} (pos {pos}, region {state_name}): "
                      f"Best action -> {best_action} (pos {best_pos}), Q={best_q:.2f}")
    
    print("\n" + "=" * 60)


def main(alpha=0.1, gamma=0.9, exp_suffix="", grid_size=(20, 20), file_suffix="",
         maze_generator='kruskal', weight_approach='random', save_visualization=True):
    """
    Run experiment 2 with Environment 3: A -> C
    
    Args:
        alpha: learning rate
        gamma: discount factor
        exp_suffix: suffix for experiment directory name
        grid_size: maze dimensions for Environment 3 (default: 20x20)
        file_suffix: suffix for individual file names (for parameter grid)
        maze_generator: maze generation algorithm ('kruskal')
        weight_approach: weight assignment approach ('random', 'checkered')
        save_visualization: whether to save visualization plots (default: True)
    """
    # set random seed for reproducibility
    set_random_seed(42)
    
    print("=" * 60)
    print("Experiment 2 (Environment 3): Changed Configuration")
    print("=" * 60)
    print("Initial region: A")
    print("Target region: C")
    print(f"Maze size: {grid_size}")
    print(f"Maze generator: {maze_generator}")
    print(f"Weight approach: {weight_approach}")
    print()
    
    # setup experiment
    initial_region = 'A'
    target_region = 'C'
    # Include grid size in experiment name
    grid_str = f"grid{grid_size[0]}x{grid_size[1]}"
    exp_name = f"env3_exp2{exp_suffix}"
    save_dir = f"results/{exp_name}"
    
    # create output directory
    os.makedirs(save_dir, exist_ok=True)
    
    # create environment
    env = GridWorld3(grid_size=grid_size, 
                     initial_region=initial_region, 
                     target_region=target_region,
                     maze_generator=maze_generator,
                     weight_approach=weight_approach,
                     random_seed=42)
    
    # hyperparameters (adjusted for larger state space)
    # Scale episodes based on grid size
    base_episodes = 1500
    grid_factor = (grid_size[0] * grid_size[1]) / 400  # 400 is base 20x20
    n_episodes = int(base_episodes * grid_factor)
    epsilon_start = 0.9
    epsilon_min = 0.1
    log_interval = 100
    
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
    print_q_table_summary_env3(env, final_q_table)
    
    # visualize (only if save_visualization is True)
    if save_visualization:
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
            title_suffix=f" - Experiment 2 (A -> C) - Environment 3 - {grid_str}"
        )
        
        print("Generating Q-table statistics (unified heatmap and comparison)...")
        # For large Q-tables, both heatmap and comparison are saved to the same unified file
        plot_q_table_heatmap(
            final_q_table, state_names=None,
            save_path=f'{save_dir}/qtable_statistics{viz_suffix}.png',
            show_plot=False,
            title=f" - Experiment 2 (A -> C) - Environment 3 - {grid_str}",
            initial_q_table=initial_q_table  # Pass initial Q-table for comparison
        )
        
        print("Generating maze policy visualization...")
        plot_maze_policy(
            env, final_q_table, initial_region, target_region,
            save_path=f'{save_dir}/final_policy{viz_suffix}.png',
            show_plot=False,
            title=f" - Experiment 2 (A -> C) - Environment 3 - {grid_str}"
        )
    else:
        print("\nSkipping visualization (save_visualization=False, data saved for analysis)")
    
    # save data
    print("\nSaving training data...")
    # Include grid size and parameters in training data filename
    training_name = f"{exp_name}_{grid_str}{file_suffix}" if file_suffix else f"{exp_name}_{grid_str}"
    save_training_data(
        episode_rewards, episode_steps, q_table_history, final_q_table,
        save_dir=save_dir, experiment_name=training_name, initial_q_table=initial_q_table
    )
    
    print("\n" + "=" * 60)
    print("Experiment 2 (Environment 3) completed!")
    if early_stopped:
        print(f"Training stopped early at episode {len(episode_rewards)}")
    print(f"Results saved to: {save_dir}/")
    print("=" * 60)
    
    return exp_name, save_dir


def run_parameter_grid(alpha_list=[0.05, 0.1, 0.2], gamma_list=[0.8, 0.9, 0.95], 
                       grid_size=(20, 20), maze_generator='kruskal', weight_approach='random',
                       save_visualization=False):
    """
    Run experiments with multiple parameter combinations.
    
    Args:
        alpha_list: list of learning rates to test
        gamma_list: list of discount factors to test
        grid_size: maze dimensions for Environment 3
        maze_generator: maze generation algorithm
        weight_approach: weight assignment approach
        save_visualization: whether to save visualization (default: False, only save data)
    """
    # set random seed for reproducibility
    set_random_seed(42)
    
    grid_str = f"grid{grid_size[0]}x{grid_size[1]}"
    print("=" * 70)
    print(f"Multi-Parameter Experiment: Experiment 2 (A -> C) - {grid_str}")
    print("=" * 70)
    print(f"Testing {len(alpha_list)} alpha values × {len(gamma_list)} gamma values")
    print(f"Total combinations: {len(alpha_list) * len(gamma_list)}")
    print(f"Maze generator: {maze_generator}, Weight approach: {weight_approach}")
    print(f"Save visualization: {save_visualization}")
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
                                         file_suffix=file_suffix,
                                         maze_generator=maze_generator,
                                         weight_approach=weight_approach,
                                         save_visualization=save_visualization)
                
                # Load training data to get performance metrics
                import pickle
                import os
                training_file = os.path.join(save_dir, f"{exp_name}_{grid_str}{file_suffix}_training_data.pkl")
                avg_reward = 0
                if os.path.exists(training_file):
                    with open(training_file, 'rb') as f:
                        data = pickle.load(f)
                        if 'episode_rewards' in data and len(data['episode_rewards']) > 0:
                            avg_reward = np.mean(data['episode_rewards'][-100:])  # Last 100 episodes
                
                results.append({
                    'alpha': alpha,
                    'gamma': gamma,
                    'grid_size': grid_size,
                    'maze_generator': maze_generator,
                    'weight_approach': weight_approach,
                    'exp_name': exp_name,
                    'save_dir': save_dir,
                    'avg_reward': avg_reward,
                    'success': True
                })
            except Exception as e:
                print(f"Error running alpha={alpha}, gamma={gamma}, grid={grid_size}: {e}")
                import traceback
                traceback.print_exc()
                results.append({
                    'alpha': alpha,
                    'gamma': gamma,
                    'grid_size': grid_size,
                    'maze_generator': maze_generator,
                    'weight_approach': weight_approach,
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
    if any(r['success'] for r in results):
        best = max([r for r in results if r['success']], key=lambda x: x.get('avg_reward', 0))
        print(f"Best performance: alpha={best['alpha']}, gamma={best['gamma']}, reward={best.get('avg_reward', 0):.2f}")
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

