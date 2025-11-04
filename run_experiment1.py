"""
Experiment 1: Original configuration
Initial state: F (index 5)
Target state: C (index 2)
"""

from environment import GridWorld
from q_learning import train_q_learning, set_random_seed
from visualization import plot_convergence, plot_q_table_heatmap, plot_policy, plot_q_table_comparison
from data_utils import save_training_data
from experiment_utils import print_q_table_summary, print_experiment_stats


def main(alpha=0.1, gamma=0.9, exp_suffix=""):
    """
    Run experiment 1: F -> C
    
    Args:
        alpha: learning rate
        gamma: discount factor
        exp_suffix: suffix for experiment directory name
    """
    # set random seed for reproducibility
    set_random_seed(42)
    
    print("=" * 60)
    print("Experiment 1: Original Configuration")
    print("=" * 60)
    print("Initial state: F (index 5)")
    print("Target state: C (index 2)")
    print()
    
    # setup experiment
    initial_state = 5  # F
    target_state = 2   # C
    exp_name = f"env1_exp1{exp_suffix}"
    save_dir = f"results/{exp_name}"
    
    # create environment
    env = GridWorld(initial_state=initial_state, target_state=target_state)
    
    # hyperparameters
    n_episodes = 1000
    epsilon_start = 0.9
    epsilon_min = 0.1
    log_interval = 100
    
    print(f"Hyperparameters:")
    print(f"  Episodes: {n_episodes}")
    print(f"  Learning rate (alpha): {alpha}")
    print(f"  Discount factor (gamma): {gamma}")
    print(f"  Epsilon start: {epsilon_start}, min: {epsilon_min}")
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
    
    # print Q-table summary
    print_q_table_summary(env, final_q_table)
    
    # visualize
    print("\nGenerating convergence plot...")
    plot_convergence(
        episode_rewards, episode_steps, q_table_history,
        save_path=f'{save_dir}/convergence_plot.png',
        show_plot=False,
        title_suffix=" - Experiment 1 (F -> C)"
    )
    
    print("Generating Q-table heatmap...")
    plot_q_table_heatmap(
        final_q_table, state_names=env.states,
        save_path=f'{save_dir}/qtable_heatmap.png',
        show_plot=False,
        title=" - Experiment 1 (F -> C)"
    )
    
    print("Generating policy visualization...")
    plot_policy(
        env, final_q_table, initial_state, target_state,
        save_path=f'{save_dir}/final_policy.png',
        show_plot=False,
        title=" - Experiment 1 (F -> C)"
    )
    
    print("Generating Q-table comparison...")
    plot_q_table_comparison(
        initial_q_table, final_q_table, state_names=env.states,
        save_path=f'{save_dir}/qtable_comparison.png',
        show_plot=False,
        title=" - Experiment 1 (F -> C)"
    )
    
    # save data
    print("\nSaving training data...")
    save_training_data(
        episode_rewards, episode_steps, q_table_history, final_q_table,
        save_dir=save_dir, experiment_name=exp_name, initial_q_table=initial_q_table
    )
    
    print("\n" + "=" * 60)
    print("Experiment 1 completed!")
    if early_stopped:
        print(f"Training stopped early at episode {len(episode_rewards)}")
    print(f"Results saved to: {save_dir}/")
    print("=" * 60)
    
    return exp_name, save_dir


def run_parameter_grid(alpha_list=[0.1], gamma_list=[0.9]):
    """
    Run experiments with multiple parameter combinations.
    
    Args:
        alpha_list: list of learning rates to test
        gamma_list: list of discount factors to test
    """
    # set random seed for reproducibility
    set_random_seed(42)
    
    print("=" * 70)
    print("Multi-Parameter Experiment: Experiment 1 (F -> C)")
    print("=" * 70)
    print(f"Testing {len(alpha_list)} alpha values × {len(gamma_list)} gamma values")
    print(f"Total combinations: {len(alpha_list) * len(gamma_list)}")
    print()
    
    results = []
    
    for alpha in alpha_list:
        for gamma in gamma_list:
            # create suffix for directory name
            exp_suffix = f"_alpha{alpha}_gamma{gamma}"
            exp_suffix = exp_suffix.replace(".", "_")  # replace dots for directory names
            
            print("\n" + "=" * 70)
            print(f"Running: alpha={alpha}, gamma={gamma}")
            print("=" * 70)
            
            try:
                exp_name, save_dir = main(alpha=alpha, gamma=gamma, exp_suffix=exp_suffix)
                results.append({
                    'alpha': alpha,
                    'gamma': gamma,
                    'exp_name': exp_name,
                    'save_dir': save_dir,
                    'success': True
                })
            except Exception as e:
                print(f"Error running alpha={alpha}, gamma={gamma}: {e}")
                results.append({
                    'alpha': alpha,
                    'gamma': gamma,
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
    print("Results saved in:")
    for r in results:
        if r['success']:
            print(f"  alpha={r['alpha']}, gamma={r['gamma']}: {r['save_dir']}")
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    import sys
    
    # check if running multi-parameter experiment
    if len(sys.argv) > 1 and sys.argv[1] == "--grid":
        # run with default parameter grid
        alpha_list = [0.05, 0.1, 0.2]
        gamma_list = [0.8, 0.9, 0.95]
        run_parameter_grid(alpha_list, gamma_list)
    else:
        # run single experiment with default parameters
        main()

