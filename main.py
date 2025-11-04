"""
One-click script to run all experiments.
Executes experiments for both Environment 1 and Environment 2.
"""

import sys
from q_learning import set_random_seed
from run_experiment1 import main as run_exp1_env1, run_parameter_grid as run_exp1_grid_env1
from run_experiment2 import main as run_exp2_env1, run_parameter_grid as run_exp2_grid_env1
from run_experiment1_env2 import main as run_exp1_env2
from run_experiment2_env2 import main as run_exp2_env2

# set random seed for reproducibility (will be set again in each experiment, but good practice)
set_random_seed(42)


def run_all_experiments(alpha=0.1, gamma=0.9):
    """
    Run all experiments for both environments sequentially.
    
    Args:
        alpha: learning rate
        gamma: discount factor
    """
    print("=" * 70)
    print("RUNNING ALL EXPERIMENTS (Environment 1 & Environment 2)")
    print("=" * 70)
    print(f"Parameters: alpha={alpha}, gamma={gamma}")
    print()
    
    results = {}
    
    # Environment 1 experiments
    print("\n" + "=" * 70)
    print("ENVIRONMENT 1: Abstract Room-based Environment")
    print("=" * 70)
    
    # Experiment 1 (Env1)
    print("\n" + "-" * 70)
    print("ENVIRONMENT 1 - EXPERIMENT 1 (F -> C)")
    print("-" * 70)
    try:
        exp1_name, exp1_dir = run_exp1_env1(alpha=alpha, gamma=gamma)
        results['env1_exp1'] = {
            'success': True,
            'name': exp1_name,
            'dir': exp1_dir
        }
        print(f"\n✓ Environment 1 - Experiment 1 completed")
        print(f"  Results: {exp1_dir}")
    except Exception as e:
        print(f"\n✗ Environment 1 - Experiment 1 failed: {e}")
        results['env1_exp1'] = {'success': False, 'error': str(e)}
    
    # Experiment 2 (Env1)
    print("\n" + "-" * 70)
    print("ENVIRONMENT 1 - EXPERIMENT 2 (A -> C)")
    print("-" * 70)
    try:
        exp2_name, exp2_dir = run_exp2_env1(alpha=alpha, gamma=gamma)
        results['env1_exp2'] = {
            'success': True,
            'name': exp2_name,
            'dir': exp2_dir
        }
        print(f"\n✓ Environment 1 - Experiment 2 completed")
        print(f"  Results: {exp2_dir}")
    except Exception as e:
        print(f"\n✗ Environment 1 - Experiment 2 failed: {e}")
        results['env1_exp2'] = {'success': False, 'error': str(e)}
    
    # Environment 2 experiments
    print("\n" + "=" * 70)
    print("ENVIRONMENT 2: Grid-based Environment")
    print("=" * 70)
    
    # Experiment 1 (Env2)
    print("\n" + "-" * 70)
    print("ENVIRONMENT 2 - EXPERIMENT 1 (F -> C)")
    print("-" * 70)
    try:
        exp1_name, exp1_dir = run_exp1_env2(alpha=alpha, gamma=gamma)
        results['env2_exp1'] = {
            'success': True,
            'name': exp1_name,
            'dir': exp1_dir
        }
        print(f"\n✓ Environment 2 - Experiment 1 completed")
        print(f"  Results: {exp1_dir}")
    except Exception as e:
        print(f"\n✗ Environment 2 - Experiment 1 failed: {e}")
        results['env2_exp1'] = {'success': False, 'error': str(e)}
    
    # Experiment 2 (Env2)
    print("\n" + "-" * 70)
    print("ENVIRONMENT 2 - EXPERIMENT 2 (A -> C)")
    print("-" * 70)
    try:
        exp2_name, exp2_dir = run_exp2_env2(alpha=alpha, gamma=gamma)
        results['env2_exp2'] = {
            'success': True,
            'name': exp2_name,
            'dir': exp2_dir
        }
        print(f"\n✓ Environment 2 - Experiment 2 completed")
        print(f"  Results: {exp2_dir}")
    except Exception as e:
        print(f"\n✗ Environment 2 - Experiment 2 failed: {e}")
        results['env2_exp2'] = {'success': False, 'error': str(e)}
    
    # Print final summary
    print("\n" + "=" * 70)
    print("ALL EXPERIMENTS SUMMARY")
    print("=" * 70)
    
    print("\nEnvironment 1:")
    for exp_key in ['env1_exp1', 'env1_exp2']:
        status = '✓ SUCCESS' if results[exp_key]['success'] else '✗ FAILED'
        print(f"  {exp_key}: {status}")
        if results[exp_key]['success']:
            print(f"    Results: {results[exp_key]['dir']}")
        else:
            print(f"    Error: {results[exp_key]['error']}")
    
    print("\nEnvironment 2:")
    for exp_key in ['env2_exp1', 'env2_exp2']:
        status = '✓ SUCCESS' if results[exp_key]['success'] else '✗ FAILED'
        print(f"  {exp_key}: {status}")
        if results[exp_key]['success']:
            print(f"    Results: {results[exp_key]['dir']}")
        else:
            print(f"    Error: {results[exp_key]['error']}")
    
    print("=" * 70)
    
    return results


def run_all_parameter_grids(alpha_list=[0.1], gamma_list=[0.9]):
    """
    Run both experiments with multiple parameter combinations.
    
    Args:
        alpha_list: list of learning rates to test
        gamma_list: list of discount factors to test
    """
    print("=" * 70)
    print("RUNNING ALL EXPERIMENTS WITH PARAMETER GRIDS")
    print("=" * 70)
    print(f"Testing {len(alpha_list)} alpha values × {len(gamma_list)} gamma values")
    print(f"Total combinations per experiment: {len(alpha_list) * len(gamma_list)}")
    print(f"Total experiments: {2 * len(alpha_list) * len(gamma_list)}")
    print()
    
    results = {}
    
    # run experiment 1 grid
    print("\n" + "=" * 70)
    print("STARTING EXPERIMENT 1 PARAMETER GRID (F -> C)")
    print("=" * 70)
    try:
        exp1_results = run_exp1_grid(alpha_list, gamma_list)
        results['exp1'] = exp1_results
        print(f"\n✓ Experiment 1 parameter grid completed")
    except Exception as e:
        print(f"\n✗ Experiment 1 parameter grid failed: {e}")
        results['exp1'] = {'error': str(e)}
    
    # run experiment 2 grid
    print("\n" + "=" * 70)
    print("STARTING EXPERIMENT 2 PARAMETER GRID (A -> C)")
    print("=" * 70)
    try:
        exp2_results = run_exp2_grid(alpha_list, gamma_list)
        results['exp2'] = exp2_results
        print(f"\n✓ Experiment 2 parameter grid completed")
    except Exception as e:
        print(f"\n✗ Experiment 2 parameter grid failed: {e}")
        results['exp2'] = {'error': str(e)}
    
    # print final summary
    print("\n" + "=" * 70)
    print("ALL PARAMETER GRID EXPERIMENTS SUMMARY")
    print("=" * 70)
    
    if 'exp1' in results and isinstance(results['exp1'], list):
        exp1_success = sum(1 for r in results['exp1'] if r.get('success', False))
        print(f"Experiment 1: {exp1_success}/{len(results['exp1'])} successful")
    
    if 'exp2' in results and isinstance(results['exp2'], list):
        exp2_success = sum(1 for r in results['exp2'] if r.get('success', False))
        print(f"Experiment 2: {exp2_success}/{len(results['exp2'])} successful")
    
    print("=" * 70)
    
    return results


def main():
    """Main entry point."""
    if len(sys.argv) > 1:
        if sys.argv[1] == "--grid":
            # run with default parameter grid
            alpha_list = [0.05, 0.1, 0.2]
            gamma_list = [0.8, 0.9, 0.95]
            run_all_parameter_grids(alpha_list, gamma_list)
        elif sys.argv[1] == "--help" or sys.argv[1] == "-h":
            print("Usage:")
            print("  python main.py              # Run both experiments with default parameters")
            print("  python main.py --grid       # Run parameter grids for both experiments")
            print("  python main.py --help       # Show this help message")
        else:
            print(f"Unknown argument: {sys.argv[1]}")
            print("Use --help for usage information")
    else:
        # run both experiments with default parameters
        run_all_experiments()


if __name__ == "__main__":
    main()

