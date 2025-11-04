"""
Main script to run all experiments for Environment 2 (Grid-based Environment).
Runs both Experiment 1 (F -> C) and Experiment 2 (A -> C).
Supports multiple grid sizes and parameter combinations.
"""

import sys
from q_learning_optimized import set_random_seed
from run_experiment1_env2 import main as run_exp1_env2, run_parameter_grid as run_exp1_grid_env2
from run_experiment2_env2 import main as run_exp2_env2, run_parameter_grid as run_exp2_grid_env2

# set random seed for reproducibility
set_random_seed(42)

# Default parameter grid for experiments
DEFAULT_ALPHA_LIST = [0.05, 0.1, 0.2]
DEFAULT_GAMMA_LIST = [0.8, 0.9, 0.95]

# Three map sizes: small (current), medium, large
MAP_SIZES = [
    (20, 20),  # Small - current map
    (25, 25),  # Medium - larger and more complex
    (30, 30),  # Large - even larger and more complex
]


def run_all_grids_and_parameters(alpha_list=None, gamma_list=None):
    """
    Run all experiments with all grid sizes and parameter combinations.
    
    Args:
        alpha_list: list of learning rates to test (default: DEFAULT_ALPHA_LIST)
        gamma_list: list of discount factors to test (default: DEFAULT_GAMMA_LIST)
    """
    if alpha_list is None:
        alpha_list = DEFAULT_ALPHA_LIST
    if gamma_list is None:
        gamma_list = DEFAULT_GAMMA_LIST
    
    total_experiments = len(MAP_SIZES) * 2 * len(alpha_list) * len(gamma_list)
    print("=" * 70)
    print("RUNNING ALL EXPERIMENTS: ENVIRONMENT 2")
    print("Grid-based Environment with Multiple Map Sizes")
    print("=" * 70)
    print(f"Map sizes: {MAP_SIZES}")
    print(f"Alpha values: {alpha_list}")
    print(f"Gamma values: {gamma_list}")
    print(f"Experiments per map size: 2 (Exp1 + Exp2)")
    print(f"Parameter combinations per experiment: {len(alpha_list) * len(gamma_list)}")
    print(f"Total experiments: {total_experiments}")
    print("=" * 70)
    print()
    
    all_results = {}
    
    for grid_size in MAP_SIZES:
        grid_str = f"grid{grid_size[0]}x{grid_size[1]}"
        print("\n" + "=" * 70)
        print(f"MAP SIZE: {grid_size} ({grid_str})")
        print("=" * 70)
        
        # Experiment 1 (F -> C)
        print("\n" + "-" * 70)
        print(f"EXPERIMENT 1 (F -> C) - {grid_str}")
        print("-" * 70)
        try:
            exp1_results = run_exp1_grid_env2(alpha_list, gamma_list, grid_size=grid_size)
            all_results[f'exp1_{grid_str}'] = {
                'grid_size': grid_size,
                'experiment': 'F -> C',
                'results': exp1_results
            }
            exp1_success = sum(1 for r in exp1_results if r.get('success', False))
            print(f"✓ Experiment 1 ({grid_str}): {exp1_success}/{len(exp1_results)} successful")
        except Exception as e:
            print(f"✗ Experiment 1 ({grid_str}) failed: {e}")
            all_results[f'exp1_{grid_str}'] = {
                'grid_size': grid_size,
                'experiment': 'F -> C',
                'error': str(e)
            }
        
        # Experiment 2 (A -> C)
        print("\n" + "-" * 70)
        print(f"EXPERIMENT 2 (A -> C) - {grid_str}")
        print("-" * 70)
        try:
            exp2_results = run_exp2_grid_env2(alpha_list, gamma_list, grid_size=grid_size)
            all_results[f'exp2_{grid_str}'] = {
                'grid_size': grid_size,
                'experiment': 'A -> C',
                'results': exp2_results
            }
            exp2_success = sum(1 for r in exp2_results if r.get('success', False))
            print(f"✓ Experiment 2 ({grid_str}): {exp2_success}/{len(exp2_results)} successful")
        except Exception as e:
            print(f"✗ Experiment 2 ({grid_str}) failed: {e}")
            all_results[f'exp2_{grid_str}'] = {
                'grid_size': grid_size,
                'experiment': 'A -> C',
                'error': str(e)
            }
    
    # Print final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY: ALL EXPERIMENTS")
    print("=" * 70)
    for key, value in all_results.items():
        if 'results' in value:
            success_count = sum(1 for r in value['results'] if r.get('success', False))
            total_count = len(value['results'])
            print(f"{key}: {success_count}/{total_count} successful")
        else:
            print(f"{key}: Failed - {value.get('error', 'Unknown error')}")
    print("=" * 70)
    
    return all_results


def main():
    """Main entry point for Environment 2 (all experiments)."""
    if len(sys.argv) > 1:
        if sys.argv[1] == "--single":
            # run single experiment with default parameters (alpha=0.1, gamma=0.9, grid=20x20)
            results = {}
            
            print("=" * 70)
            print("RUNNING ALL EXPERIMENTS: ENVIRONMENT 2 (SINGLE PARAMETER)")
            print("Grid-based Environment")
            print("Using default parameters: alpha=0.1, gamma=0.9, grid=20x20")
            print("=" * 70)
            print()
            
            # Experiment 1: F -> C
            print("-" * 70)
            print("EXPERIMENT 1: F -> C")
            print("-" * 70)
            try:
                exp1_name, exp1_dir = run_exp1_env2()
                results['exp1'] = {'success': True, 'name': exp1_name, 'dir': exp1_dir}
                print(f"\n✓ Experiment 1 completed: {exp1_dir}")
            except Exception as e:
                results['exp1'] = {'success': False, 'error': str(e)}
                print(f"\n✗ Experiment 1 failed: {e}")
            
            print()
            
            # Experiment 2: A -> C
            print("-" * 70)
            print("EXPERIMENT 2: A -> C")
            print("-" * 70)
            try:
                exp2_name, exp2_dir = run_exp2_env2()
                results['exp2'] = {'success': True, 'name': exp2_name, 'dir': exp2_dir}
                print(f"\n✓ Experiment 2 completed: {exp2_dir}")
            except Exception as e:
                results['exp2'] = {'success': False, 'error': str(e)}
                print(f"\n✗ Experiment 2 failed: {e}")
            
            print("\n" + "=" * 70)
            print("ALL EXPERIMENTS COMPLETED")
            print("=" * 70)
            
            return results
        else:
            print("Usage: python main_env2.py [--single]")
            print("  --single: Run single experiments with default parameters")
            print("  (no args): Run all experiments with all grid sizes and parameter combinations")
            return
    else:
        # run all experiments with all grid sizes and parameter combinations
        return run_all_grids_and_parameters()


if __name__ == "__main__":
    main()
