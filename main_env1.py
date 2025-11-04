"""
Main script to run all experiments for Environment 1 (Abstract Room-based Environment).
Runs both Experiment 1 (F -> C) and Experiment 2 (A -> C).
By default, runs with multiple parameter combinations (alpha and gamma).
"""

import sys
from q_learning import set_random_seed
from run_experiment1 import main as run_exp1_env1, run_parameter_grid as run_exp1_grid_env1
from run_experiment2 import main as run_exp2_env1, run_parameter_grid as run_exp2_grid_env1

# set random seed for reproducibility
set_random_seed(42)

# Default parameter grid for experiments
DEFAULT_ALPHA_LIST = [0.05, 0.1, 0.2]
DEFAULT_GAMMA_LIST = [0.8, 0.9, 0.95]


def main():
    """Main entry point for Environment 1 (all experiments)."""
    if len(sys.argv) > 1:
        if sys.argv[1] == "--single":
            # run single experiment with default parameters (alpha=0.1, gamma=0.9)
            results = {}
            
            print("=" * 70)
            print("RUNNING ALL EXPERIMENTS: ENVIRONMENT 1 (SINGLE PARAMETER)")
            print("Abstract Room-based Environment")
            print("Using default parameters: alpha=0.1, gamma=0.9")
            print("=" * 70)
            print()
            
            # Experiment 1: F -> C
            print("-" * 70)
            print("EXPERIMENT 1: F -> C")
            print("-" * 70)
            try:
                exp1_name, exp1_dir = run_exp1_env1()
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
                exp2_name, exp2_dir = run_exp2_env1()
                results['exp2'] = {'success': True, 'name': exp2_name, 'dir': exp2_dir}
                print(f"\n✓ Experiment 2 completed: {exp2_dir}")
            except Exception as e:
                results['exp2'] = {'success': False, 'error': str(e)}
                print(f"\n✗ Experiment 2 failed: {e}")
            
            # Summary
            print("\n" + "=" * 70)
            print("ENVIRONMENT 1 SUMMARY")
            print("=" * 70)
            for exp_key in ['exp1', 'exp2']:
                if exp_key in results:
                    status = '✓ SUCCESS' if results[exp_key]['success'] else '✗ FAILED'
                    print(f"{exp_key}: {status}")
                    if results[exp_key]['success']:
                        print(f"  Results: {results[exp_key]['dir']}")
                    else:
                        print(f"  Error: {results[exp_key]['error']}")
            print("=" * 70)
        elif sys.argv[1] == "--grid":
            # run with default parameter grid (same as default behavior now)
            print("=" * 70)
            print("RUNNING PARAMETER GRID FOR ENVIRONMENT 1")
            print("=" * 70)
            print()
            run_exp1_grid_env1(DEFAULT_ALPHA_LIST, DEFAULT_GAMMA_LIST)
            print()
            run_exp2_grid_env1(DEFAULT_ALPHA_LIST, DEFAULT_GAMMA_LIST)
        elif sys.argv[1] == "--help" or sys.argv[1] == "-h":
            print("Usage:")
            print("  python main_env1.py              # Run all experiments with multiple parameter combinations (default)")
            print("  python main_env1.py --grid       # Same as default (parameter grid)")
            print("  python main_env1.py --single     # Run single experiment with default parameters (alpha=0.1, gamma=0.9)")
            print("  python main_env1.py --help       # Show this help message")
        else:
            print(f"Unknown argument: {sys.argv[1]}")
            print("Use --help for usage information")
    else:
        # Default: run all experiments with multiple parameter combinations
        print("=" * 70)
        print("RUNNING PARAMETER GRID FOR ENVIRONMENT 1")
        print("Abstract Room-based Environment")
        print(f"Testing {len(DEFAULT_ALPHA_LIST)} alpha values × {len(DEFAULT_GAMMA_LIST)} gamma values")
        print(f"Alpha: {DEFAULT_ALPHA_LIST}")
        print(f"Gamma: {DEFAULT_GAMMA_LIST}")
        print("=" * 70)
        print()
        
        # Run parameter grid for Experiment 1
        print("-" * 70)
        print("EXPERIMENT 1: F -> C (Parameter Grid)")
        print("-" * 70)
        exp1_results = run_exp1_grid_env1(DEFAULT_ALPHA_LIST, DEFAULT_GAMMA_LIST)
        
        print()
        
        # Run parameter grid for Experiment 2
        print("-" * 70)
        print("EXPERIMENT 2: A -> C (Parameter Grid)")
        print("-" * 70)
        exp2_results = run_exp2_grid_env1(DEFAULT_ALPHA_LIST, DEFAULT_GAMMA_LIST)
        
        # Summary
        print("\n" + "=" * 70)
        print("ENVIRONMENT 1 SUMMARY")
        print("=" * 70)
        print(f"Experiment 1: {sum(1 for r in exp1_results if r['success'])}/{len(exp1_results)} successful")
        print(f"Experiment 2: {sum(1 for r in exp2_results if r['success'])}/{len(exp2_results)} successful")
        print()
        print("All results saved in results/ directory with parameter suffixes")
        print("=" * 70)


if __name__ == "__main__":
    main()

