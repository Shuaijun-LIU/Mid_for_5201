"""
Main script to run all experiments for Environment 3 (Maze-based Environment).
Runs both Experiment 1 (F -> C) and Experiment 2 (A -> C).
Supports multiple maze sizes and parameter combinations.
"""

import sys
from q_learning_optimized import set_random_seed
from run_experiment1_env3 import main as run_exp1_env3, run_parameter_grid as run_exp1_grid_env3
from run_experiment2_env3 import main as run_exp2_env3, run_parameter_grid as run_exp2_grid_env3

# set random seed for reproducibility
set_random_seed(42)

# Default parameter grid for experiments
DEFAULT_ALPHA_LIST = [0.05, 0.1, 0.2]
DEFAULT_GAMMA_LIST = [0.8, 0.9, 0.95]

# Maze generation parameters (for different maze structures)
MAZE_GENERATORS = ['kruskal']  # Can add more generators in the future
WEIGHT_APPROACHES = ['random', 'checkered']  # Different weight assignment methods

# Three maze sizes: small, medium, large
MAZE_SIZES = [
    (20, 20),  # Small - base maze
    (25, 25),  # Medium - larger and more complex
    (30, 30),  # Large - even larger and more complex
]


def run_all_mazes_and_parameters(alpha_list=None, gamma_list=None, 
                                 maze_generators=None, weight_approaches=None):
    """
    Run all experiments with all maze sizes and parameter combinations.
    All experiments save training data for analysis, but visualization is only saved
    for the best parameter combination at the largest maze size.
    
    Args:
        alpha_list: list of learning rates to test (default: DEFAULT_ALPHA_LIST)
        gamma_list: list of discount factors to test (default: DEFAULT_GAMMA_LIST)
        maze_generators: list of maze generation algorithms (default: MAZE_GENERATORS)
        weight_approaches: list of weight assignment approaches (default: WEIGHT_APPROACHES)
    """
    if alpha_list is None:
        alpha_list = DEFAULT_ALPHA_LIST
    if gamma_list is None:
        gamma_list = DEFAULT_GAMMA_LIST
    if maze_generators is None:
        maze_generators = MAZE_GENERATORS
    if weight_approaches is None:
        weight_approaches = WEIGHT_APPROACHES
    
    # Calculate total experiments
    total_experiments = (len(MAZE_SIZES) * 2 * len(alpha_list) * len(gamma_list) * 
                        len(maze_generators) * len(weight_approaches))
    
    print("=" * 70)
    print("RUNNING ALL EXPERIMENTS: ENVIRONMENT 3")
    print("Maze-based Environment with Multiple Maze Sizes and Parameters")
    print("=" * 70)
    print(f"Maze sizes: {MAZE_SIZES}")
    print(f"Alpha values: {alpha_list}")
    print(f"Gamma values: {gamma_list}")
    print(f"Maze generators: {maze_generators}")
    print(f"Weight approaches: {weight_approaches}")
    print(f"Experiments per configuration: 2 (Exp1 + Exp2)")
    print(f"Total experiments: {total_experiments}")
    print("Note: All training data saved, visualization only for best params at largest size")
    print("=" * 70)
    print()
    
    all_results = {}
    all_performance = []  # Track all results for finding best parameters
    
    # Run all experiments without visualization (save data only)
    for maze_size in MAZE_SIZES:
        for maze_generator in maze_generators:
            for weight_approach in weight_approaches:
                maze_str = f"grid{maze_size[0]}x{maze_size[1]}"
                config_str = f"{maze_generator}_{weight_approach}"
                
                print("\n" + "=" * 70)
                print(f"CONFIGURATION: {maze_str} - {config_str}")
                print("=" * 70)
                
                # Experiment 1 (F -> C)
                print("\n" + "-" * 70)
                print(f"EXPERIMENT 1 (F -> C) - {maze_str} - {config_str}")
                print("-" * 70)
                try:
                    exp1_results = run_exp1_grid_env3(alpha_list, gamma_list, grid_size=maze_size,
                                                    maze_generator=maze_generator,
                                                    weight_approach=weight_approach,
                                                    save_visualization=False)  # No visualization for all runs
                    
                    # Store results with configuration info
                    for r in exp1_results:
                        if r.get('success', False):
                            all_performance.append({
                                **r,
                                'experiment': 'F -> C',
                                'config_str': config_str
                            })
                    
                    all_results[f'exp1_{maze_str}_{config_str}'] = {
                        'maze_size': maze_size,
                        'maze_generator': maze_generator,
                        'weight_approach': weight_approach,
                        'experiment': 'F -> C',
                        'results': exp1_results
                    }
                    exp1_success = sum(1 for r in exp1_results if r.get('success', False))
                    print(f"✓ Experiment 1 ({maze_str} - {config_str}): {exp1_success}/{len(exp1_results)} successful")
                except Exception as e:
                    print(f"✗ Experiment 1 ({maze_str} - {config_str}) failed: {e}")
                    import traceback
                    traceback.print_exc()
                    all_results[f'exp1_{maze_str}_{config_str}'] = {
                        'maze_size': maze_size,
                        'maze_generator': maze_generator,
                        'weight_approach': weight_approach,
                        'experiment': 'F -> C',
                        'error': str(e)
                    }
                
                # Experiment 2 (A -> C)
                print("\n" + "-" * 70)
                print(f"EXPERIMENT 2 (A -> C) - {maze_str} - {config_str}")
                print("-" * 70)
                try:
                    exp2_results = run_exp2_grid_env3(alpha_list, gamma_list, grid_size=maze_size,
                                                    maze_generator=maze_generator,
                                                    weight_approach=weight_approach,
                                                    save_visualization=False)  # No visualization for all runs
                    
                    # Store results with configuration info
                    for r in exp2_results:
                        if r.get('success', False):
                            all_performance.append({
                                **r,
                                'experiment': 'A -> C',
                                'config_str': config_str
                            })
                    
                    all_results[f'exp2_{maze_str}_{config_str}'] = {
                        'maze_size': maze_size,
                        'maze_generator': maze_generator,
                        'weight_approach': weight_approach,
                        'experiment': 'A -> C',
                        'results': exp2_results
                    }
                    exp2_success = sum(1 for r in exp2_results if r.get('success', False))
                    print(f"✓ Experiment 2 ({maze_str} - {config_str}): {exp2_success}/{len(exp2_results)} successful")
                except Exception as e:
                    print(f"✗ Experiment 2 ({maze_str} - {config_str}) failed: {e}")
                    import traceback
                    traceback.print_exc()
                    all_results[f'exp2_{maze_str}_{config_str}'] = {
                        'maze_size': maze_size,
                        'maze_generator': maze_generator,
                        'weight_approach': weight_approach,
                        'experiment': 'A -> C',
                        'error': str(e)
                    }
    
    # Find best parameters at largest maze size
    largest_size = max(MAZE_SIZES)
    largest_size_performance = [r for r in all_performance 
                               if r.get('grid_size') == largest_size and r.get('success', False)]
    
    if largest_size_performance:
        # Find best parameter combination (highest average reward)
        best_result = max(largest_size_performance, key=lambda x: x.get('avg_reward', 0))
        
        print("\n" + "=" * 70)
        print("BEST PARAMETER COMBINATION (at largest size)")
        print("=" * 70)
        print(f"Best parameters: alpha={best_result['alpha']}, gamma={best_result['gamma']}")
        print(f"Maze generator: {best_result['maze_generator']}, Weight approach: {best_result['weight_approach']}")
        print(f"Experiment: {best_result.get('experiment', 'Unknown')}")
        print(f"Average reward (last 100 episodes): {best_result.get('avg_reward', 0):.2f}")
        print("=" * 70)
        print("\nRe-running with best parameters to save visualization...")
        
        # Re-run with best parameters and save visualization
        # Create file suffix for visualization files
        file_suffix = f"_alpha{best_result['alpha']}_gamma{best_result['gamma']}"
        file_suffix = file_suffix.replace(".", "_")
        
        if best_result.get('experiment') == 'F -> C':
            run_exp1_env3(
                alpha=best_result['alpha'],
                gamma=best_result['gamma'],
                grid_size=largest_size,
                maze_generator=best_result['maze_generator'],
                weight_approach=best_result['weight_approach'],
                file_suffix=file_suffix,
                save_visualization=True  # Save visualization for best parameters
            )
        else:
            run_exp2_env3(
                alpha=best_result['alpha'],
                gamma=best_result['gamma'],
                grid_size=largest_size,
                maze_generator=best_result['maze_generator'],
                weight_approach=best_result['weight_approach'],
                file_suffix=file_suffix,
                save_visualization=True  # Save visualization for best parameters
            )
    
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
    """Main entry point for Environment 3 (all experiments)."""
    if len(sys.argv) > 1:
        if sys.argv[1] == "--single":
            # run single experiment with default parameters (alpha=0.1, gamma=0.9, maze=20x20)
            results = {}
            
            print("=" * 70)
            print("RUNNING ALL EXPERIMENTS: ENVIRONMENT 3 (SINGLE PARAMETER)")
            print("Maze-based Environment")
            print("Using default parameters: alpha=0.1, gamma=0.9, maze=20x20")
            print("=" * 70)
            print()
            
            # Experiment 1: F -> C
            print("-" * 70)
            print("EXPERIMENT 1: F -> C")
            print("-" * 70)
            try:
                exp1_name, exp1_dir = run_exp1_env3()
                results['exp1'] = {'success': True, 'name': exp1_name, 'dir': exp1_dir}
                print(f"\n✓ Experiment 1 completed: {exp1_dir}")
            except Exception as e:
                results['exp1'] = {'success': False, 'error': str(e)}
                print(f"\n✗ Experiment 1 failed: {e}")
                import traceback
                traceback.print_exc()
            
            print()
            
            # Experiment 2: A -> C
            print("-" * 70)
            print("EXPERIMENT 2: A -> C")
            print("-" * 70)
            try:
                exp2_name, exp2_dir = run_exp2_env3()
                results['exp2'] = {'success': True, 'name': exp2_name, 'dir': exp2_dir}
                print(f"\n✓ Experiment 2 completed: {exp2_dir}")
            except Exception as e:
                results['exp2'] = {'success': False, 'error': str(e)}
                print(f"\n✗ Experiment 2 failed: {e}")
                import traceback
                traceback.print_exc()
            
            print("\n" + "=" * 70)
            print("ALL EXPERIMENTS COMPLETED")
            print("=" * 70)
            
            return results
        elif sys.argv[1] == "--help" or sys.argv[1] == "-h":
            print("Usage:")
            print("  python main_env3.py              # Run all experiments with all maze sizes and parameter combinations")
            print("  python main_env3.py --single     # Run single experiments with default parameters (alpha=0.1, gamma=0.9, maze=20x20)")
            print("  python main_env3.py --help       # Show this help message")
        else:
            print(f"Unknown argument: {sys.argv[1]}")
            print("Use --help for usage information")
    else:
        # run all experiments with all maze sizes and parameter combinations
        return run_all_mazes_and_parameters()


if __name__ == "__main__":
    main()

