"""
Analyze parameter combinations (alpha, gamma) for Q-learning performance.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
from collections import defaultdict
import re


def load_all_results(exp_dir, exp_base_name):
    """load all parameter combination results."""
    results = {}
    # Pattern to match: exp_base_name_grid20x20_alpha0_1_gamma0_9_training_data.pkl
    # or: exp_base_name_alpha0_1_gamma0_9_training_data.pkl (for backward compatibility)
    pattern1 = re.compile(rf'{re.escape(exp_base_name)}_grid(\d+)x(\d+)_alpha([\d_]+)_gamma([\d_]+)_training_data\.pkl')
    pattern2 = re.compile(rf'{re.escape(exp_base_name)}_alpha([\d_]+)_gamma([\d_]+)_training_data\.pkl')
    
    for filename in os.listdir(exp_dir):
        match1 = pattern1.match(filename)
        match2 = pattern2.match(filename) if not match1 else None
        
        if match1:
            grid_rows = int(match1.group(1))
            grid_cols = int(match1.group(2))
            alpha = float(match1.group(3).replace('_', '.'))
            gamma = float(match1.group(4).replace('_', '.'))
            grid_size = (grid_rows, grid_cols)
            filepath = os.path.join(exp_dir, filename)
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            key = f"grid{grid_rows}x{grid_cols}_alpha{alpha}_gamma{gamma}"
            results[key] = {
                'alpha': alpha, 'gamma': gamma, 'grid_size': grid_size,
                'episode_rewards': data['episode_rewards'],
                'episode_steps': data['episode_steps'],
                'final_q_table': data['final_q_table']
            }
        elif match2:
            # Backward compatibility: no grid size in filename
            alpha = float(match2.group(1).replace('_', '.'))
            gamma = float(match2.group(2).replace('_', '.'))
            filepath = os.path.join(exp_dir, filename)
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            key = f"alpha{alpha}_gamma{gamma}"
            results[key] = {
                'alpha': alpha, 'gamma': gamma, 'grid_size': None,
                'episode_rewards': data['episode_rewards'],
                'episode_steps': data['episode_steps'],
                'final_q_table': data['final_q_table']
            }
    return results


def compute_metrics(episode_rewards, episode_steps, window=100):
    """compute performance metrics."""
    n = len(episode_rewards)
    avg_reward_last = np.mean(episode_rewards[-window:])
    avg_steps_last = np.mean(episode_steps[-window:])
    
    # find convergence episode with improved method
    # Use a more strict criterion: reward should be within 2% of final average
    # and maintain stability over multiple windows
    target = avg_reward_last * 0.98
    stability_windows = 3  # Number of consecutive windows that must meet target
    conv_ep = n
    
    for i in range(window, n - stability_windows * window // 2):
        # Check if current window meets target
        current_avg = np.mean(episode_rewards[max(0, i-window):i])
        if current_avg >= target:
            # Check stability: verify that subsequent windows also meet target
            stable = True
            for offset in range(1, stability_windows):
                check_idx = i + offset * window // 2
                if check_idx >= n:
                    stable = False
                    break
                check_avg = np.mean(episode_rewards[max(0, check_idx-window):check_idx])
                if check_avg < target * 0.98:  # Allow slight variation
                    stable = False
                    break
            if stable:
                conv_ep = i
                break
    
    # If no convergence found with strict criterion, use original method as fallback
    if conv_ep == n:
        target_fallback = avg_reward_last * 0.95
        for i in range(window, n):
            if np.mean(episode_rewards[i-window:i]) >= target_fallback:
                conv_ep = i
                break
    
    return {
        'avg_reward_last': avg_reward_last,
        'avg_steps_last': avg_steps_last,
        'convergence_episode': conv_ep
    }


def plot_comparison(results, exp_name, save_dir="results"):
    """plot parameter comparison charts."""
    if not results:
        return None
    
    metrics = {k: compute_metrics(v['episode_rewards'], v['episode_steps']) 
               for k, v in results.items()}
    
    fig = plt.figure(figsize=(16, 12))
    
    # convergence curves
    ax1 = plt.subplot(3, 3, 1)
    for k, data in results.items():
        r = data['episode_rewards']
        w = max(1, len(r) // 50)
        r_smooth = np.convolve(r, np.ones(w)/w, mode='valid') if len(r) > w else r
        ax1.plot(r_smooth, label=f'α={data["alpha"]}, γ={data["gamma"]}', alpha=0.7)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward (MA)')
    ax1.set_title('Convergence: Rewards')
    ax1.legend(fontsize=8, ncol=2)
    ax1.grid(True, alpha=0.3)
    
    ax2 = plt.subplot(3, 3, 2)
    for k, data in results.items():
        s = data['episode_steps']
        w = max(1, len(s) // 50)
        s_smooth = np.convolve(s, np.ones(w)/w, mode='valid') if len(s) > w else s
        ax2.plot(s_smooth, label=f'α={data["alpha"]}, γ={data["gamma"]}', alpha=0.7)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Steps (MA)')
    ax2.set_title('Convergence: Steps')
    ax2.legend(fontsize=8, ncol=2)
    ax2.grid(True, alpha=0.3)
    
    # heatmaps
    alphas = sorted(set(d['alpha'] for d in results.values()))
    gammas = sorted(set(d['gamma'] for d in results.values()))
    
    ax3 = plt.subplot(3, 3, 3)
    reward_mat = np.zeros((len(alphas), len(gammas)))
    for k, data in results.items():
        i, j = alphas.index(data['alpha']), gammas.index(data['gamma'])
        reward_mat[i, j] = metrics[k]['avg_reward_last']
    im = ax3.imshow(reward_mat, aspect='auto', cmap='viridis')
    ax3.set_xticks(range(len(gammas)))
    ax3.set_xticklabels([f'{g:.2f}' for g in gammas])
    ax3.set_yticks(range(len(alphas)))
    ax3.set_yticklabels([f'{a:.2f}' for a in alphas])
    ax3.set_xlabel('Gamma')
    ax3.set_ylabel('Alpha')
    ax3.set_title('Final Reward')
    plt.colorbar(im, ax=ax3)
    
    ax4 = plt.subplot(3, 3, 4)
    steps_mat = np.zeros((len(alphas), len(gammas)))
    for k, data in results.items():
        i, j = alphas.index(data['alpha']), gammas.index(data['gamma'])
        steps_mat[i, j] = metrics[k]['avg_steps_last']
    im = ax4.imshow(steps_mat, aspect='auto', cmap='plasma_r')
    ax4.set_xticks(range(len(gammas)))
    ax4.set_xticklabels([f'{g:.2f}' for g in gammas])
    ax4.set_yticks(range(len(alphas)))
    ax4.set_yticklabels([f'{a:.2f}' for a in alphas])
    ax4.set_xlabel('Gamma')
    ax4.set_ylabel('Alpha')
    ax4.set_title('Final Steps')
    plt.colorbar(im, ax=ax4)
    
    ax5 = plt.subplot(3, 3, 5)
    conv_mat = np.zeros((len(alphas), len(gammas)))
    for k, data in results.items():
        i, j = alphas.index(data['alpha']), gammas.index(data['gamma'])
        conv_mat[i, j] = metrics[k]['convergence_episode']
    
    # Check if convergence episode has meaningful variation
    conv_values = [m['convergence_episode'] for m in metrics.values()]
    conv_std = np.std(conv_values)
    conv_range = np.max(conv_values) - np.min(conv_values)
    
    # Get total episodes from first result to check variation threshold
    sample_episodes = len(next(iter(results.values()))['episode_rewards'])
    
    # If variation is too small (less than 5% of total episodes or std < 10),
    # use a different metric: stability index (coefficient of variation in last 50% of episodes)
    if conv_range < sample_episodes * 0.05 and conv_std < 10:
        # Replace with stability metric: std of last 50% of rewards
        ax5.clear()
        stability_mat = np.zeros((len(alphas), len(gammas)))
        for k, data in results.items():
            i, j = alphas.index(data['alpha']), gammas.index(data['gamma'])
            # Use coefficient of variation in last half of episodes as stability metric
            rewards_last_half = data['episode_rewards'][len(data['episode_rewards'])//2:]
            if np.mean(rewards_last_half) > 0:
                stability = np.std(rewards_last_half) / np.mean(rewards_last_half)
            else:
                stability = 0
            stability_mat[i, j] = stability
        im = ax5.imshow(stability_mat, aspect='auto', cmap='viridis_r')  # Lower is better
        ax5.set_title('Reward Stability (CV, lower=better)')
    else:
        im = ax5.imshow(conv_mat, aspect='auto', cmap='coolwarm')
        ax5.set_title('Convergence Episode')
    
    ax5.set_xticks(range(len(gammas)))
    ax5.set_xticklabels([f'{g:.2f}' for g in gammas])
    ax5.set_yticks(range(len(alphas)))
    ax5.set_yticklabels([f'{a:.2f}' for a in alphas])
    ax5.set_xlabel('Gamma')
    ax5.set_ylabel('Alpha')
    plt.colorbar(im, ax=ax5)
    
    # parameter effects
    ax6 = plt.subplot(3, 3, 6)
    alpha_rewards = defaultdict(list)
    for k, data in results.items():
        alpha_rewards[data['alpha']].append(metrics[k]['avg_reward_last'])
    alphas_sorted = sorted(alpha_rewards.keys())
    ax6.plot(alphas_sorted, [np.mean(alpha_rewards[a]) for a in alphas_sorted], 'o-', linewidth=2)
    ax6.set_xlabel('Alpha')
    ax6.set_ylabel('Avg Reward')
    ax6.set_title('Effect of Alpha')
    ax6.grid(True, alpha=0.3)
    
    ax7 = plt.subplot(3, 3, 7)
    gamma_rewards = defaultdict(list)
    for k, data in results.items():
        gamma_rewards[data['gamma']].append(metrics[k]['avg_reward_last'])
    gammas_sorted = sorted(gamma_rewards.keys())
    ax7.plot(gammas_sorted, [np.mean(gamma_rewards[g]) for g in gammas_sorted], 'o-', linewidth=2)
    ax7.set_xlabel('Gamma')
    ax7.set_ylabel('Avg Reward')
    ax7.set_title('Effect of Gamma')
    ax7.grid(True, alpha=0.3)
    
    # ranking table
    ax8 = plt.subplot(3, 3, 8)
    ax8.axis('off')
    sorted_params = sorted(metrics.items(), key=lambda x: x[1]['avg_reward_last'], reverse=True)
    table_data = [['Rank', 'α', 'γ', 'Reward', 'Steps', 'Conv']]
    for rank, (k, m) in enumerate(sorted_params, 1):
        d = results[k]
        table_data.append([str(rank), f'{d["alpha"]:.2f}', f'{d["gamma"]:.2f}',
                          f"{m['avg_reward_last']:.2f}", f"{m['avg_steps_last']:.2f}",
                          str(m['convergence_episode'])])
    table = ax8.table(cellText=table_data[1:], colLabels=table_data[0],
                     cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
    table.set_fontsize(8)
    table.scale(1, 1.5)
    
    # summary stats
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')
    all_rewards = [m['avg_reward_last'] for m in metrics.values()]
    all_steps = [m['avg_steps_last'] for m in metrics.values()]
    summary = f"""Summary Statistics

Combinations: {len(results)}
Reward: μ={np.mean(all_rewards):.2f}, σ={np.std(all_rewards):.2f}
Range: [{np.min(all_rewards):.2f}, {np.max(all_rewards):.2f}]
Steps: μ={np.mean(all_steps):.2f}, σ={np.std(all_steps):.2f}
Range: [{np.min(all_steps):.2f}, {np.max(all_steps):.2f}]"""
    ax9.text(0.1, 0.5, summary, fontsize=10, verticalalignment='center', family='monospace')
    
    # No title for the entire figure (subplots can have titles if needed)
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, f'{exp_name}_parameter_analysis.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    # Save EPS format with same name
    eps_path = save_path.replace('.png', '.eps')
    plt.savefig(eps_path, format='eps', bbox_inches='tight')
    plt.close()
    
    return metrics


def save_markdown_report(results, metrics, exp_name, exp_title, save_dir="results"):
    """save analysis results as markdown file."""
    sorted_params = sorted(metrics.items(), key=lambda x: x[1]['avg_reward_last'], reverse=True)
    
    md_content = f"# Parameter Analysis: {exp_title}\n\n"
    md_content += f"## Summary\n\n"
    md_content += f"- **Total combinations**: {len(results)}\n"
    
    all_rewards = [m['avg_reward_last'] for m in metrics.values()]
    all_steps = [m['avg_steps_last'] for m in metrics.values()]
    md_content += f"- **Reward**: mean={np.mean(all_rewards):.2f}, std={np.std(all_rewards):.2f}, "
    md_content += f"range=[{np.min(all_rewards):.2f}, {np.max(all_rewards):.2f}]\n"
    md_content += f"- **Steps**: mean={np.mean(all_steps):.2f}, std={np.std(all_steps):.2f}, "
    md_content += f"range=[{np.min(all_steps):.2f}, {np.max(all_steps):.2f}]\n\n"
    
    md_content += "## Ranking\n\n"
    md_content += "| Rank | Alpha | Gamma | Final Reward | Final Steps | Conv Episode |\n"
    md_content += "|------|-------|-------|--------------|-------------|--------------|\n"
    for rank, (k, m) in enumerate(sorted_params, 1):
        d = results[k]
        md_content += f"| {rank} | {d['alpha']:.2f} | {d['gamma']:.2f} | "
        md_content += f"{m['avg_reward_last']:.2f} | {m['avg_steps_last']:.2f} | "
        md_content += f"{m['convergence_episode']} |\n"
    
    md_content += "\n## Key Findings\n\n"
    best = sorted_params[0]
    worst = sorted_params[-1]
    best_data = results[best[0]]
    worst_data = results[worst[0]]
    
    md_content += f"- **Best**: α={best_data['alpha']:.2f}, γ={best_data['gamma']:.2f} "
    md_content += f"(reward={best[1]['avg_reward_last']:.2f}, steps={best[1]['avg_steps_last']:.2f})\n"
    md_content += f"- **Worst**: α={worst_data['alpha']:.2f}, γ={worst_data['gamma']:.2f} "
    md_content += f"(reward={worst[1]['avg_reward_last']:.2f}, steps={worst[1]['avg_steps_last']:.2f})\n"
    
    # alpha effect
    alpha_rewards = defaultdict(list)
    for k, data in results.items():
        alpha_rewards[data['alpha']].append(metrics[k]['avg_reward_last'])
    best_alpha = max(alpha_rewards.items(), key=lambda x: np.mean(x[1]))
    md_content += f"- **Alpha effect**: {best_alpha[0]:.2f} performs best (avg reward={np.mean(best_alpha[1]):.2f})\n"
    
    # gamma effect
    gamma_rewards = defaultdict(list)
    for k, data in results.items():
        gamma_rewards[data['gamma']].append(metrics[k]['avg_reward_last'])
    best_gamma = max(gamma_rewards.items(), key=lambda x: np.mean(x[1]))
    md_content += f"- **Gamma effect**: {best_gamma[0]:.2f} performs best (avg reward={np.mean(best_gamma[1]):.2f})\n"
    
    md_content += f"\n*Visualization saved to: `{exp_name}_parameter_analysis.png`*\n"
    
    save_path = os.path.join(save_dir, f'{exp_name}_parameter_analysis.md')
    with open(save_path, 'w') as f:
        f.write(md_content)
    print(f"Markdown report saved to: {save_path}")


def analyze_multi_grid(results_dict, exp_base_name, exp_title):
    """Analyze results across multiple grid sizes."""
    # Group results by grid size
    grid_results = {}
    for key, data in results_dict.items():
        if 'grid_size' in data and data['grid_size']:
            grid_size = data['grid_size']
            grid_str = f"grid{grid_size[0]}x{grid_size[1]}"
            if grid_str not in grid_results:
                grid_results[grid_str] = {}
            grid_results[grid_str][key] = data
    
    if not grid_results:
        return None
    
    # Analyze each grid size separately
    for grid_str, grid_data in grid_results.items():
        print(f"\nAnalyzing {exp_title} - {grid_str}")
        metrics = plot_comparison(grid_data, f"{exp_base_name}_{grid_str}")
        save_markdown_report(grid_data, metrics, f"{exp_base_name}_{grid_str}", 
                            f"{exp_title} - {grid_str}")
    
    # Cross-grid comparison if multiple grids exist
    if len(grid_results) > 1:
        print(f"\nGenerating cross-grid comparison for {exp_title}...")
        plot_cross_grid_comparison(grid_results, exp_base_name, exp_title)
    
    return grid_results


def plot_cross_grid_comparison(grid_results, exp_base_name, exp_title):
    """Plot comparison across different grid sizes."""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Extract metrics for each grid
    grid_sizes = sorted(grid_results.keys())
    metrics_by_grid = {}
    
    for grid_str in grid_sizes:
        metrics = {k: compute_metrics(v['episode_rewards'], v['episode_steps']) 
                  for k, v in grid_results[grid_str].items()}
        metrics_by_grid[grid_str] = metrics
    
    # Plot 1: Average reward by grid size
    ax1 = axes[0, 0]
    grid_avg_rewards = []
    for grid_str in grid_sizes:
        rewards = [m['avg_reward_last'] for m in metrics_by_grid[grid_str].values()]
        grid_avg_rewards.append(np.mean(rewards))
    ax1.bar(range(len(grid_sizes)), grid_avg_rewards)
    ax1.set_xticks(range(len(grid_sizes)))
    ax1.set_xticklabels(grid_sizes, rotation=45, ha='right')
    ax1.set_ylabel('Average Final Reward')
    ax1.set_title('Average Reward by Grid Size')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Average steps by grid size
    ax2 = axes[0, 1]
    grid_avg_steps = []
    for grid_str in grid_sizes:
        steps = [m['avg_steps_last'] for m in metrics_by_grid[grid_str].values()]
        grid_avg_steps.append(np.mean(steps))
    ax2.bar(range(len(grid_sizes)), grid_avg_steps)
    ax2.set_xticks(range(len(grid_sizes)))
    ax2.set_xticklabels(grid_sizes, rotation=45, ha='right')
    ax2.set_ylabel('Average Final Steps')
    ax2.set_title('Average Steps by Grid Size')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Convergence episode by grid size
    ax3 = axes[1, 0]
    grid_avg_conv = []
    for grid_str in grid_sizes:
        convs = [m['convergence_episode'] for m in metrics_by_grid[grid_str].values()]
        grid_avg_conv.append(np.mean(convs))
    ax3.bar(range(len(grid_sizes)), grid_avg_conv)
    ax3.set_xticks(range(len(grid_sizes)))
    ax3.set_xticklabels(grid_sizes, rotation=45, ha='right')
    ax3.set_ylabel('Average Convergence Episode')
    ax3.set_title('Convergence Speed by Grid Size')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Parameter sensitivity by grid size
    ax4 = axes[1, 1]
    # Group by alpha and show effect across grids
    alpha_values = sorted(set(data['alpha'] for grid_data in grid_results.values() 
                              for data in grid_data.values()))
    for alpha in alpha_values:
        alpha_rewards = []
        for grid_str in grid_sizes:
            rewards = [m['avg_reward_last'] 
                     for k, m in metrics_by_grid[grid_str].items() 
                     if grid_results[grid_str][k]['alpha'] == alpha]
            if rewards:
                alpha_rewards.append(np.mean(rewards))
            else:
                alpha_rewards.append(0)
        ax4.plot(range(len(grid_sizes)), alpha_rewards, 'o-', label=f'α={alpha}')
    ax4.set_xticks(range(len(grid_sizes)))
    ax4.set_xticklabels(grid_sizes, rotation=45, ha='right')
    ax4.set_ylabel('Average Reward')
    ax4.set_title('Alpha Sensitivity Across Grid Sizes')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # No title for the entire figure (subplots can have titles if needed)
    plt.tight_layout()
    
    save_path = os.path.join('results', f'{exp_base_name}_cross_grid_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    # Save EPS format with same name
    eps_path = save_path.replace('.png', '.eps')
    plt.savefig(eps_path, format='eps', bbox_inches='tight')
    plt.close()
    print(f"Cross-grid comparison saved to: {save_path} and {eps_path}")


def main():
    """main function."""
    experiments = [
        ('results/env1_exp1', 'env1_exp1', 'Experiment 1 (F → C)'),
        ('results/env1_exp2', 'env1_exp2', 'Experiment 2 (A → C)'),
        ('results/env2_exp1', 'env2_exp1', 'Experiment 1 (F → C) - Environment 2'),
        ('results/env2_exp2', 'env2_exp2', 'Experiment 2 (A → C) - Environment 2'),
    ]
    
    for exp_dir, exp_base_name, exp_title in experiments:
        if os.path.exists(exp_dir):
            print(f"\nAnalyzing: {exp_title}")
            results = load_all_results(exp_dir, exp_base_name)
            if results:
                # Check if we have multiple grid sizes
                grid_sizes = set(data.get('grid_size') for data in results.values() 
                                if data.get('grid_size') is not None)
                if len(grid_sizes) > 1:
                    # Multi-grid analysis
                    analyze_multi_grid(results, exp_base_name, exp_title)
                else:
                    # Single grid or backward compatibility
                    metrics = plot_comparison(results, exp_base_name)
                    save_markdown_report(results, metrics, exp_base_name, exp_title)
            else:
                print(f"No results found in {exp_dir}")
    
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
