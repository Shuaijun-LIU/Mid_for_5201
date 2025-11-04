"""
Visualization functions for Q-learning convergence analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from q_learning import compute_q_table_change


def compute_moving_average(data, window_size=50):
    """
    Compute moving average of data.
    
    Args:
        data: list or array of values
        window_size: size of moving average window
    
    Returns:
        array of moving average values
    """
    data = np.array(data)
    if len(data) < window_size:
        # if data is shorter than window, just return simple average
        return np.full(len(data), np.mean(data))
    
    moving_avg = []
    for i in range(len(data)):
        start = max(0, i - window_size + 1)
        end = i + 1
        moving_avg.append(np.mean(data[start:end]))
    
    return np.array(moving_avg)


def compute_convergence_metrics(q_table_history):
    """
    Compute convergence metrics from Q-table history.
    
    Args:
        q_table_history: list of (episode, q_table) tuples
    
    Returns:
        tuple of (episodes, mean_diffs, max_diffs)
    """
    episodes = []
    mean_diffs = []
    max_diffs = []
    
    for i in range(1, len(q_table_history)):
        ep1, q1 = q_table_history[i-1]
        ep2, q2 = q_table_history[i]
        
        mean_diff, max_diff = compute_q_table_change(q1, q2)
        
        episodes.append(ep2)
        mean_diffs.append(mean_diff)
        max_diffs.append(max_diff)
    
    return np.array(episodes), np.array(mean_diffs), np.array(max_diffs)


def plot_convergence(episode_rewards, episode_steps, q_table_history,
                     save_path=None, show_plot=True, title_suffix=""):
    """
    Plot convergence metrics for Q-learning.
    
    Creates subplots showing:
    1. Q-table change over episodes (mean and max)
    2. Episode rewards (raw and moving average)
    3. Episode steps (raw and moving average)
    
    Args:
        episode_rewards: list of rewards per episode
        episode_steps: list of steps per episode
        q_table_history: list of (episode, q_table) tuples
        save_path: path to save figure (None to not save)
        show_plot: whether to display plot
        title_suffix: additional text for title
    """
    # compute convergence metrics from Q-table history
    conv_episodes, mean_diffs, max_diffs = compute_convergence_metrics(q_table_history)
    
    # compute moving averages
    window_size = min(50, len(episode_rewards) // 10)  # adaptive window size
    if window_size < 5:
        window_size = 5
    
    reward_ma = compute_moving_average(episode_rewards, window_size)
    steps_ma = compute_moving_average(episode_steps, window_size)
    
    # create figure with subplots
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))
    
    # plot 1: Q-table convergence
    ax1 = axes[0]
    ax1.plot(conv_episodes, mean_diffs, 'b-', label='Mean Q-table change', linewidth=1.5)
    ax1.plot(conv_episodes, max_diffs, 'r-', label='Max Q-table change', linewidth=1.5)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Q-table Change')
    ax1.set_title(f'Q-table Convergence{title_suffix}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')  # log scale for better visualization
    
    # plot 2: Episode rewards
    ax2 = axes[1]
    episodes_reward = np.arange(len(episode_rewards))
    ax2.plot(episodes_reward, episode_rewards, 'lightblue', alpha=0.3, label='Raw rewards')
    ax2.plot(episodes_reward, reward_ma, 'b-', label=f'Moving average (window={window_size})', linewidth=2)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Total Reward')
    ax2.set_title(f'Episode Rewards{title_suffix}')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # plot 3: Episode steps
    ax3 = axes[2]
    episodes_steps = np.arange(len(episode_steps))
    ax3.plot(episodes_steps, episode_steps, 'lightcoral', alpha=0.3, label='Raw steps')
    ax3.plot(episodes_steps, steps_ma, 'r-', label=f'Moving average (window={window_size})', linewidth=2)
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Steps to Goal')
    ax3.set_title(f'Episode Steps{title_suffix}')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Convergence plot saved to {save_path}")
    
    # show plot
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_q_table_heatmap(q_table, state_names=None, save_path=None, show_plot=True, title=""):
    """
    Plot Q-table as heatmap.
    
    Args:
        q_table: Q-table array (n_states x n_states)
        state_names: list of state names for labels
        save_path: path to save figure
        show_plot: whether to display plot
        title: plot title
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # create heatmap
    im = ax.imshow(q_table, cmap='viridis', aspect='auto')
    
    # set labels
    if state_names is None:
        state_names = [f'S{i}' for i in range(q_table.shape[0])]
    
    ax.set_xticks(np.arange(len(state_names)))
    ax.set_yticks(np.arange(len(state_names)))
    ax.set_xticklabels(state_names)
    ax.set_yticklabels(state_names)
    
    # rotate labels for better readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Q-value', rotation=270, labelpad=20)
    
    # add text annotations
    threshold = q_table.max() / 2.0
    for i in range(len(state_names)):
        for j in range(len(state_names)):
            text = ax.text(j, i, f'{q_table[i, j]:.1f}',
                          ha="center", va="center", color="white" if q_table[i, j] > threshold else "black",
                          fontsize=8)
    
    ax.set_title(f'Q-table Heatmap{title}')
    ax.set_xlabel('Action (Target State)')
    ax.set_ylabel('State')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Q-table heatmap saved to {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_q_table_comparison(initial_q_table, final_q_table, state_names=None, 
                           save_path=None, show_plot=True, title=""):
    """
    Plot comparison between initial and final Q-tables.
    
    Args:
        initial_q_table: initial Q-table (before training, all zeros)
        final_q_table: final Q-table (after convergence)
        state_names: list of state names for labels
        save_path: path to save figure
        show_plot: whether to display plot
        title: plot title
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    if state_names is None:
        state_names = [f'S{i}' for i in range(initial_q_table.shape[0])]
    
    # Plot 1: Initial Q-table
    im1 = axes[0].imshow(initial_q_table, cmap='viridis', aspect='auto')
    axes[0].set_xticks(np.arange(len(state_names)))
    axes[0].set_yticks(np.arange(len(state_names)))
    axes[0].set_xticklabels(state_names)
    axes[0].set_yticklabels(state_names)
    plt.setp(axes[0].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    cbar1 = plt.colorbar(im1, ax=axes[0])
    cbar1.set_label('Q-value', rotation=270, labelpad=20)
    axes[0].set_title(f'Initial Q-table (Before Training){title}')
    axes[0].set_xlabel('Action (Target State)')
    axes[0].set_ylabel('State')
    
    # Add text annotations for initial Q-table
    for i in range(len(state_names)):
        for j in range(len(state_names)):
            text = axes[0].text(j, i, f'{initial_q_table[i, j]:.1f}',
                          ha="center", va="center", color="white" if initial_q_table[i, j] > initial_q_table.max() / 2.0 else "black",
                          fontsize=8)
    
    # Plot 2: Final Q-table
    im2 = axes[1].imshow(final_q_table, cmap='viridis', aspect='auto')
    axes[1].set_xticks(np.arange(len(state_names)))
    axes[1].set_yticks(np.arange(len(state_names)))
    axes[1].set_xticklabels(state_names)
    axes[1].set_yticklabels(state_names)
    plt.setp(axes[1].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    cbar2 = plt.colorbar(im2, ax=axes[1])
    cbar2.set_label('Q-value', rotation=270, labelpad=20)
    axes[1].set_title(f'Final Q-table (After Convergence){title}')
    axes[1].set_xlabel('Action (Target State)')
    axes[1].set_ylabel('State')
    
    # Add text annotations for final Q-table
    threshold = final_q_table.max() / 2.0
    for i in range(len(state_names)):
        for j in range(len(state_names)):
            text = axes[1].text(j, i, f'{final_q_table[i, j]:.1f}',
                          ha="center", va="center", color="white" if final_q_table[i, j] > threshold else "black",
                          fontsize=8)
    
    # Plot 3: Difference (Final - Initial)
    q_diff = final_q_table - initial_q_table
    im3 = axes[2].imshow(q_diff, cmap='RdBu_r', aspect='auto', 
                        vmin=-abs(q_diff).max(), vmax=abs(q_diff).max())
    axes[2].set_xticks(np.arange(len(state_names)))
    axes[2].set_yticks(np.arange(len(state_names)))
    axes[2].set_xticklabels(state_names)
    axes[2].set_yticklabels(state_names)
    plt.setp(axes[2].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    cbar3 = plt.colorbar(im3, ax=axes[2])
    cbar3.set_label('Q-value Change', rotation=270, labelpad=20)
    axes[2].set_title(f'Q-table Change (Final - Initial){title}')
    axes[2].set_xlabel('Action (Target State)')
    axes[2].set_ylabel('State')
    
    # Add text annotations for difference
    diff_threshold = abs(q_diff).max() / 2.0
    for i in range(len(state_names)):
        for j in range(len(state_names)):
            text = axes[2].text(j, i, f'{q_diff[i, j]:.1f}',
                          ha="center", va="center", 
                          color="white" if abs(q_diff[i, j]) > diff_threshold else "black",
                          fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Q-table comparison saved to {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_policy(env, q_table, initial_state, target_state, save_path=None, show_plot=True, title=""):
    """
    Plot maze layout with rooms, walls, passages, and optimal path.
    
    Args:
        env: environment object
        q_table: Q-table array (n_states x n_states)
        initial_state: initial state index
        target_state: target state index
        save_path: path to save figure
        show_plot: whether to display plot
        title: plot title
    """
    from matplotlib.patches import FancyArrowPatch, Circle, Rectangle, Patch

    fig, ax = plt.subplots(figsize=(10, 6))

    # Room layout: F at top center, C at bottom right
    # Connections: A->B,D; B->A,C,E; C->B,F; D->A,E; E->B,D,F; F->C,E
    room_bounds = {
        'A': [(7.5, 4.5), (10, 6)],   # top right
        'B': [(0, 2.5), (3, 4.5)],    # top left
        'C': [(7.5, 0), (10, 2.5)],   # bottom right
        'D': [(3, 2.5), (5, 4.5)],    # middle left
        'E': [(5, 0), (7.5, 2.5)],    # middle bottom
        'F': [(3, 4.5), (7.5, 6)],    # top center
    }
    
    # Draw room backgrounds
    room_colors = {
        'A': '#E8F4F8', 'B': '#F0F8E8', 'C': '#FFF8E8',
        'D': '#F8E8F0', 'E': '#E8F8F0', 'F': '#F8F0E8'
    }
    for name, ((x1, y1), (x2, y2)) in room_bounds.items():
        rect = Rectangle((x1, y1), x2-x1, y2-y1, 
                         facecolor=room_colors[name], 
                         edgecolor='gray', linewidth=1, alpha=0.5, zorder=0)
        ax.add_patch(rect)
    
    # Draw outer walls
    outer_walls = [
        [(0, 0), (10, 0)],   # bottom
        [(10, 0), (10, 6)],  # right
        [(10, 6), (0, 6)],   # top
        [(0, 6), (0, 0)]      # left
    ]
    for (x1, y1), (x2, y2) in outer_walls:
        ax.plot([x1, x2], [y1, y2], color='black', lw=5, zorder=10)

    # Draw internal walls with gaps for passages
    # Vertical walls: gaps at y=2.5-4.5 for passages
    internal_walls = [
        # Vertical walls (with gaps at y=2.5-4.5)
        [(3, 0), (3, 2.5)],      # left wall below passage
        [(3, 4.5), (3, 6)],      # left wall above passage
        [(5, 0), (5, 2.5)],      # middle wall below passage
        [(5, 4.5), (5, 6)],      # middle wall above passage
        [(7.5, 0), (7.5, 2.5)],  # right wall below passage
        [(7.5, 4.5), (7.5, 6)],  # right wall above passage
        # Horizontal walls at y=2.5 (gaps: B-E at x=3-5, D-E at x=5)
        [(0, 2.5), (3, 2.5)],    # left part
        [(5, 2.5), (7.5, 2.5)],  # right part (gap 3-5 already exists from vertical walls)
        # Horizontal walls at y=4.5 (gaps: A-B at x=3, E-F at x=5, C-F at x=7.5, A-D at x~6.25)
        [(0, 4.5), (3, 4.5)],    # left part (gap at x=3 for A-B)
        [(5, 4.5), (7.5, 4.5)],  # middle part (gaps exist at x=5 and x=7.5)
        [(7.5, 4.5), (10, 4.5)], # right part
    ]
    for (x1, y1), (x2, y2) in internal_walls:
        ax.plot([x1, x2], [y1, y2], color='black', lw=4, zorder=10)
    
    # Define passage locations at wall gaps
    # Format: (state1, state2): (x, y, direction) where direction is 'h' or 'v'
    passages_info = {
        (0, 1): (3, 4.5, 'h'),      # A-B: horizontal passage at x=3, y=4.5
        (1, 0): (3, 4.5, 'h'),
        (0, 3): (6.25, 4.5, 'h'),  # A-D: horizontal passage
        (3, 0): (6.25, 4.5, 'h'),
        (1, 2): (7.5, 2.5, 'v'),   # B-C: vertical passage at x=7.5, y=2.5
        (2, 1): (7.5, 2.5, 'v'),
        (1, 4): (4, 2.5, 'h'),      # B-E: horizontal passage at x=4, y=2.5
        (4, 1): (4, 2.5, 'h'),
        (2, 5): (7.5, 4.5, 'v'),   # C-F: vertical passage at x=7.5, y=4.5
        (5, 2): (7.5, 4.5, 'v'),
        (3, 4): (5, 2.5, 'h'),     # D-E: horizontal passage at x=5, y=2.5
        (4, 3): (5, 2.5, 'h'),
        (4, 5): (5, 4.5, 'v'),     # E-F: vertical passage at x=5, y=4.5
        (5, 4): (5, 4.5, 'v'),
    }
    
    # Draw passage areas covering full wall gaps
    # Each passage spans the entire opening between adjacent rooms
    drawn_passages = set()  # Avoid drawing duplicates
    
    # Vertical passages: gaps in vertical walls (x=3, x=5, x=7.5)
    # These span the full height from y=2.5 to y=4.5
    vertical_passage_x = [3, 5, 7.5]
    for vx in vertical_passage_x:
        rect = Rectangle((vx - 0.5, 2.5), 
                       1.0, 2.0,  # Full height of gap (2.5 to 4.5)
                       facecolor='#90EE90', edgecolor='green', 
                       linewidth=2, alpha=0.5, zorder=8)
        ax.add_patch(rect)
    
    # Horizontal passages: gaps in horizontal walls (y=2.5, y=4.5)
    # These span appropriate widths based on room connections
    
    # Passage at y=2.5: B-E connection (x=3 to x=5)
    rect = Rectangle((3, 2.5 - 0.5), 
                   2.0, 1.0,  # Full width from x=3 to x=5
                   facecolor='#90EE90', edgecolor='green', 
                   linewidth=2, alpha=0.5, zorder=8)
    ax.add_patch(rect)
    
    # Passage at y=4.5: multiple connections
    # A-B at x=3, E-F at x=5, C-F at x=7.5, A-D around x=6.25
    # Draw continuous passage covering these areas
    rect = Rectangle((3, 4.5 - 0.5), 
                   4.5, 1.0,  # From x=3 to x=7.5
                   facecolor='#90EE90', edgecolor='green', 
                   linewidth=2, alpha=0.5, zorder=8)
    ax.add_patch(rect)

    # Room labels at room centers
    labels = {}
    for name, ((x1, y1), (x2, y2)) in room_bounds.items():
        labels[name] = ((x1 + x2) / 2, (y1 + y2) / 2)

    # Highlight initial and target states
    state_to_name = {i: env.states[i] for i in range(env.n_states)}
    initial_name = state_to_name[initial_state]
    target_name = state_to_name[target_state]

    for name, (x, y) in labels.items():
        if name == initial_name:
            circle = Circle((x, y), 0.4, color='#FFD700', alpha=0.6, zorder=1)
            ax.add_patch(circle)
        elif name == target_name:
            circle = Circle((x, y), 0.4, color='#FF6347', alpha=0.6, zorder=1)
            ax.add_patch(circle)
        
        ax.text(x, y, name, ha='center', va='center', fontsize=20, weight='bold', zorder=2)

    # Compute optimal policy from Q-table
    centers = {
        0: labels['A'], 1: labels['B'], 2: labels['C'],
        3: labels['D'], 4: labels['E'], 5: labels['F']
    }

    arrow_style = dict(arrowstyle='->', color='#32CD32', lw=3, zorder=5)
    optimal_policy = {}
    for s in range(env.n_states):
        actions = env.get_valid_actions(s)
        if not actions:
            continue
        q_values = [q_table[s, a] for a in actions]
        best_a = actions[np.argmax(q_values)]
        optimal_policy[s] = best_a

    # Find optimal path
    path = [initial_state]
    current = initial_state
    visited = set()
    while current != target_state and current not in visited:
        visited.add(current)
        if current in optimal_policy:
            current = optimal_policy[current]
            path.append(current)
        else:
            break

    # Draw path with curved arrows
    for i in range(len(path) - 1):
        s1, s2 = path[i], path[i + 1]
        x1, y1 = centers[s1]
        x2, y2 = centers[s2]
        
        dx = x2 - x1
        dy = y2 - y1
        
        # Use curved path for longer distances
        if abs(dx) > 3 or abs(dy) > 3:
            connectionstyle = "arc3,rad=0.2" if abs(dx) > 1 and abs(dy) > 1 else None
            ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), 
                                       connectionstyle=connectionstyle,
                                       **arrow_style))
        else:
            ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), **arrow_style))

    # Add legend
    legend_elements = [
        Patch(facecolor='black', edgecolor='black', label='Walls'),
        Patch(facecolor='#90EE90', edgecolor='green', label='Passages', alpha=0.7),
        Patch(facecolor='#32CD32', edgecolor='#32CD32', label='Optimal Path')
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10, 
             framealpha=0.9, bbox_to_anchor=(0.02, 0.98))
    
    # Final adjustments
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-0.5, 6.5)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Add title
    if path[-1] == target_state:
        path_str = " → ".join([env.states[s] for s in path])
        ax.set_title(f"Maze Layout Visualization {title}\nOptimal Path: {path_str}", 
                     fontsize=14, weight='bold', pad=15)
    else:
        ax.set_title(f"Maze Layout Visualization {title}", fontsize=14, weight='bold', pad=15)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved maze visualization → {save_path}")
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_grid_policy(env, q_table, initial_region, target_region, 
                     save_path=None, show_plot=True, title=""):
    """
    Plot grid-based environment with cells, walls, regions, and optimal path.
    For Environment 2 (GridWorld2).
    
    Args:
        env: GridWorld2 environment object
        q_table: Q-table array (n_states x n_states)
        initial_region: initial region name
        target_region: target region name
        save_path: path to save figure
        show_plot: whether to display plot
        title: plot title
    """
    from matplotlib.patches import Rectangle, Circle, Patch
    
    fig, ax = plt.subplots(figsize=(12, 12))
    
    grid = env.get_grid()
    rows, cols = grid.shape
    
    # Define colors for regions
    region_colors = {
        0: '#E8F4F8',  # A - light blue
        1: '#F0F8E8',  # B - light green
        2: '#FFF8E8',  # C - light yellow
        3: '#F8E8F0',  # D - light pink
        4: '#E8F8F0',  # E - light cyan
        5: '#F8F0E8',  # F - light orange
    }
    
    # Draw grid cells
    for r in range(rows):
        for c in range(cols):
            cell_value = grid[r, c]
            x, y = c, rows - 1 - r  # Flip y-axis for display
            
            if cell_value == 1:
                # Wall: black
                rect = Rectangle((x, y), 1, 1, 
                               facecolor='black', edgecolor='gray', 
                               linewidth=0.5, zorder=1)
            elif cell_value >= 2:
                # Region: colored
                region_idx = cell_value - 2
                color = region_colors.get(region_idx, 'white')
                rect = Rectangle((x, y), 1, 1, 
                               facecolor=color, edgecolor='gray', 
                               linewidth=0.5, alpha=0.7, zorder=1)
            else:
                # Empty: white
                rect = Rectangle((x, y), 1, 1, 
                               facecolor='white', edgecolor='gray', 
                               linewidth=0.5, zorder=1)
            ax.add_patch(rect)
    
    # Find optimal path
    initial_pos = env._get_region_center(env.region_map[initial_region])
    target_pos = env._get_region_center(env.region_map[target_region])
    initial_state = env._pos_to_state(initial_pos)
    target_state = env._pos_to_state(target_pos)
    
    # Compute optimal policy from Q-table
    path = [initial_state]
    current = initial_state
    visited = set()
    max_steps = rows * cols  # Prevent infinite loops
    
    while current != target_state and len(path) < max_steps and current not in visited:
        visited.add(current)
        actions = env.get_valid_actions(current)
        if not actions:
            break
        
        # Find best action
        q_values = [q_table[current][a] for a in actions]
        best_action = actions[np.argmax(q_values)]
        
        # Check if best action leads to target region
        next_pos = env._state_to_pos(best_action)
        region = env._get_region_at_pos(next_pos)
        if region == env.region_map[target_region]:
            path.append(best_action)
            break
        
        current = best_action
        path.append(current)
    
    # Draw path
    path_positions = [env._state_to_pos(s) for s in path]
    for i in range(len(path_positions) - 1):
        r1, c1 = path_positions[i]
        r2, c2 = path_positions[i + 1]
        y1, x1 = rows - 1 - r1, c1
        y2, x2 = rows - 1 - r2, c2
        
        # Draw arrow
        ax.annotate('', xy=(x2+0.5, y2+0.5), xytext=(x1+0.5, y1+0.5),
                   arrowprops=dict(arrowstyle='->', color='#32CD32', lw=2, zorder=10))
    
    # Highlight initial and target positions
    for r, c in [initial_pos, target_pos]:
        y, x = rows - 1 - r, c
        if (r, c) == initial_pos:
            circle = Circle((x+0.5, y+0.5), 0.3, color='#FFD700', 
                          alpha=0.8, zorder=5)
            ax.text(x+0.5, y+0.5, initial_region, ha='center', va='center',
                   fontsize=12, weight='bold', zorder=6)
        else:
            circle = Circle((x+0.5, y+0.5), 0.3, color='#FF6347', 
                          alpha=0.8, zorder=5)
            ax.text(x+0.5, y+0.5, target_region, ha='center', va='center',
                   fontsize=12, weight='bold', zorder=6)
        ax.add_patch(circle)
    
    # Add region labels
    region_labels = ['A', 'B', 'C', 'D', 'E', 'F']
    for region_idx in range(6):
        region_id = region_idx + 2
        region_cells = np.argwhere(grid == region_id)
        if len(region_cells) > 0:
            center = region_cells[len(region_cells) // 2]
            r, c = center
            y, x = rows - 1 - r, c
            ax.text(x+0.5, y+0.5, region_labels[region_idx], 
                   ha='center', va='center', fontsize=10, weight='bold', zorder=4)
    
    # Add legend
    legend_elements = [
        Patch(facecolor='black', edgecolor='black', label='Walls'),
        Patch(facecolor='#32CD32', edgecolor='#32CD32', label='Optimal Path'),
        Patch(facecolor='#FFD700', edgecolor='#FFD700', label='Initial'),
        Patch(facecolor='#FF6347', edgecolor='#FF6347', label='Target')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10, 
             framealpha=0.9)
    
    # Final adjustments
    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Add title
    if len(path) > 1:
        ax.set_title(f"Grid-based Environment Visualization {title}\n"
                    f"Path: {initial_region} -> {target_region} ({len(path)-1} steps)", 
                    fontsize=14, weight='bold', pad=15)
    else:
        ax.set_title(f"Grid-based Environment Visualization {title}", 
                    fontsize=14, weight='bold', pad=15)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved grid visualization → {save_path}")
    if show_plot:
        plt.show()
    else:
        plt.close()

