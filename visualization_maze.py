"""
Visualization functions for Environment 3 (Maze-based environment).
"""

try:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Patch
    HAS_MATPLOTLIB = True
except:
    plt = None
    HAS_MATPLOTLIB = False

from maze3.maze import Maze
from maze3.util import Coordinates


def plot_maze_policy(env, q_table, initial_region, target_region,
                     save_path=None, show_plot=True, title=""):
    """
    Plot maze with walls, entrances/exits, and optimal policy path from Q-table.
    
    Args:
        env: GridWorld3 environment object
        q_table: Q-table array (n_states x n_states)
        initial_region: initial region name
        target_region: target region name
        save_path: path to save figure
        show_plot: whether to display plot
        title: plot title
    """
    if not HAS_MATPLOTLIB:
        print("Warning: matplotlib not available, skipping visualization")
        return
    
    maze = env.get_maze()
    cell_size = 1
    
    # Create figure
    fig = plt.figure(figsize=(7, 7 * maze.rowNum() / maze.colNum()))
    ax = plt.axes()
    ax.set_aspect("equal")
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    
    # Plot walls
    plot_maze_walls(ax, maze, cell_size)
    
    # Plot entrances and exits
    plot_maze_entrances_exits(ax, maze, cell_size)
    
    # Find and plot optimal path from Q-table
    path = extract_path_from_q_table(env, q_table, initial_region, target_region)
    if path:
        plot_q_learning_path(ax, path, cell_size)
    
    # Highlight initial and target positions
    plot_initial_target_positions(ax, env, cell_size, initial_region, target_region)
    
    if title:
        plt.suptitle(title, fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        # Save EPS format with same name
        eps_path = save_path.replace('.png', '.eps')
        plt.savefig(eps_path, format='eps', bbox_inches='tight')
        print(f"Maze policy visualization saved to {save_path} and {eps_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_maze_walls(ax, maze, cell_size):
    """Plot walls of the maze."""
    for r in range(0, maze.rowNum()):
        for c in range(0, maze.colNum()):
            # Top wall
            if maze.hasWall(Coordinates(r-1, c), Coordinates(r, c)):
                ax.plot([(c+1)*cell_size, (c+1+1)*cell_size],
                       [(r+1)*cell_size, (r+1)*cell_size], color="k", linewidth=2)
            # Left wall
            if maze.hasWall(Coordinates(r, c-1), Coordinates(r, c)):
                ax.plot([(c+1)*cell_size, (c+1)*cell_size],
                       [(r+1)*cell_size, (r+1+1)*cell_size], color="k", linewidth=2)
    
    # Bottom boundary
    for c in range(0, maze.colNum()):
        if maze.hasWall(Coordinates(maze.rowNum()-1, c), Coordinates(maze.rowNum(), c)):
            ax.plot([(c+1)*cell_size, (c+1+1)*cell_size],
                   [(maze.rowNum()+1)*cell_size, (maze.rowNum()+1)*cell_size], color="k", linewidth=2)
    
    # Right boundary
    for r in range(0, maze.rowNum()):
        if maze.hasWall(Coordinates(r, maze.colNum()-1), Coordinates(r, maze.colNum())):
            ax.plot([(maze.colNum()+1)*cell_size, (maze.colNum()+1)*cell_size],
                   [(r+1)*cell_size, (r+1+1)*cell_size], color="k", linewidth=2)


def plot_maze_entrances_exits(ax, maze, cell_size):
    """Plot entrances and exits with arrows."""
    # Plot entrances
    for ent in maze.getEntrances():
        # Upwards arrow
        if ent.getRow() == -1:
            ax.arrow((ent.getCol()+1.5)*cell_size, (ent.getRow()+1)*cell_size, 
                    0, cell_size*0.6, head_width=0.15*cell_size, head_length=0.15*cell_size,
                    fc='green', ec='green', linewidth=2)
        # Downwards arrow
        elif ent.getRow() == maze.rowNum():
            ax.arrow((ent.getCol()+1.5)*cell_size, (ent.getRow()+2)*cell_size, 
                    0, -cell_size*0.6, head_width=0.15*cell_size, head_length=0.15*cell_size,
                    fc='green', ec='green', linewidth=2)
        # Rightward arrow
        elif ent.getCol() == -1:
            ax.arrow((ent.getCol()+1)*cell_size, (ent.getRow()+1.5)*cell_size, 
                    cell_size*0.6, 0, head_width=0.15*cell_size, head_length=0.15*cell_size,
                    fc='green', ec='green', linewidth=2)
        # Leftward arrow
        elif ent.getCol() == maze.colNum():
            ax.arrow((ent.getCol()+2)*cell_size, (ent.getRow()+1.5)*cell_size, 
                    -cell_size*0.6, 0, head_width=0.15*cell_size, head_length=0.15*cell_size,
                    fc='green', ec='green', linewidth=2)
    
    # Plot exits
    for ext in maze.getExits():
        # Downwards arrow
        if ext.getRow() == -1:
            ax.arrow((ext.getCol()+1.5)*cell_size, (ext.getRow()+1.8)*cell_size, 
                    0, -cell_size*0.6, head_width=0.15*cell_size, head_length=0.15*cell_size,
                    fc='red', ec='red', linewidth=2)
        # Upwards arrow
        elif ext.getRow() == maze.rowNum():
            ax.arrow((ext.getCol()+1.5)*cell_size, (ext.getRow()+1.2)*cell_size, 
                    0, cell_size*0.6, head_width=0.15*cell_size, head_length=0.15*cell_size,
                    fc='red', ec='red', linewidth=2)
        # Leftward arrow
        elif ext.getCol() == -1:
            ax.arrow((ext.getCol())*cell_size, (ext.getRow()+1.5)*cell_size, 
                    -cell_size*0.6, 0, head_width=0.15*cell_size, head_length=0.15*cell_size,
                    fc='red', ec='red', linewidth=2)
        # Rightward arrow
        elif ext.getCol() == maze.colNum():
            ax.arrow((ext.getCol()+1.2)*cell_size, (ext.getRow()+1.5)*cell_size, 
                    cell_size*0.6, 0, head_width=0.15*cell_size, head_length=0.15*cell_size,
                    fc='red', ec='red', linewidth=2)


def extract_path_from_q_table(env, q_table, initial_region, target_region):
    """
    Extract optimal path from Q-table using greedy policy.
    
    Args:
        env: GridWorld3 environment object
        q_table: Q-table array
        initial_region: initial region name
        target_region: target region name
    
    Returns:
        list of Coordinates representing the path
    """
    initial_state = env._pos_to_state(env.initial_pos)
    target_state = env._pos_to_state(env.target_pos)
    
    path = []
    current_state = initial_state
    visited = set()
    max_steps = env.n_states  # Prevent infinite loops
    
    while current_state != target_state and len(path) < max_steps and current_state not in visited:
        visited.add(current_state)
        path.append(Coordinates(env._state_to_pos(current_state)[0], 
                               env._state_to_pos(current_state)[1]))
        
        # Get valid actions
        valid_actions = env.get_valid_actions(current_state)
        if not valid_actions:
            break
        
        # Find best action according to Q-table
        q_values = [q_table[current_state][a] for a in valid_actions]
        best_action = valid_actions[q_values.index(max(q_values))]
        
        # Check if we reached target
        if best_action == target_state:
            path.append(Coordinates(env._state_to_pos(best_action)[0],
                                  env._state_to_pos(best_action)[1]))
            break
        
        current_state = best_action
    
    return path


def plot_q_learning_path(ax, path, cell_size):
    """
    Plot Q-learning path as a series of circles with gradient coloring.
    
    Args:
        ax: matplotlib axes
        path: list of Coordinates representing the path
        cell_size: size of each cell
    """
    if len(path) == 0:
        return
    
    base_color = (0.2, 0.6, 1.0)  # Blue color
    
    for circle_num in range(len(path)):
        # Calculate gradient factor based on progress along the path
        if len(path) > 1:
            gradient_factor = circle_num / (len(path) - 1)
        else:
            gradient_factor = 0
        
        # Create gradient color (darker to brighter)
        color = tuple([gradient_factor * base + (1 - gradient_factor) * 0.3 
                      for base in base_color])
        
        # Draw circle
        circle = Circle(((path[circle_num].getCol() + 1.5) * cell_size,
                        (path[circle_num].getRow() + 1.5) * cell_size), 
                       0.2 * cell_size, fc=color, ec='darkblue', 
                       linewidth=1, alpha=0.7, zorder=5)
        ax.add_patch(circle)
    
    # Draw connecting lines between path points
    if len(path) > 1:
        for i in range(len(path) - 1):
            x1 = (path[i].getCol() + 1.5) * cell_size
            y1 = (path[i].getRow() + 1.5) * cell_size
            x2 = (path[i+1].getCol() + 1.5) * cell_size
            y2 = (path[i+1].getRow() + 1.5) * cell_size
            ax.plot([x1, x2], [y1, y2], 'b-', linewidth=2, alpha=0.5, zorder=4)


def plot_initial_target_positions(ax, env, cell_size, initial_region, target_region):
    """
    Highlight initial and target positions.
    
    Args:
        ax: matplotlib axes
        env: GridWorld3 environment object
        cell_size: size of each cell
        initial_region: initial region name
        target_region: target region name
    """
    # Highlight initial position
    init_pos = env.initial_pos
    init_circle = Circle(((init_pos[1] + 1.5) * cell_size,
                         (init_pos[0] + 1.5) * cell_size),
                        0.4 * cell_size, fc='gold', ec='orange', 
                        linewidth=2, alpha=0.8, zorder=6)
    ax.add_patch(init_circle)
    ax.text((init_pos[1] + 1.5) * cell_size, (init_pos[0] + 1.5) * cell_size,
           initial_region, ha='center', va='center', fontsize=12, 
           weight='bold', zorder=7)
    
    # Highlight target position
    target_pos = env.target_pos
    target_circle = Circle(((target_pos[1] + 1.5) * cell_size,
                           (target_pos[0] + 1.5) * cell_size),
                          0.4 * cell_size, fc='#FF6347', ec='darkred', 
                          linewidth=2, alpha=0.8, zorder=6)
    ax.add_patch(target_circle)
    ax.text((target_pos[1] + 1.5) * cell_size, (target_pos[0] + 1.5) * cell_size,
           target_region, ha='center', va='center', fontsize=12, 
           weight='bold', zorder=7)


def plot_maze_only(env, save_path=None, show_plot=True, title=""):
    """
    Plot just the maze structure (walls, entrances, exits) without path.
    
    Args:
        env: GridWorld3 environment object
        save_path: path to save figure
        show_plot: whether to display plot
        title: plot title
    """
    if not HAS_MATPLOTLIB:
        print("Warning: matplotlib not available, skipping visualization")
        return
    
    maze = env.get_maze()
    cell_size = 1
    
    # Create figure
    fig = plt.figure(figsize=(7, 7 * maze.rowNum() / maze.colNum()))
    ax = plt.axes()
    ax.set_aspect("equal")
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    
    # Plot walls
    plot_maze_walls(ax, maze, cell_size)
    
    # Plot entrances and exits
    plot_maze_entrances_exits(ax, maze, cell_size)
    
    if title:
        plt.suptitle(title, fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        eps_path = save_path.replace('.png', '.eps')
        plt.savefig(eps_path, format='eps', bbox_inches='tight')
        print(f"Maze visualization saved to {save_path} and {eps_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()

