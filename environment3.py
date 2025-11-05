"""
Maze-based environment for Q-learning assignment (Environment 3).
Uses maze generation algorithms (Kruskal) to create mazes with walls and passages.
Compatible with existing Q-learning code.
"""

import numpy as np
from maze3.maze import Maze
from maze3.util import Coordinates
from maze3.generator.mazeGenerator import MazeGenerator


class GridWorld3:
    """
    Maze-based world environment using generated mazes.
    Each cell is a state. Actions are target state indices (adjacent cells without walls).
    Compatible with existing Q-learning code.
    """
    
    def __init__(self, grid_size=(20, 20), initial_region='F', target_region='C',
                 maze_generator='kruskal', weight_approach='random', random_seed=None):
        """
        Initialize the maze-based environment.
        
        Args:
            grid_size: tuple of (rows, cols) for maze dimensions (default: 20x20)
            initial_region: starting region name ('A' through 'F')
            target_region: goal region name ('A' through 'F')
            maze_generator: maze generation algorithm ('kruskal')
            weight_approach: weight assignment approach ('random', 'checkered', or None)
            random_seed: random seed for maze generation (optional)
        """
        self.grid_rows, self.grid_cols = grid_size
        self.grid_size = grid_size
        
        # Region names (compatible with GridWorld)
        self.states = ['A', 'B', 'C', 'D', 'E', 'F']
        self.regions = self.states
        self.region_map = {r: i for i, r in enumerate(self.regions)}
        
        # Set random seed if provided
        if random_seed is not None:
            import random
            random.seed(random_seed)
        
        # Create and generate maze
        self.maze = Maze(self.grid_rows, self.grid_cols, weight_approach)
        
        # Generate maze using specified algorithm
        generator = MazeGenerator(maze_generator)
        generator.generateMaze(self.maze)
        
        # Set initial and target regions
        self.initial_region = initial_region
        self.target_region = target_region
        
        # Map regions to entrance/exit positions on maze boundaries
        self._setup_entrances_exits()
        
        # Set initial and target positions
        self.initial_pos = self._get_entrance_position(initial_region)
        self.current_pos = self.initial_pos
        self.target_pos = self._get_exit_position(target_region)
        
        # Reward parameters (compatible with GridWorld)
        self.reward_goal = 100.0
        self.reward_step = -1.0
        self.reward_invalid = -10.0
        
        # Total number of states (all cells)
        self.n_states = self.grid_rows * self.grid_cols
        
        # Pre-compute and cache valid actions for each state (major performance optimization)
        self._valid_actions_cache = {}
        self._build_valid_actions_cache()
    
    def _setup_entrances_exits(self):
        """
        Setup entrance and exit positions for each region on maze boundaries.
        Maps regions A-F to boundary positions.
        """
        rows, cols = self.grid_rows, self.grid_cols
        
        # Define region to boundary position mapping
        # Region A: top right -> entrance at top boundary, right side
        # Region B: top left -> entrance at top boundary, left side
        # Region C: bottom right -> exit at bottom boundary, right side
        # Region D: top center -> entrance at left boundary, top side
        # Region E: middle bottom -> exit at bottom boundary, center
        # Region F: middle left -> entrance at left boundary, middle
        
        self.region_entrances = {
            'A': Coordinates(-1, cols * 3 // 4),      # Top boundary, right side
            'B': Coordinates(-1, cols // 4),         # Top boundary, left side
            'C': Coordinates(rows, cols * 3 // 4),   # Bottom boundary, right side (exit)
            'D': Coordinates(rows // 6, -1),         # Left boundary, top
            'E': Coordinates(rows, cols // 2),      # Bottom boundary, center (exit)
            'F': Coordinates(rows // 2, -1),        # Left boundary, middle
        }
        
        self.region_exits = {
            'A': Coordinates(rows // 6, cols),        # Right boundary, top
            'B': Coordinates(rows // 3, -1),        # Left boundary, middle-top
            'C': Coordinates(rows, cols * 3 // 4),  # Bottom boundary, right side
            'D': Coordinates(-1, cols // 2),        # Top boundary, center
            'E': Coordinates(rows * 2 // 3, cols),   # Right boundary, bottom
            'F': Coordinates(rows // 2, cols),       # Right boundary, middle
        }
        
        # Add entrances and exits to maze
        for region in self.regions:
            if region in self.region_entrances:
                self.maze.addEntrance(self.region_entrances[region])
            if region in self.region_exits:
                self.maze.addExit(self.region_exits[region])
    
    def _get_entrance_position(self, region):
        """Get initial position (cell) from entrance coordinate."""
        if region not in self.region_entrances:
            # Default to center if region not found
            return (self.grid_rows // 2, self.grid_cols // 2)
        
        entrance = self.region_entrances[region]
        # Convert boundary coordinate to actual cell position
        if entrance.getRow() == -1:  # Top boundary
            return (0, entrance.getCol())
        elif entrance.getRow() == self.grid_rows:  # Bottom boundary
            return (self.grid_rows - 1, entrance.getCol())
        elif entrance.getCol() == -1:  # Left boundary
            return (entrance.getRow(), 0)
        elif entrance.getCol() == self.grid_cols:  # Right boundary
            return (entrance.getRow(), self.grid_cols - 1)
        else:
            return (entrance.getRow(), entrance.getCol())
    
    def _get_exit_position(self, region):
        """Get target position (cell) from exit coordinate."""
        if region not in self.region_exits:
            # Default to center if region not found
            return (self.grid_rows // 2, self.grid_cols // 2)
        
        exit_coord = self.region_exits[region]
        # Convert boundary coordinate to actual cell position
        if exit_coord.getRow() == -1:  # Top boundary
            return (0, exit_coord.getCol())
        elif exit_coord.getRow() == self.grid_rows:  # Bottom boundary
            return (self.grid_rows - 1, exit_coord.getCol())
        elif exit_coord.getCol() == -1:  # Left boundary
            return (exit_coord.getRow(), 0)
        elif exit_coord.getCol() == self.grid_cols:  # Right boundary
            return (exit_coord.getRow(), self.grid_cols - 1)
        else:
            return (exit_coord.getRow(), exit_coord.getCol())
    
    def reset(self):
        """Reset environment to initial position."""
        self.current_pos = self.initial_pos
        return self._pos_to_state(self.current_pos)
    
    def _pos_to_state(self, pos):
        """Convert position (row, col) to state index."""
        return pos[0] * self.grid_cols + pos[1]
    
    def _state_to_pos(self, state):
        """Convert state index to position (row, col)."""
        row = state // self.grid_cols
        col = state % self.grid_cols
        return (row, col)
    
    def _pos_to_coord(self, pos):
        """Convert position tuple to Coordinates object."""
        return Coordinates(pos[0], pos[1])
    
    def _build_valid_actions_cache(self):
        """
        Pre-compute valid actions for all states (called once during initialization).
        This avoids repeated wall checks and coordinate conversions during training.
        """
        print("Pre-computing valid actions cache for all states...")
        for state in range(self.n_states):
            pos = self._state_to_pos(state)
            coord = self._pos_to_coord(pos)
            valid_actions = []
            
            # Get neighbors from maze (all adjacent cells)
            neighbors = self.maze.neighbours(coord)
            
            # Filter to only valid cells (within bounds and no wall)
            for neighbor in neighbors:
                n_row = neighbor.getRow()
                n_col = neighbor.getCol()
                
                # Only include cells within the actual maze (not boundary)
                if 0 <= n_row < self.grid_rows and 0 <= n_col < self.grid_cols:
                    # Check if there's no wall between current and neighbor
                    if not self.maze.hasWall(coord, neighbor):
                        target_state = self._pos_to_state((n_row, n_col))
                        valid_actions.append(target_state)
            
            self._valid_actions_cache[state] = valid_actions
        print(f"Valid actions cache built for {len(self._valid_actions_cache)} states.")
    
    def get_valid_actions(self, state):
        """
        Get list of valid actions from a state (optimized: uses pre-computed cache).
        Actions are target state indices (adjacent cells without walls).
        Compatible with Q-learning code.
        
        Args:
            state: state index (cell position)
        
        Returns:
            list of valid action (target state indices)
        """
        # Return cached valid actions (major performance improvement)
        return self._valid_actions_cache.get(state, [])
    
    def step(self, action):
        """
        Execute one step in the environment (optimized: uses cached valid actions).
        Action is target state index (compatible with Q-learning).
        
        Args:
            action: target state index (adjacent cell)
        
        Returns:
            tuple of (next_state, reward, done, info)
        """
        current_state = self._pos_to_state(self.current_pos)
        
        # Fast validation: check if action is in cached valid actions
        if action not in self._valid_actions_cache.get(current_state, []):
            # Invalid action (not in valid actions list)
            reward = self.reward_invalid
            done = False
            next_state = current_state
        else:
            # Valid move (action is in cached valid actions, so it's guaranteed valid)
            target_pos = self._state_to_pos(action)
            self.current_pos = target_pos
            next_state = action
            
            # Check if reached target position
            if target_pos == self.target_pos:
                reward = self.reward_goal
                done = True
            else:
                reward = self.reward_step
                done = False
        
        return next_state, reward, done, {}
    
    def get_state_name(self, state_idx):
        """Get region name for a state (approximate, for compatibility)."""
        # Since maze cells don't have explicit regions, return approximate region
        pos = self._state_to_pos(state_idx)
        row, col = pos
        
        # Approximate region based on position
        if row < self.grid_rows // 3:
            if col > self.grid_cols * 2 // 3:
                return 'A'
            elif col < self.grid_cols // 3:
                return 'B'
            else:
                return 'D'
        elif row > self.grid_rows * 2 // 3:
            if col > self.grid_cols * 2 // 3:
                return 'C'
            else:
                return 'E'
        else:
            if col < self.grid_cols // 3:
                return 'F'
            else:
                return 'E'
    
    def get_current_state_name(self):
        """Get current state name."""
        return self.get_state_name(self._pos_to_state(self.current_pos))
    
    def is_terminal(self, state):
        """Check if state is at target position."""
        pos = self._state_to_pos(state)
        return pos == self.target_pos
    
    def get_maze(self):
        """Get maze object for visualization."""
        return self.maze
    
    def is_valid_action(self, state, action):
        """Check if action is valid from given state."""
        valid_actions = self.get_valid_actions(state)
        return action in valid_actions

