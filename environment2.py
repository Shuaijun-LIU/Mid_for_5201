"""
Grid-based environment for Q-learning assignment (Environment 2).
Map consists of many small cells, with 6 regions (A-F) each containing multiple cells.
Black cells represent walls that cannot be traversed.
"""

import numpy as np


class GridWorld2:
    """
    Grid-based world environment with 6 regions (A-F).
    Each cell is a state. Actions are target state indices (adjacent cells).
    Compatible with existing Q-learning code.
    """
    
    def __init__(self, grid_size=(20, 20), initial_region='F', target_region='C'):
        """
        Initialize the grid-based environment.
        
        Args:
            grid_size: tuple of (rows, cols) for grid dimensions
            initial_region: starting region name ('A' through 'F')
            target_region: goal region name ('A' through 'F')
        """
        self.grid_rows, self.grid_cols = grid_size
        self.grid_size = grid_size
        
        # Region names (compatible with GridWorld)
        self.states = ['A', 'B', 'C', 'D', 'E', 'F']
        self.regions = self.states
        self.region_map = {r: i for i, r in enumerate(self.regions)}
        
        # Initialize grid: 0=empty, 1=wall, 2-7=regions A-F
        self.grid = np.zeros(grid_size, dtype=int)
        
        # Define regions and walls
        self._setup_grid()
        
        # Set initial and target regions
        self.initial_region = initial_region
        self.target_region = target_region
        self.initial_pos = self._get_region_center(self.region_map[initial_region])
        self.current_pos = self.initial_pos
        
        # Reward parameters (compatible with GridWorld)
        self.reward_goal = 100.0
        self.reward_step = -1.0
        self.reward_invalid = -10.0
        
        # Total number of states (all cells)
        self.n_states = self.grid_rows * self.grid_cols
    
    def _setup_grid(self):
        """Setup grid with regions and walls."""
        rows, cols = self.grid_rows, self.grid_cols
        
        # Define region boundaries based on 6-room layout
        # Region A: top right
        self._fill_region(2, (0, cols*3//4), (rows//3, cols), 'A')
        
        # Region B: top left
        self._fill_region(3, (0, 0), (rows//3, cols//2), 'B')
        
        # Region C: bottom right
        self._fill_region(4, (rows*2//3, cols*3//4), (rows, cols), 'C')
        
        # Region D: middle left
        self._fill_region(5, (rows//3, 0), (rows*2//3, cols//3), 'D')
        
        # Region E: middle bottom
        self._fill_region(6, (rows*2//3, cols//3), (rows, cols*2//3), 'E')
        
        # Region F: top center
        self._fill_region(7, (0, cols//2), (rows//3, cols*3//4), 'F')
        
        # Add walls between regions with passages
        self._add_walls()
    
    def _fill_region(self, region_id, start, end, region_name):
        """Fill a rectangular region with region_id."""
        r1, c1 = start
        r2, c2 = end
        # Fill region cells
        self.grid[r1:r2, c1:c2] = region_id
    
    def _add_walls(self):
        """Add walls between regions with passages."""
        rows, cols = self.grid_rows, self.grid_cols
        
        # Vertical walls with gaps for passages
        # Left wall (x = cols//3) with gap at middle
        self.grid[0:rows//3, cols//3] = 1  # top part
        self.grid[rows*2//3:, cols//3] = 1  # bottom part
        
        # Middle wall (x = cols//2) with gap
        self.grid[0:rows//3, cols//2] = 1  # top part
        self.grid[rows*2//3:, cols//2] = 1  # bottom part
        
        # Right wall (x = cols*2//3) with gap
        self.grid[0:rows*2//3, cols*2//3] = 1  # top part
        self.grid[rows*4//5:, cols*2//3] = 1  # bottom part
        
        # Horizontal walls with gaps for passages
        # Top wall (y = rows//3) with gaps
        self.grid[rows//3, 0:cols//3] = 1  # left part
        self.grid[rows//3, cols//2:cols*2//3] = 1  # middle part
        self.grid[rows//3, cols*4//5:] = 1  # right part
        
        # Bottom wall (y = rows*2//3) with gaps
        self.grid[rows*2//3, 0:cols//3] = 1  # left part
        self.grid[rows*2//3, cols//2:cols*2//3] = 1  # middle part
        self.grid[rows*2//3, cols*4//5:] = 1  # right part
    
    def _get_region_center(self, region_idx):
        """Get center position of a region."""
        region_id = region_idx + 2
        region_cells = np.argwhere(self.grid == region_id)
        if len(region_cells) > 0:
            center = region_cells[len(region_cells) // 2]
            return tuple(center)
        return (self.grid_rows // 2, self.grid_cols // 2)
    
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
    
    def _is_valid_position(self, pos):
        """Check if position is valid and traversable."""
        row, col = pos
        if row < 0 or row >= self.grid_rows or col < 0 or col >= self.grid_cols:
            return False
        # Check if cell is wall
        if self.grid[row, col] == 1:
            return False
        return True
    
    def _get_region_at_pos(self, pos):
        """Get region index (0-5) at given position."""
        row, col = pos
        if not self._is_valid_position(pos):
            return None
        cell_value = self.grid[row, col]
        if cell_value >= 2:
            return cell_value - 2  # Convert to region index (0-5)
        return None
    
    def get_valid_actions(self, state):
        """
        Get list of valid actions from a state.
        Actions are target state indices (adjacent cells).
        Compatible with Q-learning code.
        
        Args:
            state: state index (cell position)
        
        Returns:
            list of valid action (target state indices)
        """
        pos = self._state_to_pos(state)
        valid_actions = []
        
        # Check 4 directions: up, down, left, right
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for dr, dc in directions:
            new_pos = (pos[0] + dr, pos[1] + dc)
            if self._is_valid_position(new_pos):
                target_state = self._pos_to_state(new_pos)
                valid_actions.append(target_state)
        
        return valid_actions
    
    def step(self, action):
        """
        Execute one step in the environment.
        Action is target state index (compatible with Q-learning).
        
        Args:
            action: target state index (adjacent cell)
        
        Returns:
            tuple of (next_state, reward, done, info)
        """
        target_pos = self._state_to_pos(action)
        
        # Check if action is valid (adjacent and not wall)
        if not self._is_valid_position(target_pos):
            # Invalid action: stay in place, get penalty
            reward = self.reward_invalid
            done = False
            next_state = self._pos_to_state(self.current_pos)
        else:
            # Check if target is adjacent
            dr = abs(target_pos[0] - self.current_pos[0])
            dc = abs(target_pos[1] - self.current_pos[1])
            if dr + dc != 1:  # Not adjacent
                reward = self.reward_invalid
                done = False
                next_state = self._pos_to_state(self.current_pos)
            else:
                # Valid move
                self.current_pos = target_pos
                next_state = action
                
                # Check if reached target region
                region = self._get_region_at_pos(target_pos)
                if region is not None and region == self.region_map[self.target_region]:
                    reward = self.reward_goal
                    done = True
                else:
                    reward = self.reward_step
                    done = False
        
        return next_state, reward, done, {}
    
    def get_state_name(self, state_idx):
        """Get region name for a state, compatible with GridWorld."""
        pos = self._state_to_pos(state_idx)
        if not self._is_valid_position(pos):
            return None
        region = self._get_region_at_pos(pos)
        if region is not None:
            return self.regions[region]
        return None
    
    def get_current_state_name(self):
        """Get current state name."""
        return self.get_state_name(self._pos_to_state(self.current_pos))
    
    def is_terminal(self, state):
        """Check if state is in target region."""
        pos = self._state_to_pos(state)
        region = self._get_region_at_pos(pos)
        if region is not None:
            return region == self.region_map[self.target_region]
        return False
    
    def get_grid(self):
        """Get grid for visualization."""
        return self.grid.copy()
    
    def is_valid_action(self, state, action):
        """Check if action is valid from given state."""
        valid_actions = self.get_valid_actions(state)
        return action in valid_actions

