"""
Grid world environment for Q-learning assignment.
Represents a 6-room layout (A-F) with walls and passages between rooms.
"""


class GridWorld:
    """Simple grid world environment with 6 states (A through F)."""
    
    def __init__(self, initial_state=5, target_state=2):
        """
        Initialize the grid world environment.
        
        Args:
            initial_state: starting state index (0=A, 1=B, 2=C, 3=D, 4=E, 5=F)
            target_state: goal state index
        """
        # state names for display
        self.states = ['A', 'B', 'C', 'D', 'E', 'F']
        self.n_states = len(self.states)
        
        # define connections between rooms
        # based on the description: walls with narrow passages
        # this is a reasonable assumption, might need to adjust later
        self.connections = {
            0: [1, 3],      # A -> B, D
            1: [0, 2, 4],   # B -> A, C, E
            2: [1, 5],      # C -> B, F
            3: [0, 4],      # D -> A, E
            4: [1, 3, 5],   # E -> B, D, F
            5: [2, 4]       # F -> C, E
        }
        
        # set initial and target states
        self.initial_state = initial_state
        self.target_state = target_state
        self.current_state = initial_state
        
        # reward parameters
        self.reward_goal = 100.0      # reward for reaching goal
        self.reward_step = -1.0       # small penalty per step (encourage efficiency)
        self.reward_invalid = -10.0   # penalty for invalid action
    
    def reset(self):
        """Reset environment to initial state."""
        self.current_state = self.initial_state
        return self.current_state
    
    def is_valid_action(self, state, action):
        """
        Check if an action is valid from given state.
        
        Args:
            state: current state index
            action: target state index (the action is to move to this state)
        
        Returns:
            True if action is valid, False otherwise
        """
        if state not in self.connections:
            return False
        return action in self.connections[state]
    
    def get_valid_actions(self, state):
        """
        Get list of valid actions from a state.
        
        Args:
            state: state index
        
        Returns:
            list of valid action (target state indices)
        """
        return self.connections.get(state, [])
    
    def step(self, action):
        """
        Execute one step in the environment.
        
        Args:
            action: target state index to move to
        
        Returns:
            tuple of (next_state, reward, done, info)
        """
        # check if action is valid
        if not self.is_valid_action(self.current_state, action):
            # invalid action: stay in place, get penalty
            reward = self.reward_invalid
            done = False
            next_state = self.current_state
        else:
            # valid action: move to target state
            next_state = action
            self.current_state = next_state
            
            # check if reached goal
            if next_state == self.target_state:
                reward = self.reward_goal
                done = True
            else:
                reward = self.reward_step
                done = False
        
        return next_state, reward, done, {}
    
    def get_state_name(self, state_idx):
        """Get state name string for display."""
        if 0 <= state_idx < len(self.states):
            return self.states[state_idx]
        return None
    
    def get_current_state_name(self):
        """Get current state name."""
        return self.get_state_name(self.current_state)
    
    def is_terminal(self, state):
        """Check if state is terminal (goal state)."""
        return state == self.target_state

