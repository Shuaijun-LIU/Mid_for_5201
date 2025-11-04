"""
Optimized Q-learning algorithm implementation for performance.
Uses sparse Q-table and reduced overhead for faster training.
"""

import numpy as np
import random
from collections import defaultdict

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    # Fallback progress bar if tqdm is not available
    def tqdm(iterable, **kwargs):
        return iterable


def set_random_seed(seed=42):
    """
    Set random seed for reproducibility.
    
    Args:
        seed: random seed value
    """
    np.random.seed(seed)
    random.seed(seed)


class QLearningAgent:
    """Optimized Q-learning agent with sparse Q-table and epsilon-greedy action selection."""
    
    def __init__(self, n_states, alpha=0.1, gamma=0.9, 
                 epsilon_start=0.9, epsilon_min=0.1, epsilon_decay='linear'):
        """
        Initialize Q-learning agent with sparse Q-table.
        
        Args:
            n_states: number of states in the environment (for compatibility)
            alpha: learning rate
            gamma: discount factor
            epsilon_start: initial exploration rate
            epsilon_min: minimum exploration rate
            epsilon_decay: decay method ('linear' or 'exponential')
        """
        self.n_states = n_states  # Kept for compatibility
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        
        # Use sparse Q-table: only store valid (state, action) pairs
        # defaultdict(lambda: defaultdict(float)) creates nested dicts with 0.0 default
        self.q_table = defaultdict(lambda: defaultdict(float))
    
    def choose_action(self, state, env):
        """
        Choose action using epsilon-greedy policy.
        
        Args:
            state: current state index
            env: environment object
        
        Returns:
            action (target state index) or None if no valid actions
        """
        valid_actions = env.get_valid_actions(state)
        
        if not valid_actions:
            return None
        
        # epsilon-greedy: explore or exploit
        if random.random() < self.epsilon:
            # explore: random valid action
            return random.choice(valid_actions)
        else:
            # exploit: choose best action according to Q-table
            # get Q values for valid actions (use 0.0 if not in sparse table)
            q_values = [self.q_table[state][a] for a in valid_actions]
            max_q = max(q_values)
            
            # if multiple actions have same max Q, randomly choose one
            best_actions = [a for a in valid_actions if self.q_table[state][a] == max_q]
            return random.choice(best_actions)
    
    def update(self, state, action, reward, next_state, env):
        """
        Update Q-table using Q-learning update rule.
        
        Args:
            state: current state index
            action: action taken (target state index)
            reward: reward received
            next_state: next state index
            env: environment object
        """
        # current Q value (defaultdict returns 0.0 if not present)
        current_q = self.q_table[state][action]
        
        # max Q value for next state (only consider valid actions)
        if env.is_terminal(next_state):
            # terminal state: no future reward
            max_next_q = 0.0
        else:
            valid_next_actions = env.get_valid_actions(next_state)
            if valid_next_actions:
                # Get Q values for valid actions (defaultdict returns 0.0 if not present)
                max_next_q = max([self.q_table[next_state][a] for a in valid_next_actions])
            else:
                max_next_q = 0.0
        
        # Q-learning update: Q(s,a) = Q(s,a) + α[r + γ*max(Q(s',a')) - Q(s,a)]
        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        self.q_table[state][action] = new_q
    
    def decay_epsilon(self, episode, total_episodes):
        """
        Decay epsilon based on episode number.
        
        Args:
            episode: current episode number (0-indexed)
            total_episodes: total number of episodes
        """
        if self.epsilon_decay == 'linear':
            # linear decay
            progress = episode / total_episodes
            self.epsilon = max(self.epsilon_min, 
                             self.epsilon_start - (self.epsilon_start - self.epsilon_min) * progress)
        elif self.epsilon_decay == 'exponential':
            # exponential decay
            decay_rate = 0.995  # adjust this to control decay speed
            self.epsilon = max(self.epsilon_min, self.epsilon * decay_rate)
    
    def get_q_table(self):
        """
        Get current Q-table as dense numpy array for compatibility.
        Converts sparse representation to dense n_states × n_states matrix.
        """
        # Convert sparse dict to dense numpy array for compatibility
        q_table_dense = np.zeros((self.n_states, self.n_states))
        for state in self.q_table:
            for action in self.q_table[state]:
                q_table_dense[state][action] = self.q_table[state][action]
        return q_table_dense
    
    def get_q_table_sparse(self):
        """Get current Q-table in sparse format (for internal use)."""
        # Return a copy of the nested dict structure
        return {state: dict(actions) for state, actions in self.q_table.items()}


def run_episode(env, agent, max_steps=1000):
    """
    Run one episode of Q-learning (optimized - no changes needed here).
    
    Args:
        env: environment object
        agent: Q-learning agent
        max_steps: maximum steps per episode to prevent infinite loops
    
    Returns:
        tuple of (total_reward, num_steps)
    """
    state = env.reset()
    total_reward = 0
    steps = 0
    
    while steps < max_steps:
        # choose action
        action = agent.choose_action(state, env)
        if action is None:
            break
        
        # take step in environment
        next_state, reward, done, _ = env.step(action)
        
        # update Q-table
        agent.update(state, action, reward, next_state, env)
        
        # move to next state
        state = next_state
        total_reward += reward
        steps += 1
        
        # check if episode finished
        if done:
            break
    
    return total_reward, steps


def train_q_learning(env, n_episodes=1000, alpha=0.1, gamma=0.9,
                     epsilon_start=0.9, epsilon_min=0.1, log_interval=100,
                     early_stop=True, early_stop_patience=100, early_stop_threshold=1e-4,
                     random_seed=42, early_stop_check_interval=10):
    """
    Train Q-learning agent (optimized version).
    
    Args:
        env: environment object
        n_episodes: number of training episodes
        alpha: learning rate
        gamma: discount factor
        epsilon_start: initial exploration rate
        epsilon_min: minimum exploration rate
        log_interval: interval for logging and saving Q-table
        early_stop: enable early stopping if True
        early_stop_patience: number of consecutive episodes to check for early stop
        early_stop_threshold: Q-table change threshold for early stopping
        random_seed: random seed for reproducibility (default: 42)
        early_stop_check_interval: check early stopping every N episodes (default: 10)
    
    Returns:
        tuple of (episode_rewards, episode_steps, q_table_history, early_stopped, initial_q_table)
    """
    # set random seed for reproducibility
    set_random_seed(random_seed)
    
    n_states = env.n_states
    
    # create agent
    agent = QLearningAgent(n_states, alpha=alpha, gamma=gamma,
                          epsilon_start=epsilon_start, epsilon_min=epsilon_min)
    
    # save initial Q-table (all zeros, converted to dense format)
    initial_q_table = agent.get_q_table()
    
    # storage for results
    episode_rewards = []
    episode_steps = []
    q_table_history = []  # list of (episode, q_table) tuples
    
    # early stopping variables (optimized: check less frequently)
    previous_q_table_sparse = None
    early_stopped = False
    q_change_history = []  # store Q-table changes for sliding window
    
    print(f"Starting optimized Q-learning training with {n_episodes} episodes...")
    print(f"Parameters: alpha={alpha}, gamma={gamma}, epsilon_start={epsilon_start}, epsilon_min={epsilon_min}")
    if early_stop:
        print(f"Early stopping: enabled (patience={early_stop_patience}, threshold={early_stop_threshold}, check_interval={early_stop_check_interval})")
    print("-" * 60)
    
    # Calculate max steps per episode based on environment size
    # For grid environments: max_steps = n_states * 2 (safety factor)
    max_steps_per_episode = min(1000, n_states * 2)
    
    # Use tqdm for progress bar if available
    episode_iterator = range(n_episodes)
    if HAS_TQDM:
        episode_iterator = tqdm(episode_iterator, desc="Training", unit="ep")
    
    for episode in episode_iterator:
        # run one episode
        total_reward, steps = run_episode(env, agent, max_steps=max_steps_per_episode)
        
        # record metrics
        episode_rewards.append(total_reward)
        episode_steps.append(steps)
        
        # decay epsilon
        agent.decay_epsilon(episode, n_episodes)
        
        # save Q-table periodically for convergence analysis (only when needed)
        if episode % log_interval == 0 or episode == n_episodes - 1:
            q_table_dense = agent.get_q_table()  # Convert to dense only when saving
            q_table_history.append((episode, q_table_dense))
        
        # early stopping check: only check every N episodes (optimized)
        if early_stop and episode > 0 and episode % early_stop_check_interval == 0:
            # Get sparse Q-table representation for comparison
            current_q_table_sparse = agent.get_q_table_sparse()
            
            if previous_q_table_sparse is not None:
                # Compute change only for states/actions that exist in either table
                all_states = set(previous_q_table_sparse.keys()) | set(current_q_table_sparse.keys())
                total_diff = 0.0
                count = 0
                
                for state in all_states:
                    prev_actions = previous_q_table_sparse.get(state, {})
                    curr_actions = current_q_table_sparse.get(state, {})
                    all_actions = set(prev_actions.keys()) | set(curr_actions.keys())
                    
                    for action in all_actions:
                        prev_q = prev_actions.get(action, 0.0)
                        curr_q = curr_actions.get(action, 0.0)
                        total_diff += abs(curr_q - prev_q)
                        count += 1
                
                if count > 0:
                    mean_change = total_diff / count
                    q_change_history.append(mean_change)
                    
                    # keep only last 'patience' number of changes
                    if len(q_change_history) > early_stop_patience:
                        q_change_history.pop(0)
                    
                    # check early stopping condition: if we have enough history
                    if len(q_change_history) >= early_stop_patience:
                        # compute average change over the last 'patience' checks
                        avg_change = np.mean(q_change_history)
                        
                        # if average change is below threshold, stop training
                        if avg_change < early_stop_threshold:
                            early_stopped = True
                            if HAS_TQDM:
                                tqdm.write(f"\nEarly stopping triggered at episode {episode+1}")
                                tqdm.write(f"Average Q-table change over last {early_stop_patience} checks: {avg_change:.8f}")
                                tqdm.write(f"Threshold: {early_stop_threshold}")
                            else:
                                print(f"\nEarly stopping triggered at episode {episode+1}")
                                print(f"Average Q-table change over last {early_stop_patience} checks: {avg_change:.8f}")
                                print(f"Threshold: {early_stop_threshold}")
                            # save final Q-table if not already saved
                            saved_episodes = [ep for ep, _ in q_table_history]
                            if episode not in saved_episodes:
                                q_table_dense = agent.get_q_table()
                                q_table_history.append((episode, q_table_dense))
                            break
            
            # update previous Q-table for next check (store sparse representation)
            previous_q_table_sparse = current_q_table_sparse
        
        # detailed progress at log_interval (only if not using tqdm or at log points)
        if not HAS_TQDM and (episode + 1) % log_interval == 0:
            avg_reward = np.mean(episode_rewards[-log_interval:])
            avg_steps = np.mean(episode_steps[-log_interval:])
            progress_pct = ((episode + 1) / n_episodes) * 100
            print(f"Episode {episode+1}/{n_episodes} ({progress_pct:.1f}%) | "
                  f"Avg reward: {avg_reward:.2f} | "
                  f"Avg steps: {avg_steps:.1f} | "
                  f"Epsilon: {agent.epsilon:.3f}")
    
    if HAS_TQDM:
        print()  # Newline after tqdm
    print("-" * 60)
    if early_stopped:
        print(f"Training completed early at episode {len(episode_rewards)}!")
    else:
        print("Training completed!")
    
    return episode_rewards, episode_steps, q_table_history, early_stopped, initial_q_table


def compute_q_table_change(q_table1, q_table2):
    """
    Compute change between two Q-tables (dense numpy arrays).
    
    Args:
        q_table1: first Q-table (dense numpy array)
        q_table2: second Q-table (dense numpy array)
    
    Returns:
        mean absolute difference and max absolute difference
    """
    diff = np.abs(q_table2 - q_table1)
    mean_diff = np.mean(diff)
    max_diff = np.max(diff)
    return mean_diff, max_diff

