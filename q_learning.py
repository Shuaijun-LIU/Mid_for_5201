"""
Q-learning algorithm implementation.
"""

import numpy as np
import random


def set_random_seed(seed=42):
    """
    Set random seed for reproducibility.
    
    Args:
        seed: random seed value
    """
    np.random.seed(seed)
    random.seed(seed)


class QLearningAgent:
    """Q-learning agent with epsilon-greedy action selection."""
    
    def __init__(self, n_states, alpha=0.1, gamma=0.9, 
                 epsilon_start=0.9, epsilon_min=0.1, epsilon_decay='linear'):
        """
        Initialize Q-learning agent.
        
        Args:
            n_states: number of states in the environment
            alpha: learning rate
            gamma: discount factor
            epsilon_start: initial exploration rate
            epsilon_min: minimum exploration rate
            epsilon_decay: decay method ('linear' or 'exponential')
        """
        self.n_states = n_states
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        
        # initialize Q-table: Q[state][action]
        # actions are target state indices, so Q-table is n_states x n_states
        self.q_table = np.zeros((n_states, n_states))
    
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
            # get Q values for valid actions
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
        # current Q value
        current_q = self.q_table[state][action]
        
        # max Q value for next state (only consider valid actions)
        if env.is_terminal(next_state):
            # terminal state: no future reward
            max_next_q = 0
        else:
            valid_next_actions = env.get_valid_actions(next_state)
            if valid_next_actions:
                max_next_q = max([self.q_table[next_state][a] for a in valid_next_actions])
            else:
                max_next_q = 0
        
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
        """Get current Q-table."""
        return self.q_table.copy()


def run_episode(env, agent):
    """
    Run one episode of Q-learning.
    
    Args:
        env: environment object
        agent: Q-learning agent
    
    Returns:
        tuple of (total_reward, num_steps)
    """
    state = env.reset()
    total_reward = 0
    steps = 0
    
    while True:
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
                     random_seed=42):
    """
    Train Q-learning agent.
    
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
    
    Returns:
        tuple of (episode_rewards, episode_steps, q_table_history, early_stopped, initial_q_table)
    """
    # set random seed for reproducibility
    set_random_seed(random_seed)
    
    n_states = env.n_states
    
    # create agent
    agent = QLearningAgent(n_states, alpha=alpha, gamma=gamma,
                          epsilon_start=epsilon_start, epsilon_min=epsilon_min)
    
    # save initial Q-table (all zeros before training)
    initial_q_table = agent.get_q_table().copy()
    
    # storage for results
    episode_rewards = []
    episode_steps = []
    q_table_history = []  # list of (episode, q_table) tuples
    
    # early stopping variables
    previous_q_table = None
    early_stopped = False
    q_change_history = []  # store Q-table changes for sliding window
    
    print(f"Starting Q-learning training with {n_episodes} episodes...")
    print(f"Parameters: alpha={alpha}, gamma={gamma}, epsilon_start={epsilon_start}, epsilon_min={epsilon_min}")
    if early_stop:
        print(f"Early stopping: enabled (patience={early_stop_patience}, threshold={early_stop_threshold})")
    print("-" * 60)
    
    for episode in range(n_episodes):
        # run one episode
        total_reward, steps = run_episode(env, agent)
        
        # record metrics
        episode_rewards.append(total_reward)
        episode_steps.append(steps)
        
        # decay epsilon
        agent.decay_epsilon(episode, n_episodes)
        
        # get current Q-table
        current_q_table = agent.get_q_table()
        
        # save Q-table periodically for convergence analysis
        if episode % log_interval == 0 or episode == n_episodes - 1:
            q_table_history.append((episode, current_q_table.copy()))
        
        # early stopping check: compute Q-table change every episode
        if early_stop and episode > 0:
            # compute change from previous episode
            if previous_q_table is not None:
                mean_change, max_change = compute_q_table_change(previous_q_table, current_q_table)
                q_change_history.append(mean_change)
                
                # keep only last 'patience' number of changes
                if len(q_change_history) > early_stop_patience:
                    q_change_history.pop(0)
                
                # check early stopping condition: if we have enough history
                if len(q_change_history) >= early_stop_patience:
                    # compute average change over the last 'patience' episodes
                    avg_change = np.mean(q_change_history)
                    
                    # if average change is below threshold, stop training
                    if avg_change < early_stop_threshold:
                        early_stopped = True
                        print(f"\nEarly stopping triggered at episode {episode+1}")
                        print(f"Average Q-table change over last {early_stop_patience} episodes: {avg_change:.8f}")
                        print(f"Threshold: {early_stop_threshold}")
                        # save final Q-table if not already saved
                        saved_episodes = [ep for ep, _ in q_table_history]
                        if episode not in saved_episodes:
                            q_table_history.append((episode, current_q_table.copy()))
                        break
        
        # update previous Q-table for next iteration
        previous_q_table = current_q_table.copy()
        
        # print progress
        if (episode + 1) % log_interval == 0:
            avg_reward = np.mean(episode_rewards[-log_interval:])
            avg_steps = np.mean(episode_steps[-log_interval:])
            print(f"Episode {episode+1}/{n_episodes} | "
                  f"Avg reward: {avg_reward:.2f} | "
                  f"Avg steps: {avg_steps:.1f} | "
                  f"Epsilon: {agent.epsilon:.3f}")
    
    print("-" * 60)
    if early_stopped:
        print(f"Training completed early at episode {len(episode_rewards)}!")
    else:
        print("Training completed!")
    
    return episode_rewards, episode_steps, q_table_history, early_stopped, initial_q_table


def compute_q_table_change(q_table1, q_table2):
    """
    Compute change between two Q-tables.
    
    Args:
        q_table1: first Q-table
        q_table2: second Q-table
    
    Returns:
        mean absolute difference and max absolute difference
    """
    diff = np.abs(q_table2 - q_table1)
    mean_diff = np.mean(diff)
    max_diff = np.max(diff)
    return mean_diff, max_diff

