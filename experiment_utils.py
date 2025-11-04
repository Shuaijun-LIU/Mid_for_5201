"""
Shared utility functions for experiment scripts.
"""

import numpy as np


def print_q_table_summary(env, q_table):
    """Print Q-table in human-readable format."""
    print("\n" + "=" * 60)
    print("Final Q-table Summary:")
    print("=" * 60)
    
    for state_idx in range(env.n_states):
        state_name = env.get_state_name(state_idx)
        valid_actions = env.get_valid_actions(state_idx)
        
        if not valid_actions:
            continue
        
        # get Q values for valid actions
        q_values = [(action, q_table[state_idx][action]) for action in valid_actions]
        q_values.sort(key=lambda x: x[1], reverse=True)  # sort by Q value
        
        print(f"\nState {state_name} (index {state_idx}):")
        for action, q_val in q_values:
            action_name = env.get_state_name(action)
            print(f"  -> {action_name}: {q_val:8.4f}")
        
        # mark optimal action
        if q_values:
            best_action, best_q = q_values[0]
            best_action_name = env.get_state_name(best_action)
            print(f"  [Optimal: {best_action_name} with Q = {best_q:.4f}]")
    
    print("\n" + "=" * 60)


def print_experiment_stats(episode_rewards, episode_steps):
    """Print experiment statistics."""
    print("\n" + "=" * 60)
    print("Training Statistics:")
    print("=" * 60)
    print(f"Total episodes: {len(episode_rewards)}")
    print(f"Average reward (all): {np.mean(episode_rewards):.4f}")
    print(f"Average reward (last 100): {np.mean(episode_rewards[-100:]):.4f}")
    print(f"Average steps (all): {np.mean(episode_steps):.4f}")
    print(f"Average steps (last 100): {np.mean(episode_steps[-100:]):.4f}")
    print(f"Min steps: {np.min(episode_steps)}")
    print(f"Max steps: {np.max(episode_steps)}")
    print("=" * 60)

