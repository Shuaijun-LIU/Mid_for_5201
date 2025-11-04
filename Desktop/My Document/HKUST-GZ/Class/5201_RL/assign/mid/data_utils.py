"""
Utility functions for saving and loading training data.
"""

import numpy as np
import pickle
import os


def save_training_data(episode_rewards, episode_steps, q_table_history, 
                      final_q_table, save_dir, experiment_name, initial_q_table=None):
    """
    Save training data to files.
    
    Args:
        episode_rewards: list of rewards per episode
        episode_steps: list of steps per episode
        q_table_history: list of (episode, q_table) tuples
        final_q_table: final Q-table
        save_dir: directory to save files
        experiment_name: name of experiment (for file naming)
        initial_q_table: initial Q-table (before training, all zeros)
    """
    # create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # save initial Q-table if provided
    if initial_q_table is not None:
        initial_qtable_path = os.path.join(save_dir, f'{experiment_name}_qtable_initial.npy')
        np.save(initial_qtable_path, initial_q_table)
        print(f"Initial Q-table saved to {initial_qtable_path}")
    
    # save final Q-table as numpy file
    qtable_path = os.path.join(save_dir, f'{experiment_name}_qtable_final.npy')
    np.save(qtable_path, final_q_table)
    print(f"Final Q-table saved to {qtable_path}")
    
    # save training data as pickle
    training_data = {
        'episode_rewards': episode_rewards,
        'episode_steps': episode_steps,
        'q_table_history': q_table_history,
        'final_q_table': final_q_table
    }
    if initial_q_table is not None:
        training_data['initial_q_table'] = initial_q_table
    data_path = os.path.join(save_dir, f'{experiment_name}_training_data.pkl')
    with open(data_path, 'wb') as f:
        pickle.dump(training_data, f)
    print(f"Training data saved to {data_path}")
    
    # save rewards and steps as text file (human readable)
    txt_path = os.path.join(save_dir, f'{experiment_name}_rewards_steps.txt')
    with open(txt_path, 'w') as f:
        f.write("Episode\tReward\tSteps\n")
        for i, (r, s) in enumerate(zip(episode_rewards, episode_steps)):
            f.write(f"{i+1}\t{r:.2f}\t{s}\n")
    print(f"Rewards and steps saved to {txt_path}")


def load_training_data(load_dir, experiment_name):
    """
    Load training data from files.
    
    Args:
        load_dir: directory containing saved files
        experiment_name: name of experiment
    
    Returns:
        dictionary with training data
    """
    data_path = os.path.join(load_dir, f'{experiment_name}_training_data.pkl')
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    return data

