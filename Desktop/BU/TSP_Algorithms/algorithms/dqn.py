import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random


class DQN(nn.Module):
    """
    Deep Q-Network model using fully connected layers and LSTM to capture sequential dependencies.
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, hidden=None):
        x = torch.relu(self.fc1(x))
        x, hidden = self.lstm(x.unsqueeze(0), hidden)  # Add batch dimension for LSTM
        x = self.fc2(x.squeeze(0))  # Remove batch dimension
        return x, hidden


class DQNTSP:
    """
    Solves the Traveling Salesman Problem (TSP) using Deep Q-Learning.
    """
    def __init__(self, distance_matrix, episodes=500, max_steps=100, gamma=0.99, lr=0.001, batch_size=32, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.1):
        self.distance_matrix = distance_matrix
        self.num_cities = len(distance_matrix)
        self.episodes = episodes
        self.max_steps = max_steps
        self.gamma = gamma
        self.lr = lr
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Initialize the DQN and optimizer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQN(self.num_cities, 128, self.num_cities).to(self.device)
        self.target_model = DQN(self.num_cities, 128, self.num_cities).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

        # Replay memory
        self.memory = deque(maxlen=5000)
        self.update_target_model()

    def update_target_model(self):
        """
        Updates the target model to match the current model parameters.
        """
        self.target_model.load_state_dict(self.model.state_dict())

    def choose_action(self, state, visited, hidden=None):
        """
        Chooses the next action based on the current state and epsilon-greedy policy.
        """
        if len(visited) == self.num_cities:
            return None, hidden  # No actions available when all cities are visited
        if np.random.rand() < self.epsilon:
            # Random choice among unvisited cities
            action = np.random.choice([i for i in range(self.num_cities) if i not in visited])
        else:
            # Choose action with the highest Q-value
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32).to(self.device)
                q_values, hidden = self.model(state_tensor, hidden)
                for i in visited:
                    q_values[i] = float('-inf')  # Mask visited cities
                action = torch.argmax(q_values).item()
        return action, hidden

    def store_experience(self, state, action, reward, next_state, done):
        """
        Stores a single experience in the replay memory.
        """
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        """
        Performs experience replay to train the model on past experiences.
        """
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        # Efficiently convert batch to tensors
        states, actions, rewards, next_states, dones = map(
            lambda x: np.array(x, dtype=np.float32), zip(*batch)
        )
        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)

        # Compute current Q-values
        current_q_values, _ = self.model(states)
        current_q_values = current_q_values.gather(1, actions.unsqueeze(1)).squeeze()

        # Compute target Q-values
        with torch.no_grad():
            next_q_values, _ = self.target_model(next_states)
            max_next_q_values = next_q_values.max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * max_next_q_values

        # Update model
        loss = self.criterion(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def solve_tsp(self):
        """
        Solves the TSP using Deep Q-Learning.
        """
        best_path = None
        best_cost = float('inf')
        cost_history = []

        for episode in range(self.episodes):
            state = np.zeros(self.num_cities)
            visited = set()
            path = []
            hidden = None
            current_city = np.random.choice(range(self.num_cities))
            path.append(current_city)
            visited.add(current_city)

            for step in range(self.max_steps):
                action, hidden = self.choose_action(state, visited, hidden)
                if action is None:
                    break

                # Reward is negative distance to encourage shorter paths
                reward = -self.distance_matrix[current_city, action]

                next_state = state.copy()
                next_state[action] = 1
                done = len(visited) == self.num_cities - 1
                self.store_experience(state, action, reward, next_state, done)
                self.replay()

                state = next_state
                current_city = action
                path.append(action)
                visited.add(action)

                if done:
                    # Close the loop by returning to the start
                    path.append(path[0])
                    # Correctly calculate the total path cost
                    total_cost = sum(
                        self.distance_matrix[path[i], path[i + 1]] for i in range(len(path) - 1)
                    )
                    if total_cost < best_cost:
                        best_cost = total_cost
                        best_path = path
                    break

            cost_history.append(best_cost)
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            self.update_target_model()

        return best_path[:-1], best_cost, cost_history  # Remove the duplicated start point