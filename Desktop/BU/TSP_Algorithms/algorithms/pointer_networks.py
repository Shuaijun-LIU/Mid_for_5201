import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset


class PointerNetwork(nn.Module):
    """
    Pointer Network implementation for solving the TSP.
    """
    def __init__(self, input_dim, hidden_dim):
        super(PointerNetwork, self).__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.pointer = nn.Linear(hidden_dim, hidden_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, inputs):
        encoder_outputs, (hidden, cell) = self.encoder(inputs)

        decoder_input = encoder_outputs[:, 0, :].unsqueeze(1)  # Start from the first point
        path_logits = []

        for _ in range(inputs.size(1)):
            decoder_output, (hidden, cell) = self.decoder(decoder_input, (hidden, cell))
            scores = torch.matmul(decoder_output, encoder_outputs.transpose(1, 2))
            path_logits.append(scores)
            decoder_input = torch.matmul(self.softmax(scores), encoder_outputs)

        path_logits = torch.cat(path_logits, dim=1)
        return path_logits


class PointerNetworksTSP:
    def __init__(self, num_cities, input_dim=2, hidden_dim=128, learning_rate=0.001, num_epochs=1000, batch_size=32):
        self.num_cities = num_cities
        self.model = PointerNetwork(input_dim, hidden_dim)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.loss_fn = nn.CrossEntropyLoss()
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def fit(self, coordinates):
        """
        Train the Pointer Network with the given city coordinates.
        """
        coordinates = torch.tensor(coordinates, dtype=torch.float32).to(self.device)
        # Ensure shape is [batch_size, num_cities, input_dim]
        coordinates = coordinates.unsqueeze(0).repeat(self.batch_size, 1, 1)  # Add batch dimension
        target = torch.arange(self.num_cities).repeat(self.batch_size, 1).to(self.device)

        dataset = TensorDataset(coordinates, target)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model.train()
        for epoch in range(self.num_epochs):
            total_loss = 0
            for batch_coordinates, batch_target in dataloader:
                self.optimizer.zero_grad()
                outputs = self.model(batch_coordinates)
                outputs = outputs.view(-1, self.num_cities)
                batch_target = batch_target.view(-1)
                loss = self.loss_fn(outputs, batch_target)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            if (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch + 1}/{self.num_epochs}, Loss: {total_loss:.4f}")

    def predict(self, coordinates):
        """
        Predict the TSP solution for the given city coordinates.
        """
        self.model.eval()
        with torch.no_grad():
            coordinates = torch.tensor(coordinates, dtype=torch.float32).to(self.device).unsqueeze(0)
            outputs = self.model(coordinates)
            path = torch.argmax(outputs, dim=-1).squeeze().tolist()
        return path


def solve_tsp_pointer_network(coordinates, num_epochs=1000, batch_size=32, learning_rate=0.001, hidden_dim=128):
    """
    Solves the TSP using Pointer Networks.
    """
    num_cities = len(coordinates)
    tsp_solver = PointerNetworksTSP(
        num_cities=num_cities,
        input_dim=2,
        hidden_dim=hidden_dim,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        batch_size=batch_size
    )
    coordinates = np.array(coordinates)  # Ensure coordinates are in correct array format
    tsp_solver.fit(coordinates)

    best_path = tsp_solver.predict(coordinates)
    best_cost = sum(
        np.linalg.norm(np.array(coordinates[best_path[i]]) - np.array(coordinates[best_path[(i + 1) % num_cities]]))
        for i in range(num_cities)
    )
    return best_path + [best_path[0]], best_cost
