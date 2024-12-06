import numpy as np
import pickle
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class InertiaModel:
    def __init__(self, dataset_path, model_path='inertia_nn_model.pth'):
        self.dataset_path = dataset_path
        self.model_path = model_path
        self.scaler = StandardScaler()
        self.model = self._build_model()
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self._load_dataset()
        self._prepare_data()

    def _build_model(self):
        class InertiaNN(nn.Module):
            def __init__(self, input_size, output_size):
                super(InertiaNN, self).__init__()
                self.fc1 = nn.Linear(input_size, 64)
                self.fc2 = nn.Linear(64, 64)
                self.fc3 = nn.Linear(64, output_size)

            def forward(self, x):
                x = torch.relu(self.fc1(x))
                x = torch.relu(self.fc2(x))
                x = self.fc3(x)
                return x

        return InertiaNN(input_size=4, output_size=9)  # Adjust input and output sizes as needed

    def _load_dataset(self):
        with open(self.dataset_path, 'rb') as f:
            self.dataset = pickle.load(f)

    def _prepare_data(self):
        X = np.array([data[:4] for data in self.dataset])  # Inputs: pitch, roll, yaw, fill_level
        y = np.array([data[4].flatten() for data in self.dataset])  # Outputs: flattened inertia matrix

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

        self.X_train = torch.tensor(X_train, dtype=torch.float32)
        self.X_test = torch.tensor(X_test, dtype=torch.float32)
        self.y_train = torch.tensor(y_train, dtype=torch.float32)
        self.y_test = torch.tensor(y_test, dtype=torch.float32)

        self.train_loader = DataLoader(TensorDataset(self.X_train, self.y_train), batch_size=32, shuffle=True)
        self.test_loader = DataLoader(TensorDataset(self.X_test, self.y_test), batch_size=32, shuffle=False)

    def train(self, num_epochs=100):
        for epoch in tqdm(range(num_epochs), desc="Training"):
            self.model.train()
            for inputs, targets in self.train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()

    def evaluate(self):
        self.model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for inputs, targets in self.test_loader:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                test_loss += loss.item()

        test_loss /= len(self.test_loader)
        print(f"Mean Squared Error on Test Set: {test_loss}")

    def save_model(self):
        torch.save(self.model.state_dict(), self.model_path)

    def load_model(self):
        self.model.load_state_dict(torch.load(self.model_path, weights_only=True))
        self.model.eval()

    def predict(self, input_data):
        input_scaled = self.scaler.transform([input_data])
        input_tensor = torch.tensor(input_scaled, dtype=torch.float32)
        with torch.no_grad():
            predicted_inertia = self.model(input_tensor)
        return predicted_inertia.numpy()

    def compare_ground_truth(self):
        selected_index = np.random.randint(0, len(self.dataset))
        ground_truth_data = self.dataset[selected_index]
        ground_truth_input = ground_truth_data[:4]
        ground_truth_inertia = ground_truth_data[4].flatten()

        predicted_inertia = self.predict(ground_truth_input)

        print("Ground truth input (pitch, roll, yaw, fill_level):")
        print(ground_truth_input)

        print("Ground truth inertia matrix (flattened):")
        print(ground_truth_inertia)

        print("Predicted inertia matrix (flattened):")
        print(predicted_inertia)

        mse = np.mean((predicted_inertia - ground_truth_inertia) ** 2)
        print(f"Mean Squared Error for the selected example: {mse}")

# Example usage
model = InertiaModel(dataset_path='dataset_new_new.pkl')
model.load_model()
#model.train(num_epochs=200) #100 : 630
model.evaluate()
#model.save_model()
model.compare_ground_truth()