import numpy as np
import pickle
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Load dataset
with open('dataset_new_new.pkl', 'rb') as f:
    dataset = pickle.load(f)

# Prepare data for training
X = np.array([data[:4] for data in dataset])  # Inputs: pitch, roll, yaw, fill_level
y = np.array([data[4].flatten() for data in dataset])  # Outputs: flattened inertia matrix

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the input data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# Create DataLoader
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define the neural network model
class InertiaNN(nn.Module):
    def __init__(self):
        super(InertiaNN, self).__init__()
        self.fc1 = nn.Linear(X_train.shape[1], 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, y_train.shape[1])  # Adjust output layer to match flattened inertia matrix size

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = InertiaNN()

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 100
for epoch in tqdm(range(num_epochs), desc="Training"):
    model.train()
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

# Evaluate the model
model.eval()
test_loss = 0.0
with torch.no_grad():
    for inputs, targets in test_loader:
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        test_loss += loss.item()

test_loss /= len(test_loader)
print(f"Mean Squared Error on Test Set: {test_loss}")

# Save the trained model
torch.save(model.state_dict(), 'inertia_nn_model.pth')

# Load the trained model
loaded_model = InertiaNN()
loaded_model.load_state_dict(torch.load('inertia_nn_model.pth', weights_only=True))
loaded_model.eval()

# Select a ground truth inertia matrix from the dataset
# Change this index to select a different example - Randomly select an index between 0 and len(dataset)
selected_index =   np.random.randint(0, len(dataset))
ground_truth_data = dataset[selected_index]
ground_truth_input = ground_truth_data[:4]
ground_truth_inertia = ground_truth_data[4].flatten()

# Scale the ground truth input
ground_truth_input_scaled = scaler.transform([ground_truth_input])
ground_truth_input_tensor = torch.tensor(ground_truth_input_scaled, dtype=torch.float32)

# Predict the inertia matrix using the neural network
with torch.no_grad():
    predicted_inertia = loaded_model(ground_truth_input_tensor)

# Print the results
print("Ground truth input (pitch, roll, yaw, fill_level):")
print(ground_truth_input)

print("Ground truth inertia matrix (flattened):")
print(ground_truth_inertia)

print("Predicted inertia matrix (flattened):")
print(predicted_inertia.numpy())

# Calculate and print the Mean Squared Error for the selected example
mse = np.mean((predicted_inertia.numpy() - ground_truth_inertia) ** 2)
print(f"Mean Squared Error for the selected example: {mse}")