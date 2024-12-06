import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class BaseFeatureExtractor(nn.Module):
    def __init__(self, input_shape):
        super(BaseFeatureExtractor, self).__init__()
        self.output_size = 512
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        self.input_shape = input_shape
        
        self._conv_output_size = self.output_size_of_convolution(input_shape)
        self.fc1 = nn.Linear(self._conv_output_size, self.output_size)
        
    def output_size_of_convolution(self, shape):
        input = torch.zeros(1, *shape)
        out1 = self.conv1(input)
        out2 = self.conv2(out1)
        out3 = self.conv3(out2)
        conv_output_size = int(np.prod(out3.size()))
        return conv_output_size

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return x
    
    def return_input_shape(self):
        return self.input_shape
    
class QNetwork(nn.Module):
    def __init__(self, num_actions, input_shape):
        super(QNetwork, self).__init__()
        self.feature_extractor = BaseFeatureExtractor(input_shape)
        self.fc2 = nn.Linear(self.feature_extractor.output_size, num_actions)

    def forward(self, x):
        x = self.feature_extractor(x)
        return self.fc2(x)
    
class SimpleQNetwork_LunarLander(nn.Module):
    def __init__(self, num_actions, input_shape):
        super(SimpleQNetwork_LunarLander, self).__init__()
        self.input_shape = input_shape
        self.fc1 = nn.Linear(np.prod(input_shape), 128)  # First fully connected layer
        self.fc2 = nn.Linear(128, 128)  # Second fully connected layer
        self.fc3 = nn.Linear(128, num_actions)  # Output layer

    def forward(self, x):
        x = x.view(x.size(0), -1) # Flatten the input
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class PER_buffer:
    def __init__(self, parameters):
        self.update_params(parameters)
        self.buffer = []
        self.priorities = np.zeros((self.capacity,), dtype=np.float32)
        self.pos = 0

    def update_params(self, parameters):
        self.alpha = parameters["alpha"]
        self.beta = parameters["beta"]
        self.beta_increment_per_sampling = parameters["beta_increment_per_sampling"]
        self.capacity = parameters["capacity"]

    def add(self, state, action, reward, next_state, done, td_error):
        # Ensure states and next_states are np.ndarray
        state = np.array(state, dtype=np.float32)
        next_state = np.array(next_state, dtype=np.float32)

        # Replace the oldest experience with the new one
        if len(self.buffer) == self.capacity:
            self.buffer[self.pos] = (state, action, reward, next_state, done)
        # Add a new experience
        else:
            self.buffer.append((state, action, reward, next_state, done))

        priority = td_error + 1e-5
        self.priorities[self.pos] = priority
        self.pos = (self.pos + 1) % self.capacity
        
    def update_priorities(self, indices, priorities):
        # Update the priorities of the experiences
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority

    def sample(self, batch_size):
        # Check if the replay buffer is empty or has fewer samples than required
        if len(self.buffer) == 0:
            raise ValueError("Replay buffer is empty. Cannot sample.")

        if len(self.buffer) < batch_size:
            raise ValueError(f"Replay buffer has {len(self.buffer)} samples, but batch size is {batch_size}.")
        
        # Proceed to sample experiences from the buffer
        probabilities = self.priorities[:len(self.buffer)] ** self.alpha
        probabilities /= probabilities.sum()

        if len(probabilities) != len(self.buffer):
            raise ValueError("'a' and 'p' must have the same size.")

        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        samples = [self.buffer[idx] for idx in indices]
        total = len(self.buffer)
        weights = (total * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)

        states, actions, rewards, next_states, dones = zip(*samples)

        # Ensure all states and next_states have the same shape
        # Debugging step to identify inconsistent state shapes
        if len(states) == 0:
            raise ValueError("No states to sample.")

        for i, state in enumerate(states):
            if not isinstance(state, np.ndarray):
                raise ValueError(f"State {i} is of type {type(state)} instead of np.ndarray")

        max_shape = max(state.shape for state in states if isinstance(state, np.ndarray))
        states = [np.resize(state, max_shape) if state.shape != max_shape else state for state in states]

        try:
            states = np.stack([np.array(state, dtype=np.float32) for state in states])
            next_states = np.stack([np.array(next_state, dtype=np.float32) for next_state in next_states])
        except ValueError as e:
            print(f"Error stacking states: {e}")
            raise ValueError("Inconsistent state shapes, unable to stack.")

        # Increment beta
        self.beta = min(1.0, self.beta + self.beta_increment_per_sampling)

        return states, actions, rewards, next_states, dones, indices, weights


'''
Features
- Q-Network with target network
- Epsilon-greedy policy
- PER buffer
'''
class BasicAgent:
    def __init__(self,
                 env,
                 parameters):
        self.env = env

        number_of_actions = env.action_space.n
        input_shape = env.observation_space.shape
        
        self.q_network = SimpleQNetwork_LunarLander(number_of_actions,
                                  input_shape)
        
        self.target_q_network = SimpleQNetwork_LunarLander(number_of_actions,
                                         input_shape)
        
        self.replay_buffer = PER_buffer(parameters)
        
        self.update_params(parameters)

        self.step = 0

    def update_params(self,
                      parameters):
        self.learning_rate          = parameters["learning_rate"]
        self.optimizer              = torch.optim.Adam(self.q_network.parameters(),
                                                       lr=self.learning_rate)
        self.batch_size             = parameters["batch_size"]
        self.gamma                  = parameters["gamma"]
        self.epsilon                = parameters["epsilon"]
        self.epsilon_decay          = parameters["epsilon_decay"]
        self.epsilon_min            = parameters["epsilon_min"]
        self.update_target_every    = parameters["update_target_every"]      

    def act(self, state_observation):
        epsilon_old = self.epsilon
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        # Epsilon-greedy policy
        if np.random.rand() <= epsilon_old:
            return self.env.action_space.sample()
        else:
            return self.get_action(state_observation)
        
    def get_action(self,
                   state_observation):
        # Get the action with the highest Q-value
        state_observation = torch.tensor(state_observation, dtype=torch.float32)
        state_observation = state_observation.unsqueeze(0)
        q_values = self.q_network(state_observation)
        return torch.argmax(q_values).item()
    
    def update_target_network(self):
        if self.step % self.update_target_every == 0:
            # Copy the weights from the Q-network to the target Q-network
            self.target_q_network.load_state_dict(self.q_network.state_dict())

    def training_step(self):
        states, actions, rewards, next_states, dones, indices, weights = self.replay_buffer.sample(self.batch_size)

        # Convert to PyTorch tensors
        states = torch.tensor(states, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)
        weights = torch.tensor(weights, dtype=torch.float32)

        # Compute Q-values
        q_values = self.q_network(states).gather(1, actions).squeeze(1)
        # Compute target Q-values
        next_q_values = self.target_q_network(next_states).max(1)[0]
        target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))

        # Compute loss
        loss = (weights * (q_values - target_q_values.detach()).pow(2)).mean()

        # Optimize the Q-network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update priorities
        td_errors = (q_values - target_q_values).detach().cpu().numpy()
        self.replay_buffer.update_priorities(indices, td_errors)

    def train(self,
              number_of_episodes):
        self.results = {
            "rewards": []
        }
        for episode in tqdm(range(number_of_episodes)):
            state = self.env.reset()
            done = False
            episode_reward = 0
            while not done:
                action = self.act(state)
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                self.replay_buffer.add(state, action, reward, next_state, done, 0)
                episode_reward += reward
                state = next_state
                # Ensure the replay buffer has enough samples before training
                if len(self.replay_buffer.buffer) >= self.batch_size:
                    self.training_step()
            print(f"Episode: {episode}, Reward: {episode_reward}")
            self.results["rewards"].append(episode_reward)

    def plot_results(self):
        plt.plot(self.results["rewards"])
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title("Training Results")
        plt.show()