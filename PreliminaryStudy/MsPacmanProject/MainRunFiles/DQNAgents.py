import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from MainRunFiles.ReplayBuffer import ReplayBuffer, PrioritizedReplayBuffer
from MainRunFiles.ICM import IntrinsicCuriosityModule
from MainRunFiles.QNetwork import PNNQNetwork, PNNDuelingQNetwork
import concurrent.futures

class DQNAgentBase:
    def __init__(self,
                 feature_extractor,
                 num_actions,
                 params,
                 Network,
                 task_id,
                 is_dueling_PNN,
                 num_tasks,
                 pnn_bool,
                 task_specific_layers_names_list,
                 flexible_pnn_network): # Actual flexible_PNN network
        self.task_specific_layers = task_specific_layers_names_list
        if not pnn_bool:
            self.q_network = Network(feature_extractor, num_actions).to(params["device"])
            self.target_q_network = Network(feature_extractor, num_actions).to(params["device"])
            self.target_q_network.load_state_dict(self.q_network.state_dict())
        else:
            if flexible_pnn_network is not None:
                # PNN flex
                self.q_network = flexible_pnn_network.to(params["device"])
                self.target_q_network = flexible_pnn_network.to(params["device"])
                self.target_q_network.load_state_dict(self.q_network.state_dict())
                self.q_network.task_id = task_id
                self.target_q_network.task_id = task_id
            else:
                if is_dueling_PNN:
                    # PNN Duel
                    self.q_network = PNNDuelingQNetwork(feature_extractor, num_actions, num_tasks, task_id).to(params["device"])
                    self.target_q_network = PNNDuelingQNetwork(feature_extractor, num_actions, num_tasks, task_id).to(params["device"])
                else:
                    # PNN Q
                    self.q_network = PNNQNetwork(feature_extractor, num_actions, num_tasks, task_id).to(params["device"])
                    self.target_q_network = PNNQNetwork(feature_extractor, num_actions, num_tasks, task_id).to(params["device"])
            self.target_q_network.load_state_dict(self.q_network.state_dict())

        self.device = params["device"]
        self.num_actions = num_actions
        self.learning_rate = params["learning_rate"]
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        
        self.batch_size = params["batch_size"]
        self.gamma = params["gamma"]
        self.epsilon = params["epsilon"]
        self.epsilon_decay = params["epsilon_decay"]
        self.epsilon_min = params["epsilon_min"]
        self.update_target_every = params["update_target_every"]
        self.steps = 0

    def act(self, state, task_id = None):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.num_actions)
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            if task_id is not None:
                return self.q_network(state, task_id).argmax().item()
            else:
                return self.q_network(state).argmax().item()

    def update_target_network(self):
        if self.steps % self.update_target_every == 0:
            state_dict = self.q_network.state_dict()
            filtered_state_dict = {}
            for key, value in state_dict.items():
                if not any(prefix in key for prefix in self.task_specific_layers):
                    filtered_state_dict[key] = value
            self.target_q_network.load_state_dict(filtered_state_dict)

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


# DQN Agent all together
class DQNAgentUnified(DQNAgentBase):
    def __init__(self,
                 feature_extractor,
                 num_actions,
                 params,
                 configuration,
                 Network = None,
                 is_dueling_PNN = False,
                 num_tasks = 3,
                 task_id = None,
                 pnn_bool = False,
                 task_specific_layers_names_list = [],
                 flexible_pnn_network = None):
        super(DQNAgentUnified, self).__init__(feature_extractor,
                                              num_actions,
                                              params,
                                              Network,
                                              task_id,
                                              is_dueling_PNN,
                                              num_tasks,                                           
                                              pnn_bool,
                                              task_specific_layers_names_list,
                                              flexible_pnn_network)      
        if pnn_bool: 
            print("PNN is enabled.")
            if Network and flexible_pnn_network is None:
                raise ValueError("Flexible PNN network or Network is required for training.") 
        # Replay buffer
        self.replay_buffer_size = params["replay_buffer_size"]
        self.PER_bool = configuration.get("PER_bool", False)
        if self.PER_bool: # Prioritized Experience Replay
            self.alpha = params.get("alpha", 0.6)
            self.beta = params.get("beta", 0.4)
            self.beta_increment_per_sampling = params.get("beta_increment_per_sampling", 0.001)
            self.replay_buffer = PrioritizedReplayBuffer(self.replay_buffer_size, self.alpha)
        else: # Regular Replay Buffer
            self.replay_buffer = ReplayBuffer(self.replay_buffer_size)
        
        # Double Q-learning
        self.double_q_learning_bool = configuration.get("double_q_learning_bool", False)

        # Intrinsic Curiosity Module
        self.icm_bool = configuration.get("icm_bool", False)
        self.pnn_bool = pnn_bool
        if self.icm_bool:
            self.eta = params.get("eta", 0.2)
            if not self.pnn_bool:
                input_shape = self.q_network.feature_extractor.return_input_shape()
            else:
                input_shape = self.q_network.base_network.feature_extractor.return_input_shape()
            self.icm_bool = configuration.get("icm_bool", False)
            icm_learning_rate = params.get("icm_learning_rate", 1e-3)
            self.eta = params.get("eta", 0.1)
            self.icm = IntrinsicCuriosityModule(feature_extractor, num_actions, input_shape, self.device)
            self.icm_optimizer = optim.Adam(self.icm.parameters(), lr=icm_learning_rate)
            self.icm_update_every = params.get("icm_update_every", 5)

    def compute_intrinsic_rewards_batch(self, states, next_states, actions):
        states_tensor = torch.FloatTensor(states).to(self.device)
        next_states_tensor = torch.FloatTensor(next_states).to(self.device)
        actions_tensor = torch.LongTensor(actions).to(self.device)
        intrinsic_rewards = []

        with torch.no_grad():
            for state, next_state, action in zip(states_tensor, next_states_tensor, actions_tensor):
                _, predicted_next_state, phi_next_state = self.icm.forward(state.unsqueeze(0), 
                                                                                        next_state.unsqueeze(0), 
                                                                                        action.unsqueeze(0))
                intrinsic_reward = F.mse_loss(predicted_next_state, phi_next_state).item()
                intrinsic_rewards.append(intrinsic_reward)

        return intrinsic_rewards


    def async_icm_calculation(self, states, next_states, actions):
        # https://www.geeksforgeeks.org/how-to-use-threadpoolexecutor-in-python3/
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(self.compute_intrinsic_rewards_batch, states, next_states, actions)
            return future.result()

    def train(self, task_id = None):
        if len(self.replay_buffer) < self.batch_size:
            return
        
        if self.PER_bool:
            self.beta = min(1.0, self.beta + self.beta_increment_per_sampling)
            states, actions, rewards, next_states, dones, indices, weights = self.replay_buffer.sample(self.batch_size, self.beta)
            weights = torch.FloatTensor(weights).to(self.device)
        else:
            states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
            weights = None
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        if task_id is not None:
            q_values = self.q_network(states, task_id).gather(1, actions).squeeze(1)
        else:
            q_values = self.q_network(states).gather(1, actions).squeeze(1)

        if self.icm_bool:
            # icm update every
            if self.steps % self.icm_update_every == 0:
                intrinsic_rewards = self.async_icm_calculation(states.cpu().numpy(), 
                                                            next_states.cpu().numpy(), 
                                                            actions.cpu().numpy())
                intrinsic_rewards = torch.FloatTensor(intrinsic_rewards).to(self.device)
                rewards += self.eta * intrinsic_rewards
        
        if self.double_q_learning_bool:
            # Double Q-Learning
            next_actions = self.q_network(next_states).argmax(1).unsqueeze(1)
            if task_id is not None:
                next_q_values = self.target_q_network(next_states, task_id).gather(1, next_actions).squeeze(1)
            else:
                next_q_values = self.target_q_network(next_states).gather(1, next_actions).squeeze(1)
        else:
            # Regular DQN
            if task_id is not None:
                next_q_values = self.target_q_network(next_states, task_id).max(1)[0]
            else:
                next_q_values = self.target_q_network(next_states).max(1)[0]
        
        # Target Q-value
        target_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        if self.PER_bool:
            # PER: Apply importance-sampling weights
            loss = (F.mse_loss(q_values, target_q_values.detach(), reduction='none') * weights).mean()
            td_errors = torch.abs(target_q_values.detach() - q_values.detach()).cpu().numpy()
            self.replay_buffer.update_priorities(indices, td_errors)
        else:
            loss = F.mse_loss(q_values, target_q_values)

        # Optimize the network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.steps += 1
        self.update_target_network()
        self.decay_epsilon()