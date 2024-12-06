import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import os
import cv2
import imageio
import pickle

from MainRunFiles.QNetwork import QNetwork, DuelingQNetwork, BaseFeatureExtractor, PNNDuelingQNetwork, PNNQNetwork
from MainRunFiles.DQNAgents import DQNAgentUnified
from MainRunFiles.ReplayBuffer import PrioritizedReplayBuffer

class DQNTrainer:
    def __init__(self,
                 input_shape,
                 num_actions,
                 DQN_params,
                 environment,
                 num_episodes,
                 render_frequency,
                 file_path,
                 environment_name,
                 Wrapper,
                 model_save_interval = 100,
                 configuration = {
                        "double_q_learning_bool": False,
                        "icm_bool": False,
                        "PER_bool": False,
                        "DuelingQNetwork": False
                 },
                 config_name = 'NoName',
                 plot_bool = True,
                 fps = 30,
                 task_id = None,
                 pnn_bool = False,
                 num_tasks = 3,
                 task_specific_layers_names_list = [],
                 flexible_pnn_network = None): #Pre-Built network
        print('PNN bool:', pnn_bool)
        if pnn_bool:
            self.task_specific_layers_names_list = task_specific_layers_names_list
        
        # Initialize the agent, environment, and training parameters
        self.env = environment
        self.num_episodes = num_episodes
        self.render_frequency = render_frequency
        self.per_enabled = configuration.get("PER_bool", False)
        self.dueling_enabled = configuration.get("DuelingQNetwork", False)
        self.pnn_bool = pnn_bool
        feature_extractor = BaseFeatureExtractor(input_shape)
        self.number_of_actions = num_actions
        self.flexible_pnn_bool = flexible_pnn_network is not None

        if self.flexible_pnn_bool:
            self.agent = DQNAgentUnified(feature_extractor,
                                            num_actions,
                                            DQN_params,
                                            configuration,
                                            Network = None,
                                            is_dueling_PNN=None,
                                            num_tasks=num_tasks,
                                            task_id=task_id,
                                            pnn_bool=True,
                                            task_specific_layers_names_list = self.task_specific_layers_names_list,
                                            flexible_pnn_network = flexible_pnn_network)    
        else:
            if self.dueling_enabled: # Dueling Q Network
                if not self.pnn_bool:
                    self.agent = DQNAgentUnified(feature_extractor,
                                                num_actions,
                                                DQN_params,
                                                configuration,
                                                Network = DuelingQNetwork,
                                                is_dueling_PNN=True,
                                                num_tasks=num_tasks,
                                                task_id=task_id,
                                                pnn_bool=False,
                                                )           

                elif self.pnn_bool:
                    if task_id is None:
                        raise ValueError("Task ID must be provided for Progressive Neural Networks.")
                    self.agent = DQNAgentUnified(feature_extractor,
                                                num_actions,
                                                DQN_params,
                                                configuration,
                                                Network = PNNDuelingQNetwork,
                                                is_dueling_PNN=True,
                                                num_tasks=num_tasks,
                                                task_id=task_id,
                                                pnn_bool=True,
                                                task_specific_layers_names_list = self.task_specific_layers_names_list)    
            else:
                if not self.pnn_bool:
                    self.agent = DQNAgentUnified(feature_extractor,
                                                num_actions,
                                                DQN_params,
                                                configuration,
                                                Network = QNetwork,
                                                is_dueling_PNN=False,
                                                num_tasks=num_tasks,
                                                task_id=task_id,
                                                pnn_bool=False)
                elif self.pnn_bool:
                    self.agent = DQNAgentUnified(feature_extractor,
                                                num_actions,
                                                DQN_params,
                                                configuration,
                                                Network = PNNQNetwork,
                                                is_dueling_PNN=False,
                                                num_tasks=num_tasks,
                                                task_id=task_id,
                                                pnn_bool=True,
                                                task_specific_layers_names_list = self.task_specific_layers_names_list)    
            
        # Task-specific parameters
        self.task_id = task_id
        
        # Initialize the input shape and wrapper
        self.input_shape = input_shape
        self.Wrapper = Wrapper

        # Initialize the file path for saving the model
        self.save_path = f"{file_path}/{environment_name}/{config_name}"
        self.model_save_interval = model_save_interval
        self.fps = fps

        # Display booleans
        self.plot_bool = plot_bool

        # Initialize logs for metrics
        self.episode_rewards = []
        self.action_log = []
        self.state_log = []
        self.number_of_steps_log = []

        # Initialize a variable to track consecutive positive rewards
        self.positive_reward_streak = 0

    def load_network(self,
                    model_path_Network,
                    currentNetwork,
                    icm_update=False,
                    freeze_feature_extractor=False):
        """
        - model_path_Network: str, path to the pre-trained model file (.pt).
        - currentNetwork: current network (e.g.) self.agent.q_network
        - freeze_feature_extractor: bool, whether to freeze the feature extractor layers during fine-tuning.
        """
        pretrained_dict = torch.load(model_path_Network,
                                        map_location='cpu',
                                        weights_only=True)
        model_dict = currentNetwork.state_dict()

        pretrained_dict = {}
        for key, value in pretrained_dict.items():
            if key in model_dict and value.size() == model_dict[key].size():
                pretrained_dict[key] = value

        model_dict.update(pretrained_dict)
        currentNetwork.load_state_dict(model_dict)

        # Freezeee feature extractor
        if freeze_feature_extractor:
            if not self.pnn_bool:
                params = currentNetwork.feature_extractor.parameters()
            else:   
                params = currentNetwork.base_network.feature_extractor.parameters()
            for param in params:
                param.requires_grad = False

        # Reinit out layers
        if not icm_update:
            if not self.pnn_bool:
                pass # CHECK!!!
                # Reinitialize layers for standard QNetwork or DuelingQNetwork
                if not self.dueling_enabled:
                    # Reinitialize the output layer for QNetwork
                    torch.nn.init.kaiming_normal_(currentNetwork.fc2.weight, nonlinearity='relu')
                    torch.nn.init.constant_(currentNetwork.fc2.bias, 0)
                else:
                    print("DuelingQNetwork reinitialization")
                    # Reinitialize the value and advantage streams for DuelingQNetwork
                    torch.nn.init.kaiming_normal_(currentNetwork.value_stream.weight, nonlinearity='relu')
                    torch.nn.init.constant_(currentNetwork.value_stream.bias, 0)
                    torch.nn.init.kaiming_normal_(currentNetwork.advantage_stream.weight, nonlinearity='relu')
                    torch.nn.init.constant_(currentNetwork.advantage_stream.bias, 0)
            else:
                for task_id in range(1, currentNetwork.num_tasks + 1):
                    if hasattr(currentNetwork.base_network, f"{task_id}_value_stream"):
                        value_stream = getattr(currentNetwork.base_network, f"{task_id}_value_stream")
                        torch.nn.init.kaiming_normal_(value_stream.weight, nonlinearity='relu')
                        torch.nn.init.constant_(value_stream.bias, 0)
                    
                    if hasattr(currentNetwork.base_network, f"{task_id}_advantage_stream"):
                        advantage_stream = getattr(currentNetwork.base_network, f"{task_id}_advantage_stream")
                        torch.nn.init.kaiming_normal_(advantage_stream.weight, nonlinearity='relu')
                        torch.nn.init.constant_(advantage_stream.bias, 0)
                        
                    if hasattr(currentNetwork.base_network, f"{task_id}_output_layer"):
                        output_layer = getattr(currentNetwork.base_network, f"{task_id}_output_layer")
                        torch.nn.init.kaiming_normal_(output_layer.weight, nonlinearity='relu')
                        torch.nn.init.constant_(output_layer.bias, 0)
            print(f"Loaded pre-trained weights from {model_path_Network}. Freeze feature extractor: {freeze_feature_extractor}")

    def load_buffer_into_agent(self, agent, filename):
        # Load
        with open(filename, 'rb') as f:
            loaded_buffer = pickle.load(f)

        # Clear
        agent.replay_buffer.buffer.clear()

        # Transfer
        for i, experience in enumerate(loaded_buffer.buffer):
            state, action, reward, next_state, done = experience
            if isinstance(agent.replay_buffer, PrioritizedReplayBuffer):
                if hasattr(loaded_buffer, 'priorities'):
                    # Use the corresponding priority from the loaded buffer
                    td_error = loaded_buffer.priorities[i]
                    agent.replay_buffer.add(state, action, reward, next_state, done, td_error)
                else:
                    # DEBUG: REPLAY vs PRIO BUFFER changes - not rly needed if always using PER
                    default_td_error = 1.0
                    agent.replay_buffer.add(state, action, reward, next_state, done, default_td_error)
            else:
                # Standard replay buffer
                agent.replay_buffer.add(state, action, reward, next_state, done)
        # PER ?
        if isinstance(agent.replay_buffer, PrioritizedReplayBuffer) and isinstance(loaded_buffer, PrioritizedReplayBuffer):
            agent.replay_buffer.priorities[:len(loaded_buffer.buffer)] = loaded_buffer.priorities[:len(loaded_buffer.buffer)]


    def load_pretrained_weights_transfer_learning(self,
                                                model_path_QNetwork,
                                                model_path_target_QNetwork,
                                                model_path_target_Buffer,
                                                model_path_target_ICM = None,
                                                freeze_feature_extractor=False):

            # Agent
            self.load_network(model_path_QNetwork, self.agent.q_network, freeze_feature_extractor)

            # Target
            self.load_network(model_path_target_QNetwork, self.agent.target_q_network, freeze_feature_extractor)

            # ICM
            if self.agent.icm_bool:
                self.load_network(model_path_target_ICM,
                                  self.agent.icm,
                                  freeze_feature_extractor = False, # As ICM
                                    icm_update = True)

            # Buffer
            self.load_buffer_into_agent(self.agent, model_path_target_Buffer)

    def calculate_td_error(self, state, action, reward, next_state, done, task_id = None):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.agent.device)

        if self.pnn_bool:
            if task_id is None:
                task_id = self.agent.q_network.task_id
            with torch.no_grad():
                q_value = self.agent.q_network(state_tensor, task_id).gather(1, torch.LongTensor([[action]]).to(self.agent.device)).item()
                next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.agent.device)
                with torch.no_grad():
                    next_q_value = self.agent.target_q_network(next_state_tensor, task_id).max(1)[0].item()
                td_error = abs(reward + (1 - done) * self.agent.gamma * next_q_value - q_value)
        else:
            with torch.no_grad():
                q_value = self.agent.q_network(state_tensor).gather(1, torch.LongTensor([[action]]).to(self.agent.device)).item()
                next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.agent.device)
                with torch.no_grad():
                    next_q_value = self.agent.target_q_network(next_state_tensor).max(1)[0].item()
                td_error = abs(reward + (1 - done) * self.agent.gamma * next_q_value - q_value)
        return td_error
    
    def save_render(self, frames, episode):
        video_path = os.path.join(self.save_path, f"episode_{episode}.mp4")
        with imageio.get_writer(video_path, fps=30, macro_block_size=1) as video_writer:
            for frame in frames:
                if frame is not None:
                    video_writer.append_data(frame)
                else:
                    print("Warning.")
        print(f"Saved video for Episode {episode} at {video_path}")
    
    def preprocess_render_next_state(self, state):
        if state.shape[-1] == 3:
            state = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
        target_size = (self.input_shape[1], self.input_shape[2])
        state = cv2.resize(state, target_size)
        state = np.expand_dims(state, axis=0)        
        return state

    def training_loop(self):
        for episode in range(self.num_episodes):
            # 0. Reset the environment & select if rendering
            # OPTIONAL: Render the environment every "render_frequency" episodes
            render_episode = (episode + 1) % self.render_frequency == 0
            frames = []
            if render_episode:
                _, _ = self.env.reset()
                # Temporarily create a new environment with render_mode='human'
                render_env = gym.make(self.env.spec.id, render_mode='rgb_array')
                render_env.metadata['render_fps'] = self.fps
                wrapped_render_env = self.Wrapper(render_env)
                state, _ = wrapped_render_env.reset()
            else:
                state, _ = self.env.reset()

            total_reward = 0
            steps_in_episode = 0
            
            while True: # Loop through the episode until it is done
                # 2. Choose an action
                action = self.agent.act(state)

                # 3. Take a step in the environment
                if render_episode:
                    next_state, reward, terminated, truncated, info = render_env.step(action)
                    next_state = self.preprocess_render_next_state(next_state)
                    frame = render_env.render()  # Get the frame
                    frames.append(frame)
                else:
                    next_state, reward, terminated, truncated, info = self.env.step(action)

                done = terminated or truncated

                self.action_log.append(action)
                self.state_log.append(state)

                # 4. Add experience to the replay buffer
                if self.per_enabled:
                    td_error = self.calculate_td_error(state, action, reward, next_state, done)
                    self.agent.replay_buffer.add(state, action, reward, next_state, done, td_error)
                else:
                    self.agent.replay_buffer.add(state, action, reward, next_state, done)

                # Train the agent
                self.agent.train()

                state = next_state
                total_reward += reward
                steps_in_episode += 1

                if done:
                    print(f"Episode {episode + 1}/{self.num_episodes}, Total Reward: {total_reward}, Steps in Episode: {steps_in_episode}, Exploration Rate: {self.agent.epsilon}")
                    
                    self.number_of_steps_log.append(steps_in_episode)
                    self.episode_rewards.append(total_reward)

                    if render_episode:
                        self.save_render(frames, episode + 1)
                    break
                if render_episode:
                    render_env.close()
            # Just in case manually stopped training early lol.
            if (episode + 1) % self.model_save_interval == 0: 
                self.save_model()
                self.plot_rewards()
        self.save_model()
        self.plot_rewards()

    def plot_rewards(self):
        plt.figure(figsize=(12, 8))
        plt.plot(self.episode_rewards, label='Episode Rewards')
        plt.xlabel('Episode', fontsize=14)
        plt.ylabel('Total Reward', fontsize=14)
        plt.title('Rewards per Episode', fontsize=16)
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{self.save_path}\Rewards.png")

        if self.plot_bool:
            plt.show()
        else:
            plt.close()

    def save_model(self):
        model_path = f"{self.save_path}/Episode_{len(self.episode_rewards)}_Agent.pt"
        torch.save(self.agent.q_network.state_dict(), model_path)
        print(f"Model saved to {model_path}")

        model_path = f"{self.save_path}/Episode_{len(self.episode_rewards)}_Target.pt"
        torch.save(self.agent.target_q_network.state_dict(), model_path)
        print(f"Model saved to {model_path}")

        buffer = self.agent.replay_buffer
        buffer_path = f"{self.save_path}/Episode_{len(self.episode_rewards)}_ReplayBuffer.pkl"
        with open(buffer_path, 'wb') as f:
            pickle.dump(buffer, f)
        print(f"Replay Buffer saved to {buffer_path}")

        if hasattr(self.agent, 'icm') and self.agent.icm is not None:
            icm_path = f"{self.save_path}/Episode_{len(self.episode_rewards)}_ICM.pt"
            torch.save(self.agent.icm.state_dict(), icm_path)
            print(f"Model saved to {icm_path}")

        rewards_path = f"{self.save_path}/Episode_{len(self.episode_rewards)}_Rewards.txt"
        with open(rewards_path, "w") as file:
            for reward in self.episode_rewards:
                file.write(str(reward) + "\n")


    def load_model(self, path):
        self.agent.q_network.load_state_dict(torch.load(path, map_location=self.agent.device))
        self.agent.target_q_network.load_state_dict(self.agent.q_network.state_dict())
        print(f"Model loaded from {path}")