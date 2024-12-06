import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class BaseFeatureExtractor(nn.Module):
    def __init__(self, input_shape):
        super(BaseFeatureExtractor, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        self.input_shape = input_shape
        
        self._conv_output_size = self.outputsizeofconvolution(input_shape)
        self.fc1 = nn.Linear(self._conv_output_size, 512)
        
    def outputsizeofconvolution(self, shape):
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
    def __init__(self, feature_extractor, num_actions, freeze_feature_extractor=False):
        super(QNetwork, self).__init__()
        self.feature_extractor = feature_extractor
        if freeze_feature_extractor:
            self.freeze_layers(self.feature_extractor)
        self.fc2 = nn.Linear(512, num_actions) # 512, IF CHANGES CHANGE THIS

    def freeze_layers(self, module):
        for param in module.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.feature_extractor(x)
        return self.fc2(x)

# Dueling DQN
class DuelingQNetwork(nn.Module):
    def __init__(self, feature_extractor, num_actions, freeze_feature_extractor=False):
        super(DuelingQNetwork, self).__init__()
        self.feature_extractor = feature_extractor
        if freeze_feature_extractor:
            self.freeze_layers(self.feature_extractor)
        self.value_stream = nn.Linear(512, 1)
        self.advantage_stream = nn.Linear(512, num_actions)

    def freeze_layers(self, module):
        for param in module.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.feature_extractor(x)
        value = self.value_stream(x)                # V(s)
        advantage = self.advantage_stream(x)        # A(s, a)
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values


### PNN : Progressive Neural Networks ###
class PNNQNetwork(nn.Module):
    def __init__(self, feature_extractor, num_actions, num_tasks, task_id):
        super(PNNQNetwork, self).__init__()
        self.base_network = QNetwork(feature_extractor, num_actions)
        self.num_tasks = num_tasks
        self.task_id = str(task_id)

        # Create task-specific output layers for each task
        for task in range(1, num_tasks + 1):
            setattr(self.base_network, f"{task}_output_layer", nn.Linear(512, num_actions))


    def forward(self, x, task_id=None):
        if task_id is not None:
            self.task_id = str(task_id)
        x = self.base_network.feature_extractor(x)
        q_values = getattr(self.base_network, f"{self.task_id}_output_layer")(x)
        return q_values

class PNNDuelingQNetwork(nn.Module):
    def __init__(self, feature_extractor, num_actions, num_tasks, task_id):
        super(PNNDuelingQNetwork, self).__init__()
        self.base_network = DuelingQNetwork(feature_extractor, num_actions)
        self.num_tasks = num_tasks

        for task in range(1, num_tasks + 1):
            setattr(self.base_network, f"{task}_value_stream", nn.Linear(512, 1))
            setattr(self.base_network, f"{task}_advantage_stream", nn.Linear(512, num_actions))
        self.task_id = str(task_id)

    def forward(self, x, task_id=None):
        if task_id is not None:
            self.task_id = str(task_id)
        x = self.base_network.feature_extractor(x)
        value = getattr(self.base_network, f"{self.task_id}_value_stream")(x)
        advantage = getattr(self.base_network, f"{self.task_id}_advantage_stream")(x)
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values
    

class QNetwork_NoOutput(nn.Module):
    def __init__(self, feature_extractor):
        super(QNetwork_NoOutput, self).__init__()
        self.feature_extractor = feature_extractor

    def forward(self, x):
        x = self.feature_extractor(x)
        return x  # Output layer handled by PNNFlexibleQNetwork

class PNNFlexibleQNetwork(nn.Module):
    def __init__(self, feature_extractor, num_tasks, task_architectures, task_num_actions):
        super(PNNFlexibleQNetwork, self).__init__()
        self.base_network = QNetwork_NoOutput(feature_extractor)
        self.num_tasks = num_tasks
        self.task_architectures = task_architectures  # List indicating 'dueling' or 'normal' for each task
        self.task_num_actions = task_num_actions  # List of num_actions for each task
        for task_id in range(1, num_tasks + 1):
            num_actions = task_num_actions[task_id - 1]
            if self.task_architectures[task_id - 1] == 'dueling':
                setattr(self.base_network, f"{task_id}_value_stream", nn.Linear(512, 1))
                setattr(self.base_network, f"{task_id}_advantage_stream", nn.Linear(512, num_actions))
            else:
                setattr(self.base_network, f"{task_id}_output_layer", nn.Linear(512, num_actions))

        self.task_id = None

    def forward(self, x, task_id=None):
        if task_id is not None:
            self.task_id = task_id
        task_id = str(self.task_id)
        x = self.base_network(x)
        if self.task_architectures[int(task_id) - 1] == 'dueling':
            value = getattr(self.base_network, f"{task_id}_value_stream")(x)
            advantage = getattr(self.base_network, f"{task_id}_advantage_stream")(x)
            q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        else:
            q_values = getattr(self.base_network, f"{task_id}_output_layer")(x)
        return q_values