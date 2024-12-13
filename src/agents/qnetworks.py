import torch
import torch.nn as nn
import torch.nn.functional as F

class PolicyNetwork(nn.Module):
    '''
    This class represents the policy network; essentially an actor.
    params:
    state_dim: dimension of the state space [int]
    action_dim: dimension of the action space [int]
    hidden_dim: dimension of the hidden layers [int]
    stochastic: whether the policy is stochastic or deterministic [bool]

    returns:
    mean: mean of the action distribution [tensor]
    std: standard deviation of the action distribution [tensor]
    '''
    def __init__(self,
                 state_dim,
                 action_dim,
                 hidden_dim=256,
                 stochastic=True):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        self.stochastic = stochastic
    
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mean = self.mean(x)
        
        if self.stochastic:
            log_std = self.log_std(x)
            std = torch.exp(log_std)
            return mean, std
        else:
            return mean
        
# Critic tings

class BaseCriticNetwork(nn.Module):
    '''
    This class represents the base critic network.
    params:
    state_dim: dimension of the state space [int]
    action_dim: dimension of the action space [int]
    hidden_dim: dimension of the hidden layers [int]
    action_dim_size_output: [bool] - determines whether complete critic or has other additional output

    returns:
    x: output of the critic network [tensor]
    '''
    def __init__(self, state_dim, action_dim, hidden_dim=256, action_dim_size_output = False):
        super(BaseCriticNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        if action_dim_size_output:
            self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward_base(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        if hasattr(self, 'fc3'):
            x = self.fc3(x)
        return x

class DoubleCriticNetwork(BaseCriticNetwork):
    '''
    This class implements a double critic network.
    params:
    state_dim: dimension of the state space [int]
    action_dim: dimension of the action space [int]
    hidden_dim: dimension of the hidden layers [int]

    returns:
    Q_1: Q-function of the first critic [tensor]
    Q_2: Q-function of the second critic [tensor]
    '''
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(DoubleCriticNetwork, self).__init__(state_dim, action_dim, hidden_dim)
        self.Q_1 = nn.Linear(hidden_dim, 1)
        self.Q_2 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        x = self.forward_base(state, action)
        Q_1 = self.Q_1(x)
        Q_2 = self.Q_2(x)
        return Q_1, Q_2

class DistributionalCriticNetwork(BaseCriticNetwork):
    '''
    This class implements a distributional critic network.
    params:
    state_dim: dimension of the state space [int]
    action_dim: dimension of the action space [int]
    hidden_dim: dimension of the hidden layers [int]
    num_points: number of points in the distribution [int]
    support_range: range of the support [tuple]

    returns:
    dist: probability distribution over the points [tensor]

    notes:
    support is a tensor showing the bins of the distribution (e.g. [-5, -4, -3, ..., 3, 4, 5])
    -5 is the first bin, -4 is the second bin, etc.
    The bins represent the value of the Q-function at that point.
    '''
    def __init__(self, state_dim, action_dim, hidden_dim=256, num_points=51, support_range=(-10, 10)):
        super(DistributionalCriticNetwork, self).__init__(state_dim, action_dim, hidden_dim)
        self.num_points = num_points
        self.support_range = support_range
        self.delta_z = (support_range[1] - support_range[0]) / (num_points - 1)
        self.support = torch.linspace(support_range[0], support_range[1], num_points)
        self.distributional_output = nn.Linear(hidden_dim, num_points)

    def forward(self, state, action):
        x = self.forward_base(state, action) # Basic critic pass
        distributional_output = F.softmax(self.distributional_output(x), dim=1) # Convert to probability distributions
        dist = distributional_output.clamp(min=1e-3) # Clamped to prevent numerical instabilities
        return dist # Probability distribution over the points
    
class DoubleDistributionalCriticNetwork(BaseCriticNetwork):
    '''
    This class implements a double distributional critic network.
    params:
    state_dim: dimension of the state space [int]
    action_dim: dimension of the action space [int]
    hidden_dim: dimension of the hidden layers [int]
    num_points: number of points in the distribution [int]
    support_range: range of the support [tuple]

    returns:
    dist_1: probability distribution of the first critic [tensor]
    dist_2: probability distribution of the second critic [tensor]
    '''
    def __init__(self, state_dim, action_dim, hidden_dim=256, num_points=51, support_range=(-10, 10)):
        super(DoubleDistributionalCriticNetwork, self).__init__(state_dim, action_dim, hidden_dim)
        self.num_points = num_points
        self.support_range = support_range
        self.delta_z = (support_range[1] - support_range[0]) / (num_points - 1)
        self.support = torch.linspace(support_range[0], support_range[1], num_points)
        self.distributional_output_1 = nn.Linear(hidden_dim, num_points)
        self.distributional_output_2 = nn.Linear(hidden_dim, num_points)

    def forward(self, state, action):
        x = self.forward_base(state, action) # Basic critic pass
        dist_1 = F.softmax(self.distributional_output_1(x), dim=1) # Convert to probability distributions
        dist_2 = F.softmax(self.distributional_output_2(x), dim=1) # Convert to probability distributions
        dist_1 = dist_1.clamp(min=1e-3) # Clamped to prevent numerical instabilities
        dist_2 = dist_2.clamp(min=1e-3) # Clamped to prevent numerical instabilities
        return dist_1, dist_2
