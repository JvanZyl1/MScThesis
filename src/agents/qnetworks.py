import jax.numpy as jnp
import jax

class PolicyNetwork:
    '''
    This class represents the policy network; essentially an actor.
    params:
    state_dim: dimension of the state space [int]
    action_dim: dimension of the action space [int]
    hidden_dim: dimension of the hidden layers [int]
    stochastic: whether the policy is stochastic or deterministic [bool]

    returns:
    mean: mean of the action distribution [jnp.array]
    std: standard deviation of the action distribution [jnp.array]
    '''
    def __init__(self,
                 state_dim,
                 action_dim,
                 hidden_dim=256,
                 stochastic=True,
                 rng_key=None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.stochastic = stochastic
        self.key = rng_key
        self.params = {
            "fc1": jax.random.normal(rng_key, (state_dim, hidden_dim)),
            "fc2": jax.random.normal(rng_key, (hidden_dim, hidden_dim)),
            "mean": jax.random.normal(rng_key, (hidden_dim, action_dim)),
            "log_std": jax.random.normal(rng_key, (hidden_dim, action_dim))
        }

    def forward(self, state):
        x = jax.nn.relu(jnp.dot(state, self.params["fc1"]))
        x = jax.nn.relu(jnp.dot(x, self.params["fc2"]))
        mean = jnp.dot(x, self.params["mean"])

        if self.stochastic:
            log_std = jnp.dot(x, self.params["log_std"])
            std = jnp.exp(log_std)
            return mean, std
        else:
            return mean

class CriticNetwork:
    '''
    This class represents the critic network, to be used by others
    params:
    state_dim: dimension of the state space [int]
    action_dim: dimension of the action space [int]
    hidden_dim: dimension of the hidden layers [int]

    returns:
    x: output of the critic network [jnp.array]
    '''
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.params = {
            "fc1": jax.random.normal(jax.random.PRNGKey(0), (state_dim + action_dim, hidden_dim)),
            "fc2": jax.random.normal(jax.random.PRNGKey(1), (hidden_dim, hidden_dim))
        }

    def forward_base(self, state, action):
        x = jnp.concatenate([state, action], axis=-1)
        x = jax.nn.relu(jnp.dot(x, self.params["fc1"]))
        x = jax.nn.relu(jnp.dot(x, self.params["fc2"]))
        return x
    
class BaseCriticNetwork(CriticNetwork):
    '''
    No fancy stuff
    '''
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        
    
    def compute_td_target(reward, next_state, not_done, target_critic, gamma):
        # Basic:  Q(s, a) = r + \gamma * Q(s', a')
        '''
        params:
        reward: Reward [jnp.array]
        next_state: Next state [jnp.array]
        not_done: Not done flag [jnp.array]
        target_critic: Target critic network [CriticNetwork]
        gamma: Discount factor [float]
        '''
        next_action = target_critic.policy(next_state)
        next_q_value = target_critic.forward(next_state, next_action)
        td_target = reward + gamma * not_done * next_q_value
        return td_target

class DoubleCriticNetwork(CriticNetwork):
    '''
    This class implements a double critic network.
    params:
    state_dim: dimension of the state space [int]
    action_dim: dimension of the action space [int]
    hidden_dim: dimension of the hidden layers [int]

    returns:
    Q_1: Q-function of the first critic [jnp.array]
    Q_2: Q-function of the second critic [jnp.array]
    '''
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__(state_dim, action_dim, hidden_dim)
        self.params.update({
            "Q_1": jax.random.normal(jax.random.PRNGKey(2), (hidden_dim, 1)),
            "Q_2": jax.random.normal(jax.random.PRNGKey(3), (hidden_dim, 1))
        })

    def forward(self, state, action):
        x = self.forward_base(state, action)
        Q_1 = jnp.dot(x, self.params["Q_1"])
        Q_2 = jnp.dot(x, self.params["Q_2"])
        return Q_1, Q_2

class DistributionalCriticNetwork(CriticNetwork):
    '''
    This class implements a distributional critic network.
    params:
    state_dim: dimension of the state space [int]
    action_dim: dimension of the action space [int]
    hidden_dim: dimension of the hidden layers [int]
    num_points: number of points in the distribution [int]
    support_range: range of the support [tuple]

    returns:
    dist: probability distribution over the points [jnp.array]

    notes:
    support is a tensor showing the bins of the distribution (e.g. [-5, -4, -3, ..., 3, 4, 5])
    -5 is the first bin, -4 is the second bin, etc.
    The bins represent the value of the Q-function at that point.
    '''
    def __init__(self, state_dim, action_dim, hidden_dim=256, num_points=51, support_range=(-10, 10)):
        super().__init__(state_dim, action_dim, hidden_dim)
        self.num_points = num_points
        self.support_range = support_range
        self.delta_z = (support_range[1] - support_range[0]) / (num_points - 1)
        self.support = jnp.linspace(support_range[0], support_range[1], num_points)
        self.params.update({
            "distributional_output": jax.random.normal(jax.random.PRNGKey(4), (hidden_dim, num_points))
        })

    def forward(self, state, action):
        '''
        This function calculates the probability distribution of the critic.
        params:
        state: state input [jnp.array]
        action: action input [jnp.array]

        returns:
        dist: probability distribution over the points [jnp.array]
        '''
        x = self.forward_base(state, action)
        distributional_output = jax.nn.softmax(jnp.dot(x, self.params["distributional_output"]), axis=-1)
        dist = jnp.clip(distributional_output, a_min=1e-3)
        return dist

class DoubleDistributionalCriticNetwork(CriticNetwork):
    '''
    This class implements a double distributional critic network.
    params:
    state_dim: dimension of the state space [int]
    action_dim: dimension of the action space [int]
    hidden_dim: dimension of the hidden layers [int]
    num_points: number of points in the distribution [int]
    support_range: range of the support [tuple]

    returns:
    dist_1: probability distribution of the first critic [jnp.array]
    dist_2: probability distribution of the second critic [jnp.array]
    '''
    def __init__(self, state_dim, action_dim, hidden_dim=256, num_points=51, support_range=(-10, 10)):
        super().__init__(state_dim, action_dim, hidden_dim)
        self.num_points = num_points
        self.support_range = support_range
        self.delta_z = (support_range[1] - support_range[0]) / (num_points - 1)
        self.support = jnp.linspace(support_range[0], support_range[1], num_points)
        self.params.update({
            "distributional_output_1": jax.random.normal(jax.random.PRNGKey(5), (hidden_dim, num_points)),
            "distributional_output_2": jax.random.normal(jax.random.PRNGKey(6), (hidden_dim, num_points))
        })

    def forward(self, state, action):
        '''
        This function calculates the probability distribution of the critic.
        params:
        state: state input [jnp.array]
        action: action input [jnp.array]

        returns:
        dist_1: probability distribution of the first critic [jnp.array]
        dist_2: probability distribution of the second critic [jnp.array]

        notes:
        The probability distribution is calculated using the jax.nn.softmax function.
        Also, clipping is done to avoid numerical instability.

        '''
        x = self.forward_base(state, action)
        dist_1 = jax.nn.softmax(jnp.dot(x, self.params["distributional_output_1"]), axis=-1)
        dist_2 = jax.nn.softmax(jnp.dot(x, self.params["distributional_output_2"]), axis=-1)
        dist_1 = jnp.clip(dist_1, a_min=1e-3)
        dist_2 = jnp.clip(dist_2, a_min=1e-3)
        return dist_1, dist_2
