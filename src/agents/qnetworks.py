import jax.numpy as jnp
import jax
from typing import Tuple

class PolicyNetwork:
    '''
    This class represents the policy network; essentially an actor.
    params:
    state_dim: dimension of the state space [int]
    action_dim: dimension of the action space [int]
    hidden_dim: dimension of the hidden layers [int]
    stochastic: whether the policy is stochastic or deterministic [bool]

    returns:
    mean: mean of the action distribution [jnp.ndarray]
    std: standard deviation of the action distribution [jnp.ndarray]
    '''
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 hidden_dim: int=256,
                 stochastic: bool=True,
                 rng_key: jax.random.PRNGKey=jax.random.PRNGKey(0)):
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

    def forward(self,
                state: jnp.ndarray) -> jnp.ndarray:
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
    x: output of the critic network [jnp.ndarray]
    '''
    def __init__(self,
                state_dim: int,
                action_dim: int,
                hidden_dim: int=256):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.params = {
            "fc1": jax.random.normal(jax.random.PRNGKey(0), (state_dim + action_dim, hidden_dim)),
            "fc2": jax.random.normal(jax.random.PRNGKey(1), (hidden_dim, hidden_dim))
        }

    def forward_base(self,
                    state: jnp.ndarray,
                    action: jnp.ndarray) -> jnp.ndarray:
        x = jnp.concatenate([state, action], axis=-1)
        x = jax.nn.relu(jnp.dot(x, self.params["fc1"]))
        x = jax.nn.relu(jnp.dot(x, self.params["fc2"]))
        return x
    
class BaseCriticNetwork(CriticNetwork):
    '''
    No fancy stuff
    '''
    def __init__(self,
                state_dim: int,
                action_dim: int,
                hidden_dim: int=256):
        super().__init__(state_dim, action_dim, hidden_dim)
        self.params.update({
            "Q": jax.random.normal(jax.random.PRNGKey(2), (hidden_dim, 1))
        })

    def forward(self,
                state: jnp.ndarray,
                action: jnp.ndarray) -> jnp.ndarray:
        x = self.forward_base(state, action)
        Q = jnp.dot(x, self.params["Q"])
        return Q        
    
    def compute_td_target(self,
                        reward: jnp.ndarray,
                        next_states: jnp.ndarray,
                        next_actions: jnp.ndarray,
                        not_done: jnp.ndarray,
                        gamma: float) -> jnp.ndarray:
        # Basic:  Q(s, a) = r + \gamma * Q(s', a')
        '''
        This function calculates the TD target for the critic.

        params:
        reward: Reward [jnp.ndarray]
        next_states: Next state [jnp.ndarray]
        next_actions: Next action [jnp.ndarray]
        not_done: Not done flag [jnp.ndarray]
        gamma: Discount factor [float]

        returns:
        td_target: TD target [jnp.ndarray]
        '''
        next_q_value = self.forward(next_states, next_actions)
        td_target = reward + gamma * not_done * next_q_value
        return td_target
    
    def compute_loss(self,
                    states: jnp.ndarray,
                    actions: jnp.ndarray,
                    target_q_value: jnp.ndarray) -> float:
        # Get current Q-values from the critic for the sampled state-action pairs
        q_value = self.critic_network.forward(states, actions)

        # Compute the critic loss (MSE between current Q-values and the TD target)
        critic_loss = jnp.mean((q_value - target_q_value) ** 2)
        return critic_loss

class DoubleCriticNetwork(CriticNetwork):
    '''
    This class implements a double critic network.
    params:
    state_dim: dimension of the state space [int]
    action_dim: dimension of the action space [int]
    hidden_dim: dimension of the hidden layers [int]

    returns:
    Q_1: Q-function of the first critic [jnp.ndarray]
    Q_2: Q-function of the second critic [jnp.ndarray]
    '''
    def __init__(self,
                state_dim: int,
                action_dim: int,
                hidden_dim: int=256):
        super().__init__(state_dim, action_dim, hidden_dim)
        self.params.update({
            "Q_1": jax.random.normal(jax.random.PRNGKey(2), (hidden_dim, 1)),
            "Q_2": jax.random.normal(jax.random.PRNGKey(3), (hidden_dim, 1))
        })

    def forward(self,
                state: jnp.ndarray,
                action: jnp.ndarray) -> jnp.ndarray:
        x = self.forward_base(state, action)
        Q_1 = jnp.dot(x, self.params["Q_1"])
        Q_2 = jnp.dot(x, self.params["Q_2"])
        return Q_1, Q_2
    
    # Double: Q(s,a) = r + \gamma * min(Q_1(s', a'), Q_2(s', a'))
    def compute_td_target(self,
                        reward: jnp.ndarray,
                        next_states: jnp.ndarray,
                        next_actions: jnp.ndarray,
                        not_done: jnp.ndarray,
                        gamma: float) -> jnp.ndarray:
        next_q1, next_q2 = self.forward(next_states, next_actions)
        next_q_min = jnp.minimum(next_q1, next_q2)
        td_target = reward + gamma * not_done * next_q_min
        return td_target
    
    def compute_loss(self,
                    states: jnp.ndarray,
                    actions: jnp.ndarray,
                    target_q_value: jnp.ndarray) -> float:
        # Get current Q-values from the critic for the sampled state-action pairs
        Q_1, Q_2 = self.critic_network.forward(states, actions)

        # Compute the critic loss (MSE between current Q-values and the TD target)
        critic_loss = jnp.mean((Q_1 - target_q_value) ** 2) + jnp.mean((Q_2 - target_q_value) ** 2)
        return critic_loss

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
    dist: probability distribution over the points [jnp.ndarray]

    notes:
    support is a tensor showing the bins of the distribution (e.g. [-5, -4, -3, ..., 3, 4, 5])
    -5 is the first bin, -4 is the second bin, etc.
    The bins represent the value of the Q-function at that point.
    '''
    def __init__(self,
                state_dim: int,
                action_dim: int,
                hidden_dim: int=256,
                num_points: int=51,
                support_range: tuple=(-10, 10)):
        super().__init__(state_dim, action_dim, hidden_dim)
        self.num_points = num_points
        self.support_range = support_range
        self.delta_z = (support_range[1] - support_range[0]) / (num_points - 1)
        self.support = jnp.linspace(support_range[0], support_range[1], num_points)
        self.params.update({
            "distributional_output": jax.random.normal(jax.random.PRNGKey(4), (hidden_dim, num_points))
        })

    def forward(self,
                state: jnp.ndarray,
                action: jnp.ndarray) -> jnp.ndarray:
        '''
        This function calculates the probability distribution of the critic.
        params:
        state: state input [jnp.ndarray]
        action: action input [jnp.ndarray]

        returns:
        dist: probability distribution over the points [jnp.ndarray]
        '''
        x = self.forward_base(state, action)
        distributional_output = jax.nn.softmax(jnp.dot(x, self.params["distributional_output"]), axis=-1)
        dist = jnp.clip(distributional_output, a_min=1e-3)
        return dist
    
    # Distributional: Q_Z(s, a) = r + \gamma * (1 - done) * z_i
    def compute_td_target(self,
                        reward: jnp.ndarray,
                        next_states: jnp.ndarray,
                        next_actions: jnp.ndarray,
                        not_done: jnp.ndarray,
                        gamma: float) -> jnp.ndarray:
        '''
        Compute the TD target for a distributional critic.

        params:
        reward: Reward from the sampled transition [jnp.ndarray]
        next_states: Next state from the sampled transition [jnp.ndarray]
        next_actions: Next action from the sampled transition [jnp.ndarray]
        not_done: Not done flags indicating whether episodes are ongoing [jnp.ndarray]
        gamma: Discount factor for future rewards [float]

        returns:
        projected_dist: Projected probability distribution for the TD target [jnp.ndarray]

        steps:
        - Compute the next action using the target policy.
        - Get the distributional Q-values for the next state-action pair.
        - Compute a "projected" distribution that aligns with the critic's fixed support.
        '''
        # Get the distributional Q-values.
        next_dist = self.forward(next_states, next_actions)  

        # Initialize the projected distribution with zeros.
        projected_dist = jnp.zeros_like(next_dist)

        # Iterate over each point in the fixed support of the distribution.
        for i, z_i in enumerate(self.support):
            # Compute the target value (tz) for each point in the support.
            tz = reward + gamma * not_done * z_i

            # Clamp the target value to ensure it stays within the range of the support.
            tz = jnp.clip(tz, self.support[0], self.support[-1])

            # Map the target value (tz) to the corresponding bin in the fixed support.
            b = (tz - self.support[0]) / self.delta_z

            # Determine the indices of the lower and upper bins.
            lower, upper = jnp.floor(b).astype(int), jnp.ceil(b).astype(int)

            # Distribute the probability mass across the lower and upper bins proportionally.
            projected_dist[:, lower] += next_dist[:, i] * (upper - b)
            projected_dist[:, upper] += next_dist[:, i] * (b - lower)

        return projected_dist
    
    def compute_loss(self,
                    states: jnp.ndarray,
                    actions: jnp.ndarray,
                    target_distribution: jnp.ndarray) -> float:
        '''
        Compute the loss for a distributional critic.
        params:
        states: States from the sampled transitions [jnp.ndarray]
        actions: Actions from the sampled transitions [jnp.ndarray]
        target_distribution: Target probability distribution [jnp.ndarray]

        returns:
        critic_loss: Loss value for the distributional critic [float]
        '''
        # Get the predicted distributional Q-values for the current state-action pairs
        predicted_distribution = self.forward(states, actions)

        # Ensure numerical stability by adding a small epsilon to avoid log(0)
        epsilon = 1e-6
        predicted_distribution = jnp.clip(predicted_distribution, a_min=epsilon, a_max=1.0)

        # Compute the KL divergence between the target and predicted distributions
        critic_loss = jnp.sum(target_distribution * jnp.log(target_distribution / predicted_distribution), axis=-1)
        
        # Return the mean loss across the batch
        return jnp.mean(critic_loss)



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
    dist_1: probability distribution of the first critic [jnp.ndarray]
    dist_2: probability distribution of the second critic [jnp.ndarray]
    '''
    def __init__(self,
                state_dim: int,
                action_dim: int,
                hidden_dim: int=256,
                num_points: int=51,
                support_range: tuple=(-10, 10)):
        super().__init__(state_dim, action_dim, hidden_dim)
        self.num_points = num_points
        self.support_range = support_range
        self.delta_z = (support_range[1] - support_range[0]) / (num_points - 1)
        self.support = jnp.linspace(support_range[0], support_range[1], num_points)
        self.params.update({
            "distributional_output_1": jax.random.normal(jax.random.PRNGKey(5), (hidden_dim, num_points)),
            "distributional_output_2": jax.random.normal(jax.random.PRNGKey(6), (hidden_dim, num_points))
        })

    def forward(self,
                state: jnp.ndarray,
                action: jnp.ndarray) -> jnp.ndarray:
        '''
        This function calculates the probability distribution of the critic.
        params:
        state: state input [jnp.ndarray]
        action: action input [jnp.ndarray]

        returns:
        dist_1: probability distribution of the first critic [jnp.ndarray]
        dist_2: probability distribution of the second critic [jnp.ndarray]

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

    # Double Distributional: Q_Z(s, a) = r + \gamma * (1 - done) * min(z_1', z_2')
    def compute_td_target(self,
                        reward: jnp.ndarray,
                        next_states: jnp.ndarray,
                        next_actions: jnp.ndarray,
                        not_done: jnp.ndarray,
                        gamma: float) -> jnp.ndarray:
        '''
        Compute the TD target for a DOUBLE distributional critic.
        '''
        # Get the distributional Q-values.
        next_dist_1, next_dist_2 = self.forward(next_states, next_actions)  

        # Minimum distribution
        next_dist_min = jnp.minimum(next_dist_1, next_dist_2)

        # Initialize the projected distribution with zeros.
        projected_dist = jnp.zeros_like(next_dist_min)

        # Iterate over each point in the fixed support of the distribution.
        for i, z_i in enumerate(self.support):
            # Compute the target value (tz) for each point in the support.
            tz = reward + gamma * not_done * z_i

            # Clamp the target value to ensure it stays within the range of the support.
            tz = jnp.clip(tz, self.support[0], self.support[-1])

            # Map the target value (tz) to the corresponding bin in the fixed support.
            b = (tz - self.support[0]) / self.delta_z

            # Determine the indices of the lower and upper bins.
            lower, upper = jnp.floor(b).astype(int), jnp.ceil(b).astype(int)

            # Distribute the probability mass across the lower and upper bins proportionally.
            projected_dist[:, lower] += next_dist_min[:, i] * (upper - b)
            projected_dist[:, upper] += next_dist_min[:, i] * (b - lower)

        return projected_dist

    def compute_loss(self,
                    states: jnp.ndarray,
                    actions: jnp.ndarray,
                    target_distribution: jnp.ndarray) -> float:
        '''
        Compute the loss for a double distributional critic.
        '''
        # Get the predicted distributional Q-values for the current state-action pairs
        predicted_distribution_1, predicted_distribution_2 = self.forward(states, actions)

        # Ensure numerical stability by adding a small epsilon to avoid log(0)
        epsilon = 1e-6
        predicted_distribution_1 = jnp.clip(predicted_distribution_1, a_min=epsilon, a_max=1.0)
        predicted_distribution_2 = jnp.clip(predicted_distribution_2, a_min=epsilon, a_max=1.0)

        # Compute the KL divergence between the target and predicted distributions
        critic_loss = jnp.sum(target_distribution * jnp.log(target_distribution / predicted_distribution_1), axis=-1) + \
                      jnp.sum(target_distribution * jnp.log(target_distribution / predicted_distribution_2), axis=-1)
        
        # Return the mean loss across the batch
        return jnp.mean(critic_loss)