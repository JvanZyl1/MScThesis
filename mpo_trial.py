import jax.numpy as jnp
import jax
from flax import linen as nn
jax.clear_caches()
from collections import deque
from typing import Tuple
import scipy

# Helper functions
@jax.jit
def kl_divergence_multivariate_gaussian(mean_p: jnp.ndarray,
                                        std_p: jnp.ndarray,
                                        mean_q: jnp.ndarray,
                                        std_q: jnp.ndarray) -> jnp.ndarray:
    '''
    Compute the KL divergence between two multivariate Gaussian distributions
    with diagonal covariance matrices.

    params:
    mean_p: Mean of distribution P [jnp.ndarray]
    std_p: Standard deviation of distribution P [jnp.ndarray]
    mean_q: Mean of distribution Q [jnp.ndarray]
    std_q: Standard deviation of distribution Q [jnp.ndarray]

    returns:
    kl: KL divergence between P and Q [float]

    notes:
    The KL divergence for multivariate Gaussians with diagonal covariance is:
    D_KL(P || Q) = log(std_q) - log(std_p) + (std_p^2 + (mu_p - mu_q)^2) / (2 * std_q^2) - 0.5
    '''
    log_std_p = jnp.log(std_p)
    log_std_q = jnp.log(std_q)
    variance_p = std_p ** 2
    variance_q = std_q ** 2

    # Compute KL divergence
    kl = (
        log_std_q - log_std_p  # Log std ratio
        + (variance_p + (mean_p - mean_q) ** 2) / (2 * variance_q)  # Variance and mean term
        - 0.5  # Normalization constant
    )
    return kl.sum(axis=-1)  # Sum over action dimensions

@jax.jit
def gaussian_likelihood(actions: jnp.ndarray,
                        mean: jnp.ndarray,
                        std: jnp.ndarray) -> jnp.ndarray:
    '''
    Compute the log likelihood of actions under a Gaussian distribution.

    params:
    actions: Sampled actions [jnp.ndarray]
    mean: Mean of the Gaussian distribution [jnp.ndarray]
    std: Standard deviation of the Gaussian distribution [jnp.ndarray]

    returns:
    log_prob: Log likelihood of the actions [jnp.ndarray]

    notes:
    The Gaussian log likelihood is given by:
    log_prob = -0.5 * ((actions - mean)^2 / (std^2) + 2 * log(std) + log(2 * pi))
    '''
    log_prob = -0.5 * (
        ((actions - mean) ** 2) / (std ** 2)  # Quadratic term
        + 2 * jnp.log(std)  # Log scale normalization
        + jnp.log(2 * jnp.pi)  # Constant factor
    )
    return log_prob.sum(axis=-1)  # Sum over the action dimensions

@jax.jit
def clip_grads(grads: jnp.ndarray, max_norm: float) -> jnp.ndarray:
    """
    Clips gradients by their global norm to ensure stability.

    Args:
        grads: Gradients to clip, as a PyTree.
        max_norm: Maximum norm for the gradients.

    Returns:
        Clipped gradients with the same structure as the input.
    """
    # Compute the global norm
    norm = jnp.sqrt(sum(jnp.sum(jnp.square(x)) for x in jax.tree_util.tree_leaves(grads)))
    # Compute scaling factor
    scale = jnp.minimum(1.0, max_norm / (norm + 1e-6))
    # Scale gradients
    clipped_grads = jax.tree_util.tree_map(lambda x: x * scale, grads)
    return clipped_grads



class Actor(nn.Module):
    """
    This class represents the policy network; essentially an actor.

    Attributes:
    state_dim: dimension of the state space [int]
    action_dim: dimension of the action space [int]
    hidden_dim: dimension of the hidden layers [int]
    stochastic: whether the policy is stochastic or deterministic [bool]

    Returns:
    mean: mean of the action distribution [jnp.ndarray]
    std: standard deviation of the action distribution [jnp.ndarray]
    
    Example usage:
    # Initialize the model
    actor = Actor(state_dim=state_dim, action_dim=action_dim, hidden_dim=hidden_dim, stochastic=stochastic)

    # Example input state
    state = jnp.ones((1, state_dim))

    # Initialize parameters
    params = actor.init(rng, state)

    # Get mean and std from the policy network; through applying params which are the weights and bias' of the network.
    mean, std = actor.apply(params, state)
    """
    state_dim: int
    action_dim: int
    hidden_dim: int = 256
    stochastic: bool = True

    @nn.compact
    def __call__(self,
                 state : jnp.ndarray) -> jnp.ndarray:
        # Hidden layers
        x = nn.Dense(self.hidden_dim)(state)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)

        # Output layers for mean and std deviation
        mean = nn.Dense(self.action_dim)(x)
        if self.stochastic:
            log_std = self.param("log_std", lambda rng, shape: -jnp.ones(shape), (self.action_dim,))
            std = jnp.exp(log_std)
        else:
            std = jnp.zeros_like(mean)

        return mean, std

class DoubleDistributionalCritic(nn.Module):
    """
    Double distributional critic module.

    Attributes:
    state_dim: Dimension of the state space [int]
    action_dim: Dimension of the action space [int]
    hidden_dim: Dimension of the hidden layers [int]
    num_points: number of points in the distribution [int]
    v_min: Minimum value for the critic distribution [float]
    v_max: Maximum value for the critic distribution [float]
    activation_fn: Activation function to use [callable]

    Returns:
    q1: First distributional Q-value (logits over points) [jnp.ndarray]
    q2: Second distributional Q-value (logits over points) [jnp.ndarray]
    z: Support of the distribution [jnp.ndarray]

    Notes:
    support is a tensor showing the bins of the distribution (e.g. [-5, -4, -3, ..., 3, 4, 5])
    -5 is the first bin, -4 is the second bin, etc.
    The bins represent the value of the Q-function at that point.

    v_min, v_max is the range for the support
    """
    state_dim: int
    action_dim: int
    hidden_dim: int = 256
    num_points: int = 51
    v_min: float = -10.0
    v_max: float = 10.0
    activation_fn: callable = nn.relu

    @nn.compact
    def __call__(self,
                 state : jnp.ndarray,
                 action : jnp.ndarray) -> jnp.ndarray:
        # Concatenate state and action
        x = jnp.concatenate([state, action], axis=-1)

        # Define support of the distribution
        z = jnp.linspace(self.v_min, self.v_max, self.num_points)

        # First critic network (Q1)
        q1 = nn.Dense(self.hidden_dim)(x)
        q1 = self.activation_fn(q1)
        q1 = nn.Dense(self.hidden_dim)(q1)
        q1 = self.activation_fn(q1)
        q1 = nn.Dense(self.num_points)(q1)  # Distribution 1 output

        # Second critic network (Q2)
        q2 = nn.Dense(self.hidden_dim)(x)
        q2 = self.activation_fn(q2)
        q2 = nn.Dense(self.hidden_dim)(q2)
        q2 = self.activation_fn(q2)
        q2 = nn.Dense(self.num_points)(q2)  # Distribution 2 output

        return q1, q2, z

@jax.jit
def project_distribution(next_dist: jnp.ndarray,
                         z: jnp.ndarray,
                         rewards: jnp.ndarray,
                         gamma: float,
                         dones: jnp.ndarray) -> jnp.ndarray:
    """
    Projects the target distribution (Tz) onto the fixed support of z.

    Args:
        next_dist: Target distribution from the next state-action pair [batch, num_points].
        z: Support of the distribution [num_points].
        rewards: Observed rewards [batch].
        gamma: Discount factor [float].
        dones: Done flags, indicating terminal states [batch].

    Returns:
        projected_dist: Projected distribution [batch, num_points].
    """
    # Compute Tz (Bellman update of the support)
    Tz = rewards[:, None] + gamma * z[None, :] * (1.0 - dones[:, None])
    Tz = jnp.clip(Tz, z[0], z[-1])  # Clip to the support range

    # Calculate the projection
    b = (Tz - z[0]) / (z[1] - z[0])  # Relative positions on the support
    lower = jnp.floor(b).astype(jnp.int32)  # Lower indices
    upper = jnp.ceil(b).astype(jnp.int32)  # Upper indices

    lower = jnp.clip(lower, 0, z.shape[0] - 1)  # Bound indices within range
    upper = jnp.clip(upper, 0, z.shape[0] - 1)

    # Initialize the projected distribution
    projected_dist = jnp.zeros_like(next_dist)

    # Distribute the probabilities across the lower and upper bins
    l_indices = jnp.arange(next_dist.shape[0])[:, None], lower
    u_indices = jnp.arange(next_dist.shape[0])[:, None], upper

    # Update the projected distribution
    projected_dist = projected_dist.at[l_indices].add(next_dist * (upper - b))
    projected_dist = projected_dist.at[u_indices].add(next_dist * (b - lower))

    # Ensure numerical stability
    epsilon = 1e-6
    projected_dist = jnp.clip(projected_dist, a_min=epsilon, a_max=1.0)
    projected_dist /= jnp.sum(projected_dist, axis=-1, keepdims=True)  # Normalize to a valid distribution

    return projected_dist


@jax.jit
def calculate_td_error(q1_logits : jnp.ndarray,
                    q2_logits : jnp.ndarray,
                    z : jnp.ndarray,
                    rewards : jnp.ndarray, 
                    not_done : jnp.ndarray,
                    next_dist : jnp.ndarray,
                    gamma : float =0.99) -> jnp.ndarray:
    """
    Calculates the temporal difference (TD) error for the distributional critic.

    Args:
        q1_logits: Logits for the first Q-value distribution [batch, num_points].
        q2_logits: Logits for the second Q-value distribution [batch, num_points].
        z: Support of the distribution [num_points].
        rewards: Observed rewards [batch].
        not_done: Not done [batch]. This masks out terminal states by setting their influence to zero.
        next_dist: Target distribution from the next state-action pair [batch, num_points].
        gamma: Discount factor [float].

    Returns:
        td_error_q1: TD error for the first Q-value distribution [batch].
        td_error_q2: TD error for the second Q-value distribution [batch].
    """
    # Compute the softmax probabilities with numerical stability
    q1_logits = q1_logits - jnp.max(q1_logits, axis=-1, keepdims=True)  # Normalize logits
    q2_logits = q2_logits - jnp.max(q2_logits, axis=-1, keepdims=True)  # Normalize logits
    q1_probs = jax.nn.softmax(q1_logits, axis=-1)
    q2_probs = jax.nn.softmax(q2_logits, axis=-1)

    # Target distribution (Tz)
    Tz = rewards[:, None] + gamma * z[None, :] * not_done[:, None]
    Tz = jnp.clip(Tz, a_min=z[0], a_max=z[-1])

    # Project Tz onto the support z
    b = (Tz - z[0]) / (z[1] - z[0])  # Relative positions on the support
    lower = jnp.floor(b).astype(jnp.int32)
    upper = jnp.ceil(b).astype(jnp.int32)

    lower  = jnp.clip(lower, 0, z.shape[0] - 1)
    upper = jnp.clip(upper, 0, z.shape[0] - 1)

    # Vectorized projection
    # Initialize the projected distribution
    target_dist = jnp.zeros_like(next_dist)
    # Indices for lower and upper bounds
    l_indices = jnp.arange(next_dist.shape[0])[:, None], lower
    u_indices = jnp.arange(next_dist.shape[0])[:, None], upper
    # Update the projected distribution at lower indices
    target_dist = target_dist.at[l_indices].add(next_dist * (upper - b))
    # Update the projected distribution at upper indices
    target_dist = target_dist.at[u_indices].add(next_dist * (b - lower))

    # Debugging shapes (use sparingly to avoid flooding)
    assert q1_probs.shape == target_dist.shape, "Mismatch in shapes of critic 1 and target distribution"
    assert q2_probs.shape == target_dist.shape, "Mismatch in shapes of critic 2 and target distribution"

    # Clip distributions for numerical stability
    epsilon_clip = 1e-6
    q1_probs = jnp.clip(q1_probs, a_min=epsilon_clip, a_max=1.0)
    q2_probs = jnp.clip(q2_probs, a_min=epsilon_clip, a_max=1.0)
    target_dist = jnp.clip(target_dist, a_min=epsilon_clip, a_max=1.0)


    # Compute TD error through the KL divergence
    epsilon = 1e-8
    td_error_q1 = jnp.sum(target_dist * jnp.log(q1_probs + epsilon), axis=-1)
    td_error_q2 = jnp.sum(target_dist * jnp.log(q2_probs + epsilon), axis=-1)

    # Note that critic loss is the sum of the TD errors
    return td_error_q1, td_error_q2


@jax.jit
def compute_critic_loss(q1_logits: jnp.ndarray,
                        q2_logits: jnp.ndarray,
                        target_dist: jnp.ndarray,
                        weights: jnp.ndarray) -> jnp.ndarray:
    """
    Computes the loss for a distributional critic using KL divergence.

    Args:
        q1_logits: Logits for the first Q-value distribution [batch, num_points].
        q2_logits: Logits for the second Q-value distribution [batch, num_points].
        target_dist: Target probability distribution [batch, num_points].
        weights: Importance sampling weights for prioritized replay [batch].

    Returns:
        critic_loss: The critic loss value.
    """
    # Convert logits to probabilities using softmax
    q1_probs = jax.nn.softmax(q1_logits, axis=-1)
    q2_probs = jax.nn.softmax(q2_logits, axis=-1)

    # Clip for numerical stability
    epsilon = 1e-6
    q1_probs = jnp.clip(q1_probs, a_min=epsilon, a_max=1.0)
    q2_probs = jnp.clip(q2_probs, a_min=epsilon, a_max=1.0)
    target_dist = jnp.clip(target_dist, a_min=epsilon, a_max=1.0)

    # Compute KL divergence for each distribution
    kl_loss_q1 = jnp.sum(target_dist * jnp.log(target_dist / q1_probs), axis=-1)
    kl_loss_q2 = jnp.sum(target_dist * jnp.log(target_dist / q2_probs), axis=-1)

    # Weighted loss for prioritized replay
    critic_loss = jnp.mean(weights * (kl_loss_q1 + kl_loss_q2) / 2.0)

    return critic_loss


class PrioritizedReplayBuffer:
    '''
    A prioritized replay buffer for reinforcement learning, with optional n-step return computation.

    Args:
        capacity (int): Maximum number of transitions the buffer can hold.
        n_step (int): Number of steps for n-step return calculation.
        gamma (float): Discount factor for future rewards.
        alpha (float): Priority exponent for sampling distribution.
        beta (float): Initial importance-sampling weight exponent.
        beta_decay (float): Increment rate for beta towards 1.

    Attributes:
        buffer (list): Stores transitions as (state, action, reward, next_state, done).
        priorities (jax.numpy.ndarray): Stores transition priorities.
        n_step_buffer (deque): Temporary buffer for n-step transitions.
        pos (int): Current position for circular buffer insertion.
        alpha (float): Priority exponent for sampling.
        beta (float): Importance-sampling weight exponent.
        beta_decay (float): Increment rate for beta towards 1.
        n_step (int): Number of steps for n-step return.
        gamma (float): Discount factor for n-step computation.
    '''

    def __init__(self, capacity: int, n_step: int = 1, gamma: float = 0.99,
                 alpha: float = 0.6, beta: float = 0.4, beta_decay: float = 0.001):
        self.capacity = capacity
        self.buffer = []
        self.priorities = jnp.zeros((capacity,), dtype=jnp.float32)
        self.n_step_buffer = deque(maxlen=n_step) if n_step > 1 else None
        self.pos = 0
        self.alpha = alpha
        self.beta = beta
        self.beta_decay = beta_decay
        self.n_step = n_step
        self.gamma = gamma

    def __call__(self, batch_size: int, rng_key: jax.random.PRNGKey) -> Tuple[jnp.ndarray]: # SAMPLE
        '''
        Sample a batch of transitions from the buffer.

        Args:
            batch_size (int): Number of transitions to sample.
            rng_key (jax.random.PRNGKey): Random number generator key.

        Returns:
            Tuple containing:
                states, actions, rewards, next_states, dones, indices, weights.
        '''
        if len(self.buffer) < batch_size:
            raise ValueError("Not enough samples in the buffer to sample a batch")

        priorities = self.priorities[:len(self.buffer)]
        if jnp.sum(priorities) == 0:
            probabilities = jnp.ones_like(priorities) / len(priorities)
        else:
            probabilities = (priorities ** self.alpha) / jnp.sum(priorities ** self.alpha)
        
        indices = jax.random.choice(rng_key, len(self.buffer), shape=(batch_size,), p=probabilities)
        samples = [self.buffer[idx] for idx in indices]

        weights = (probabilities[indices] * len(self.buffer)) ** (-self.beta)
        weights /= jnp.max(weights)
        self.beta = min(1.0, self.beta + self.beta_decay)

        states, actions, rewards, next_states, dones = map(jnp.array, zip(*samples))
        return (jnp.stack(states), jnp.stack(actions), jnp.stack(rewards),
                jnp.stack(next_states), jnp.stack(dones), indices, weights)

    def _compute_n_step(self) -> Tuple[float, jnp.ndarray, bool]:
        '''
        Compute n-step returns for the oldest transition in the n-step buffer.

        Returns:
            Tuple containing:
                reward (float): n-step discounted reward.
                next_state (jnp.ndarray): State after n steps.
                done (bool): Done flag after n steps.
        '''
        reward, next_state, done = self.n_step_buffer[-1][-3:]
        for transition in reversed(list(self.n_step_buffer)[:-1]):
            r, n_s, d = transition[-3:]
            reward = r + self.gamma * reward * (1 - d)
            next_state, done = (n_s, d) if d else (next_state, done)
        return reward, next_state, done

    def add(self, state: jnp.ndarray, action: jnp.ndarray, reward: float,
            next_state: jnp.ndarray, done: bool, td_error: float):
        '''
        JIT-compiled method to add a transition to the buffer.

        Args:
            state (jnp.ndarray): Current state.
            action (jnp.ndarray): Executed action.
            reward (float): Observed reward.
            next_state (jnp.ndarray): Next state.
            done (bool): Done flag.
            td_error (float): Temporal difference error for priority.
        '''

        # Handle n-step logic
        if self.n_step > 1:
            self.n_step_buffer.append((state, action, reward, next_state, done))
            if len(self.n_step_buffer) < self.n_step:
                return
            reward, next_state, done = self._compute_n_step()
            state, action, _, _, _ = self.n_step_buffer[0]

        # Add transition to the buffer
        if len(self.buffer) == self.capacity:
            self.buffer[self.pos] = (state, action, reward, next_state, done)
        else:
            self.buffer.append((state, action, reward, next_state, done))

        # Update priorities
        priority = (td_error + 1e-5) ** self.alpha
        self.priorities = self.priorities.at[self.pos].set(priority)

        # Update position for circular buffer
        self.pos = (self.pos + 1) % self.capacity

    def update_priorities(self, indices: jnp.ndarray, priorities: jnp.ndarray):
        '''
        Update priorities for specific buffer indices.

        Args:
            indices (jnp.ndarray): List of indices to update.
            priorities (jnp.ndarray): New priority values.
        '''
        self.priorities = self.priorities.at[indices].set(priorities)

    def __len__(self) -> float:
        '''Return the current size of the buffer.'''
        return len(self.buffer)
    

class HybridMPO:

    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 buffer_size: int,
                 hidden_dim_actor: int = 256,
                 gamma: float = 0.99,
                 alpha: float = 0.6,
                 beta: float = 0.4,
                 beta_decay: float = 0.001,
                 batch_size: int = 256,
                 tau: float = 0.005,
                 critic_lr: float = 3e-4,
                 critic_grad_max_norm : float = 10,
                 temperature_updated_learned_bool : bool = False):  # Added critic learning rate

        self.critic = DoubleDistributionalCritic(state_dim=state_dim, action_dim=action_dim)
        self.actor = Actor(state_dim=state_dim, action_dim=action_dim, hidden_dim=hidden_dim_actor, stochastic=True)
        self.buffer = PrioritizedReplayBuffer(capacity=buffer_size,
                                              n_step=1,
                                              gamma=gamma,
                                              alpha=alpha,
                                              beta=beta,
                                              beta_decay=beta_decay)

        self.gamma = gamma  # Discount factor
        self.batch_size = batch_size
        self.tau = tau
        self.critic_lr = critic_lr
        self.critic_grad_max_norm = critic_grad_max_norm
        self.rng_key = jax.random.PRNGKey(0)

        # Initialize parameters
        self.actor_params = self.actor.init(self.rng_key, jnp.zeros((1, state_dim)))
        self.critic_params = self.critic.init(self.rng_key, jnp.zeros((1, state_dim)), jnp.zeros((1, action_dim)))
        self.target_critic_params = self.critic_params

        # Temperature update method
        self.temperature_updated_learned_bool = temperature_updated_learned_bool
        self.kl_constraint_temperature = action_dim # Desired entropy

        # Intialize temperature parameter
        self.temp = 1.0
        self.temp_lr = 1e-3

        # Define action sample size
        self.action_min = -1.0  # Example bounds for normalized actions
        self.action_max = 1.0
        self.action_sample_size = 10

    def critic_update(self):
        """
        Updates the critic network parameters using KL divergence loss.

        Returns:
            critic_loss: The loss value of the critic network.
        """
        # 1. Sample a batch of transitions from the replay buffer
        # Extract states, actions, rewards, next states, dones, indices, and importance sampling weights
        states, actions, rewards, next_states, dones, indices, weights = self.buffer(self.batch_size, self.rng_key)

        # 2. Calculate TD errors and target distributions
        # Apply the critic to compute the Q-value logits and distributional support (z) for the current state-action pairs
        q1_logits, q2_logits, z = self.critic.apply(self.critic_params, states, actions)
        
        # `not_done` is 1 for non-terminal states, used to mask terminal states in updates
        not_done = 1 - dones

        # Apply the target critic to compute Q-value logits for the next state-action pairs
        next_q1_logits, next_q2_logits, _ = self.critic.apply(self.target_critic_params, next_states, actions)
        
        # Calculate the target distribution as the minimum between the two Q-value distributions
        next_dist = jnp.minimum(jax.nn.softmax(next_q1_logits, axis=-1), jax.nn.softmax(next_q2_logits, axis=-1))
        
        # Project the next state distribution to align with the support of the current distribution
        target_dist = project_distribution(next_dist, z, rewards, self.gamma, dones)

        # Calculate TD errors for both Q-value distributions
        # These errors quantify the difference between the current and target distributions
        td_errors_q1, td_errors_q2 = calculate_td_error(q1_logits, q2_logits, z, rewards, not_done, next_dist, self.gamma)

        # 3. Update replay buffer
        # Update the priorities in the replay buffer using the TD errors to focus learning on important samples
        priorities = td_errors_q1 + td_errors_q2
        self.buffer.update_priorities(indices, priorities)

        # 4. Compute critic loss
        # Calculate the critic loss as the KL divergence between the predicted and target distributions
        critic_loss = compute_critic_loss(q1_logits, q2_logits, target_dist, weights)

        # 5. Perform gradient descent with gradient clipping
        # Compute gradients of the critic loss with respect to the critic parameters
        gradients = jax.grad(lambda params: compute_critic_loss(*self.critic.apply(params, states, actions)[:2], target_dist, weights))(self.critic_params)

        # Clip gradients to prevent exploding gradients and improve stability
        clipped_gradients = clip_grads(gradients, max_norm=self.critic_grad_max_norm)
        
        # Update the critic parameters using the clipped gradients and learning rate
        self.critic_params = jax.tree_util.tree_map(
            lambda param, grad: param - self.critic_lr * grad, self.critic_params, clipped_gradients
        )

        # 6. Soft update the target network parameters
        # Perform a soft update on the target network parameters to slowly track the main critic network
        self.target_critic_params = jax.tree_util.tree_map(
            lambda target, current: self.tau * current + (1 - self.tau) * target,
            self.target_critic_params,
            self.critic_params
        )

        # Return the computed critic loss for logging or monitoring
        return critic_loss
    
    def actor_forward(self, params: dict, state: jnp.ndarray) -> Tuple[jnp.ndarray]:
        return self.actor.apply(params, state)


    def sample_actions(self, states: jnp.ndarray) -> jnp.ndarray:
        mean, std = self.actor_forward(self.actor_params, states)
        actions = jax.random.normal(self.rng_key, mean.shape) * std + mean
        actions = jnp.clip(actions, self.action_min, self.action_max)
        return actions

    
    # LEARNED TEMPERATURE
    def temperature_update_learned(self,
                                   q_logits : jnp.ndarray):
        """
        Learn the temperature parameter through gradient descent.
        """
        target_entropy = self.kl_constraint_temperature  # Desired entropy
        current_entropy = -jnp.mean(jax.scipy.special.logsumexp(q_logits / self.temp, axis=1))
        temp_loss = (current_entropy - target_entropy) ** 2

        temp_grad = jax.grad(lambda temp: temp_loss)(self.temp)
        self.temp -= self.temp_lr * temp_grad  # Update using gradient descent
        self.temp = jnp.clip(self.temp, 1e-6, None)  # Avoid instability

    def temperature_update(self, q_logits: jnp.ndarray) -> None:
        def dual(temp: float) -> float:
            dual_value = (
                temp * self.kl_constraint_temperature
                + temp * jnp.mean(
                    jax.scipy.special.logsumexp(q_logits / temp, axis=1)
                    - jnp.log(self.action_sample_size)
                )
            )
            return jnp.squeeze(dual_value)  # ensure scalar

        def dual_grad(temp: float) -> float:
            return jax.grad(dual)(temp)

        initial_temp = self.temp
        bounds = [(1e-6, None)]
        res = scipy.optimize.minimize(
            fun=lambda x: dual(x),
            x0=initial_temp,
            jac=lambda x: dual_grad(x),
            bounds=bounds,
            method="SLSQP"
        )
        self.temp = jnp.array(res.x)

    def e_step(self, states: jnp.ndarray):
        """
        Perform the E-step (Target policy update).

        Args:
            states: Batch of states for the E-step [jnp.ndarray].

        Returns:
            weights: Weights for the sampled actions [jnp.ndarray].
            sampled_actions: Sampled actions for each state [jnp.ndarray].
        """
        # 1) Sample actions from the current policy
        sampled_actions = self.sample_actions(states)

        # 2) Compute the Q-values for the current state-action pairs
        q1_logits, q2_logits, _ = self.critic.apply(self.critic_params, states, sampled_actions)

        # 3) Combine Q-values (e.g., take the minimum for conservative Q-learning)
        q_logits = jnp.minimum(q1_logits, q2_logits)

        # 4) Update temperature
        if not self.temperature_updated_learned_bool:
            self.temperature_update(q_logits)
        else:
            self.temperature_update_learned(q_logits)
        
        # 5) Compute weights for the sampled actions
        weights = jax.nn.softmax(q_logits / self.temp, axis=1)
        weights = jnp.expand_dims(weights, axis=-1)  # Add dimension for compatibility

        # 6) Stop gradient for weights and temperature to ensure stability
        weights = jax.lax.stop_gradient(weights)
        self.temp = jax.lax.stop_gradient(self.temp)

        return weights, sampled_actions


'''
    def m_step():


def main():


if __name__ == "__main__":
    main()

'''