import jax.numpy as jnp
import jax
import scipy
from typing import Tuple

from src.agents.qnetworks import PolicyNetwork                                                                        # Actor network
from src.agents.qnetworks import DistributionalCriticNetwork, DoubleCriticNetwork, DoubleDistributionalCriticNetwork, BaseCriticNetwork  # Critic networks
from src.agents.buffers import ReplayBuffer, PrioritizedReplayBuffer                                                  # Replay buffer

class MPOLearner:
    '''
    MPO Learner with options for PER buffer, target networks, double critics, distributional critics, and N-step returns.
    '''
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 config: dict):
        '''
        Initialize the MPO Learner.
        params:
        state_dim: Dimension of the state space [int]
        action_dim: Dimension of the action space [int]
        config: Configuration dictionary with options [dict]
        - replay_buffer: Type of replay buffer [str]
        - buffer_capacity: Capacity of the replay buffer [int]
        - n_step: N-step return [int]
        - gamma: Discount factor [float]
        - hidden_dim: Dimension of the hidden layers [int]
        - target_networks: Use target networks [bool]
        - critic_type: Type of critic network [str]
        - num_points: Number of points for distributional critics [int]
        - support_range: Range of support for distributional critics [tuple]
        - critic_lr: Learning rate for the critic network [float]
        - tau: Soft update factor [float]
        - seed: Random seed [int]
        - per_alpha: PER alpha parameter [float]
        # Note what about per_beta?
        '''
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config

        # Replay Buffer
        if config['replay_buffer'] == 'uniform':
            self.replay_buffer = ReplayBuffer(
                capacity=config['buffer_capacity'],
                n_step=config['n_step'],
                gamma=config['gamma']
            )
        elif config['replay_buffer'] == 'per':
            self.replay_buffer = PrioritizedReplayBuffer(
                capacity=config['buffer_capacity'],
                n_step=config['n_step'],
                gamma=config['gamma'],
                alpha=config['per_alpha']
            )
        else:
            raise ValueError('Invalid replay buffer type.')

        # Actor: stochastic true by default as MPO.
        self.policy_network = PolicyNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=config['hidden_dim'],
            stochastic=True,
            rng_key=jax.random.PRNGKey(0)
        )

        # Target Networks
        if config['target_networks']:
            self.target_policy_network = PolicyNetwork(
                state_dim=state_dim,
                action_dim=action_dim,
                hidden_dim=config['hidden_dim'],
                stochastic=True,
                rng_key=jax.random.PRNGKey(1)
            )

        # Critic
        if config['critic_type'] =='basic':
            self.critic_network = BaseCriticNetwork(
                state_dim=state_dim,
                action_dim=action_dim,
                hidden_dim=config['hidden_dim']
            )
            self.target_critic_network = BaseCriticNetwork(
                state_dim=state_dim,
                action_dim=action_dim,
                hidden_dim=config['hidden_dim']
            )
        elif config['critic_type'] == 'double':
            self.critic_network = DoubleCriticNetwork(
                state_dim=state_dim,
                action_dim=action_dim,
                hidden_dim=config['hidden_dim']
            )
            self.target_critic_network = DoubleCriticNetwork(
                state_dim=state_dim,
                action_dim=action_dim,
                hidden_dim=config['hidden_dim']
            )
        elif config['critic_type'] == 'distributional':
            self.critic_network = DistributionalCriticNetwork(
                state_dim=state_dim,
                action_dim=action_dim,
                hidden_dim=config['hidden_dim'],
                num_points=config['num_points'],
                support_range=config['support_range']
            )
            self.target_critic_network = DistributionalCriticNetwork(
                state_dim=state_dim,
                action_dim=action_dim,
                hidden_dim=config['hidden_dim'],
                num_points=config['num_points'],
                support_range=config['support_range']
            )
        elif config['critic_type'] == 'double_distributional':
            self.critic_network = DoubleDistributionalCriticNetwork(
                state_dim=state_dim,
                action_dim=action_dim,
                hidden_dim=config['hidden_dim'],
                num_points=config['num_points'],
                support_range=config['support_range']
            )
            self.target_critic_network = DoubleDistributionalCriticNetwork(
                state_dim=state_dim,
                action_dim=action_dim,
                hidden_dim=config['hidden_dim'],
                num_points=config['num_points'],
                support_range=config['support_range']
            )
        else:
            raise ValueError('Invalid critic type.')

        # Optimization parameters
        self.gamma = config['gamma']
        self.critic_lr = config['critic_lr']
        self.tau = config['tau']
        self.n_step = config['n_step']
        self.rng_key = jax.random.PRNGKey(config['seed'])

        # Action ranges
        self.action_min = config['action_min']
        self.action_max = config['action_max']

    def select_action(self,
                      state: jnp.ndarray) -> jnp.ndarray:
        '''
        Select an action using the policy network.
        params:
        state: Current state [jnp.ndarray]
        action_min: Minimum values of the action ranges [jnp.ndarray]
        action_max: Maximum values of the action ranges [jnp.ndarray]

        returns:
        action: Selected action [jnp.ndarray]
        '''
        mean, std = self.policy_network.forward(state)
        action = mean + std * jax.random.normal(self.rng_key, shape=mean.shape)
        action = jnp.clip(action, self.action_min, self.action_max)
        return action

    def store_transition(self,
                         state: jnp.ndarray,
                         action: jnp.ndarray,
                         reward: float,
                         next_state: jnp.ndarray,
                         done: bool):
        '''
        Store a transition in the replay buffer.
        params:
        state: Current state [jnp.ndarray]
        action: Executed action [jnp.ndarray]
        reward: Received reward [float]
        next_state: Next state [jnp.ndarray]
        done: Terminal flag [bool]
        '''
        self.replay_buffer.add(state, action, reward, next_state, done)

    def critic_update(self,
                      states: jnp.ndarray,
                      actions: jnp.ndarray,
                      rewards: jnp.ndarray,
                      next_states: jnp.ndarray,
                      dones: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        '''
        Update the critic network.
        params:
        states: States sampled from the replay buffer [jnp.ndarray]
        actions: Actions sampled from the replay buffer [jnp.ndarray]
        rewards: Rewards sampled from the replay buffer [jnp.ndarray]
        next_states: Next states sampled from the replay buffer [jnp.ndarray]
        dones: Done flags sampled from the replay buffer [jnp.ndarray]

        returns:
        target_q_values: Target Q-values for the critic update [jnp.ndarray]
        critic_loss: Loss of the critic network [jnp.ndarray]

        Note, different critic configurations:
        - Double critic
        - Distributional critic
        - Double distributional critic
        - Normal critic
        '''
        # 1. Compute the target TD
        next_actions = self.policy(next_states)  # Compute the next action using the policy.
        # Compute target TD

        # Compute the target TD using the double critic's compute_td_target method
        target_q_values = self.critic_network.compute_td_target(
            rewards, next_states, next_actions, 1 - dones, self.gamma
        ) # or for distributional critics finds the target distribution

        critic_loss = self.critic_network.loss(states, actions, target_q_values)

        # Perform a gradient step to minimize the critic loss
        # Compute the gradient of the critic loss w.r.t. the critic network parameters
        grad = jax.grad(lambda params: critic_loss)(self.critic_network.params)
        # Update the critic network parameters using the gradient
        # lambda function updates via SGD: param - lr * gradient

        self.critic_network.params = jax.tree_util.tree_map(
            lambda param, gradient: param - self.critic_lr * gradient, self.critic_network.params, grad
        )
        return target_q_values, critic_loss
    


    def e_step(
        self,
        states: jnp.ndarray,
        batch_size: int,
        action_sample_size: int
    ) -> Tuple[float, jnp.ndarray, jnp.ndarray]:
        '''
        Perform the E-step of MPO.

        params:
        states: Batch of states for the E-step [jnp.ndarray]
        batch_size: Number of states in the batch [int]
        action_sample_size: Number of actions to sample per state [int]

        returns:
        temp: Updated temperature parameter [float]
        weights: Weights for the sampled actions [jnp.ndarray]
        sampled_actions: Sampled actions for each state [jnp.ndarray]
        '''
        # Sample actions from the policy network
        mean, log_std = self.policy_network.forward(states)
        std = jnp.exp(log_std)
        sampled_actions = mean + std * jax.random.normal(self.rng, (batch_size, action_sample_size, self.action_dim))
        sampled_actions = jnp.clip(sampled_actions, -self.max_action, self.max_action)

        # Evaluate Q-values for sampled actions
        q_values = []
        for i in range(action_sample_size):
            q_value = self.critic_network.forward(states, sampled_actions[:, i, :])
            if isinstance(q_value, tuple):  # Double critic
                q_value = jnp.minimum(q_value[0], q_value[1])
            q_values.append(q_value)
        q_values = jnp.stack(q_values, axis=1)  # Shape: (batch_size, action_sample_size)

        # Minimize the dual function to update temperature
        def dual(temp):
            softmax_weights = jax.nn.softmax(q_values / temp, axis=1)
            dual_value = temp * self.eps_eta + temp * jnp.mean(
                jax.scipy.special.logsumexp(q_values / temp, axis=1) - jnp.log(action_sample_size)
            )
            return dual_value

        # Compute the gradient of the dual function
        dual_grad = jax.grad(dual)
        # Perform optimization (e.g., using SLSQP as in your example)
        bounds = [(1e-6, None)]  # Avoid numerical instability
        res = scipy.minimize(lambda x: dual(x), self.temp, jac=lambda x: dual_grad(x), bounds=bounds, method="SLSQP")
        self.temp = jax.lax.stop_gradient(res.x)

        # Compute weights for the sampled actions
        weights = jax.nn.softmax(q_values / self.temp, axis=1)
        weights = jnp.expand_dims(weights, axis=-1)  # Add an extra dimension for broadcasting

        # Stop gradient for weights and temperature
        weights = jax.lax.stop_gradient(weights)
        self.temp = jax.lax.stop_gradient(self.temp)

        return self.temp, weights, sampled_actions

    def update(self, batch_size):
        '''
        Update the policy and critic networks.
        params:
        batch_size: Batch size [int]
        '''
        # Sample a batch from the replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size, self.rng_key)

        # 1. Perform the critic update (Q-function update)
        target_q_values, critic_loss = self.critic_update(states, actions, rewards, next_states, dones)

        # 2. Perform the E-step (Target policy update)
        weights, sampled_actions = self.e_step(states, target_q_values)

        # 3. Perform the M-step (Parametric policy update)
        self.m_step(states, weights, sampled_actions)

        # 4. Update the temperature parameter
        self.temperature_update()



   

    def m_step(self, batch_size):
        '''
        NOT DONE
        Perform an M-step of the MPO algorithm.
        i.e. parametric policy update through minimisation of the KL divergence of the target and parametric policy netwrks,
        using KL divergence constraints to stay within trust region
        params:
        '''




    def temperature_update(self):
        '''
        NOT DONE
        Update the temperature parameter.
        '''

    def actor_update(self):
        '''
        NOT DONE
        Update the actor network.
        '''