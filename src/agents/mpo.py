import jax.numpy as jnp
import jax
import scipy
from typing import Tuple

from flax.core.frozen_dict import FrozenDict
import optax
from functools import partial

from src.agents.qnetworks import PolicyNetwork                                                                        # Actor network
from src.agents.qnetworks import DistributionalCriticNetwork, DoubleCriticNetwork, DoubleDistributionalCriticNetwork, BaseCriticNetwork  # Critic networks
from src.agents.buffers import ReplayBuffer, PrioritizedReplayBuffer                                                  # Replay buffer
from src.utils.gaussian_likelihood import gaussian_likelihood                                                         # Gaussian likelihood
from src.utils.kl_divergence import kl_divergence_multivariate_gaussian, std_lagrange_step        # KL divergence
from src.utils.clip_gradients import clip_grads                                                                       # Gradient clipping

jax.clear_backends()

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
                alpha=config['per_alpha'],
                beta=config['beta'],
                beta_decay=config['beta_decay']
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
        assert self.policy_network.params is not None, "Policy network parameters are not initialized."

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

        # Action ranges: nice to have as (0, 1) so are normalised values
        self.action_min = config['action_min']
        self.action_max = config['action_max']

        # KL divergence constraints
        self.kl_constraint_mean = config['kl_constraint_mean']
        self.kl_constraint_std = config['kl_constraint_std']
        self.kl_constraint_temperature = config['kl_constraint_temperature']

        # Action sampling size for the E-step: determines the number of actions to sample per state
        self.action_sample_size = config['action_sample_size']

        # Temperature parameter for the E-step
        self.temp = 1.0 # Check this value

        # Actor optimizer
        self.actor_optimizer = optax.adam(config['actor_lr'])
        self.actor_optimizer_state = self.actor_optimizer.init(self.policy_network.params)

        # Lagrange multipliers for the KL constraints
        self.mean_lagrange_optimizer = optax.adam(config['mean_lr'])
        self.std_lagrange_optimizer = optax.adam(config['std_lr'])

        # Lagrange multipliers
        self.mean_lagrange_params = jnp.array(1.0)  # Initial value
        self.std_lagrange_params = jnp.array(1.0)

        # Optimizer state
        self.mean_lagrange_opt_state = self.mean_lagrange_optimizer.init(self.mean_lagrange_params)
        self.std_lagrange_opt_state = self.std_lagrange_optimizer.init(self.std_lagrange_params)




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
        mean, std = self.policy_network.forward(next_states)  # Get mean and std from policy
        next_actions = mean + std * jax.random.normal(self.rng_key, shape=mean.shape)  # Sample actions

        # Compute the target TD using the double critic's compute_td_target method
        target_q_values = self.critic_network.compute_td_target(
            rewards, next_states, next_actions, 1 - dones, self.gamma
        ) # or for distributional critics finds the target distribution

        critic_loss = self.critic_network.compute_loss(states, actions, target_q_values)

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

        REFERENCE: https://github.com/henry-prior/jax-rl/blob/master/jax_rl/MPO.py

        '''
        # Sample actions from the policy network
        mean, std = self.policy_network.forward(states)
        sampled_actions = mean + std * jax.random.normal(self.rng_key, (batch_size, action_sample_size, self.action_dim))
        sampled_actions = jnp.clip(sampled_actions, self.action_min, self.action_max)

        # Evaluate Q-values for sampled actions
        q_values = []
        for i in range(action_sample_size):
            q_value = self.critic_network.evaluate_q_value(states, sampled_actions[:, i, :])
            q_values.append(q_value)
        q_values = jnp.stack(q_values, axis=1)  # Shape: (batch_size, action_sample_size)

        # Define the dual function for temperature optimization
        def dual(temp: float) -> float:
            '''
            Computes the dual function for temperature optimization.

            params:
            temp: Current temperature parameter [float]

            returns:
            dual_value: Value of the dual function [float]
            '''
            dual_value = temp * self.kl_constraint_temperature + temp * jnp.mean(
                jax.scipy.special.logsumexp(q_values / temp, axis=1) - jnp.log(action_sample_size)
            )
            return jax.device_get(dual_value).item()

        # Compute the gradient of the dual function
        dual_grad = jax.grad(dual)

        # Perform optimization (e.g., using SLSQP as in your example)
        bounds = [(1e-6, None)]  # Avoid numerical instability
        res = scipy.optimize.minimize(
            fun=lambda x: dual(x), 
            x0=self.temp, 
            jac=lambda x: dual_grad(x), 
            bounds=bounds, 
            method="SLSQP"
        )

        # Update the temperature parameter
        self.temp = jax.lax.stop_gradient(res.x)

        # Compute weights for the sampled actions
        weights = jax.nn.softmax(q_values / self.temp, axis=1)
        weights = jnp.expand_dims(weights, axis=-1)  

        # Stop gradient for weights and temperature
        weights = jax.lax.stop_gradient(weights)
        self.temp = jax.lax.stop_gradient(self.temp)

        return weights, sampled_actions
    
    def actor_loss_fcn(self,
                    states: jnp.ndarray,
                    sampled_actions: jnp.ndarray,
                    weights_sampled_actions: jnp.ndarray
                    ) -> Tuple[float, Tuple]:
        '''
        align the parametric policy (neural network) with the non-parametric target policy (from the E-step)
        while maintaining stability via KL divergence constraints

        params:
        states: Batch of states for the actor update [jnp.ndarray]
        sampled_actions: Sampled actions for each state [jnp.ndarray]
        weights_sampled_actions: Weights for the sampled actions [jnp.ndarray]

        returns:
        actor_loss: Loss of the actor network [float]
        mu_lagrange_optimizer: Optimizer for the mean KL constraint [Tuple]
        sig_lagrange_optimizer: Optimizer for the std KL constraint [Tuple]
        '''
        def mean_lagrange_step(opt_state: optax.OptState, params: optax.Params, reg: float) -> Tuple[optax.Params, optax.OptState]:
            '''
            Update the mean Lagrange multiplier using the KL constraint violation.

            params:
            opt_state: Optimizer state [optax.OptState]
            params: Current parameters of the optimizer [optax.Params]
            reg: Regularization term (eps_mu - KL_mu) [float]

            returns:
            Updated parameters and optimizer state with adjusted Lagrange multiplier.
            '''
            def loss_fn(mean_lagrange_params):
                # Scale the Lagrange multiplier by 1.0
                return jnp.sum(mean_lagrange_params * reg)

            grad = jax.grad(loss_fn)(params)
            updates, new_opt_state = self.mean_lagrange_optimizer.update(grad, opt_state, params)
            new_params = optax.apply_updates(params, updates)
            return new_params, new_opt_state

        # 1. Calculate the distribution of the actor network for the current policy
        current_mean, current_std = self.policy_network.forward(states)  # Predict mean and std for actions
        current_std = jnp.clip(current_std, 1e-6, None)  # Ensure numerical stability for standard deviation

        # 2. Compute the target policy distribution (detached to prevent updates)
        target_mean, target_std = self.target_policy_network.forward(states)  # Predict target mean and std
        target_mean = jax.lax.stop_gradient(target_mean)        # Prevent gradients from flowing into target
        target_std = jax.lax.stop_gradient(jnp.clip(target_std, 1e-6, None))  # Stop gradient for target std

        # Ensure numerical stability by clipping standard deviations
        current_std = jnp.clip(current_std, 1e-6, 1e1)
        target_std = jnp.clip(target_std, 1e-6, 1e1)

        # Debug prints to check the forward pass of the policy network
        #print("Forward pass - current policy network:")
        #print("Current mean shape:", current_mean.shape)
        #print("Current std shape:", current_std.shape)

        #print("Forward pass - target policy network:")
        #print("Target mean shape:", target_mean.shape)
        #print("Target std shape:", target_std.shape)

        # 3. Compute the actor loss via gaussian likelihood of the sampled actions
        actor_log_prob = gaussian_likelihood(sampled_actions, target_mean, current_std) + gaussian_likelihood(sampled_actions, current_mean, target_std)

        # Ensure numerical stability by clipping log probabilities
        actor_log_prob = jnp.clip(actor_log_prob, -1e1, 1e1)

        # 4. Compute the KL divergence between the target and current policy
        KL_divergence_mean = kl_divergence_multivariate_gaussian(target_mean, target_std, current_mean, target_std).mean()
        KL_divergence_std  = kl_divergence_multivariate_gaussian(target_mean, target_std, target_mean, current_std).mean()

        # Ensure numerical stability by clipping KL divergence values
        KL_divergence_mean = jnp.clip(KL_divergence_mean, 0, 1e1)
        KL_divergence_std = jnp.clip(KL_divergence_std, 0, 1e1)

        # Update Lagrange multipliers
        reg_mean = self.kl_constraint_mean - jax.lax.stop_gradient(KL_divergence_mean)
        self.mean_lagrange_params, self.mean_lagrange_opt_state = mean_lagrange_step(
            self.mean_lagrange_opt_state,
            self.mean_lagrange_params,
            reg_mean,
        )

        reg_std = self.kl_constraint_std - jax.lax.stop_gradient(KL_divergence_std)
        self.std_lagrange_params, self.std_lagrange_opt_state = mean_lagrange_step(
            self.std_lagrange_opt_state,
            self.std_lagrange_params,
            reg_std,
        )

        # 5. Compute actor loss
        actor_loss = -(actor_log_prob * weights_sampled_actions).sum(axis=1).mean()

        # Add KL constraints
        actor_loss -= jax.lax.stop_gradient(self.mean_lagrange_params) * (self.kl_constraint_mean - KL_divergence_mean)
        actor_loss -= jax.lax.stop_gradient(self.std_lagrange_params) * (self.kl_constraint_std - KL_divergence_std)

        return actor_loss, (self.mean_lagrange_optimizer, self.std_lagrange_optimizer)

    def m_step(
        self,
        states: jnp.ndarray,
        weights: jnp.ndarray,
        sampled_actions: jnp.ndarray
    ) -> float:
        """
        Perform the M-step of MPO.

        Args:
            states (jnp.ndarray): Batch of states for the M-step.
            weights (jnp.ndarray): Weights for the sampled actions.
            sampled_actions (jnp.ndarray): Sampled actions for each state.

        Returns:
            float: The actor loss.
        """

        # Define the actor loss function with params as an argument
        def actor_loss_fcn(params, states, sampled_actions, weights):
            """
            Compute actor loss and update Lagrange multipliers.
            """
            # Temporarily update the policy network parameters
            original_params = self.policy_network.params
            self.policy_network.params = params
            actor_loss, (updated_mean_optimizer, updated_std_optimizer) = self.actor_loss_fcn(
                states,
                sampled_actions,
                weights
            )
            # Restore the original parameters
            self.policy_network.params = original_params
            return actor_loss, (updated_mean_optimizer, updated_std_optimizer)

        # Compute gradients of the actor loss with respect to policy network parameters
        grad_fn = jax.value_and_grad(actor_loss_fcn, has_aux=True)
        (actor_loss, (self.mu_lagrange_optimizer, self.sig_lagrange_optimizer)), grads = grad_fn(
            self.policy_network.params,
            states,
            sampled_actions,
            weights
        )

        # Debug: Log structures before processing
        #print("Before processing:")
        #print("Parameter tree structure:", jax.tree_util.tree_structure(self.policy_network.params))
        #print("Gradient tree structure:", jax.tree_util.tree_structure(grads))

        # Replace None gradients with zeros and ensure structure matches
        grads = jax.tree_util.tree_map(
            lambda g, p: g if g is not None else jnp.zeros_like(p),
            grads,
            self.policy_network.params
        )

        # Debug: Log structures after processing
        #print("After processing:")
        #print("Parameter tree structure:", jax.tree_util.tree_structure(self.policy_network.params))
        #print("Gradient tree structure:", jax.tree_util.tree_structure(grads))

        # Debugging Individual Components: #print shapes of parameters and corresponding gradients
        def print_shapes(param, grad):
            #print(f"Parameter shape: {param.shape}, Gradient shape: {grad.shape}")
            pass

        jax.tree_util.tree_map(print_shapes, self.policy_network.params, grads)

        # Check for matching tree structures
        assert jax.tree_util.tree_structure(self.policy_network.params) == jax.tree_util.tree_structure(grads), \
            "Gradient and parameter tree structures still do not match after processing."

        # Clip gradients and apply updates
        clipped_grads = clip_grads(grads, 40.0)
        updates, new_opt_state = self.actor_optimizer.update(clipped_grads, self.actor_optimizer_state)
        self.actor_optimizer_state = new_opt_state

        def safe_apply_updates(params, updates):
            return jax.tree_util.tree_map(lambda p, u: p if u is None else optax.apply_updates(p, u), params, updates)
        self.policy_network.params = safe_apply_updates(self.policy_network.params, updates)

        return actor_loss



    def update(self, batch_size):
        '''
        Update the policy and critic networks.
        params:
        batch_size: Batch size [int]
        '''
        # Sample a batch from the replay buffer
        sample_result = self.replay_buffer.sample(batch_size, self.rng_key)
        if isinstance(self.replay_buffer, PrioritizedReplayBuffer):
            states, actions, rewards, next_states, dones, indices, weights = sample_result
        else:
            states, actions, rewards, next_states, dones = sample_result

        # 1. Perform the critic update (Q-function update)
        target_q_values, critic_loss = self.critic_update(states, actions, rewards, next_states, dones)

        # 2. Perform the E-step (Target policy update)
        weights, sampled_actions = self.e_step(states, batch_size, self.action_sample_size)

        # 3. Perform the M-step (Parametric policy update)
        actor_loss = self.m_step(states, weights, sampled_actions)
        # 4. Update the priorities if PER is enabled
        if isinstance(self.replay_buffer, PrioritizedReplayBuffer):
            # Debugging: #print shapes of target Q-values and critic network outputs
            #print("Shapes:")
            #print("Target Q-values shape:", target_q_values.shape)
            #print("Critic network output shape:", self.critic_network(states, actions).shape)
            #print("Indices shape:", indices.shape)
            # Calculate TD errors for priority updates
            scalar_target_q_values = jnp.sum(target_q_values * self.critic_network.support, axis=-1)  # Shape: (10,)
            #print("Scalar target Q-values shape:", scalar_target_q_values.shape)
            critic_output_q_values = self.critic_network.evaluate_q_value(states, actions)           # Shape: (10,)
            #print("Critic output Q-values shape:", critic_output_q_values.shape)
            td_errors = jnp.abs(scalar_target_q_values - critic_output_q_values).flatten()           # Shape: (10,)
            #print("TD errors shape:", td_errors.shape)
            self.replay_buffer.update_priorities(indices, td_errors)

        # 5. Update the target networks
        if self.config['target_networks']:
            self.target_policy_network = self.target_policy_network.replace(
                params=jax.tree_util.tree_map(
                    lambda p, tp: self.tau * p + (1 - self.tau) * tp,
                    self.policy_network.params,
                    self.target_policy_network.params
                )
            )

            self.target_critic_network = self.target_critic_network.replace(
                params=jax.tree_util.tree_map(
                    lambda p, tp: self.tau * p + (1 - self.tau) * tp,
                    self.critic_network.params,
                    self.target_critic_network.params
                )
            )
        return critic_loss, actor_loss