import jax.numpy as jnp
import jax

from src.agents.qnetworks import PolicyNetwork                                                                        # Actor network
from src.agents.qnetworks import DistributionalCriticNetwork, DoubleCriticNetwork, DoubleDistributionalCriticNetwork, BaseCriticNetwork  # Critic networks
from src.agents.buffers import ReplayBuffer, PrioritizedReplayBuffer                                                  # Replay buffer

class MPOLearner:
    '''
    MPO Learner with options for PER buffer, target networks, double critics, distributional critics, and N-step returns.
    '''
    def __init__(self, state_dim, action_dim, config):
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
        - learning_rate: Learning rate [float]
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
        self.learning_rate = config['learning_rate']
        self.tau = config['tau']
        self.n_step = config['n_step']
        self.rng_key = jax.random.PRNGKey(config['seed'])

        # Action ranges
        self.action_min = config['action_min']
        self.action_max = config['action_max']

    def select_action(self, state):
        '''
        Select an action using the policy network.
        params:
        state: Current state [jnp.array]
        action_min: Minimum values of the action ranges [jnp.array]
        action_max: Maximum values of the action ranges [jnp.array]

        returns:
        action: Selected action [jnp.array]
        '''
        mean, std = self.policy_network.forward(state)
        action = mean + std * jax.random.normal(self.rng_key, shape=mean.shape)
        action = jnp.clip(action, self.action_min, self.action_max)
        return action

    def store_transition(self, state, action, reward, next_state, done):
        '''
        Store a transition in the replay buffer.
        params:
        state: Current state [jnp.array]
        action: Executed action [jnp.array]
        reward: Received reward [float]
        next_state: Next state [jnp.array]
        done: Terminal flag [bool]
        '''
        self.replay_buffer.add(state, action, reward, next_state, done)

    def critic_update(self, states, actions, rewards, next_states, dones):
        '''
        Update the critic network.
        params:
        states: States sampled from the replay buffer [jnp.array]
        actions: Actions sampled from the replay buffer [jnp.array]
        rewards: Rewards sampled from the replay buffer [jnp.array]
        next_states: Next states sampled from the replay buffer [jnp.array]
        dones: Done flags sampled from the replay buffer [jnp.array]

        returns:
        target_q_values: Target Q-values for the critic update [jnp.array]

        Note, different critic configurations:
        - Double critic
        - Distributional critic
        - Double distributional critic
        - Normal critic
        '''
        # 1. Compute the target TD
        next_action = self.policy(next_state)  # Compute the next action using the policy.
        

        

        

        # Double Distributional: Q_Z(s, a) = r + \gamma * (1 - done) * min(z_1', z_2')
        def compute_td_target_double_distributional(reward, next_state, not_done, target_critic, gamma, support, delta_z):
            next_action = target_critic.policy(next_state)  # Compute next action
            next_dist1, next_dist2 = target_critic.forward(next_state, next_action)  # Two distributions

            # Minimum distribution
            next_dist_min = jnp.minimum(next_dist1, next_dist2)

            # Compute projected distribution
            projected_dist = jnp.zeros_like(next_dist_min)
            for i, z_i in enumerate(support):
                tz = reward + gamma * not_done * z_i
                tz = jnp.clip(tz, support[0], support[-1])  # Clamp to the range of support
                b = (tz - support[0]) / delta_z  # Map to bin
                lower, upper = jnp.floor(b).astype(int), jnp.ceil(b).astype(int)
                projected_dist[:, lower] += next_dist_min[:, i] * (upper - b)
                projected_dist[:, upper] += next_dist_min[:, i] * (b - lower)
            return projected_dist

        # Compute the target Q-values


        # 2. Compute the critic loss


        

        return target_q_values

    def update(self, batch_size):
        '''
        Update the policy and critic networks.
        params:
        batch_size: Batch size [int]
        '''
        # Sample a batch from the replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size, self.rng_key)

        # 1. Perform the critic update (Q-function update)
        target_q_values = self.critic_update(states, actions, rewards, next_states, dones)

        # 2. Perform the E-step (Target policy update)
        weights, sampled_actions = self.e_step(states, target_q_values)

        # 3. Perform the M-step (Parametric policy update)
        self.m_step(states, weights, sampled_actions)

        # 4. Update the temperature parameter
        self.temperature_update()



    def e_step(self, batch_size):
        '''
        NOT DONE
        Perform an E-step of the MPO algorithm.
        i.e. target policy update
        \zeta_{\text{S, target}}(s_t; \theta^{\zeta_{\text{target}}}) \varpropto  \exp\left(\frac{Q(s, a)}{\nu}\right)

        params:
        '''

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