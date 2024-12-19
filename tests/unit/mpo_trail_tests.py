import jax
import gymnasium as gym
from src.agents.mpo import MPOLearner
from src.agents.qnetworks import DoubleDistributionalCriticNetwork
from configs.agents_parameters import config

jax.clear_caches()

import jax.numpy as jnp

# RUN: python -m tests.unit.mpo_trail_tests

def test_mpo_trainer():
    # Initialize the environment
    env = gym.make("LunarLanderContinuous-v2")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # Configure the MPO learner
    agent = MPOLearner(
        state_dim=state_dim,
        action_dim=action_dim,
        config=config,
    )

    # Create dummy inputs
    batch_size = 10
    states = jax.random.normal(jax.random.PRNGKey(0), (batch_size, state_dim))
    actions = jax.random.normal(jax.random.PRNGKey(1), (batch_size, action_dim))
    rewards = jax.random.normal(jax.random.PRNGKey(2), (batch_size,))
    next_states = jax.random.normal(jax.random.PRNGKey(3), (batch_size, state_dim))
    dones = jax.random.randint(jax.random.PRNGKey(4), (batch_size,), 0, 2)

    # Test critic update
    critic_loss = agent.critic_update(states, actions, rewards, next_states, dones)
    assert critic_loss.shape == (), "Critic loss should be a scalar"

    # Test actor update
    agent.e_step(states, batch_size, config['action_sample_size'])
    actor_loss = agent.m_step(states, agent.weights, agent.sampled_actions)
    assert actor_loss.shape == (), "Actor loss should be a scalar"

    # Test DoubleDistributionalCriticNetwork
    critic = DoubleDistributionalCriticNetwork(state_dim, action_dim, config['hidden_dim'], config['num_points'], config['support_range'])
    dist_1, dist_2 = critic.forward(states, actions)
    assert dist_1.shape == (batch_size, config['num_points']), "Distribution 1 shape mismatch"
    assert dist_2.shape == (batch_size, config['num_points']), "Distribution 2 shape mismatch"

    q_value = critic.evaluate_q_value(states, actions)
    assert q_value.shape == (batch_size, 1), "Q-value shape mismatch"

    td_target = critic.compute_td_target(rewards, next_states, actions, 1 - dones, config['gamma'])
    assert td_target.shape == (batch_size, config['num_points']), "TD target shape mismatch"

    print("All tests passed!")

if __name__ == "__main__":
    test_mpo_trainer()