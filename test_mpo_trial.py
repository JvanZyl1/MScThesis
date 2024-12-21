import jax
import unittest
from mpo_trial import HybridMPO, DoubleDistributionalCritic, Actor, PrioritizedReplayBuffer

import jax.numpy as jnp

class TestHybridMPO(unittest.TestCase):

    def setUp(self):
        self.state_dim = 4
        self.action_dim = 2
        self.buffer_size = 1000
        self.hidden_dim_actor = 256
        self.gamma = 0.99
        self.alpha = 0.6
        self.beta = 0.4
        self.beta_decay = 0.001
        self.batch_size = 256
        self.tau = 0.005
        self.critic_lr = 3e-4
        self.critic_grad_max_norm = 10
        self.temperature_updated_learned_bool = False

        self.hybrid_mpo = HybridMPO(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            buffer_size=self.buffer_size,
            hidden_dim_actor=self.hidden_dim_actor,
            gamma=self.gamma,
            alpha=self.alpha,
            beta=self.beta,
            beta_decay=self.beta_decay,
            batch_size=self.batch_size,
            tau=self.tau,
            critic_lr=self.critic_lr,
            critic_grad_max_norm=self.critic_grad_max_norm,
            temperature_updated_learned_bool=self.temperature_updated_learned_bool
        )

    def test_critic_update(self):
        # Add some dummy data to the buffer
        for _ in range(self.buffer_size):
            state = jnp.zeros(self.state_dim)
            action = jnp.zeros(self.action_dim)
            reward = 0.0
            next_state = jnp.zeros(self.state_dim)
            done = False
            td_error = 0.0
            self.hybrid_mpo.buffer.add(state, action, reward, next_state, done, td_error)

        # Perform a critic update
        critic_loss = self.hybrid_mpo.critic_update()
        self.assertIsNotNone(critic_loss)

    def test_sample_actions(self):
        states = jnp.zeros((self.batch_size, self.state_dim))
        actions = self.hybrid_mpo.sample_actions(states)
        self.assertEqual(actions.shape, (self.batch_size, self.action_dim))

    def test_temperature_update_learned(self):
        q_logits = jnp.zeros((self.batch_size, self.hybrid_mpo.critic.num_points))
        self.hybrid_mpo.temperature_update_learned(q_logits)
        self.assertGreaterEqual(self.hybrid_mpo.temp, 1e-6)

    def test_temperature_update(self):
        q_logits = jnp.zeros((self.batch_size, self.hybrid_mpo.critic.num_points))
        self.hybrid_mpo.temperature_update(q_logits)
        self.assertGreaterEqual(self.hybrid_mpo.temp, 1e-6)

    def test_e_step(self):
        states = jnp.zeros((self.batch_size, self.state_dim))
        weights, sampled_actions = self.hybrid_mpo.e_step(states)
        self.assertEqual(weights.shape, (self.batch_size, self.hybrid_mpo.critic.num_points, 1))
        self.assertEqual(sampled_actions.shape, (self.batch_size, self.action_dim))

if __name__ == "__main__":
    unittest.main()