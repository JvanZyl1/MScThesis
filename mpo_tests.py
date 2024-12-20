import jax
import jax.numpy as jnp
from collections import deque

from mpo_trial import PrioritizedReplayBuffer, kl_divergence_multivariate_gaussian, gaussian_likelihood
from mpo_trial import DoubleDistributionalCritic, calculate_td_error

def test_prioritized_replay_buffer_initialization():
    buffer = PrioritizedReplayBuffer(capacity=10, n_step=3, gamma=0.99)
    assert len(buffer) == 0, "Buffer should be empty after initialization."
    assert buffer.n_step == 3, "n_step should be initialized correctly."
    assert buffer.capacity == 10, "Capacity should match initialization value."
    assert isinstance(buffer.n_step_buffer, deque), "n_step_buffer should be a deque."

def test_add_and_sample():
    buffer = PrioritizedReplayBuffer(capacity=5, n_step=3, gamma=0.99)
    rng_key = jax.random.PRNGKey(0)
    td_error = 1.0

    # Add transitions
    for i in range(5):
        buffer.add(
            state=jnp.array([i]), 
            action=jnp.array([i]), 
            reward=i, 
            next_state=jnp.array([i + 1]), 
            done=False, 
            td_error=td_error
        )

    # Sample
    batch_size = 3
    states, actions, rewards, next_states, dones, indices, weights = buffer(batch_size, rng_key)
    assert states.shape == (batch_size, 1), "States batch size mismatch."
    assert actions.shape == (batch_size, 1), "Actions batch size mismatch."

def test_update_priorities():
    buffer = PrioritizedReplayBuffer(capacity=5)
    for i in range(5):
        buffer.add(
            state=jnp.array([i]),
            action=jnp.array([i]),
            reward=i,
            next_state=jnp.array([i + 1]),
            done=False,
            td_error=1.0,
        )
    indices = jnp.array([0, 1, 2])
    new_priorities = jnp.array([0.1, 0.5, 0.9])
    buffer.update_priorities(indices, new_priorities)
    assert jnp.allclose(buffer.priorities[indices], new_priorities), "Priorities not updated correctly."

def test_kl_divergence_multivariate_gaussian():
    mean_p = jnp.array([0.0, 0.0])
    std_p = jnp.array([1.0, 1.0])
    mean_q = jnp.array([1.0, 1.0])
    std_q = jnp.array([2.0, 2.0])
    kl = kl_divergence_multivariate_gaussian(mean_p, std_p, mean_q, std_q)
    assert kl.shape == (), "KL divergence should return a scalar."
    assert kl > 0, "KL divergence should be positive."

def test_gaussian_likelihood():
    actions = jnp.array([1.0, 2.0])
    mean = jnp.array([0.0, 0.0])
    std = jnp.array([1.0, 1.0])
    log_likelihood = gaussian_likelihood(actions, mean, std)
    assert log_likelihood.shape == (), "Log likelihood should return a scalar."
    assert log_likelihood < 0, "Log likelihood should be negative."

def test_double_distributional_critic_td_error():
    critic = DoubleDistributionalCritic(state_dim=4, action_dim=2)
    rng_key = jax.random.PRNGKey(0)
    params = critic.init(rng_key, jnp.ones((1, 4)), jnp.ones((1, 2)))
    state = jnp.ones((1, 4))
    action = jnp.ones((1, 2))
    q1, q2, z = critic.apply(params, state, action)

    rewards = jnp.array([1.0])
    not_done = jnp.array([1.0])
    next_dist = jnp.ones_like(q1) / q1.shape[-1]  # Uniform distribution
    td_error_q1, td_error_q2 = calculate_td_error(q1, q2, z, rewards, not_done, next_dist)

    assert td_error_q1.shape == (1,), "TD error shape mismatch for q1."
    assert td_error_q2.shape == (1,), "TD error shape mismatch for q2."


def test_prioritized_replay_buffer_n_step():
    buffer = PrioritizedReplayBuffer(capacity=10, n_step=3, gamma=0.99)
    td_error = 1.0
    # Add transitions
    buffer.add(jnp.array([0]), jnp.array([0]), 1.0, jnp.array([1]), False, td_error)
    buffer.add(jnp.array([1]), jnp.array([1]), 1.0, jnp.array([2]), False, td_error)
    buffer.add(jnp.array([2]), jnp.array([2]), 1.0, jnp.array([3]), False, td_error)

    reward, next_state, done = buffer._compute_n_step()
    assert reward == 1.0 + 0.99 * 1.0 + 0.99 ** 2 * 1.0, "Incorrect n-step reward calculation."
    assert jnp.allclose(next_state, jnp.array([3])), "Incorrect n-step next state."

# Run Tests
if __name__ == "__main__":
    test_prioritized_replay_buffer_initialization()
    test_add_and_sample()
    test_update_priorities()
    test_kl_divergence_multivariate_gaussian()
    test_gaussian_likelihood()
    test_double_distributional_critic_td_error()
    test_prioritized_replay_buffer_n_step()
    print("All tests passed!")
