import jax
import jax.numpy as jnp
from collections import deque

from mpo_trial import PrioritizedReplayBuffer, kl_divergence_multivariate_gaussian, gaussian_likelihood
from mpo_trial import DoubleDistributionalCritic, calculate_td_error, clip_grads, project_distribution
from mpo_trial import compute_critic_loss, HybridMPO

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

def test_clip_grads(rng):
    # Create dummy gradients
    grads = {
        "w1": jax.random.normal(rng, (3, 3)),
        "w2": jax.random.normal(rng, (3,))
    }
    max_norm = 1.0

    # Clip gradients
    clipped_grads = clip_grads(grads, max_norm)

    # Compute norms
    original_norm = jnp.sqrt(sum(jnp.sum(jnp.square(x)) for x in jax.tree_util.tree_leaves(grads)))
    clipped_norm = jnp.sqrt(sum(jnp.sum(jnp.square(x)) for x in jax.tree_util.tree_leaves(clipped_grads)))

    # Assertions
    assert isinstance(clipped_grads, dict), "Clipped gradients are not in the same format as input"
    assert clipped_norm <= max_norm, "Clipped gradients exceed max_norm"
    assert clipped_norm <= original_norm, "Clipped gradients norm is greater than original norm"

def test_project_distribution():
    batch_size = 4
    rewards = jnp.array([1.0, 2.0, 3.0, 4.0])
    dones = jnp.array([0.0, 1.0, 0.0, 0.0])
    gamma = 0.99
    next_dist = jnp.ones((batch_size, num_points)) / num_points

    projected = project_distribution(next_dist, z, rewards, gamma, dones)

    assert projected.shape == (batch_size, num_points), "Shape mismatch in projected distribution"
    assert jnp.allclose(jnp.sum(projected, axis=-1), 1.0), "Projected distributions are not normalized"
    assert jnp.all(projected >= 0), "Projected distributions contain negative values"

def test_compute_critic_loss():
    batch_size = 4
    q1_logits = jax.random.normal(rng, (batch_size, num_points))
    q2_logits = jax.random.normal(rng, (batch_size, num_points))
    target_dist = jnp.ones((batch_size, num_points)) / num_points
    weights = jnp.ones(batch_size)

    loss = compute_critic_loss(q1_logits, q2_logits, target_dist, weights)

    assert isinstance(loss, jnp.ndarray), "Loss is not a JAX array"
    assert loss.shape == (), "Loss is not a scalar"
    assert loss >= 0, "Loss is negative"

def test_critic_update():
    hybrid_mpo = HybridMPO(
        state_dim=state_dim,
        action_dim=action_dim,
        buffer_size=1000,
        hidden_dim_actor=256,
        gamma=0.99,
        alpha=0.6,
        beta=0.4,
        beta_decay=0.001,
        batch_size=batch_size
    )

    # Fill replay buffer with dummy data
    for _ in range(100):
        state = jax.random.normal(rng, (state_dim,))
        action = jax.random.normal(rng, (action_dim,))
        reward = jax.random.uniform(rng, ())
        next_state = jax.random.normal(rng, (state_dim,))
        done = jax.random.bernoulli(rng, p=0.1)
        td_error = jax.random.uniform(rng, ())
        hybrid_mpo.buffer.add(state, action, reward, next_state, done, td_error)

    # Call critic_update
    critic_loss = hybrid_mpo.critic_update()

    assert isinstance(critic_loss, jnp.ndarray), "Critic loss is not a JAX array"
    assert critic_loss.shape == (), "Critic loss is not a scalar"
    assert critic_loss >= 0, "Critic loss is negative"


# Run Tests
if __name__ == "__main__":
    #test_prioritized_replay_buffer_initialization()
    #test_add_and_sample()
    #test_update_priorities()
    #test_kl_divergence_multivariate_gaussian()
    #test_gaussian_likelihood()
    #test_double_distributional_critic_td_error()
    #test_prioritized_replay_buffer_n_step()
    #print("All tests (1) passed!")

    # Initialize RNG
    rng = jax.random.PRNGKey(0)

    # Test Parameters
    state_dim = 4
    action_dim = 2
    batch_size = 32
    num_points = 51
    v_min = -10.0
    v_max = 10.0

    # Support for distributional critic
    z = jnp.linspace(v_min, v_max, num_points)

    test_clip_grads(rng)
    test_project_distribution()
    test_critic_update()

    print("All tests (2) passed!")
