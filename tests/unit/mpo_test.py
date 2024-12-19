import jax
import jax.numpy as jnp
from src.agents.mpo import MPOLearner
from configs.agents_parameters import config

jax.clear_caches()

# python -m tests.unit.mpo_test

def test_mpo_update():
    # Dummy inputs
    state_dim = 4
    action_dim = 2
    batch_size = 10

    # Initialize the MPOLearner
    agent = MPOLearner(
        state_dim=state_dim,
        action_dim=action_dim,
        config=config,
    )

    # Call the update method
    critic_loss, actor_loss = agent.update(batch_size)

    # Check the shapes of the outputs
    assert critic_loss.shape == (), "Critic loss should be a scalar"
    assert actor_loss.shape == (), "Actor loss should be a scalar"

    print("All tests passed!")

if __name__ == "__main__":
    test_mpo_update()