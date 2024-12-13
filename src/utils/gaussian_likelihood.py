import jax.numpy as jnp

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
        + 2 * std  # Log scale normalization
        + jnp.log(2 * jnp.pi)  # Constant factor
    )
    return log_prob.sum(axis=-1)  # Sum over the action dimensions
