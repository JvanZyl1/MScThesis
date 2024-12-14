import jax.numpy as jnp
import jax
import optax
from typing import Tuple

jax.clear_backends()

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
def std_lagrange_step(optimizer: optax.GradientTransformation, opt_state: optax.OptState, params: optax.Params, reg: float) -> Tuple[optax.Params, optax.OptState]:
    '''
    Update the variance Lagrange multiplier using the KL constraint violation.

    params:
    optimizer: Optimizer for the variance Lagrange multiplier [optax.GradientTransformation]
    opt_state: Optimizer state [optax.OptState]
    params: Current parameters of the optimizer [optax.Params]
    reg: Regularization term (eps_sig - KL_sig) [float]

    returns:
    Updated parameters and optimizer state with adjusted Lagrange multiplier.
    '''
    def loss_fn(std_lagrange_params):
        # Scale the Lagrange multiplier by 100.0
        return jnp.sum(std_lagrange_params * reg * 100.0)

    grad = jax.grad(loss_fn)(params)
    updates, new_opt_state = optimizer.update(grad, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state