import jax.numpy as jnp
import jax
from flax import optim
# Flax simplifies gradient updates with built-in optimizers, making code cleaner and easier to scale for complex updates.

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
    D_KL(P || Q) = 0.5 * [log(std_q^2 / std_p^2) - 1 + (std_p^2 + (mu_p - mu_q)^2) /std_q^2]
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
def mean_lagrange_step(optimizer: optim.Optimizer, reg: float) -> optim.Optimizer:
    '''
    Update the mean Lagrange multiplier using the KL constraint violation.

    params:
    optimizer: Optimizer for the mean Lagrange multiplier [optim.Optimizer]
    reg: Regularization term (eps_mu - KL_mu) [float]

    returns:
    Updated optimizer with adjusted Lagrange multiplier.
    '''
    def loss_fn(mean_lagrange_params):
        # Scale the Lagrange multiplier by 1.0
        return jnp.sum(mean_lagrange_params * reg)

    grad = jax.grad(loss_fn)(optimizer.target)
    return optimizer.apply_gradient(grad)

@jax.jit
def std_lagrange_step(optimizer: optim.Optimizer, reg: float) -> optim.Optimizer:
    '''
    Update the variance Lagrange multiplier using the KL constraint violation.

    params:
    optimizer: Optimizer for the variance Lagrange multiplier [optim.Optimizer]
    reg: Regularization term (eps_sig - KL_sig) [float]

    returns:
    Updated optimizer with adjusted Lagrange multiplier.
    '''
    def loss_fn(std_lagrange_params):
        # Scale the Lagrange multiplier by 100.0
        return jnp.sum(std_lagrange_params * reg * 100.0)

    grad = jax.grad(loss_fn)(optimizer.target)
    return optimizer.apply_gradient(grad)

