import jax
import jax.numpy as jnp
from typing import Any

jax.clear_caches()

def clip_grads(grads: Any, max_norm: float) -> Any:
    """
    Clip gradients to avoid exploding gradients.
    
    params:
    grads: Gradients to be clipped [Any]
    max_norm: Maximum norm for the gradients [float]
    
    returns:
    clipped_grads: Clipped gradients [Any]
    """
    # Initialize total_norm
    total_norm = 0.0

    # Compute the L2 norm of the gradients
    for g in jax.tree_util.tree_leaves(grads):
        if g is not None:  # Ensure the gradient is not None
            total_norm += jnp.sum(g ** 2)
    total_norm = jnp.sqrt(total_norm)

    # Compute the clipping factor
    clip_factor = jnp.minimum(1.0, max_norm / (total_norm + 1e-6))

    # Define a function to apply the clipping factor
    def apply_clip_factor(grad):
        return grad * clip_factor

    # Apply the clipping factor to the gradients
    clipped_grads = jax.tree_util.tree_map(apply_clip_factor, grads)
    
    return clipped_grads
