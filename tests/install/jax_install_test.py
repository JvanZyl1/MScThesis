import jax

print(jax.devices())

import jax.numpy as jnp

# Function to differentiate
def my_func(x):
    return jnp.sin(x) ** 2

# Gradient computation
grad_func = jax.grad(my_func)
x = 1.0
print(grad_func(x))  # Output: 2 * sin(x) * cos(x)

def my_func(x, y):
    return x ** 2 + y ** 3

# Compute gradients w.r.t. x and y
grad_func = jax.grad(my_func, argnums=(0, 1))
x, y = 1.0, 2.0
print(grad_func(x, y))  # Output: (2.0, 12.0)
