# Minimal GPU smoke test - let JAX use its bundled CUDA libraries
import os

# Unset CUDA paths - JAX uses its bundled CUDA libraries
# Don't override them with system CUDA paths
for key in ['LD_LIBRARY_PATH', 'CUDA_HOME', 'CUDA_ROOT', 'CUDA_PATH']:
    if key in os.environ:
        del os.environ[key]

import jax
import jax.numpy as jnp

print("JAX version:", jax.__version__)
print("Devices:", jax.devices())
print("Default backend:", jax.default_backend())

# Test GPU computation
x = jnp.ones((2048, 2048), dtype=jnp.float32)
y = x @ x
y.block_until_ready()
print("OK, matmul done on:", y.device)  # device is a property, not a function
print("Result shape:", y.shape, "sum:", float(jnp.sum(y)))

