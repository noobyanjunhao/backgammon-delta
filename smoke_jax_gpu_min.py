# Minimal GPU smoke test - no fancy CUDA path edits
import jax
import jax.numpy as jnp

print("JAX version:", jax.__version__)
print("Devices:", jax.devices())
print("Default backend:", jax.default_backend())

# Test GPU computation
x = jnp.ones((4096, 4096), dtype=jnp.float32)
y = x @ x
y.block_until_ready()
print("OK, matmul done on:", y.device())
print("Result shape:", y.shape, "sum:", float(jnp.sum(y)))

