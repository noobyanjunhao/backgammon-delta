# smoke_jax_gpu.py
import os
# Configure JAX for better GPU compatibility
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'
# Disable parallel compilation to avoid CUDA version mismatch issues
os.environ['XLA_FLAGS'] = '--xla_gpu_force_compilation_parallelism=1'

import jax
import jax.numpy as jnp

from backgammon_value_net import BackgammonValueNet, BOARD_LENGTH, CONV_INPUT_CHANNELS, AUX_INPUT_SIZE

def main():
    print("JAX version:", jax.__version__)
    print("JAX devices:", jax.devices())
    print("Default backend:", jax.default_backend())
    print("Number of devices:", len(jax.devices()))
    
    # Test basic JAX operation first
    print("\nTesting basic JAX operation...")
    test_array = jnp.array([1.0, 2.0, 3.0])
    result = jnp.sum(test_array)
    print(f"Basic JAX test passed: sum = {result}")
    
    print("\nInitializing model...")
    key = jax.random.key(0)

    # Batch=32; board planes (B, 24, C); aux (B, 6)
    print("Creating input tensors...")
    board = jnp.zeros((32, BOARD_LENGTH, CONV_INPUT_CHANNELS), dtype=jnp.float32)
    aux   = jnp.zeros((32, AUX_INPUT_SIZE), dtype=jnp.float32)
    print(f"Board shape: {board.shape}, Aux shape: {aux.shape}")

    print("Creating model...")
    model = BackgammonValueNet()
    
    print("Initializing model parameters...")
    board_flat = board.reshape(32, -1)
    init_variables = model.init(key, board_state=board_flat, aux_features=aux)
    params = init_variables['params']
    print("Model initialized successfully")
    
    print("Running forward pass...")
    v = model.apply({'params': params}, board_state=board_flat, aux_features=aux)
    print("Forward pass completed!")

    print("Forward OK. v.shape =", v.shape, "v[:5] =", v[:5])

if __name__ == "__main__":
    main()

