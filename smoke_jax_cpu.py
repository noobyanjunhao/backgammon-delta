# smoke_jax_cpu.py - CPU fallback version to test if code works
import os
# Force CPU to test if the issue is GPU-related
os.environ['JAX_PLATFORMS'] = 'cpu'

import jax
import jax.numpy as jnp

from backgammon_value_net import BackgammonValueNet, BOARD_LENGTH, CONV_INPUT_CHANNELS, AUX_INPUT_SIZE

def main():
    print("JAX version:", jax.__version__)
    print("JAX devices:", jax.devices())
    print("Default backend:", jax.default_backend())
    
    key = jax.random.key(0)
    board = jnp.zeros((32, BOARD_LENGTH, CONV_INPUT_CHANNELS), dtype=jnp.float32)
    aux   = jnp.zeros((32, AUX_INPUT_SIZE), dtype=jnp.float32)

    model = BackgammonValueNet()
    board_flat = board.reshape(32, -1)
    init_variables = model.init(key, board_state=board_flat, aux_features=aux)
    params = init_variables['params']
    v = model.apply({'params': params}, board_state=board_flat, aux_features=aux)

    print("Forward OK (CPU). v.shape =", v.shape, "v[:5] =", v[:5])

if __name__ == "__main__":
    main()

