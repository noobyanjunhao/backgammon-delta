# smoke_jax_gpu.py
import jax
import jax.numpy as jnp

from backgammon_value_net import BackgammonValueNet, BOARD_LENGTH, CONV_INPUT_CHANNELS, AUX_INPUT_SIZE

def main():
    print("JAX devices:", jax.devices())
    print("Default backend:", jax.default_backend())

    key = jax.random.key(0)

    # Batch=32; board planes (B, 24, C); aux (B, 6)
    board = jnp.zeros((32, BOARD_LENGTH, CONV_INPUT_CHANNELS), dtype=jnp.float32)
    aux   = jnp.zeros((32, AUX_INPUT_SIZE), dtype=jnp.float32)

    model = BackgammonValueNet()
    init_variables = model.init(key, board_state=board.reshape(32, -1), aux_features=aux)  # your net reshapes internally
    params = init_variables['params']
    v = model.apply({'params': params}, board_state=board.reshape(32, -1), aux_features=aux)

    print("Forward OK. v.shape =", v.shape, "v[:5] =", v[:5])

if __name__ == "__main__":
    main()

