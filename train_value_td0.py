import time
import numpy as np
import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from backgammon_engine import _new_game, _actions, _reward, _to_canonical
from backgammon_value_net import BackgammonValueNet as ValueNet

def encode(state, player):
    # canonicalize so the net always sees "current player as white"
    s = _to_canonical(state, player).astype(np.float32)
    board = s / 15.0  # simple scaling
    aux = np.zeros((8,), dtype=np.float32)  # placeholder
    return board, aux

@jax.jit
def v_apply(params, board, aux):
    return ValueNet().apply({"params": params}, board, aux).squeeze(-1)

@jax.jit
def loss_and_grads(params, board, aux, target):
    pred = v_apply(params, board, aux)
    loss = jnp.mean((pred - target) ** 2)
    return loss, jax.grad(lambda p: jnp.mean((v_apply(p, board, aux) - target) ** 2))(params)

def main(steps=5000, lr=3e-4, eps_greedy=0.10, batch=256, seed=0):
    key = jax.random.PRNGKey(seed)

    # init model
    dummy_board = jnp.zeros((28,), dtype=jnp.float32)
    dummy_aux = jnp.zeros((8,), dtype=jnp.float32)
    params = ValueNet().init(key, dummy_board, dummy_aux)["params"]
    tx = optax.adam(lr)
    st = train_state.TrainState.create(apply_fn=None, params=params, tx=tx)

    buf_board, buf_aux, buf_tgt = [], [], []

    def pick_action(state, player):
        # roll dice + enumerate legal afterstates
        dice = np.random.randint(1, 7, size=(2,), dtype=np.int8)
        afterstates, _paths = _actions(state, dice, player)
        if len(afterstates) == 0:
            return 0, state  # no-op fallback
        # epsilon-greedy: random action sometimes
        if np.random.rand() < eps_greedy:
            ns = afterstates[np.random.randint(len(afterstates))]
            r = _reward(ns, player)
            return r, ns
        # greedy by value of afterstate from next player's perspective (zero-sum)
        boards = []
        auxs = []
        for ns in afterstates:
            b,a = encode(ns, -player)  # value net predicts for side-to-move
            boards.append(b); auxs.append(a)
        boards = jnp.asarray(np.stack(boards))
        auxs = jnp.asarray(np.stack(auxs))
        vals = v_apply(st.params, boards, auxs)
        best = int(jnp.argmin(vals))  # minimize opponent's value
        ns = afterstates[best]
        r = _reward(ns, player)
        return r, ns

    t0 = time.time()
    state, player = _new_game()

    for it in range(1, steps + 1):
        b, a = encode(state, player)
        r, next_state = pick_action(state, player)

        if r != 0:
            tgt = float(r)
            buf_board.append(b); buf_aux.append(a); buf_tgt.append(tgt)
            state, player = _new_game()
        else:
            # TD(0): target = -V(next_state, next_player)
            nb, na = encode(next_state, -player)
            v_next = float(v_apply(st.params, jnp.asarray(nb), jnp.asarray(na)))
            tgt = -v_next
            buf_board.append(b); buf_aux.append(a); buf_tgt.append(tgt)
            state, player = next_state, -player

        if len(buf_tgt) >= batch:
            board = jnp.asarray(np.stack(buf_board))
            aux = jnp.asarray(np.stack(buf_aux))
            target = jnp.asarray(np.array(buf_tgt, dtype=np.float32))
            loss, grads = loss_and_grads(st.params, board, aux, target)
            st = st.apply_gradients(grads=grads)
            buf_board, buf_aux, buf_tgt = [], [], []

        if it % 500 == 0:
            dt = time.time() - t0
            print(f"step {it}/{steps}  last_loss={float(loss):.4f}  elapsed={dt:.1f}s")

    print("done")

if __name__ == "__main__":
    main()

