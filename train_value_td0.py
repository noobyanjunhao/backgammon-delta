import time
import numpy as np
import jax
import jax.numpy as jnp
import optax
from flax.training import train_state

from backgammon_engine import (
    _new_game,
    _actions,
    _to_canonical,
    NUM_CHECKERS,
    HOME_BOARD_SIZE,
    NUM_POINTS,
    W_BAR,
    B_BAR,
    W_OFF,
    B_OFF,
)
from backgammon_value_net import (
    BackgammonValueNet as ValueNet,
    BOARD_LENGTH,
    CONV_INPUT_CHANNELS,
    AUX_INPUT_SIZE,
)


def py_reward(state, player):
    """
    Pure Python / NumPy version of _reward to avoid Numba typing issues.

    state: array-like of shape (28,), dtype int8
    player: +1 or -1
    """
    s = np.asarray(state, dtype=np.int8)
    p = int(player)

    # White has borne off all checkers
    if s[W_OFF] == NUM_CHECKERS:
        # if the losing player has borne off at least one checker,
        # he loses only one point
        if s[B_OFF] < 0:
            return p
        else:
            # if the loser has not borne off any of his checkers, he is
            # gammoned and loses two points
            #
            # if the loser has not borne off any of his checkers and still has
            # a checker on the bar or in the winner's home board, he is
            # backgammoned and loses three points
            if s[B_BAR] < 0:
                return 3 * p
            for pt in range(1, int(HOME_BOARD_SIZE) + 1):
                if s[pt] < 0:
                    return 3 * p
            return 2 * p

    # Black has borne off all checkers
    if s[B_OFF] == -NUM_CHECKERS:
        if s[W_OFF] > 0:
            return -p
        else:
            if s[W_BAR] > 0:
                return -3 * p
            for pt in range(3 * int(HOME_BOARD_SIZE) + 1, int(NUM_POINTS) + 1):
                if s[pt] > 0:
                    return -3 * p
            return -2 * p

    return 0


def encode(state, player):
    """
    Encode a 28-dim canonical engine state into:
      - board planes: (BOARD_LENGTH, CONV_INPUT_CHANNELS)
      - aux features: (AUX_INPUT_SIZE,)

    This is a simple placeholder encoding that:
      - maps point occupancies (indices 1..24) into the first channel
      - uses bar/off counts in aux features.
    You should replace this with a richer encoding later.
    """
    # canonicalize so the net always sees "current player as white"
    # Ensure types are plain numpy for numba-jitted _to_canonical
    state_arr = np.asarray(state, dtype=np.int8)
    player_int = np.int8(player)
    s = _to_canonical(state_arr, player_int).astype(np.float32)
    # Flatten in case any unexpected extra dimensions creep in
    s = np.asarray(s, dtype=np.float32).ravel()

    board = np.zeros((BOARD_LENGTH, CONV_INPUT_CHANNELS), dtype=np.float32)
    # Points 1..24 go to first channel, indices 0..23
    pts = s[1 : 1 + BOARD_LENGTH]
    # Only assign if we have exactly BOARD_LENGTH points; otherwise leave zeros.
    if pts.shape[0] == BOARD_LENGTH:
        board[:, 0] = pts / 15.0

    aux = np.zeros((AUX_INPUT_SIZE,), dtype=np.float32)
    # Very simple aux: bar and off counts (scaled), plus side-to-move
    # W_BAR = 0, B_BAR = 25, W_OFF = 26, B_OFF = 27 in engine
    aux[0] = 1.0  # side to move (always +1 in canonical)
    if AUX_INPUT_SIZE > 1 and s.shape[0] > 0:
        aux[1] = float(np.asarray(s[0]).ravel()[0]) / 15.0   # bar
    if AUX_INPUT_SIZE > 2 and s.shape[0] > 25:
        aux[2] = float(np.asarray(s[25]).ravel()[0]) / 15.0  # opponent bar (canonical coords)
    if AUX_INPUT_SIZE > 3 and s.shape[0] > 26:
        aux[3] = float(np.asarray(s[26]).ravel()[0]) / 15.0  # off
    if AUX_INPUT_SIZE > 4 and s.shape[0] > 27:
        aux[4] = float(np.asarray(s[27]).ravel()[0]) / 15.0  # opponent off

    return board, aux


@jax.jit
def v_apply(params, board, aux):
    # board: (B, 24, 15); aux: (B, AUX_INPUT_SIZE)
    # Call the module with explicit keyword arguments to match __call__ signature.
    return ValueNet().apply(
        {"params": params},
        board_state=board,
        aux_features=aux,
    ).squeeze(-1)


@jax.jit
def loss_and_grads(params, board, aux, target):
    pred = v_apply(params, board, aux)
    loss = jnp.mean((pred - target) ** 2)

    def loss_fn(p):
        return jnp.mean((v_apply(p, board, aux) - target) ** 2)

    grads = jax.grad(loss_fn)(params)
    return loss, grads


def main(steps=5000, lr=3e-4, eps_greedy=0.10, batch=256, seed=0):
    key = jax.random.PRNGKey(seed)

    # init model with correctly-shaped dummy inputs
    dummy_board = jnp.zeros((1, BOARD_LENGTH, CONV_INPUT_CHANNELS), dtype=jnp.float32)
    dummy_aux = jnp.zeros((1, AUX_INPUT_SIZE), dtype=jnp.float32)
    params = ValueNet().init(
        key,
        board_state=dummy_board,
        aux_features=dummy_aux,
    )["params"]
    tx = optax.adam(lr)
    st = train_state.TrainState.create(apply_fn=None, params=params, tx=tx)

    buf_board, buf_aux, buf_tgt = [], [], []

    def pick_action(state, player):
        # roll dice + enumerate legal afterstates
        dice = np.random.randint(1, 7, size=(2,), dtype=np.int8)
        # _actions expects (state, current_player, dice)
        afterstates, _paths = _actions(state, player, dice)
        if len(afterstates) == 0:
            return 0, state  # no-op fallback
        # epsilon-greedy: random action sometimes
        if np.random.rand() < eps_greedy:
            ns = afterstates[np.random.randint(len(afterstates))]
            r = py_reward(ns, player)
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
        r = py_reward(ns, player)
        return r, ns

    t0 = time.time()
    # _new_game() returns (player, dice, state)
    player, _dice, state = _new_game()

    for it in range(1, steps + 1):
        b, a = encode(state, player)
        r, next_state = pick_action(state, player)

        if r != 0:
            tgt = float(r)
            buf_board.append(b); buf_aux.append(a); buf_tgt.append(tgt)
            # start a fresh game
            player, _dice, state = _new_game()
        else:
            # TD(0): target = -V(next_state, next_player)
            nb, na = encode(next_state, -player)
            # encode returns single state (24, 15) and (AUX_INPUT_SIZE,)
            # add batch dimension so v_apply sees shape (1, 24, 15) and (1, AUX_INPUT_SIZE)
            nb_batch = jnp.asarray(nb)[None, ...]
            na_batch = jnp.asarray(na)[None, ...]
            v_next = float(v_apply(st.params, nb_batch, na_batch)[0])
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

