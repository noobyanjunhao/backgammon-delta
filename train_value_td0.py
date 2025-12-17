import os
import time
import numpy as np
import jax
import jax.numpy as jnp
import optax
from flax.training import train_state, checkpoints

from backgammon_engine import (
    _new_game,
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
    # If state does not have full length, treat as non-terminal
    if s.shape[0] < 28:
        return 0
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


def can_bear_off_py(state, player):
    """Pure Python version of _can_bear_off."""
    s = np.asarray(state, dtype=np.int8)
    bar_index = int(W_BAR if player == 1 else B_BAR)
    # Ensure we compare scalar values, not arrays
    if int(s[bar_index]) * int(player) > 0:
        return False
    # White's home board: points 19-24 (indices 19-24)
    if player == 1:
        for i in range(1, int(NUM_POINTS) - int(HOME_BOARD_SIZE) + 1):
            if s[i] > 0:
                return False
    else:
        # Black's home board: points 1-6 (indices 1-6)
        for i in range(int(HOME_BOARD_SIZE) + 1, int(NUM_POINTS) + 1):
            if s[i] < 0:
                return False
    return True


def get_target_index_py(start_index, roll, player):
    """Pure Python version of _get_target_index."""
    target = int(start_index + roll * player)
    if 1 <= target <= int(NUM_POINTS):
        return target
    if target <= 0:
        return int(B_OFF)
    return int(W_OFF)


def is_move_legal_py(state, player, from_point, to_point):
    """Pure Python version of _is_move_legal."""
    s = np.asarray(state, dtype=np.int8)
    p = int(player)
    from_point = int(from_point)
    to_point = int(to_point)

    # Rule 1: Cannot move from an empty point
    if s[from_point] * p <= 0:
        return False

    # Rule 2: Cannot move if pieces are on the bar unless the move starts at the bar
    bar_index = W_BAR if p == 1 else B_BAR
    if s[bar_index] * p > 0 and from_point != bar_index:
        return False

    # Rule 3: Target point must not be blocked by the opponent
    if 1 <= to_point <= int(NUM_POINTS):
        target_checkers = s[to_point] * p
        if target_checkers <= -2:
            return False

    # Rule 4: Bearing off rules
    if to_point == W_OFF or to_point == B_OFF:
        off_target = W_OFF if p == 1 else B_OFF
        if to_point != off_target:
            return False
        if not can_bear_off_py(s, p):
            return False
        if p == 1:
            for i in range(int(NUM_POINTS) - int(HOME_BOARD_SIZE) + 1, from_point):
                if s[i] > 0:
                    return False
        else:
            for i in range(from_point - 1, 0, -1):
                if s[i] < 0:
                    return False

    return True


def apply_sub_move_py(state, player, from_point, to_point):
    """Pure Python version of _apply_sub_move."""
    s = np.asarray(state, dtype=np.int8)
    p = int(player)
    from_point = int(from_point)
    to_point = int(to_point)

    if not is_move_legal_py(s, p, from_point, to_point):
        return None

    next_state = s.copy()
    # Remove checker from source
    next_state[from_point] -= p

    # If bearing off
    if to_point == W_OFF or to_point == B_OFF:
        next_state[to_point] += p
    # If not bearing off
    elif 1 <= to_point <= int(NUM_POINTS):
        # Is there an opponent blot? Move it to bar
        if next_state[to_point] == -p:
            opponent_bar_index = B_BAR if p == 1 else W_BAR
            next_state[opponent_bar_index] -= p
            next_state[to_point] = p
        else:
            next_state[to_point] += p
    else:
        return None

    return next_state


def find_moves_recursive_py(state, player, remaining_dice, current_move, all_moves, all_afterstates):
    """
    Pure Python version of _find_moves_recursive.
    state: np.array int8 (28,)
    remaining_dice: list of ints
    current_move: list of (from_point, roll) tuples
    all_moves: list of move-sequences
    all_afterstates: list of resulting states
    """
    s = np.asarray(state, dtype=np.int8)
    p = int(player)

    # Base Case 1: All dice used
    if len(remaining_dice) == 0:
        all_moves.append(list(current_move))
        all_afterstates.append(s.copy())
        return

    # Optimization: Only consider unique dice values for iteration
    unique_dice = sorted(set(remaining_dice), reverse=True)

    # Try to use the largest unique die first
    for roll in unique_dice:
        possible_moves = []

        # If on bar, only consider moves from the bar
        bar_index = W_BAR if p == 1 else B_BAR
        # Guard against malformed states that are too short
        if int(bar_index) < s.size and s[int(bar_index)] * p > 0:
            from_point = bar_index
            to_point = get_target_index_py(from_point, roll, p)
            if to_point >= 0 and is_move_legal_py(s, p, from_point, to_point):
                possible_moves.append((from_point, to_point, roll))
        else:
            # If not on bar
            for from_point in range(1, int(NUM_POINTS) + 1):
                if s[from_point] * p > 0:
                    to_point = get_target_index_py(from_point, roll, p)
                    if is_move_legal_py(s, p, from_point, to_point):
                        possible_moves.append((from_point, to_point, roll))

        # Recurse on valid moves found
        for from_point, to_point, r in possible_moves:
            next_state = apply_sub_move_py(s, p, from_point, to_point)
            if next_state is None:
                continue
            new_dice_list = list(remaining_dice)
            try:
                new_dice_list.remove(r)
            except Exception:
                continue
            new_move = list(current_move)
            new_move.append((from_point, int(r)))
            find_moves_recursive_py(next_state, p, new_dice_list, new_move, all_moves, all_afterstates)

    if current_move:
        all_moves.append(list(current_move))
        all_afterstates.append(s.copy())


def actions_py(state, current_player, dice):
    """
    Pure Python version of _actions (no Numba).
    Returns (moves, afterstates) as lists.
    """
    s = np.asarray(state, dtype=np.int8)
    p = int(current_player)
    all_moves = []
    all_afterstates = []
    current_move = []

    # Defensive: if state is not a full backgammon board, treat as forced pass.
    # Engine uses indices up to B_OFF (27), so require at least 28 entries.
    min_state_size = int(B_OFF) + 1
    if s.ndim != 1 or s.size < min_state_size:
        all_moves.append([])
        all_afterstates.append(s.copy())
        return all_moves, all_afterstates

    dice_list = list(int(d) for d in dice)
    if dice_list[0] == dice_list[1]:
        dice_list = [dice_list[0]] * 4

    find_moves_recursive_py(s, p, dice_list, current_move, all_moves, all_afterstates)

    if not all_moves:
        # Forced pass: only legal move is no-op
        all_moves.append([])
        all_afterstates.append(s.copy())
        return all_moves, all_afterstates

    # Player must use maximum number of dice possible
    max_dice_used = max(len(m) for m in all_moves)
    max_len_moves = []
    max_len_afterstates = []
    for m, a in zip(all_moves, all_afterstates):
        if len(m) == max_dice_used:
            max_len_moves.append(m)
            max_len_afterstates.append(a)

    if max_dice_used == len(dice_list):
        return max_len_moves, max_len_afterstates

    # Player must use maximum number of pips possible
    total_pips = sum(dice_list)
    max_pips_used = 0
    for m in max_len_moves:
        s_pips = sum(r for (_, r) in m)
        if s_pips > max_pips_used:
            max_pips_used = s_pips
            if max_pips_used == total_pips:
                break

    max_pip_moves = []
    max_pip_afterstates = []
    for m, a in zip(max_len_moves, max_len_afterstates):
        s_pips = sum(r for (_, r) in m)
        if s_pips == max_pips_used:
            max_pip_moves.append(m)
            max_pip_afterstates.append(a)

    return max_pip_moves, max_pip_afterstates


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
        # roll dice + enumerate legal afterstates (pure Python version)
        dice = np.random.randint(1, 7, size=(2,), dtype=np.int8)
        afterstates, _paths = actions_py(state, player, dice)
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

    # Save final value network parameters
    ckpt_dir = "checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = checkpoints.save_checkpoint(
        ckpt_dir=ckpt_dir,
        target=st.params,
        step=steps,
        overwrite=True,
    )
    print(f"Saved final params to {ckpt_path}")

if __name__ == "__main__":
    main()

