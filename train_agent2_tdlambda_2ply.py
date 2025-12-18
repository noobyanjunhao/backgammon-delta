import os
# Set JAX memory settings BEFORE importing jax to avoid OOM errors
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.7'  # Use only 70% of GPU memory
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'  # Don't preallocate all memory
os.environ['XLA_FLAGS'] = '--xla_gpu_enable_command_buffer='  # Disable CUDA graphs

import time
import argparse
import numpy as np

import jax
import jax.numpy as jnp
import optax
from flax.training import checkpoints

from backgammon_engine import (
    _new_game,
    _actions,
    _to_canonical,
    _roll_dice,
    _get_target_index,
    NUM_CHECKERS,
    HOME_BOARD_SIZE,
    NUM_POINTS,
    W_BAR, B_BAR, W_OFF, B_OFF,
)
from backgammon_value_net import BackgammonValueNet, AUX_INPUT_SIZE, BOARD_LENGTH, CONV_INPUT_CHANNELS


# -----------------------------
# Reward (same as your TD code)
# -----------------------------
def py_reward(state, player):
    s = np.asarray(state, dtype=np.int8)
    p = int(player)

    # White wins
    if s[W_OFF] == NUM_CHECKERS:
        if s[B_OFF] < 0:
            return p
        else:
            if s[B_BAR] < 0:
                return 3 * p
            for pt in range(1, int(HOME_BOARD_SIZE) + 1):
                if s[pt] < 0:
                    return 3 * p
            return 2 * p

    # Black wins
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


# -----------------------------
# Dice outcomes (21 unique)
# -----------------------------
def dice21():
    # unordered dice outcomes i<=j with correct probabilities
    outs = []
    probs = []
    for i in range(1, 7):
        for j in range(i, 7):
            outs.append((j, i) if j >= i else (i, j))  # engine expects sorted hi->lo usually
            if i == j:
                probs.append(1.0 / 36.0)
            else:
                probs.append(2.0 / 36.0)
    outs = np.array(outs, dtype=np.int8)
    probs = np.array(probs, dtype=np.float32)
    return outs, probs


DICE21, PROBS21 = dice21()


# -----------------------------
# Agent 2 encoding: 24 x 15 planes + 6 aux
# Planes:
#   0: empty (shared)
#   For each category in [1..6 exact, overflow]:
#     two planes (self, opp)
# Total = 1 + 7*2 = 15
# -----------------------------
def encode_agent2(state, player):
    s = np.asarray(state, dtype=np.int8)
    # canonicalize so current player is "self" (white-positive)
    c = _to_canonical(s, np.int8(player)).astype(np.int16)

    pts = c[1:25]  # 24 points
    self_cnt = np.clip(pts, 0, 15)
    opp_cnt = np.clip(-pts, 0, 15)

    board = np.zeros((24, 15), dtype=np.float32)

    # plane 0: empty if no checkers for either player on that point
    board[:, 0] = (pts == 0).astype(np.float32)

    # categories: exact 1..6, and overflow (>6 normalized /9)
    # plane layout:
    #   cat_idx 0..6 -> planes (1+2*cat_idx)=self, (2+2*cat_idx)=opp
    exact_values = [1, 2, 3, 4, 5, 6]
    for k, v in enumerate(exact_values):
        p_self = 1 + 2 * k
        p_opp = 2 + 2 * k
        board[:, p_self] = (self_cnt == v).astype(np.float32)
        board[:, p_opp] = (opp_cnt == v).astype(np.float32)

    # overflow category (cat_idx=6)
    p_self = 1 + 2 * 6
    p_opp = 2 + 2 * 6
    board[:, p_self] = np.maximum(self_cnt - 6, 0).astype(np.float32) / 9.0
    board[:, p_opp] = np.maximum(opp_cnt - 6, 0).astype(np.float32) / 9.0

    # aux (6):
    # self bar active, self bar scale, self off scale, opp bar active, opp bar scale, opp off scale
    # canonical indices:
    #   bar (self) is c[0]
    #   bar (opp) is c[25]
    #   off (self) is c[26]
    #   off (opp) is c[27]
    aux = np.zeros((AUX_INPUT_SIZE,), dtype=np.float32)
    self_bar = float(max(c[0], 0))
    opp_bar = float(max(c[25], 0))
    self_off = float(max(c[26], 0))
    opp_off = float(max(c[27], 0))

    aux[0] = 1.0 if self_bar > 0 else 0.0
    aux[1] = self_bar / 15.0
    aux[2] = self_off / 15.0
    aux[3] = 1.0 if opp_bar > 0 else 0.0
    aux[4] = opp_bar / 15.0
    aux[5] = opp_off / 15.0

    return board, aux


# -----------------------------
# JAX model apply (fixed batch)
# -----------------------------
# Create module instance once outside JIT to avoid reinitialization issues
_value_net = BackgammonValueNet()

@jax.jit
def value_apply(params, board_batch, aux_batch):
    v = _value_net.apply({"params": params}, board_state=board_batch, aux_features=aux_batch)
    # v in [-1,1]; scale to [-3,3] (project spec)
    return 3.0 * v.squeeze(-1)


def batched_values(params, boards_np, aux_np, eval_batch=2048):
    """Evaluate many states in fixed-size batches to avoid JAX recompiles."""
    n = boards_np.shape[0]
    out = []
    for i in range(0, n, eval_batch):
        b = boards_np[i:i + eval_batch]
        a = aux_np[i:i + eval_batch]
        if b.shape[0] < eval_batch:
            pad = eval_batch - b.shape[0]
            b = np.pad(b, ((0, pad), (0, 0), (0, 0)), mode="constant")
            a = np.pad(a, ((0, pad), (0, 0)), mode="constant")
        v = value_apply(params, jnp.asarray(b, dtype=jnp.float32), jnp.asarray(a, dtype=jnp.float32))
        v = np.array(v, dtype=np.float32)[:min(eval_batch, n - i)]
        out.append(v)
    return np.concatenate(out, axis=0)


# -----------------------------
# 2-ply action selection (matches spec)
# a* = argmax_a sum_j P(d_j) [ min_aopp V(S_{a,j,aopp}) ]
# -----------------------------
def choose_action_2ply(state, player, dice, params, eval_batch):
    # legal actions for current player under current dice
    moves_a, after_a = _actions(state, np.int8(player), dice)
    moves_a = list(moves_a)
    after_a = [np.asarray(s, dtype=np.int8) for s in after_a]

    if len(after_a) == 0:
        return None, state  # pass (should be rare)

    # Build ALL opponent response states for all candidate actions and all 21 dice
    all_boards = []
    all_aux = []
    index = []  # (a_idx, j_idx, resp_k)
    group_sizes = []  # number of opponent moves for each (a,j)

    for a_idx, s_a in enumerate(after_a):
        opp_player = -player
        for j_idx, d in enumerate(DICE21):
            moves_o, after_o = _actions(s_a, np.int8(opp_player), d)
            after_o = list(after_o)

            # If opponent has no moves, treat as staying in same state
            if len(after_o) == 0:
                b, ax = encode_agent2(s_a, player)  # value from *current player's* perspective
                all_boards.append(b)
                all_aux.append(ax)
                index.append((a_idx, j_idx))
                group_sizes.append(1)
                continue

            # Opp chooses move that MINIMIZES our value after their move
            group_sizes.append(len(after_o))
            for s_ao in after_o:
                b, ax = encode_agent2(np.asarray(s_ao, dtype=np.int8), player)
                all_boards.append(b)
                all_aux.append(ax)
                index.append((a_idx, j_idx))

    boards_np = np.asarray(all_boards, dtype=np.float32)
    aux_np = np.asarray(all_aux, dtype=np.float32)

    vals = batched_values(params, boards_np, aux_np, eval_batch=eval_batch)

    # Reduce: for each (a,j) take min over opponent moves; then expectation over dice
    # We'll consume vals sequentially using group_sizes ordering.
    n_actions = len(after_a)
    min_per_aj = np.full((n_actions, len(DICE21)), np.inf, dtype=np.float32)

    ptr = 0
    for a_idx in range(n_actions):
        for j_idx in range(len(DICE21)):
            # We appended groups in (a,j) order, so just read next group
            # But note: group_sizes is in that same nested loop order.
            pass

    # reconstruct group_sizes in same order
    ptr = 0
    gs_ptr = 0
    for a_idx in range(n_actions):
        for j_idx in range(len(DICE21)):
            g = group_sizes[gs_ptr]
            gs_ptr += 1
            vblock = vals[ptr:ptr + g]
            ptr += g
            min_per_aj[a_idx, j_idx] = float(np.min(vblock))

    expected = (min_per_aj * PROBS21[None, :]).sum(axis=1)  # (n_actions,)
    best = int(np.argmax(expected))
    return moves_a[best], after_a[best]


# -----------------------------
# TD(lambda) update on params
# z_t is eligibility trace tree
# Classical:
#   z <- gamma*lam*z + grad(V(s))
#   delta = r + gamma*V(s') - V(s)
#   params <- params + alpha * delta * z
# -----------------------------
def tree_zeros_like(tree):
    return jax.tree_util.tree_map(lambda x: jnp.zeros_like(x), tree)

def tree_add(a, b):
    return jax.tree_util.tree_map(lambda x, y: x + y, a, b)

def tree_scale(a, c):
    return jax.tree_util.tree_map(lambda x: c * x, a)

def tree_axpy(a, x, y):
    # y + a*x
    return jax.tree_util.tree_map(lambda xx, yy: a * xx + yy, x, y)

@jax.jit
def value_and_grad(params, board, aux):
    def v_fn(p):
        v = _value_net.apply({"params": p}, board_state=board, aux_features=aux)
        return (3.0 * v.squeeze(-1)).mean()
    v, g = jax.value_and_grad(v_fn)(params)
    return v, g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=200000)
    ap.add_argument("--alpha", type=float, default=1e-3)      # TD step size
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--lam", type=float, default=0.85)        # TD(lambda)
    ap.add_argument("--eval_batch", type=int, default=2048, help="Batch size for value evaluation (default: 2048)")
    ap.add_argument("--log_every", type=int, default=200, help="Print progress every N steps (default: 200)")
    ap.add_argument("--save_every", type=int, default=2000, help="Save checkpoint every N steps (default: 2000)")
    ap.add_argument("--seed", type=int, default=0, help="Random seed (default: 0)")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    key = jax.random.PRNGKey(args.seed)

    # init net
    model = BackgammonValueNet()
    dummy_board = jnp.zeros((1, BOARD_LENGTH, CONV_INPUT_CHANNELS), dtype=jnp.float32)
    dummy_aux = jnp.zeros((1, AUX_INPUT_SIZE), dtype=jnp.float32)
    params = model.init(key, board_state=dummy_board, aux_features=dummy_aux)["params"]

    # eligibility trace
    z = tree_zeros_like(params)

    # Use consistent checkpoint directory format: checkpoints_tdlambda_2ply
    ckpt_dir = os.path.abspath("checkpoints_tdlambda_2ply")
    os.makedirs(ckpt_dir, exist_ok=True)

    player, dice, state = _new_game()

    t0 = time.time()
    avg_loss = 0.0

    for step in range(1, args.steps + 1):
        # choose move using 2-ply planning (spec)
        move, next_state = choose_action_2ply(state, int(player), dice, params, args.eval_batch)

        # compute reward and terminal
        r = py_reward(next_state, int(player))
        done = (r != 0)

        # current value + grad(V(s))
        b_s, a_s = encode_agent2(state, int(player))
        v_s, g_s = value_and_grad(
            params,
            jnp.asarray(b_s[None, ...], dtype=jnp.float32),
            jnp.asarray(a_s[None, ...], dtype=jnp.float32),
        )

        if done:
            v_next = 0.0
        else:
            # value of next state from opponent-to-move perspective for TD target:
            # since we defined V as "value for side-to-move in canonical self perspective",
            # after our move, the next player is -player.
            b_n, a_n = encode_agent2(next_state, int(-player))
            v_next = float(np.array(
                value_apply(params,
                            jnp.asarray(b_n[None, ...], dtype=jnp.float32),
                            jnp.asarray(a_n[None, ...], dtype=jnp.float32))[0]
            ))

        v_s_f = float(np.array(v_s))

        delta = float(r + args.gamma * v_next - v_s_f)

        # eligibility trace update
        z = tree_add(tree_scale(z, args.gamma * args.lam), g_s)

        # params update (manual SGD in param space)
        params = tree_axpy(args.alpha * delta, z, params)

        avg_loss += (delta * delta)

        # advance environment
        if done:
            player, dice, state = _new_game()
            # optional: reset trace at episode boundary (common)
            z = tree_zeros_like(params)
        else:
            state = np.asarray(next_state, dtype=np.int8)
            player = -player
            dice = _roll_dice()

        if step % args.log_every == 0:
            dt = time.time() - t0
            rate = step / dt if dt > 0 else 0
            progress_percent = (step / args.steps) * 100
            eta_seconds = (args.steps - step) / rate if rate > 0 else float('inf')
            eta_minutes = eta_seconds / 60
            eta_hours = eta_minutes / 60
            
            print(f"[step {step:6d}/{args.steps}] "
                  f"td_err2={avg_loss/args.log_every:.6f}  "
                  f"steps/s={rate:.2f}  "
                  f"progress={progress_percent:.1f}%  "
                  f"elapsed={dt/60:.1f}min", flush=True)
            
            if eta_hours < 1:
                print(f"  ETA: {eta_minutes:.1f} min", flush=True)
            else:
                print(f"  ETA: {eta_hours:.1f} hr ({eta_minutes:.1f} min)", flush=True)
            
            avg_loss = 0.0

        if step % args.save_every == 0:
            ckpt_path = checkpoints.save_checkpoint(
                ckpt_dir=ckpt_dir,
                target=params,
                step=step,
                overwrite=True,
            )
            print(f"  💾 CHECKPOINT SAVED: {ckpt_path} @ step {step}", flush=True)

    # Save final checkpoint
    ckpt_path = checkpoints.save_checkpoint(
        ckpt_dir=ckpt_dir,
        target=params,
        step=args.steps,
        overwrite=True,
    )
    print(f"\nTraining complete. Saved final checkpoint to: {ckpt_path}")


if __name__ == "__main__":
    main()
