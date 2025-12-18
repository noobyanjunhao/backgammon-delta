import os
import time
import argparse
import numpy as np

import jax
import jax.numpy as jnp
from flax.training import checkpoints

from backgammon_engine import (
    _new_game,
    _actions,
    _to_canonical,
    _roll_dice,
    NUM_CHECKERS,
    HOME_BOARD_SIZE,
    NUM_POINTS,
    W_BAR, B_BAR, W_OFF, B_OFF,
)
from agent2_value_net import Agent2ValueNet
from agent2_encode import encode_agent2, BOARD_LENGTH, CONV_INPUT_CHANNELS, AUX_INPUT_SIZE


# -----------------------------
# Reward (same logic)
# -----------------------------
def py_reward(state, player):
    s = np.asarray(state, dtype=np.int8)
    p = int(player)

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
# 21 dice outcomes with probs
# -----------------------------
def dice21():
    outs = []
    probs = []
    for i in range(1, 7):
        for j in range(i, 7):
            # store sorted hi->lo
            outs.append((j, i))
            probs.append((1.0 / 36.0) if i == j else (2.0 / 36.0))
    return np.array(outs, dtype=np.int8), np.array(probs, dtype=np.float32)

DICE21, PROBS21 = dice21()


# encode_agent2 is now imported from agent2.agent2_encode


# -----------------------------
# JAX value apply
# -----------------------------
# Create module instance once outside JIT to avoid reinitialization issues
_value_net = Agent2ValueNet()

@jax.jit
def value_apply(params, board_batch, aux_batch):
    v = _value_net.apply({"params": params}, board_state=board_batch, aux_features=aux_batch)
    return v.squeeze(-1)  # Network already outputs [-3,3] range


def batched_values(params, boards_np, aux_np, eval_batch):
    """
    Evaluate N states with FIXED eval_batch to avoid recompiles.
    Uses large batches to reduce CPU->GPU overhead.
    """
    n = boards_np.shape[0]
    out = np.empty((n,), dtype=np.float32)

    for i in range(0, n, eval_batch):
        b = boards_np[i:i + eval_batch]
        a = aux_np[i:i + eval_batch]
        m = b.shape[0]
        if m < eval_batch:
            pad = eval_batch - m
            b = np.pad(b, ((0, pad), (0, 0), (0, 0)), mode="constant")
            a = np.pad(a, ((0, pad), (0, 0)), mode="constant")

        v = value_apply(
            params,
            jnp.asarray(b, dtype=jnp.float32),
            jnp.asarray(a, dtype=jnp.float32),
        )
        v = np.array(v, dtype=np.float32)[:m]
        out[i:i + m] = v

    return out


# -----------------------------
# Fast unique-afterstate packing (dedup)
# This directly implements the "unique afterstates" tip in the spec.
# -----------------------------
def _state_key(s: np.ndarray) -> bytes:
    # state is int8 array, make an immutable key fast
    return memoryview(np.asarray(s, dtype=np.int8)).tobytes()


def choose_action_2ply_fast(state, player, dice, params, eval_batch, debug=False):
    """
    Exact 2-ply:
      for each a:
        sum_j P(d_j) [ min_{aopp} V(S_{a,j,aopp}) ]
      choose argmax.

    Speed tricks:
      - two-pass counting and preallocation (no Python append hot path)
      - deduplicate identical afterstates across all (a,j,aopp)
      - one GPU eval for all unique states (batched)
      - CPU reduction by indices
    """
    moves_a, after_a = _actions(state, np.int8(player), dice)
    moves_a = list(moves_a)
    after_a = [np.asarray(s, dtype=np.int8) for s in after_a]
    nA = len(after_a)
    if nA == 0:
        return None, state

    opp_player = -player

    # -------- Pass 1: count how many opponent afterstates per (a,j) and collect keys for dedup
    # We'll build:
    #   group_counts[a,j] = number of opponent afterstates (before dedup)
    group_counts = np.zeros((nA, 21), dtype=np.int32)

    # We also build a dict for dedup mapping state_key -> unique_index
    uniq_index = {}
    uniq_states = []  # store representative state arrays (int8)

    # During pass1, just enumerate and populate uniq_index (dedup) and group_counts.
    for a_idx, s_a in enumerate(after_a):
        for j_idx, d in enumerate(DICE21):
            moves_o, after_o = _actions(s_a, np.int8(opp_player), d)
            after_o = list(after_o)
            if len(after_o) == 0:
                group_counts[a_idx, j_idx] = 1
                k = _state_key(s_a)
                if k not in uniq_index:
                    uniq_index[k] = len(uniq_states)
                    uniq_states.append(np.asarray(s_a, dtype=np.int8))
            else:
                group_counts[a_idx, j_idx] = len(after_o)
                for s in after_o:
                    s = np.asarray(s, dtype=np.int8)
                    k = _state_key(s)
                    if k not in uniq_index:
                        uniq_index[k] = len(uniq_states)
                        uniq_states.append(s)

    U = len(uniq_states)
    if debug:
        total_raw = int(group_counts.sum())
        print(f"2-ply enumeration: actions={nA}, raw_states={total_raw}, unique_states={U}")

    # -------- Encode all unique states once
    boards = np.empty((U, 24, 15), dtype=np.float32)
    auxs = np.empty((U, AUX_INPUT_SIZE), dtype=np.float32)
    for i, s in enumerate(uniq_states):
        b, a = encode_agent2(s, player)  # IMPORTANT: V is from current player's perspective
        boards[i] = b
        auxs[i] = a

    # -------- GPU evaluate once (or big batches)
    vals_u = batched_values(params, boards, auxs, eval_batch=eval_batch)  # (U,)

    # -------- Pass 2: compute min per (a,j) by scanning again but now only indexing vals_u
    # min_per_aj[a,j] = min over opponent moves
    min_per_aj = np.full((nA, 21), np.inf, dtype=np.float32)

    for a_idx, s_a in enumerate(after_a):
        for j_idx, d in enumerate(DICE21):
            moves_o, after_o = _actions(s_a, np.int8(opp_player), d)
            after_o = list(after_o)

            if len(after_o) == 0:
                u = uniq_index[_state_key(s_a)]
                vmin = vals_u[u]
            else:
                # compute min over opponent afterstates
                vmin = np.inf
                for s in after_o:
                    s = np.asarray(s, dtype=np.int8)
                    u = uniq_index[_state_key(s)]
                    v = vals_u[u]
                    if v < vmin:
                        vmin = v
            min_per_aj[a_idx, j_idx] = vmin

    expected = (min_per_aj * PROBS21[None, :]).sum(axis=1)  # (nA,)
    best = int(np.argmax(expected))
    return moves_a[best], after_a[best]


# -----------------------------
# TD(λ) update: keep it simple
# (Note: deep-net traces are expensive; but you asked to keep TD(λ).)
# We'll store traces in bfloat16 to cut bandwidth/mem.
# -----------------------------
def tree_zeros_like(tree):
    return jax.tree_util.tree_map(lambda x: jnp.zeros_like(x, dtype=jnp.bfloat16), tree)

def tree_scale(tree, c):
    c = jnp.asarray(c, dtype=jnp.float32)
    return jax.tree_util.tree_map(lambda x: (x.astype(jnp.float32) * c).astype(jnp.bfloat16), tree)

def tree_add(a, b):
    # a is bfloat16 trace, b is float32 grad
    return jax.tree_util.tree_map(lambda x, y: (x.astype(jnp.float32) + y).astype(jnp.bfloat16), a, b)

def tree_axpy(alpha, x, y):
    # y + alpha*x
    a = jnp.asarray(alpha, dtype=jnp.float32)
    return jax.tree_util.tree_map(lambda xx, yy: yy + a * xx.astype(jnp.float32), x, y)

@jax.jit
def value_and_grad(params, board, aux):
    def v_fn(p):
        v = _value_net.apply({"params": p}, board_state=board, aux_features=aux)
        return v.squeeze(-1).mean()  # Network already outputs [-3,3] range
    v, g = jax.value_and_grad(v_fn)(params)
    return v, g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=200000)
    ap.add_argument("--alpha", type=float, default=5e-4)   # smaller often helps stability
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--lam", type=float, default=0.85)
    ap.add_argument("--eval_batch", type=int, default=8192, help="Batch size for value evaluation (default: 8192)")
    ap.add_argument("--log_every", type=int, default=200, help="Print progress every N steps (default: 200)")
    ap.add_argument("--save_every", type=int, default=2000, help="Save checkpoint every N steps (default: 2000)")
    ap.add_argument("--seed", type=int, default=0, help="Random seed (default: 0)")
    ap.add_argument("--debug_2ply", action="store_true", help="Print 2-ply deduplication stats")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    key = jax.random.PRNGKey(args.seed)

    model = Agent2ValueNet()
    dummy_board = jnp.zeros((1, BOARD_LENGTH, CONV_INPUT_CHANNELS), dtype=jnp.float32)
    dummy_aux = jnp.zeros((1, AUX_INPUT_SIZE), dtype=jnp.float32)
    params = model.init(key, board_state=dummy_board, aux_features=dummy_aux)["params"]

    # eligibility traces
    z = tree_zeros_like(params)

    # Use consistent checkpoint directory format: checkpoints_tdlambda_2ply
    ckpt_dir = os.path.abspath("checkpoints_tdlambda_2ply")
    os.makedirs(ckpt_dir, exist_ok=True)

    player, dice, state = _new_game()

    t0 = time.time()
    td2_acc = 0.0

    for step in range(1, args.steps + 1):
        # 2-ply exact move selection (fastened, dedup, batched GPU eval)
        move, next_state = choose_action_2ply_fast(
            state, int(player), dice, params, args.eval_batch,
            debug=args.debug_2ply and (step % args.log_every == 0)
        )

        r = py_reward(next_state, int(player))
        done = (r != 0)

        # V(s) and grad
        b_s, a_s = encode_agent2(state, int(player))
        v_s, g_s = value_and_grad(
            params,
            jnp.asarray(b_s[None, ...], dtype=jnp.float32),
            jnp.asarray(a_s[None, ...], dtype=jnp.float32),
        )
        v_s_f = float(np.array(v_s))

        if done:
            v_next = 0.0
        else:
            # TD target uses value from next player's perspective
            b_n, a_n = encode_agent2(next_state, int(-player))
            v_next = float(np.array(
                value_apply(
                    params,
                    jnp.asarray(b_n[None, ...], dtype=jnp.float32),
                    jnp.asarray(a_n[None, ...], dtype=jnp.float32),
                )[0]
            ))

        delta = float(r + args.gamma * v_next - v_s_f)

        # TD(lambda) trace update
        z = tree_add(tree_scale(z, args.gamma * args.lam), g_s)

        # weight update
        params = tree_axpy(args.alpha * delta, z, params)

        td2_acc += delta * delta

        # advance env
        if done:
            player, dice, state = _new_game()
            z = tree_zeros_like(params)  # common practice to reset traces at episode end
        else:
            state = np.asarray(next_state, dtype=np.int8)
            player = -player
            dice = _roll_dice()

        if step % args.log_every == 0:
            dt = time.time() - t0
            rate = step / dt if dt > 0 else 0
            avg_td2 = td2_acc / args.log_every
            td2_acc = 0.0
            progress_percent = (step / args.steps) * 100
            eta_seconds = (args.steps - step) / rate if rate > 0 else float('inf')
            eta_minutes = eta_seconds / 60
            eta_hours = eta_minutes / 60

            print(f"[step {step:6d}/{args.steps}] "
                  f"td_err2={avg_td2:.6f}  "
                  f"steps/s={rate:.3f}  "
                  f"progress={progress_percent:.1f}%  "
                  f"elapsed={dt/60:.1f}min", flush=True)
            
            if eta_hours < 1:
                print(f"  ETA: {eta_minutes:.1f} min", flush=True)
            else:
                print(f"  ETA: {eta_hours:.1f} hr ({eta_minutes:.1f} min)", flush=True)

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
