"""
Agent 2 TD(λ) Training - Vectorized Batch Version

Runs B games in parallel with:
- Per-game eligibility traces
- Batched value/grad computation
- Shared parameter updates
- Matches specification exactly
"""

import os
import sys

# Add parent directory to path so we can import from root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set JAX memory settings BEFORE importing jax to avoid OOM errors
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.7'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_FLAGS'] = '--xla_gpu_enable_command_buffer='

import time
import argparse
import numpy as np

import jax
import jax.numpy as jnp
from flax.training import checkpoints

from backgammon_engine import (
    _new_game,
    _actions,
    _roll_dice,
    _vectorized_new_game,
    _vectorized_roll_dice,
    NUM_CHECKERS,
    HOME_BOARD_SIZE,
    NUM_POINTS,
    W_BAR, B_BAR, W_OFF, B_OFF,
)
from agent2.agent2_value_net import Agent2ValueNet
from agent2.agent2_encode import encode_agent2, BOARD_LENGTH, CONV_INPUT_CHANNELS, AUX_INPUT_SIZE


# -----------------------------
# Reward function
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
# Tree utilities for batched traces
# -----------------------------
def tree_zeros_like_params(params, B, dtype=jnp.bfloat16):
    """Create pytree with shape (B, *param_shape) for per-game traces"""
    return jax.tree_util.tree_map(lambda x: jnp.zeros((B,) + x.shape, dtype=dtype), params)

def tree_scale(t, c):
    """Scale pytree by scalar"""
    return jax.tree_util.tree_map(lambda x: x * c, t)

def tree_add(a, b):
    """Add two pytrees"""
    return jax.tree_util.tree_map(lambda x, y: x + y, a, b)

def tree_sum_over_batch(t):
    """Sum pytree over batch dimension (axis 0)"""
    return jax.tree_util.tree_map(lambda x: jnp.sum(x, axis=0), t)


# -----------------------------
# Batched value and grad computation
# -----------------------------
_value_net = Agent2ValueNet()

@jax.jit
def single_value_and_grad(params, board, aux):
    """Compute value and grad for a single state"""
    def v_fn(p):
        v = _value_net.apply({"params": p}, board_state=board[None, ...], aux_features=aux[None, ...])
        return v.squeeze(-1)[0]  # Network already outputs [-3,3] range
    v, g = jax.value_and_grad(v_fn)(params)
    return v, g

# Vectorize over batch dimension
batched_value_and_grad = jax.jit(jax.vmap(single_value_and_grad, in_axes=(None, 0, 0)))

@jax.jit
def value_apply(params, board_batch, aux_batch):
    """Batched value evaluation only (JIT-compiled)"""
    v = _value_net.apply({"params": params}, board_state=board_batch, aux_features=aux_batch)
    return v.squeeze(-1)  # (B,)

def batched_values(params, boards_np, aux_np, eval_batch):
    """
    Evaluate N states with FIXED eval_batch to avoid recompiles and OOM.
    Chunks large batches into smaller ones.
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
# TD(λ) batch update (matches spec exactly)
# -----------------------------
def apply_delta(zz, delta):
    """Broadcast delta (B,) to match zz shape (B, *param_shape)"""
    shape = (delta.shape[0],) + (1,) * (zz.ndim - 1)
    return zz * delta.reshape(shape)

@jax.jit
def td_lambda_update(params, z, v_s, g_s, r, v_next, done, alpha, gamma, lam):
    """
    Batched TD(λ) update matching specification.
    
    Args:
        params: pytree (shared across all games)
        z: pytree with leaves (B, *param_shape) - per-game traces
        v_s: (B,) - current state values
        g_s: pytree with leaves (B, *param_shape) - per-game grads
        r: (B,) - rewards
        v_next: (B,) - next state values (0 if done)
        done: (B,) bool - episode done flags
        alpha: learning rate
        gamma: discount factor
        lam: lambda (trace decay)
    
    Returns:
        params: updated shared params
        z: updated per-game traces
        delta: (B,) TD errors
    """
    # Set v_next to 0 for done states
    v_next = jnp.where(done, 0.0, v_next)
    
    # TD error: delta_i = r_i + gamma * v_next_i - v_s_i
    delta = r + gamma * v_next - v_s  # (B,)
    
    # Update traces: z_i = gamma*lam*z_i + g_i
    z = tree_add(tree_scale(z, gamma * lam), g_s)
    
    # Weight each game's trace by its TD error
    z_weighted = jax.tree_util.tree_map(lambda zz: apply_delta(zz, delta), z)
    
    # Sum over batch: params += alpha * sum_i (delta_i * z_i)
    step_dir = tree_sum_over_batch(z_weighted)
    params = jax.tree_util.tree_map(lambda p, d: p + alpha * d, params, step_dir)
    
    return params, z, delta


# -----------------------------
# Dice outcomes (needed for 2-ply)
# -----------------------------
def dice21():
    outs = []
    probs = []
    for i in range(1, 7):
        for j in range(i, 7):
            outs.append((j, i))
            probs.append((1.0 / 36.0) if i == j else (2.0 / 36.0))
    return np.array(outs, dtype=np.int8), np.array(probs, dtype=np.float32)

DICE21, PROBS21 = dice21()


# -----------------------------
# Helper for 2-ply
# -----------------------------
def _state_key(s: np.ndarray) -> bytes:
    return memoryview(np.asarray(s, dtype=np.int8)).tobytes()


# -----------------------------
# 1-ply and 2-ply action selection
# -----------------------------
def choose_action_1ply(state, player, dice, params, eval_batch=8192):
    """
    Simple 1-ply greedy action selection for fast training.
    Use 2-ply at evaluation time.
    """
    moves, afterstates = _actions(state, np.int8(player), dice)
    moves = list(moves)
    afterstates = [np.asarray(s, dtype=np.int8) for s in afterstates]
    
    if len(moves) == 0:
        return None, state
    
    # Evaluate all afterstates
    boards = np.empty((len(afterstates), BOARD_LENGTH, CONV_INPUT_CHANNELS), dtype=np.float32)
    auxs = np.empty((len(afterstates), AUX_INPUT_SIZE), dtype=np.float32)
    
    for i, s in enumerate(afterstates):
        b, a = encode_agent2(s, -player)  # Next player's perspective
        boards[i] = b
        auxs[i] = a
    
    # Batched evaluation with chunking
    vals = batched_values(params, boards, auxs, eval_batch=eval_batch)
    
    # Choose move with best value (from next player's perspective, so minimize)
    best = int(np.argmin(vals))
    return moves[best], afterstates[best]


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
# Main training loop (vectorized)
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="Agent 2 TD(λ) Vectorized Batch Training")
    ap.add_argument("--steps", type=int, default=200000, help="Total training steps")
    ap.add_argument("--batch_size", type=int, default=32, help="Number of parallel games (default: 32)")
    ap.add_argument("--alpha", type=float, default=5e-4, help="Learning rate")
    ap.add_argument("--gamma", type=float, default=1.0, help="Discount factor")
    ap.add_argument("--lam", type=float, default=0.85, help="TD(λ) lambda")
    ap.add_argument("--eval_batch", type=int, default=8192, help="Batch size for value evaluation")
    ap.add_argument("--log_every", type=int, default=200, help="Print progress every N steps")
    ap.add_argument("--save_every", type=int, default=2000, help="Save checkpoint every N steps")
    ap.add_argument("--ckpt_dir", type=str, default="checkpoints_agent2_vectorized", help="Checkpoint directory (default: checkpoints_agent2_vectorized)")
    ap.add_argument("--seed", type=int, default=0, help="Random seed")
    ap.add_argument("--use_1ply", action="store_true", help="Use 1-ply search during training (faster but deviates from spec - use only for speed testing)")
    args = ap.parse_args()
    
    rng = np.random.default_rng(args.seed)
    key = jax.random.PRNGKey(args.seed)
    
    # Initialize model
    model = Agent2ValueNet()
    dummy_board = jnp.zeros((1, BOARD_LENGTH, CONV_INPUT_CHANNELS), dtype=jnp.float32)
    dummy_aux = jnp.zeros((1, AUX_INPUT_SIZE), dtype=jnp.float32)
    params = model.init(key, board_state=dummy_board, aux_features=dummy_aux)["params"]
    
    # Initialize B games
    B = args.batch_size
    states, players, dice = _vectorized_new_game(B)
    states = np.asarray(states, dtype=np.int8)
    players = np.asarray(players, dtype=np.int32)
    dice = np.asarray(dice, dtype=np.int8)
    
    # Per-game eligibility traces
    z = tree_zeros_like_params(params, B)
    
    # Setup checkpoint directory
    ckpt_dir = os.path.abspath(args.ckpt_dir)
    os.makedirs(ckpt_dir, exist_ok=True)
    
    print(f"Agent 2 TD(λ) Vectorized Batch Training Configuration:")
    print(f"  Steps: {args.steps}")
    print(f"  Batch size (parallel games): {B}")
    print(f"  Learning rate (alpha): {args.alpha}")
    print(f"  Gamma: {args.gamma}")
    print(f"  Lambda: {args.lam}")
    print(f"  Use 2-ply: {not args.use_1ply} (spec-compliant)")
    if args.use_1ply:
        print(f"  ⚠️  WARNING: Using 1-ply deviates from spec. Use only for speed testing!")
    print(f"  Eval batch size: {args.eval_batch}")
    print(f"  Checkpoint directory: {ckpt_dir}")
    print()
    
    t0 = time.time()
    td2_acc = 0.0
    
    for step in range(1, args.steps + 1):
        # Encode all current states
        boards_s = np.empty((B, BOARD_LENGTH, CONV_INPUT_CHANNELS), dtype=np.float32)
        auxs_s = np.empty((B, AUX_INPUT_SIZE), dtype=np.float32)
        
        for i in range(B):
            b, a = encode_agent2(states[i], players[i])
            boards_s[i] = b
            auxs_s[i] = a
        
        # Batched value and grad computation (one JAX call for all games)
        v_s, g_s = batched_value_and_grad(
            params,
            jnp.asarray(boards_s, dtype=jnp.float32),
            jnp.asarray(auxs_s, dtype=jnp.float32),
        )
        v_s = np.array(v_s)  # (B,)
        
        # Choose actions and step environments
        next_states = np.empty_like(states)
        rewards = np.zeros(B, dtype=np.float32)
        dones = np.zeros(B, dtype=np.bool_)
        v_next = np.zeros(B, dtype=np.float32)
        
        for i in range(B):
            if args.use_1ply:
                # Use 1-ply (faster but deviates from spec - for speed testing only)
                move, next_state = choose_action_1ply(states[i], int(players[i]), dice[i], params, args.eval_batch)
            else:
                # Use 2-ply (spec-compliant, matches original train_agent2.py)
                move, next_state = choose_action_2ply_fast(
                    states[i], int(players[i]), dice[i], params, args.eval_batch, debug=False
                )
            
            if move is None:
                next_state = states[i]
            
            next_states[i] = np.asarray(next_state, dtype=np.int8)
            r = py_reward(next_state, int(players[i]))
            rewards[i] = float(r)
            dones[i] = (r != 0)
            
            if not dones[i]:
                # Encode next state from next player's perspective
                b_n, a_n = encode_agent2(next_state, -players[i])
                v_n = float(np.array(
                    value_apply(
                        params,
                        jnp.asarray(b_n[None, ...], dtype=jnp.float32),
                        jnp.asarray(a_n[None, ...], dtype=jnp.float32),
                    )[0]
                ))
                v_next[i] = v_n
        
        # Batched TD(λ) update
        params, z, delta = td_lambda_update(
            params, z, v_s, g_s, rewards, v_next, dones,
            args.alpha, args.gamma, args.lam
        )
        delta = np.array(delta)  # (B,)
        
        td2_acc += np.mean(delta * delta)
        
        # Reset traces for done games
        for i in range(B):
            if dones[i]:
                # Reset trace for this game (set to zeros)
                z = jax.tree_util.tree_map(
                    lambda zz: zz.at[i].set(jnp.zeros_like(zz[i])), z
                )
        
        # Advance environments
        for i in range(B):
            if dones[i]:
                # Start new game
                p, d, s = _new_game()
                players[i] = p
                dice[i] = d
                states[i] = np.asarray(s, dtype=np.int8)
            else:
                states[i] = next_states[i]
                players[i] = -players[i]
                dice[i] = _roll_dice()
        
        # Logging
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
        
        # Checkpointing
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

