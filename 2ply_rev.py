import argparse
import time
import os
# Set JAX memory settings BEFORE importing jax to avoid OOM errors
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.7'  # Use only 70% of GPU memory
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'  # Don't preallocate all memory
os.environ['XLA_FLAGS'] = '--xla_gpu_enable_command_buffer='  # Disable CUDA graphs

import numpy as np
import jax
import jax.numpy as jnp
import optax
from flax.training import train_state, checkpoints
from typing import List, Tuple

from backgammon_engine import _new_game
from backgammon_value_net import BackgammonValueNet, BOARD_LENGTH, CONV_INPUT_CHANNELS, AUX_INPUT_SIZE
from train_value_td0 import encode, actions_py, py_reward


# -------------------------
# Utilities
# -------------------------

def all_sorted_rolls_and_probs() -> Tuple[List[Tuple[int, int]], List[float]]:
    """Return all possible sorted dice rolls and their probabilities.
    
    Each sorted roll is a two-tuple (d1, d2) with d1 >= d2. A sorted roll
    corresponds to all permutations of two dice that sort to the same pair.
    There are 36 equiprobable unordered dice outcomes. For example, the sorted
    roll (3, 1) corresponds to the permutations (3,1) and (1,3), so its
    probability is 2/36. Doubles only have one permutation, so their probability
    is 1/36.
    """
    counts = {}
    for d1 in range(1, 7):
        for d2 in range(1, 7):
            a, b = sorted((d1, d2), reverse=True)
            counts[(a, b)] = counts.get((a, b), 0) + 1
    rolls = list(counts.keys())
    probs = [counts[p] / 36.0 for p in rolls]
    # Sort in descending lexicographic order (largest die first)
    rolls, probs = zip(*sorted(zip(rolls, probs), reverse=True))
    return list(rolls), list(probs)


def build_value_state(rng, lr):
    model = BackgammonValueNet()
    dummy_board = jnp.zeros((1, BOARD_LENGTH, CONV_INPUT_CHANNELS), dtype=jnp.float32)
    dummy_aux = jnp.zeros((1, AUX_INPUT_SIZE), dtype=jnp.float32)
    params = model.init(
        rng,
        board_state=dummy_board,
        aux_features=dummy_aux,
    )["params"]
    tx = optax.adam(lr)
    return train_state.TrainState.create(
        apply_fn=None,
        params=params,
        tx=tx,
    )


@jax.jit
def value_apply(params, board, aux):
    """Apply value network to a batch of states.
    
    Args:
        params: Network parameters
        board: (B, 24, 15) board representation
        aux: (B, AUX_INPUT_SIZE) auxiliary features
    
    Returns:
        (B,) array of value predictions
    """
    return BackgammonValueNet().apply(
        {"params": params},
        board_state=board,
        aux_features=aux,
    ).squeeze(-1)


@jax.jit
def loss_and_grads(params, board, aux, target):
    def loss_fn(p):
        pred = value_apply(p, board, aux)
        return jnp.mean((pred - target) ** 2)

    loss, grads = jax.value_and_grad(loss_fn)(params)
    return loss, grads


# -------------------------
# 1-PLY ACTION SELECTION (FAST, DEFAULT)
# -------------------------

def pick_action_1ply(state, player, st, eps_greedy=0.0):
    """1-ply greedy action selection with optional epsilon-greedy exploration.
    
    Args:
        state: Current board state
        player: +1 for white, -1 for black
        st: Training state with network parameters
        eps_greedy: Probability of random exploration (0.0 = pure greedy)
    
    Returns:
        (reward, next_state) tuple
    """
    dice = np.sort(np.random.randint(1, 7, size=(2,), dtype=np.int8))[::-1]
    afterstates, _ = actions_py(state, player, dice)
    
    if len(afterstates) == 0:
        return 0.0, state  # Forced pass
    
    # Epsilon-greedy exploration
    if np.random.rand() < eps_greedy:
        idx = np.random.randint(len(afterstates))
        ns = afterstates[idx]
        r = py_reward(ns, player)
        return float(r), ns
    
    # Terminal state check
    for ns in afterstates:
        r = py_reward(ns, player)
        if r != 0:
            return float(r), ns
    
    # Greedy selection
    boards, auxs = [], []
    for s in afterstates:
        b, a = encode(s, -player)  # Value net predicts for side-to-move
        boards.append(b)
        auxs.append(a)
    
    if len(boards) == 0:
        return 0.0, state
    
    boards = jnp.asarray(np.stack(boards))
    auxs = jnp.asarray(np.stack(auxs))
    
    vals = value_apply(st.params, boards, auxs)
    best = int(jnp.argmin(vals))  # Minimize opponent's value
    
    next_state = afterstates[best]
    reward = py_reward(next_state, player)
    return float(reward), next_state


# -------------------------
# OPTIONAL 2-PLY (DEBUG / EVAL ONLY)
# -------------------------

def pick_action_2ply(state, player, st, eval_batch=4096, eps_greedy=0.0, rolls=None, roll_probs=None):
    """2-ply action selection with proper dice roll probabilities.
    
    Args:
        state: Current board state
        player: +1 for white, -1 for black
        st: Training state with network parameters
        eval_batch: Batch size for GPU evaluation
        eps_greedy: Probability of random exploration
        rolls: List of sorted dice rolls (d1, d2) tuples
        roll_probs: Corresponding probabilities for each roll
    
    Returns:
        (reward, next_state) tuple
    """
    dice = np.sort(np.random.randint(1, 7, size=(2,), dtype=np.int8))[::-1]
    afterstates, _ = actions_py(state, player, dice)
    
    if len(afterstates) == 0:
        return 0.0, state  # Forced pass
    
    # Epsilon-greedy exploration
    if np.random.rand() < eps_greedy:
        idx = np.random.randint(len(afterstates))
        ns = afterstates[idx]
        r = py_reward(ns, player)
        return float(r), ns
    
    # Terminal state check
    for ns in afterstates:
        r = py_reward(ns, player)
        if r != 0:
            return float(r), ns
    
    # Get dice rolls and probabilities if not provided
    if rolls is None or roll_probs is None:
        rolls, roll_probs = all_sorted_rolls_and_probs()
    
    best_val = -1e9
    best_state = None
    
    # For each candidate afterstate, compute expected value
    for s in afterstates:
        exp_val = 0.0
        
        # For each possible opponent dice roll
        for j, (d1, d2) in enumerate(rolls):
            prob = roll_probs[j]
            opp_dice = np.array([d1, d2], dtype=np.int8)
            opp_after, _ = actions_py(s, -player, opp_dice)
            
            if len(opp_after) == 0:
                # Opponent forced pass: evaluate s with our turn
                b_s, a_s = encode(s, player)
                v = float(value_apply(
                    st.params,
                    jnp.asarray(b_s)[None, ...],
                    jnp.asarray(a_s)[None, ...],
                )[0])
                min_val = v
            else:
                # Evaluate all opponent responses
                boards, auxs = [], []
                for os in opp_after:
                    b, a = encode(os, player)
                    boards.append(b)
                    auxs.append(a)
                
                if len(boards) == 0:
                    min_val = 0.0
                else:
                    # Process in batches to avoid OOM
                    opp_vals = []
                    for i in range(0, len(boards), eval_batch):
                        batch_boards = jnp.asarray(np.stack(boards[i:i + eval_batch]))
                        batch_auxs = jnp.asarray(np.stack(auxs[i:i + eval_batch]))
                        v = value_apply(st.params, batch_boards, batch_auxs)
                        opp_vals.append(v)
                    
                    if len(opp_vals) > 0:
                        opp_vals = jnp.concatenate(opp_vals)
                        min_val = float(jnp.min(opp_vals))
                    else:
                        min_val = 0.0
            
            # Accumulate expected value with proper probability weighting
            exp_val += prob * min_val
        
        # Track best afterstate
        if exp_val > best_val:
            best_val = exp_val
            best_state = s
    
    if best_state is None:
        # Fallback: return first afterstate
        best_state = afterstates[0]
    
    reward = py_reward(best_state, player)
    return float(reward), best_state


# -------------------------
# MAIN TRAINING LOOP
# -------------------------

def main():
    parser = argparse.ArgumentParser(description="TD(0) training with optional 1-ply or 2-ply search")
    parser.add_argument("--steps", type=int, default=200_000, help="Number of training steps")
    parser.add_argument("--batch", type=int, default=4096, help="Batch size for gradient updates")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--train_search", choices=["1ply", "2ply"], default="1ply",
                        help="Search depth: 1-ply (fast) or 2-ply (slow but stronger)")
    parser.add_argument("--eval_batch", type=int, default=4096,
                        help="Batch size for 2-ply state evaluation (reduce if OOM)")
    parser.add_argument("--eps_greedy", type=float, default=0.05,
                        help="Epsilon for random exploration (0.0 = pure greedy)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    args = parser.parse_args()

    print("=" * 70)
    print(f"Training mode: {args.train_search}")
    print(f"Steps: {args.steps}, Batch: {args.batch}, Eval batch: {args.eval_batch}")
    print(f"Learning rate: {args.lr}, Epsilon-greedy: {args.eps_greedy}")
    print("=" * 70)
    
    if args.train_search == "2ply":
        print("\n⚠️  WARNING: 2-ply search is much slower but produces stronger play.")
        print("   First run will trigger JAX JIT compilation (5-30 minutes).\n")

    rng = jax.random.PRNGKey(args.seed)
    st = build_value_state(rng, args.lr)
    
    # Precompute dice rolls and probabilities for 2-ply
    rolls, roll_probs = all_sorted_rolls_and_probs()

    buf_board, buf_aux, buf_tgt = [], [], []

    player, _, state = _new_game()
    t0 = time.time()
    last_loss = 0.0

    for step in range(1, args.steps + 1):
        b, a = encode(state, player)

        if args.train_search == "2ply":
            r, next_state = pick_action_2ply(
                state, player, st, args.eval_batch, args.eps_greedy, rolls, roll_probs
            )
        else:
            r, next_state = pick_action_1ply(state, player, st, args.eps_greedy)

        if r != 0:
            tgt = float(r)
            buf_board.append(b)
            buf_aux.append(a)
            buf_tgt.append(tgt)
            player, _, state = _new_game()
        else:
            nb, na = encode(next_state, -player)
            v_next = float(value_apply(
                st.params,
                jnp.asarray(nb)[None, ...],
                jnp.asarray(na)[None, ...],
            )[0])
            tgt = -v_next

            buf_board.append(b)
            buf_aux.append(a)
            buf_tgt.append(tgt)

            state = next_state
            player = -player

        if len(buf_tgt) >= args.batch:
            board = jnp.asarray(np.stack(buf_board))
            aux = jnp.asarray(np.stack(buf_aux))
            target = jnp.asarray(np.array(buf_tgt, dtype=np.float32))

            loss, grads = loss_and_grads(st.params, board, aux, target)
            last_loss = float(loss)
            st = st.apply_gradients(grads=grads)

            buf_board.clear()
            buf_aux.clear()
            buf_tgt.clear()

        if step % 5000 == 0:
            dt = time.time() - t0
            rate = step / dt if dt > 0 else 0
            eta_seconds = (args.steps - step) / rate if rate > 0 else 0
            eta_minutes = eta_seconds / 60
            eta_hours = eta_minutes / 60
            
            print(f"[{step}/{args.steps}] loss={last_loss:.6f} rate={rate:.2f} steps/s", end="")
            if eta_hours < 1:
                print(f" ETA: {eta_minutes:.1f} min")
            else:
                print(f" ETA: {eta_hours:.1f} hr ({eta_minutes:.1f} min)")

    # Save checkpoint with method name in folder path
    ckpt_dir = os.path.abspath(f"checkpoints_{args.train_search}")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = checkpoints.save_checkpoint(
        ckpt_dir,
        st.params,
        step=args.steps,
        overwrite=True,
    )

    print(f"\nTraining complete. Saved checkpoint to: {ckpt_path}")


if __name__ == "__main__":
    main()
