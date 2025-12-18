"""
Improved TD(0) training script for the Backgammon project.

This script augments the basic one‐ply training loop found in
``train_value_td0.py`` by using a simple two‐ply lookahead during
action selection.  In place of greedily choosing the afterstate with
the lowest value for the opponent (equivalently the highest value for
the current player), we compute the expected value of each candidate
afterstate under all possible dice rolls and the opponent's optimal
response.  The candidate afterstate with the largest expected value is
selected.  A small epsilon probability is used to choose a random
legal move to encourage exploration, mirroring the original code.

The rest of the TD(0) learning loop is unchanged: we maintain a
replay buffer of encoded states and targets, compute the mean squared
error loss, and update the network with Adam.  This script also
provides a reproducible random seed and optional hyperparameters via
command line flags.

Note that this implementation is deliberately explicit rather than
fully optimised for speed.  Two‐ply search is expensive because of
the branching factor; however, it typically yields stronger play
without the need for a deeper neural network.  For production use
consider caching and vectorising portions of the search.
"""

import os
# Set JAX memory settings BEFORE importing jax to avoid OOM errors
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.7'  # Use only 70% of GPU memory
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'  # Don't preallocate all memory
os.environ['XLA_FLAGS'] = '--xla_gpu_enable_command_buffer='  # Disable CUDA graphs if needed

import time
import argparse
from typing import List, Tuple

import numpy as np
import jax
import jax.numpy as jnp
import optax
from flax.training import train_state, checkpoints

from backgammon_engine import _new_game
from backgammon_value_net import (
    BackgammonValueNet as ValueNet,
    BOARD_LENGTH,
    CONV_INPUT_CHANNELS,
    AUX_INPUT_SIZE,
)
from train_value_td0 import (
    actions_py,
    encode,
    py_reward,
    v_apply,
    loss_and_grads,
)


def all_sorted_rolls_and_probs() -> Tuple[List[Tuple[int, int]], List[float]]:
    """Return all possible sorted dice rolls and their probabilities.

    Each sorted roll is a two‐tuple ``(d1, d2)`` with ``d1 >= d2``.  A
    sorted roll corresponds to all permutations of two dice that sort to
    the same pair.  There are 36 equiprobable unordered dice outcomes.
    For example, the sorted roll (3, 1) corresponds to the permutations
    (3,1) and (1,3), so its probability is 2/36.  Doubles only have one
    permutation, so their probability is 1/36.  The resulting lists
    align by index: ``rolls[i]`` has probability ``probs[i]``.
    """
    # Accumulate counts of permutations for each sorted pair.
    counts = {}
    for d1 in range(1, 7):
        for d2 in range(1, 7):
            a, b = sorted((d1, d2), reverse=True)
            counts[(a, b)] = counts.get((a, b), 0) + 1
    # Convert to probability list (each of the 36 unordered rolls is equiprobable).
    rolls = list(counts.keys())
    probs = [counts[p] / 36.0 for p in rolls]
    # Sort the rolls in descending lexicographic order to mirror the engine's
    # convention (largest die first).
    rolls, probs = zip(*sorted(zip(rolls, probs), reverse=True))
    return list(rolls), list(probs)


class TwoPlyTD0Trainer:
    """Encapsulate the two‐ply TD(0) training process.

    This class holds the neural network parameters, optimiser state,
    and hyperparameters for the training loop.  The primary entry point
    is the :meth:`train` method, which performs a fixed number of TD(0)
    updates and periodically saves checkpoints.  The two‐ply action
    selection is implemented in :meth:`_pick_action`.
    """

    def __init__(
        self,
        steps: int = 50000,
        lr: float = 3e-4,
        eps_greedy: float = 0.05,
        batch_size: int = 256,
        eval_batch_size: int = 8192,
        seed: int = 0,
    ):
        self.steps = steps
        self.lr = lr
        self.eps_greedy = eps_greedy
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size  # Batch size for 2-ply evaluation
        self.seed = seed

        # Build all possible sorted rolls once.
        self.rolls, self.roll_probs = all_sorted_rolls_and_probs()

        # Initial network parameters and optimiser.
        key = jax.random.PRNGKey(seed)
        dummy_board = jnp.zeros((1, BOARD_LENGTH, CONV_INPUT_CHANNELS), dtype=jnp.float32)
        dummy_aux = jnp.zeros((1, AUX_INPUT_SIZE), dtype=jnp.float32)
        params = ValueNet().init(
            key, board_state=dummy_board, aux_features=dummy_aux
        )["params"]
        tx = optax.adam(self.lr)
        self.state = train_state.TrainState.create(apply_fn=None, params=params, tx=tx)

    def _pick_action(
        self, state: np.ndarray, player: int
    ) -> Tuple[float, np.ndarray]:
        """Select an action using two‐ply search.

        Args:
            state: The current board state as a 1D ``np.ndarray`` of length 28.
            player: +1 for white or −1 for black.

        Returns:
            A tuple ``(r, next_state)``.  If the chosen move ends the game
            immediately, ``r`` is the terminal reward from the viewpoint of
            ``player`` and ``next_state`` is the afterstate.  Otherwise,
            ``r`` is zero and ``next_state`` is the afterstate to be fed
            into the TD(0) update.

        Notes:
            - An ε‐greedy policy is used for exploration: with probability
              ``eps_greedy`` we select a random legal afterstate.
            - A pure forced pass (no legal moves) returns the original state.
            - The value network is queried on the resulting states in
              batches to avoid repeated CPU↔GPU transfers when using JAX.
        """
        # Roll dice for the current player (sorted descending).
        dice = np.sort(np.random.randint(1, 7, size=(2,), dtype=np.int8))[::-1]
        # Enumerate legal moves and afterstates.
        moves, afterstates = actions_py(state, player, dice)
        if len(afterstates) == 0:
            # Forced pass.
            return 0.0, state

        # ε‐greedy exploration: random afterstate with fixed probability.
        if np.random.rand() < self.eps_greedy:
            idx = np.random.randint(len(afterstates))
            ns = afterstates[idx]
            r = float(py_reward(ns, player))
            return r, ns

        # Immediate terminal check: if any afterstate yields a non‐zero
        # reward, prefer that move immediately.
        for ns in afterstates:
            r = py_reward(ns, player)
            if r != 0:
                return float(r), ns

        # Build a list of all opponent response states across all candidate
        # afterstates and all possible dice rolls.  We evaluate these in
        # batches using the value network.  We also record how the
        # computed values map back to (afterstate index, dice index).
        all_boards: List[np.ndarray] = []
        all_aux: List[np.ndarray] = []
        idx_map: dict = {}

        # Preallocate storage for per‐afterstate expected values.
        expected_vals = np.zeros(len(afterstates), dtype=np.float64)

        for i, s1 in enumerate(afterstates):
            for j, (d1, d2) in enumerate(self.rolls):
                # Generate opponent moves from the afterstate.
                opp_moves, opp_afterstates = actions_py(s1, -player, np.array([d1, d2], dtype=np.int8))
                # If the opponent has no legal move, treat it as a pass: the state
                # remains s1 and the turn comes back to the current player.  We
                # indicate this by storing None for this (i,j) key.
                if len(opp_afterstates) == 0:
                    idx_map[(i, j)] = None
                else:
                    idx_list: List[int] = []
                    for s2 in opp_afterstates:
                        # From our perspective it will be our turn next, so we
                        # canonicalise with ``player``.
                        b, a = encode(s2, player)
                        all_boards.append(b)
                        all_aux.append(a)
                        idx_list.append(len(all_boards) - 1)
                    idx_map[(i, j)] = idx_list

        # Evaluate all gathered boards in batches to avoid OOM.
        # Process in chunks of eval_batch_size to fit in GPU memory.
        if len(all_boards) > 0:
            values = []
            num_batches = (len(all_boards) + self.eval_batch_size - 1) // self.eval_batch_size
            if num_batches > 10:  # Only print if there are many batches
                print(f"    Evaluating {len(all_boards)} states in {num_batches} batches...", flush=True)
            for batch_idx in range(num_batches):
                start_idx = batch_idx * self.eval_batch_size
                end_idx = min(start_idx + self.eval_batch_size, len(all_boards))
                batch_boards = all_boards[start_idx:end_idx]
                batch_aux = all_aux[start_idx:end_idx]
                boards_arr = jnp.asarray(np.stack(batch_boards))
                aux_arr = jnp.asarray(np.stack(batch_aux))
                # v_apply expects shape (B, 24, 15) and (B, AUX_INPUT_SIZE).
                values_jax = v_apply(self.state.params, boards_arr, aux_arr)
                values.extend(np.asarray(values_jax).tolist())
            values = np.array(values, dtype=np.float32)
        else:
            values = np.zeros(0, dtype=np.float32)

        # Compute the expected value for each candidate afterstate.
        for i in range(len(afterstates)):
            exp_val = 0.0
            s1 = afterstates[i]
            for j, prob in enumerate(self.roll_probs):
                mapping = idx_map.get((i, j))
                if mapping is None:
                    # Opponent forced pass: evaluate s1 with our turn.
                    b_s1, a_s1 = encode(s1, player)
                    # Add batch dimension when calling v_apply.
                    v = float(
                        v_apply(
                            self.state.params,
                            jnp.asarray(b_s1)[None, ...],
                            jnp.asarray(a_s1)[None, ...],
                        )[0]
                    )
                    min_val = v
                else:
                    # Select the minimal value across all opponent responses.
                    vals = values[mapping]
                    min_val = float(np.min(vals))
                exp_val += prob * min_val
            expected_vals[i] = exp_val

        # Choose the afterstate with maximal expected value.
        best_idx = int(np.argmax(expected_vals))
        best_state = afterstates[best_idx]
        reward = float(py_reward(best_state, player))
        return reward, best_state

    def train(self) -> None:
        """Run the TD(0) training loop with two‐ply action selection."""
        print("=" * 70)
        print("Starting 2-ply TD(0) training")
        print(f"Steps: {self.steps}, Batch size: {self.batch_size}, Eval batch: {self.eval_batch_size}")
        print("=" * 70)
        print("\n⚠️  WARNING: First run will trigger JAX JIT compilation.")
        print("   This can take 5-30 minutes. Please be patient...")
        print("   You'll see 'Compiling...' messages if JAX is compiling.\n")
        
        # Replay buffers for batch updates.
        buf_board: List[np.ndarray] = []
        buf_aux: List[np.ndarray] = []
        buf_tgt: List[float] = []

        # Initialise the first game.
        print("Initializing first game...", flush=True)
        player, _dice, state = _new_game()
        t0 = time.time()
        last_loss = 0.0
        
        print("Starting training loop (first action selection will trigger compilation)...", flush=True)
        print("=" * 70, flush=True)

        for it in range(1, self.steps + 1):
            # Encode the current state for TD(0).
            b, a = encode(state, player)
            # Select action using two‐ply search.
            if it == 1:
                print(f"\n[Step 1] First action selection (JIT compilation may take 5-30 min)...", flush=True)
            elif it % 10 == 0:
                print(f"\n[Step {it}] Selecting action with 2-ply search...", flush=True)
            r, next_state = self._pick_action(state, player)
            if it == 1:
                print(f"[Step 1] ✓ First action completed! Training will be faster now.\n", flush=True)
            elif it % 10 == 0:
                print(f"[Step {it}] ✓ Action selected", flush=True)

            # Determine TD target: if terminal then r; else -V(next_state).
            if r != 0:
                tgt = float(r)
                # Start a new game on terminal.
                buf_board.append(b)
                buf_aux.append(a)
                buf_tgt.append(tgt)
                player, _dice, state = _new_game()
            else:
                nb, na = encode(next_state, -player)
                nb_batch = jnp.asarray(nb)[None, ...]
                na_batch = jnp.asarray(na)[None, ...]
                v_next = float(v_apply(self.state.params, nb_batch, na_batch)[0])
                tgt = -v_next
                buf_board.append(b)
                buf_aux.append(a)
                buf_tgt.append(tgt)
                # Advance to the new state and switch players.
                state, player = next_state, -player

            # Perform a gradient update when the buffer is full.
            if len(buf_tgt) >= self.batch_size:
                board_batch = jnp.asarray(np.stack(buf_board))
                aux_batch = jnp.asarray(np.stack(buf_aux))
                target_batch = jnp.asarray(np.array(buf_tgt, dtype=np.float32))
                loss, grads = loss_and_grads(self.state.params, board_batch, aux_batch, target_batch)
                last_loss = float(loss)
                self.state = self.state.apply_gradients(grads=grads)
                buf_board.clear()
                buf_aux.clear()
                buf_tgt.clear()

            # Periodic progress reporting (every 10 steps for 2-ply since it's slow).
            if it % 10 == 0:
                dt = time.time() - t0
                rate = it / dt if dt > 0 else 0
                eta_seconds = (self.steps - it) / rate if rate > 0 else 0
                eta_minutes = eta_seconds / 60
                eta_hours = eta_minutes / 60
                print(f"\n{'='*70}")
                print(f"PROGRESS UPDATE - Step {it}/{self.steps}")
                print(f"{'='*70}")
                print(f"  Loss: {last_loss:.6f}")
                print(f"  Elapsed time: {dt/60:.1f} minutes ({dt/3600:.2f} hours)")
                print(f"  Training rate: {rate:.4f} steps/second")
                if eta_hours < 1:
                    print(f"  Estimated time remaining: {eta_minutes:.1f} minutes")
                else:
                    print(f"  Estimated time remaining: {eta_hours:.1f} hours ({eta_minutes:.1f} minutes)")
                print(f"  Progress: {100.0 * it / self.steps:.2f}%")
                print(f"{'='*70}\n", flush=True)
            
            # Periodic checkpoint saving (every 1000 steps) to avoid losing progress
            if it % 1000 == 0 and it > 0:
                ckpt_dir = os.path.abspath("checkpoints_2ply")
                os.makedirs(ckpt_dir, exist_ok=True)
                ckpt_path = checkpoints.save_checkpoint(
                    ckpt_dir=ckpt_dir, target=self.state.params, step=it, overwrite=True
                )
                print(f"\n💾 CHECKPOINT SAVED at step {it}")
                print(f"   Location: {ckpt_path}")
                print(f"   You can resume from here if training is interrupted.\n", flush=True)

        # Save the final parameters.
        ckpt_dir = os.path.abspath("checkpoints_2ply")
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_path = checkpoints.save_checkpoint(
            ckpt_dir=ckpt_dir, target=self.state.params, step=self.steps, overwrite=True
        )
        print(f"Saved final parameters to {ckpt_path}")


def main():
    parser = argparse.ArgumentParser(description="Two‐ply TD(0) Backgammon training script")
    parser.add_argument("--steps", type=int, default=50000, help="Number of updates (default: 50000)")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate (default: 3e-4)")
    parser.add_argument(
        "--eps_greedy", type=float, default=0.05, help="Epsilon for random exploration (default: 0.05)"
    )
    parser.add_argument(
        "--batch", type=int, default=256, help="Batch size for optimisation (default: 256)"
    )
    parser.add_argument(
        "--eval_batch", type=int, default=8192, help="Batch size for 2-ply evaluation (default: 8192)"
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed (default: 0)")
    args = parser.parse_args()
    trainer = TwoPlyTD0Trainer(
        steps=args.steps,
        lr=args.lr,
        eps_greedy=args.eps_greedy,
        batch_size=args.batch,
        eval_batch_size=args.eval_batch,
        seed=args.seed,
    )
    trainer.train()


if __name__ == "__main__":
    main()