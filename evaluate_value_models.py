"""
Evaluate Backgammon value networks by pitting them against each other.

This script loads two sets of parameters for the value network and
implements simple one‐ply and two‐ply move selection routines.  It
simulates a specified number of games where the first agent (Agent A)
always plays as the positive player (+1, White) and the second agent
(Agent B) plays as the negative player (−1, Black).  At each turn
the appropriate move selector is invoked and the game advances until
a terminal reward is returned.  At the end the script reports the
number of wins for each agent and the average game length.

Usage:

    python evaluate_value_models.py --agent_a checkpoint_dir_A \
                                    --agent_b checkpoint_dir_B \
                                    --games 1000

If only one agent is provided, the script will pit it against a
random‐move baseline agent.  The baseline agent selects a legal
afterstate uniformly at random on every turn.

The move selection functions here mirror those used in the training
scripts:

 * ``pick_action_greedy``: one‐ply lookahead (as in the original TD0
   trainer) that chooses the afterstate which minimises the
   opponent's value.  Terminal states are preferred immediately.

 * ``pick_action_2ply``: two‐ply lookahead as described in
   ``train_value_td0_2ply.py``.  All possible dice rolls are
   considered for the opponent and the minimal value of their
   responses is averaged to obtain the expected value.

Both routines assume that the value networks estimate equity from
the perspective of the side to move, just like in training.
"""

import argparse
import os
from typing import List, Tuple, Optional

import numpy as np
import jax
import jax.numpy as jnp
from flax.training import checkpoints

from backgammon_engine import _new_game
from backgammon_value_net import (
    BackgammonValueNet as ValueNet,
    BOARD_LENGTH,
    CONV_INPUT_CHANNELS,
    AUX_INPUT_SIZE,
)
from train_value_td0 import actions_py, encode, py_reward, v_apply
from train_value_td0_2ply import all_sorted_rolls_and_probs


def load_params(ckpt_dir: str) -> dict:
    """Load model parameters from a checkpoint directory.

    Args:
        ckpt_dir: Path to a directory containing a Flax checkpoint.  The
            directory should contain a file named ``checkpoint_{step}``.

    Returns:
        A nested dictionary representing the loaded parameters.  This
        function uses a dummy model initialisation to establish the
        correct parameter structure before restoration.
    """
    # Convert to absolute path (required by Orbax/Flax checkpoints)
    ckpt_dir = os.path.abspath(ckpt_dir)
    # Create dummy parameters to get the structure right.
    key = jax.random.PRNGKey(0)
    dummy_board = jnp.zeros((1, BOARD_LENGTH, CONV_INPUT_CHANNELS), dtype=jnp.float32)
    dummy_aux = jnp.zeros((1, AUX_INPUT_SIZE), dtype=jnp.float32)
    params = ValueNet().init(key, board_state=dummy_board, aux_features=dummy_aux)["params"]
    # Restore the checkpoint into the dummy structure.
    restored = checkpoints.restore_checkpoint(ckpt_dir, target=params)
    return restored


def pick_action_greedy(
    state: np.ndarray, player: int, params: dict, eps: float = 0.0
) -> Tuple[float, np.ndarray]:
    """One‐ply greedy move selection.

    This function mirrors the behaviour of the original TD0 trainer
    (sans the bug) by enumerating all legal afterstates for the roll
    and choosing the one that minimises the opponent's value.  A
    small epsilon can be supplied to make random selections for
    exploratory play; when evaluating agents, set ``eps=0.0`` to
    disable exploration.
    """
    dice = np.sort(np.random.randint(1, 7, size=(2,), dtype=np.int8))[::-1]
    moves, afterstates = actions_py(state, player, dice)
    if len(afterstates) == 0:
        return 0.0, state
    # Random move with probability eps
    if eps > 0.0 and np.random.rand() < eps:
        idx = np.random.randint(len(afterstates))
        ns = afterstates[idx]
        r = float(py_reward(ns, player))
        return r, ns
    # Immediate terminal check
    for ns in afterstates:
        r = py_reward(ns, player)
        if r != 0:
            return float(r), ns
    # Evaluate all afterstates from the opponent's perspective.
    boards = []
    auxs = []
    for ns in afterstates:
        b, a = encode(ns, -player)
        boards.append(b)
        auxs.append(a)
    boards_arr = jnp.asarray(np.stack(boards))
    auxs_arr = jnp.asarray(np.stack(auxs))
    vals = v_apply(params, boards_arr, auxs_arr)
    vals_np = np.asarray(vals)
    # Choose the afterstate minimising the opponent's value (maximising ours).
    best_idx = int(np.argmin(vals_np))
    best_state = afterstates[best_idx]
    reward = float(py_reward(best_state, player))
    return reward, best_state


def pick_action_2ply(
    state: np.ndarray,
    player: int,
    params: dict,
    rolls: List[Tuple[int, int]],
    roll_probs: List[float],
) -> Tuple[float, np.ndarray]:
    """Two‐ply greedy move selection as in the improved TD0 trainer.

    Args:
        state: The current board as a 1D numpy array of length 28.
        player: +1 or −1, indicating whose turn it is.
        params: Parameters of the value network to evaluate states.
        rolls: List of all sorted dice roll pairs.
        roll_probs: Corresponding probabilities for each sorted roll.

    Returns:
        (r, next_state) where ``r`` is the terminal reward (if any)
        from the perspective of ``player`` and ``next_state`` is the
        afterstate to feed into the next search iteration.
    """
    # Roll dice for the current player.
    dice = np.sort(np.random.randint(1, 7, size=(2,), dtype=np.int8))[::-1]
    moves, afterstates = actions_py(state, player, dice)
    if len(afterstates) == 0:
        return 0.0, state
    # Immediate terminal check
    for ns in afterstates:
        r = py_reward(ns, player)
        if r != 0:
            return float(r), ns
    # Gather opponent responses across all candidate moves and dice.
    all_boards: List[np.ndarray] = []
    all_aux: List[np.ndarray] = []
    idx_map: dict = {}
    expected_vals = np.zeros(len(afterstates), dtype=np.float64)
    for i, s1 in enumerate(afterstates):
        for j, (d1, d2) in enumerate(rolls):
            opp_moves, opp_afterstates = actions_py(s1, -player, np.array([d1, d2], dtype=np.int8))
            if len(opp_afterstates) == 0:
                idx_map[(i, j)] = None
            else:
                idx_list: List[int] = []
                for s2 in opp_afterstates:
                    b, a = encode(s2, player)
                    all_boards.append(b)
                    all_aux.append(a)
                    idx_list.append(len(all_boards) - 1)
                idx_map[(i, j)] = idx_list
        # Evaluate all batched states in chunks to avoid OOM.
        # Use a reasonable batch size (8192) to fit in GPU memory.
        eval_batch_size = 8192
        if len(all_boards) > 0:
            values = []
            num_batches = (len(all_boards) + eval_batch_size - 1) // eval_batch_size
            for batch_idx in range(num_batches):
                start_idx = batch_idx * eval_batch_size
                end_idx = min(start_idx + eval_batch_size, len(all_boards))
                batch_boards = all_boards[start_idx:end_idx]
                batch_aux = all_aux[start_idx:end_idx]
                boards_arr = jnp.asarray(np.stack(batch_boards))
                aux_arr = jnp.asarray(np.stack(batch_aux))
                values_jax = v_apply(params, boards_arr, aux_arr)
                values.extend(np.asarray(values_jax).tolist())
            values = np.array(values, dtype=np.float32)
        else:
            values = np.zeros(0, dtype=np.float32)
    # Accumulate expected values.
    for i, s1 in enumerate(afterstates):
        ev = 0.0
        for j, prob in enumerate(roll_probs):
            mapping = idx_map.get((i, j))
            if mapping is None:
                # Forced pass: evaluate s1 on our turn
                b_s1, a_s1 = encode(s1, player)
                v = float(
                    v_apply(
                        params,
                        jnp.asarray(b_s1)[None, ...],
                        jnp.asarray(a_s1)[None, ...],
                    )[0]
                )
                min_val = v
            else:
                vals = values[mapping]
                min_val = float(np.min(vals))
            ev += prob * min_val
        expected_vals[i] = ev
    best_idx = int(np.argmax(expected_vals))
    best_state = afterstates[best_idx]
    reward = float(py_reward(best_state, player))
    return reward, best_state


def evaluate(
    agent_a_params: Optional[dict],
    agent_b_params: Optional[dict],
    games: int = 100,
    verbose: bool = False,
) -> None:
    """Simulate a series of games between two agents.

    Agent A always plays as White (+1); Agent B plays as Black (−1).
    If either ``agent_a_params`` or ``agent_b_params`` is ``None``,
    that agent selects moves uniformly at random.  Otherwise the
    corresponding network is used with a 2‐ply lookahead for Agent A
    and a 1‐ply lookahead for Agent B.  The game continues until
    ``py_reward`` returns a non‐zero value, at which point the winner
    is determined by the sign of the reward.

    Args:
        agent_a_params: Parameters for Agent A's value network.  If
            ``None``, Agent A acts randomly.
        agent_b_params: Parameters for Agent B's value network.  If
            ``None``, Agent B acts randomly.
        games: Number of games to simulate.
        verbose: If true, prints a line for each game indicating the
            winner and game length.
    """
    rolls, roll_probs = all_sorted_rolls_and_probs()
    a_wins = 0
    b_wins = 0
    total_turns = 0
    for gi in range(games):
        player, _dice, state = _new_game()
        turns = 0
        while True:
            turns += 1
            # Determine which agent is to act based on current player sign.
            if player == 1:
                # Agent A (White)
                if agent_a_params is None:
                    # Random move selection.
                    dice = np.sort(np.random.randint(1, 7, size=(2,), dtype=np.int8))[::-1]
                    _moves, afters = actions_py(state, player, dice)
                    if len(afters) == 0:
                        r, next_state = 0.0, state
                    else:
                        idx = np.random.randint(len(afters))
                        next_state = afters[idx]
                        r = float(py_reward(next_state, player))
                else:
                    # Two‐ply move selection for Agent A
                    r, next_state = pick_action_2ply(state, player, agent_a_params, rolls, roll_probs)
            else:
                # Agent B (Black)
                if agent_b_params is None:
                    dice = np.sort(np.random.randint(1, 7, size=(2,), dtype=np.int8))[::-1]
                    _moves, afters = actions_py(state, player, dice)
                    if len(afters) == 0:
                        r, next_state = 0.0, state
                    else:
                        idx = np.random.randint(len(afters))
                        next_state = afters[idx]
                        r = float(py_reward(next_state, player))
                else:
                    # One‐ply greedy move selection for Agent B
                    r, next_state = pick_action_greedy(state, player, agent_b_params, eps=0.0)
            if r != 0:
                # Determine winner based on the sign of the reward and the current player.
                if r > 0:
                    if player == 1:
                        a_wins += 1
                    else:
                        b_wins += 1
                else:
                    if player == 1:
                        b_wins += 1
                    else:
                        a_wins += 1
                total_turns += turns
                if verbose:
                    winner = "A" if (r > 0 and player == 1) or (r < 0 and player == -1) else "B"
                    print(f"Game {gi+1}: winner={winner}, turns={turns}")
                break
            else:
                # Switch turns and continue playing.
                state, player = next_state, -player
        # End of one game
    # Report summary statistics
    print(f"Agent A wins: {a_wins} / {games} ({100.0 * a_wins / games:.2f}%)")
    print(f"Agent B wins: {b_wins} / {games} ({100.0 * b_wins / games:.2f}%)")
    print(f"Average game length: {total_turns / games:.2f} turns")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Backgammon value networks.")
    parser.add_argument("--agent_a", type=str, help="Checkpoint directory for Agent A's model.")
    parser.add_argument(
        "--agent_b", type=str, default=None, help="Checkpoint directory for Agent B's model (optional)."
    )
    parser.add_argument("--games", type=int, default=100, help="Number of games to simulate (default: 100).")
    parser.add_argument(
        "--verbose", action="store_true", help="Print per‐game results instead of only summary."
    )
    args = parser.parse_args()
    agent_a_params = load_params(args.agent_a) if args.agent_a else None
    agent_b_params = load_params(args.agent_b) if args.agent_b else None
    evaluate(agent_a_params, agent_b_params, games=args.games, verbose=args.verbose)


if __name__ == "__main__":
    main()