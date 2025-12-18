"""
Compare two checkpoint files by having them play games against each other.

This script follows the standard evaluation method:
1. Freeze both checkpoints (they never change)
2. Let them play games against each other
3. Count wins, losses, gammons, backgammons
4. Determine which checkpoint is stronger

Usage:
    # Compare two checkpoints
    python compare_checkpoint_steps.py checkpoint_100k checkpoint_200k --games 1000
    
    # Compare against random baseline
    python compare_checkpoint_steps.py checkpoint_100k --random --games 500
"""

import argparse
import os
import sys
from typing import Optional, Tuple

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
from train_value_td0 import actions_py, encode, py_reward

# Fixed batch size for JIT compilation (avoids recompilation on every move)
EVAL_BATCH = 256  # Use 512 on A100 if you have more memory

# Create module instance once outside JIT to avoid reinitialization issues
_value_net = ValueNet()


@jax.jit
def value_apply_fixed(params, board, aux):
    """Apply value network to a fixed-size batch of states.
    
    This function is JIT-compiled with a fixed batch size (EVAL_BATCH)
    to avoid recompilation on every move. The input must always be
    shape (EVAL_BATCH, 24, 15) for board and (EVAL_BATCH, 6) for aux.
    
    Uses a module instance created outside JIT to avoid parameter
    initialization issues during tracing.
    """
    return _value_net.apply(
        {"params": params},
        board_state=board,
        aux_features=aux,
    ).squeeze(-1)


def pick_action_greedy(state: np.ndarray, player: int, params: dict, eps: float = 0.0) -> Tuple[float, np.ndarray]:
    """One-ply greedy move selection.
    
    Args:
        state: Current board state
        player: +1 for white, -1 for black
        params: Network parameters
        eps: Epsilon for random exploration (0.0 = pure greedy)
    
    Returns:
        (reward, next_state) tuple
    """
    dice = np.sort(np.random.randint(1, 7, size=(2,), dtype=np.int8))[::-1]
    moves, afterstates = actions_py(state, player, dice)
    
    if len(afterstates) == 0:
        return 0.0, state
    
    # Epsilon-greedy exploration
    if eps > 0.0 and np.random.rand() < eps:
        idx = np.random.randint(len(afterstates))
        ns = afterstates[idx]
        r = py_reward(ns, player)
        return float(r), ns
    
    # Terminal state check
    for ns in afterstates:
        r = py_reward(ns, player)
        if r != 0:
            return float(r), ns
    
    # Greedy selection: minimize opponent's value
    boards, auxs = [], []
    for s in afterstates:
        b, a = encode(s, -player)  # Value net predicts for side-to-move
        boards.append(b)
        auxs.append(a)
    
    if len(boards) == 0:
        return 0.0, state
    
    boards_np = np.asarray(boards, dtype=np.float32)
    auxs_np = np.asarray(auxs, dtype=np.float32)
    
    # Evaluate in FIXED batches to avoid JAX recompiling every move
    vals_all = []
    n = boards_np.shape[0]
    for i in range(0, n, EVAL_BATCH):
        b = boards_np[i:i+EVAL_BATCH]
        a = auxs_np[i:i+EVAL_BATCH]
        
        # Pad last batch to EVAL_BATCH
        if b.shape[0] < EVAL_BATCH:
            pad = EVAL_BATCH - b.shape[0]
            b = np.pad(b, ((0, pad), (0, 0), (0, 0)), mode="constant")
            a = np.pad(a, ((0, pad), (0, 0)), mode="constant")
        
        v = value_apply_fixed(params, jnp.asarray(b), jnp.asarray(a))
        v = np.array(v)  # Bring back to numpy
        vals_all.append(v[:min(EVAL_BATCH, n - i)])
    
    vals = np.concatenate(vals_all, axis=0)
    best = int(np.argmin(vals))  # Minimize opponent's value
    
    next_state = afterstates[best]
    reward = py_reward(next_state, player)
    return float(reward), next_state


def load_checkpoint(ckpt_path: str) -> dict:
    """Load checkpoint from a file path or directory.
    
    Args:
        ckpt_path: Path to checkpoint file or directory
    
    Returns:
        Model parameters dictionary
    """
    ckpt_path = os.path.abspath(ckpt_path)
    
    # Create dummy parameters to get the structure right
    key = jax.random.PRNGKey(0)
    dummy_board = jnp.zeros((1, BOARD_LENGTH, CONV_INPUT_CHANNELS), dtype=jnp.float32)
    dummy_aux = jnp.zeros((1, AUX_INPUT_SIZE), dtype=jnp.float32)
    params = ValueNet().init(key, board_state=dummy_board, aux_features=dummy_aux)["params"]
    
    try:
        # Try to restore checkpoint (works with both file paths and directories)
        restored = checkpoints.restore_checkpoint(ckpt_path, target=params)
        print(f"✓ Loaded checkpoint from: {ckpt_path}")
        return restored
    except Exception as e:
        print(f"✗ Error loading checkpoint from {ckpt_path}: {e}", file=sys.stderr)
        print(f"  Make sure the checkpoint file or directory exists.", file=sys.stderr)
        sys.exit(1)


def play_games(
    agent_a_params: Optional[dict],
    agent_b_params: Optional[dict],
    games: int = 1000,
    verbose: bool = False,
) -> Tuple[int, int, int, int, int, int, float]:
    """Play games between two agents and return statistics.
    
    Args:
        agent_a_params: Parameters for Agent A (plays as +1/White), or None for random
        agent_b_params: Parameters for Agent B (plays as -1/Black), or None for random
        games: Number of games to play
        verbose: Print per-game results
    
    Returns:
        (a_wins, b_wins, a_gammons, b_gammons, a_backgammons, b_backgammons, avg_turns) tuple
    """
    a_wins = 0
    b_wins = 0
    a_gammons = 0  # Agent A wins by gammon
    b_gammons = 0  # Agent B wins by gammon
    a_backgammons = 0  # Agent A wins by backgammon
    b_backgammons = 0  # Agent B wins by backgammon
    total_turns = 0
    
    for game in range(games):
        player, _, state = _new_game()
        turns = 0
        
        while True:
            turns += 1
            
            if player == 1:  # Agent A's turn
                if agent_a_params is None:
                    # Random move
                    dice = np.sort(np.random.randint(1, 7, size=(2,), dtype=np.int8))[::-1]
                    moves, afterstates = actions_py(state, player, dice)
                    if len(afterstates) == 0:
                        next_state = state
                        reward = 0.0
                    else:
                        next_state = afterstates[np.random.randint(len(afterstates))]
                        reward = py_reward(next_state, player)
                else:
                    # Use trained agent
                    reward, next_state = pick_action_greedy(state, player, agent_a_params, eps=0.0)
            else:  # Agent B's turn
                if agent_b_params is None:
                    # Random move
                    dice = np.sort(np.random.randint(1, 7, size=(2,), dtype=np.int8))[::-1]
                    moves, afterstates = actions_py(state, player, dice)
                    if len(afterstates) == 0:
                        next_state = state
                        reward = 0.0
                    else:
                        next_state = afterstates[np.random.randint(len(afterstates))]
                        reward = py_reward(next_state, player)
                else:
                    # Use trained agent
                    reward, next_state = pick_action_greedy(state, player, agent_b_params, eps=0.0)
            
            if reward != 0:
                # Game ended - track win type
                abs_reward = abs(reward)
                if player == 1:  # Agent A won
                    a_wins += 1
                    if abs_reward == 2:
                        a_gammons += 1
                    elif abs_reward == 3:
                        a_backgammons += 1
                else:  # Agent B won
                    b_wins += 1
                    if abs_reward == 2:
                        b_gammons += 1
                    elif abs_reward == 3:
                        b_backgammons += 1
                break
            
            state = next_state
            player = -player
        
        total_turns += turns
        
        # Progress update every 10 games (so it doesn't look frozen)
        if (game + 1) % 10 == 0:
            print(f"Played {game+1}/{games} games... A={a_wins} B={b_wins}", flush=True)
        
        if verbose and (game + 1) % 100 == 0:
            print(f"Game {game + 1}/{games}: A={a_wins} (gammons={a_gammons}, bg={a_backgammons}), "
                  f"B={b_wins} (gammons={b_gammons}, bg={b_backgammons})", flush=True)
    
    avg_turns = total_turns / games
    return a_wins, b_wins, a_gammons, b_gammons, a_backgammons, b_backgammons, avg_turns


def compare_checkpoints(ckpt_a: str, ckpt_b: str, games: int = 1000, num_runs: int = 3) -> None:
    """Compare two checkpoints by having them play games against each other.
    
    Args:
        ckpt_a: Path to first checkpoint (Agent A)
        ckpt_b: Path to second checkpoint (Agent B)
        games: Number of games per run
        num_runs: Number of evaluation runs (to account for first-mover advantage)
    """
    print("=" * 70)
    print("COMPARING CHECKPOINTS")
    print("=" * 70)
    print(f"Agent A (White): {ckpt_a}")
    print(f"Agent B (Black): {ckpt_b}")
    print(f"Games per run: {games}")
    print(f"Number of runs: {num_runs}")
    print("=" * 70)
    
    # Load checkpoints
    print("\nLoading checkpoints...")
    params_a = load_checkpoint(ckpt_a)
    params_b = load_checkpoint(ckpt_b)
    
    # Run multiple evaluation series (swap colors to account for first-mover advantage)
    print(f"\nRunning {num_runs} evaluation series...")
    a_wins_list = []
    b_wins_list = []
    a_gammons_list = []
    b_gammons_list = []
    a_backgammons_list = []
    b_backgammons_list = []
    avg_turns_list = []
    
    for run in range(num_runs):
        print(f"\nRun {run + 1}/{num_runs}:")
        # Run twice: once with A as White, once as Black
        a_wins_1, b_wins_1, a_gam_1, b_gam_1, a_bg_1, b_bg_1, turns_1 = play_games(
            params_a, params_b, games=games // 2, verbose=False
        )
        a_wins_2, b_wins_2, a_gam_2, b_gam_2, a_bg_2, b_bg_2, turns_2 = play_games(
            params_b, params_a, games=games // 2, verbose=False
        )
        
        # Aggregate: A wins when it's Agent A and wins, or when it's Agent B and Agent A loses
        a_total = a_wins_1 + b_wins_2
        b_total = b_wins_1 + a_wins_2
        a_gam_total = a_gam_1 + b_gam_2
        b_gam_total = b_gam_1 + a_gam_2
        a_bg_total = a_bg_1 + b_bg_2
        b_bg_total = b_bg_1 + a_bg_2
        avg_turns = (turns_1 + turns_2) / 2.0
        
        a_wins_list.append(a_total)
        b_wins_list.append(b_total)
        a_gammons_list.append(a_gam_total)
        b_gammons_list.append(b_gam_total)
        a_backgammons_list.append(a_bg_total)
        b_backgammons_list.append(b_bg_total)
        avg_turns_list.append(avg_turns)
        
        print(f"  {os.path.basename(ckpt_a)} wins: {a_total}, "
              f"{os.path.basename(ckpt_b)} wins: {b_total}")
        print(f"  Gammons: A={a_gam_total}, B={b_gam_total} | "
              f"Backgammons: A={a_bg_total}, B={b_bg_total} | "
              f"Avg turns: {avg_turns:.1f}")
    
    # Compute statistics
    a_mean = np.mean(a_wins_list)
    b_mean = np.mean(b_wins_list)
    a_std = np.std(a_wins_list)
    b_std = np.std(b_wins_list)
    
    a_gam_mean = np.mean(a_gammons_list)
    b_gam_mean = np.mean(b_gammons_list)
    a_bg_mean = np.mean(a_backgammons_list)
    b_bg_mean = np.mean(b_backgammons_list)
    
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"{os.path.basename(ckpt_a)} average wins: {a_mean:.1f} ± {a_std:.1f} / {games}")
    print(f"{os.path.basename(ckpt_b)} average wins: {b_mean:.1f} ± {b_std:.1f} / {games}")
    print(f"\nAverage game length: {np.mean(avg_turns_list):.2f} turns")
    
    win_rate_a = a_mean / games
    win_rate_b = b_mean / games
    
    print(f"\nWin rates:")
    print(f"  {os.path.basename(ckpt_a)}: {100.0 * win_rate_a:.2f}%")
    print(f"  {os.path.basename(ckpt_b)}: {100.0 * win_rate_b:.2f}%")
    
    print(f"\nGammon rates:")
    print(f"  {os.path.basename(ckpt_a)}: {100.0 * a_gam_mean / games:.2f}%")
    print(f"  {os.path.basename(ckpt_b)}: {100.0 * b_gam_mean / games:.2f}%")
    
    print(f"\nBackgammon rates:")
    print(f"  {os.path.basename(ckpt_a)}: {100.0 * a_bg_mean / games:.2f}%")
    print(f"  {os.path.basename(ckpt_b)}: {100.0 * b_bg_mean / games:.2f}%")
    
    # Determine which is stronger
    if win_rate_a > win_rate_b:
        improvement = (win_rate_a - win_rate_b) / win_rate_b * 100 if win_rate_b > 0 else 0
        print(f"\n✓ {os.path.basename(ckpt_a)} is stronger than {os.path.basename(ckpt_b)} by {improvement:.1f}%")
    elif win_rate_a < win_rate_b:
        improvement = (win_rate_b - win_rate_a) / win_rate_a * 100 if win_rate_a > 0 else 0
        print(f"\n✓ {os.path.basename(ckpt_b)} is stronger than {os.path.basename(ckpt_a)} by {improvement:.1f}%")
    else:
        print(f"\n→ Both checkpoints perform similarly")
    
    # Statistical test if scipy is available
    try:
        from scipy import stats
        t_stat, p_value = stats.ttest_rel(a_wins_list, b_wins_list)
        print(f"\nStatistical test (paired t-test):")
        print(f"  t-statistic: {t_stat:.4f}")
        print(f"  p-value: {p_value:.4f}")
        if p_value < 0.05:
            print(f"  → Difference is statistically significant (p < 0.05)")
        else:
            print(f"  → Difference is not statistically significant (p >= 0.05)")
            print(f"  → No real improvement detected (within noise)")
    except ImportError:
        pass
    
    print("=" * 70)


def compare_against_random(ckpt: str, games: int = 500) -> None:
    """Compare a checkpoint against random baseline."""
    print("=" * 70)
    print("EVALUATING CHECKPOINT AGAINST RANDOM BASELINE")
    print("=" * 70)
    print(f"Checkpoint: {ckpt}")
    print(f"Games: {games}")
    print("=" * 70)
    
    print("\nLoading checkpoint...")
    params = load_checkpoint(ckpt)
    
    print(f"\nRunning evaluation (checkpoint vs random)...")
    a_wins, b_wins, a_gammons, b_gammons, a_backgammons, b_backgammons, avg_turns = play_games(
        params, None, games=games, verbose=False
    )
    
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Checkpoint wins: {a_wins} / {games} ({100.0 * a_wins / games:.2f}%)")
    print(f"Random wins: {b_wins} / {games} ({100.0 * b_wins / games:.2f}%)")
    print(f"Average game length: {avg_turns:.2f} turns")
    print(f"\nGammons: Checkpoint={a_gammons}, Random={b_gammons}")
    print(f"Backgammons: Checkpoint={a_backgammons}, Random={b_backgammons}")
    
    if a_wins > b_wins:
        print(f"\n✓ Checkpoint is stronger than random baseline")
    elif a_wins < b_wins:
        print(f"\n✗ Checkpoint is weaker than random baseline (this is unusual!)")
    else:
        print(f"\n→ Checkpoint performs similarly to random baseline")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Compare two checkpoint files by having them play games",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare two checkpoints
  python compare_checkpoint_steps.py checkpoint_100k checkpoint_200k --games 1000
  
  # Compare checkpoint against random baseline
  python compare_checkpoint_steps.py checkpoint_100k --random --games 500
  
  # Use checkpoint directories
  python compare_checkpoint_steps.py checkpoints_1ply checkpoints_2ply --games 1000
        """,
    )
    parser.add_argument("checkpoint1", type=str, help="Path to first checkpoint file or directory")
    parser.add_argument("checkpoint2", type=str, nargs="?", default=None,
                        help="Path to second checkpoint file or directory (optional if --random)")
    parser.add_argument("--random", action="store_true",
                        help="Compare checkpoint1 against random baseline instead of checkpoint2")
    parser.add_argument("--games", type=int, default=1000,
                        help="Number of games per comparison (default: 1000)")
    parser.add_argument("--runs", type=int, default=3,
                        help="Number of evaluation runs (default: 3)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-game results")
    
    args = parser.parse_args()
    
    if args.random:
        compare_against_random(args.checkpoint1, args.games)
    else:
        if args.checkpoint2 is None:
            print("Error: Either provide two checkpoints or use --random flag", file=sys.stderr)
            sys.exit(1)
        compare_checkpoints(args.checkpoint1, args.checkpoint2, args.games, args.runs)


if __name__ == "__main__":
    main()
