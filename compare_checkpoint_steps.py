"""
Compare checkpoints from the same training run at different steps.

This script loads checkpoints at different training steps (e.g., step 5000 vs step 200000)
and evaluates their playing strength by having them play games against each other.

Usage:
    # Compare step 5000 vs step 200000 from 1-ply training
    python compare_checkpoint_steps.py --method 1ply --step1 5000 --step2 200000 --games 1000
    
    # Compare step 5000 vs step 200000 from 2-ply training
    python compare_checkpoint_steps.py --method 2ply --step1 5000 --step2 200000 --games 1000
    
    # Compare against random baseline
    python compare_checkpoint_steps.py --method 1ply --step1 5000 --games 500 --vs_random
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
from train_value_td0 import actions_py, encode, py_reward, v_apply
from evaluate_value_models import pick_action_greedy


def load_checkpoint_at_step(method: str, step: int) -> dict:
    """Load checkpoint at a specific step from a training method.
    
    Args:
        method: Training method name ('1ply' or '2ply')
        step: Training step number
    
    Returns:
        Model parameters dictionary
    """
    ckpt_dir = os.path.abspath(f"checkpoints_{method}")
    
    # Create dummy parameters to get the structure right
    key = jax.random.PRNGKey(0)
    dummy_board = jnp.zeros((1, BOARD_LENGTH, CONV_INPUT_CHANNELS), dtype=jnp.float32)
    dummy_aux = jnp.zeros((1, AUX_INPUT_SIZE), dtype=jnp.float32)
    params = ValueNet().init(key, board_state=dummy_board, aux_features=dummy_aux)["params"]
    
    # Try to restore checkpoint
    try:
        restored = checkpoints.restore_checkpoint(ckpt_dir, target=params, step=step)
        print(f"✓ Loaded checkpoint from {ckpt_dir} at step {step}")
        return restored
    except Exception as e:
        print(f"✗ Error loading checkpoint at step {step} from {ckpt_dir}: {e}", file=sys.stderr)
        print(f"  Make sure the checkpoint exists. Checkpoint should be at:", file=sys.stderr)
        print(f"  {ckpt_dir}/checkpoint_{step}.orbax-checkpoint", file=sys.stderr)
        sys.exit(1)


def run_evaluation(
    agent_a_params: Optional[dict],
    agent_b_params: Optional[dict],
    games: int = 100,
    verbose: bool = False,
) -> Tuple[int, int, float]:
    """Run games between two agents and return win counts.
    
    Args:
        agent_a_params: Parameters for Agent A (plays as +1/White), or None for random
        agent_b_params: Parameters for Agent B (plays as -1/Black), or None for random
        games: Number of games to play
        verbose: Print per-game results
    
    Returns:
        (a_wins, b_wins, avg_turns) tuple
    """
    a_wins = 0
    b_wins = 0
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
                # Game ended
                if player == 1:  # Agent A won
                    a_wins += 1
                else:  # Agent B won
                    b_wins += 1
                break
            
            state = next_state
            player = -player
        
        total_turns += turns
        
        if verbose and (game + 1) % 100 == 0:
            print(f"Game {game + 1}/{games}: A={a_wins}, B={b_wins}")
    
    avg_turns = total_turns / games
    return a_wins, b_wins, avg_turns


def compare_steps(method: str, step1: int, step2: int, games: int = 1000, num_runs: int = 3) -> None:
    """Compare two checkpoints from the same training run at different steps."""
    print("=" * 70)
    print("COMPARING CHECKPOINTS AT DIFFERENT STEPS")
    print("=" * 70)
    print(f"Training method: {method}")
    print(f"Step 1: {step1}")
    print(f"Step 2: {step2}")
    print(f"Games per run: {games}")
    print(f"Number of runs: {num_runs}")
    print("=" * 70)
    
    # Load checkpoints
    print("\nLoading checkpoints...")
    params1 = load_checkpoint_at_step(method, step1)
    params2 = load_checkpoint_at_step(method, step2)
    
    # Run multiple evaluation series
    print(f"\nRunning {num_runs} evaluation series...")
    step1_wins = []
    step2_wins = []
    avg_turns_list = []
    
    for run in range(num_runs):
        print(f"\nRun {run + 1}/{num_runs}:")
        # Run twice: once with step1 as Agent A, once as Agent B (to account for first-mover advantage)
        a_wins_1, b_wins_1, turns_1 = run_evaluation(params1, params2, games=games // 2, verbose=False)
        a_wins_2, b_wins_2, turns_2 = run_evaluation(params2, params1, games=games // 2, verbose=False)
        
        # Aggregate: step1 wins when it's Agent A and wins, or when it's Agent B and Agent A loses
        step1_total = a_wins_1 + b_wins_2
        step2_total = b_wins_1 + a_wins_2
        avg_turns = (turns_1 + turns_2) / 2.0
        
        step1_wins.append(step1_total)
        step2_wins.append(step2_total)
        avg_turns_list.append(avg_turns)
        
        print(f"  Step {step1} wins: {step1_total}, Step {step2} wins: {step2_total}, Avg turns: {avg_turns:.1f}")
    
    # Compute statistics
    step1_mean = np.mean(step1_wins)
    step2_mean = np.mean(step2_wins)
    step1_std = np.std(step1_wins)
    step2_std = np.std(step2_wins)
    
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Step {step1} average wins: {step1_mean:.1f} ± {step1_std:.1f} / {games}")
    print(f"Step {step2} average wins: {step2_mean:.1f} ± {step2_std:.1f} / {games}")
    print(f"Average game length: {np.mean(avg_turns_list):.2f} turns")
    
    win_rate1 = step1_mean / games
    win_rate2 = step2_mean / games
    
    print(f"\nStep {step1} win rate: {100.0 * win_rate1:.2f}%")
    print(f"Step {step2} win rate: {100.0 * win_rate2:.2f}%")
    
    if win_rate2 > win_rate1:
        improvement = (win_rate2 - win_rate1) / win_rate1 * 100 if win_rate1 > 0 else 0
        print(f"\n✓ Step {step2} is stronger than step {step1} by {improvement:.1f}%")
    elif win_rate2 < win_rate1:
        print(f"\n✗ Step {step2} is weaker than step {step1} (unusual - check training!)")
    else:
        print(f"\n→ Steps perform similarly")
    
    # Statistical test if scipy is available
    try:
        from scipy import stats
        t_stat, p_value = stats.ttest_rel(step1_wins, step2_wins)
        print(f"\nStatistical test (paired t-test):")
        print(f"  t-statistic: {t_stat:.4f}")
        print(f"  p-value: {p_value:.4f}")
        if p_value < 0.05:
            print(f"  → Difference is statistically significant (p < 0.05)")
        else:
            print(f"  → Difference is not statistically significant (p >= 0.05)")
    except ImportError:
        pass
    
    print("=" * 70)


def compare_against_random(method: str, step: int, games: int = 500) -> None:
    """Compare a checkpoint against random baseline."""
    print("=" * 70)
    print("EVALUATING CHECKPOINT AGAINST RANDOM BASELINE")
    print("=" * 70)
    print(f"Training method: {method}")
    print(f"Step: {step}")
    print(f"Games: {games}")
    print("=" * 70)
    
    print("\nLoading checkpoint...")
    params = load_checkpoint_at_step(method, step)
    
    print(f"\nRunning evaluation (checkpoint vs random)...")
    a_wins, b_wins, avg_turns = run_evaluation(params, None, games=games, verbose=False)
    
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Checkpoint wins: {a_wins} / {games} ({100.0 * a_wins / games:.2f}%)")
    print(f"Random wins: {b_wins} / {games} ({100.0 * b_wins / games:.2f}%)")
    print(f"Average game length: {avg_turns:.2f} turns")
    
    if a_wins > b_wins:
        print(f"\n✓ Checkpoint at step {step} is stronger than random baseline")
    elif a_wins < b_wins:
        print(f"\n✗ Checkpoint at step {step} is weaker than random baseline (this is unusual!)")
    else:
        print(f"\n→ Checkpoint performs similarly to random baseline")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Compare checkpoints at different training steps",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare step 5000 vs step 200000 from 1-ply training
  python compare_checkpoint_steps.py --method 1ply --step1 5000 --step2 200000 --games 1000
  
  # Compare step 5000 vs step 200000 from 2-ply training
  python compare_checkpoint_steps.py --method 2ply --step1 5000 --step2 200000 --games 1000
  
  # Compare a checkpoint against random baseline
  python compare_checkpoint_steps.py --method 1ply --step1 5000 --games 500 --vs_random
        """,
    )
    parser.add_argument("--method", type=str, choices=["1ply", "2ply"], required=True,
                        help="Training method: '1ply' or '2ply'")
    parser.add_argument("--step1", type=int, help="First checkpoint step number")
    parser.add_argument("--step2", type=int, help="Second checkpoint step number")
    parser.add_argument("--games", type=int, default=1000, help="Number of games per comparison (default: 1000)")
    parser.add_argument("--runs", type=int, default=3, help="Number of evaluation runs (default: 3)")
    parser.add_argument("--vs_random", action="store_true",
                        help="Compare step1 against random baseline instead of step2")
    
    args = parser.parse_args()
    
    if args.vs_random:
        if args.step1 is None:
            print("Error: --step1 is required when using --vs_random", file=sys.stderr)
            sys.exit(1)
        compare_against_random(args.method, args.step1, args.games)
    else:
        if args.step1 is None or args.step2 is None:
            print("Error: Both --step1 and --step2 are required for comparison", file=sys.stderr)
            sys.exit(1)
        compare_steps(args.method, args.step1, args.step2, args.games, args.runs)


if __name__ == "__main__":
    main()

