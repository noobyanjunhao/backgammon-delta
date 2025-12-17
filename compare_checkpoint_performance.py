"""
Compare the playing performance of different checkpoint models.

This script evaluates checkpoints by having them play games against each other
or against baselines (random play). It reports win rates, average game length,
and statistical significance of differences.

Usage:
    # Compare two checkpoints head-to-head
    python compare_checkpoint_performance.py --checkpoint1 checkpoints --checkpoint2 checkpoints_2ply --games 1000

    # Compare a checkpoint against random baseline
    python compare_checkpoint_performance.py --checkpoint1 checkpoints --games 500

    # Compare multiple checkpoints in a tournament
    python compare_checkpoint_performance.py --checkpoints checkpoints checkpoints_2ply checkpoint_old --games 200
"""

import argparse
import os
import sys
from typing import List, Optional, Tuple

import numpy as np
try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

from evaluate_value_models import load_params, evaluate


def run_evaluation(
    agent_a_params: Optional[dict],
    agent_b_params: Optional[dict],
    games: int = 100,
    verbose: bool = False,
) -> Tuple[int, int, float]:
    """Run evaluation and return (a_wins, b_wins, avg_turns)."""
    # Capture the output from evaluate() function
    import io
    from contextlib import redirect_stdout
    
    f = io.StringIO()
    with redirect_stdout(f):
        evaluate(agent_a_params, agent_b_params, games=games, verbose=verbose)
    
    output = f.getvalue()
    
    # Parse the output to extract wins and average turns
    a_wins = 0
    b_wins = 0
    avg_turns = 0.0
    
    for line in output.split('\n'):
        if 'Agent A wins:' in line:
            parts = line.split()
            a_wins = int(parts[2])
        elif 'Agent B wins:' in line:
            parts = line.split()
            b_wins = int(parts[2])
        elif 'Average game length:' in line:
            parts = line.split()
            avg_turns = float(parts[3])
    
    return a_wins, b_wins, avg_turns


def compare_two_checkpoints(
    ckpt1: str,
    ckpt2: str,
    games: int = 1000,
    num_runs: int = 5,
    verbose: bool = False,
) -> None:
    """Compare two checkpoints by having them play multiple match series."""
    print("=" * 70)
    print(f"COMPARING CHECKPOINT PERFORMANCE")
    print("=" * 70)
    print(f"Checkpoint 1: {ckpt1}")
    print(f"Checkpoint 2: {ckpt2}")
    print(f"Games per run: {games}")
    print(f"Number of runs: {num_runs}")
    print("=" * 70)
    
    # Load checkpoints
    print("\nLoading checkpoints...")
    try:
        params1 = load_params(ckpt1)
        print(f"✓ Loaded {ckpt1}")
    except Exception as e:
        print(f"✗ Error loading {ckpt1}: {e}", file=sys.stderr)
        sys.exit(1)
    
    try:
        params2 = load_params(ckpt2)
        print(f"✓ Loaded {ckpt2}")
    except Exception as e:
        print(f"✗ Error loading {ckpt2}: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Run multiple evaluation series
    print(f"\nRunning {num_runs} evaluation series...")
    ckpt1_wins = []
    ckpt2_wins = []
    avg_turns_list = []
    
    for run in range(num_runs):
        print(f"\nRun {run + 1}/{num_runs}:")
        # Run twice: once with ckpt1 as Agent A, once as Agent B (to account for first-mover advantage)
        a_wins_1, b_wins_1, turns_1 = run_evaluation(params1, params2, games=games // 2, verbose=False)
        a_wins_2, b_wins_2, turns_2 = run_evaluation(params2, params1, games=games // 2, verbose=False)
        
        # Aggregate: ckpt1 wins when it's Agent A and wins, or when it's Agent B and Agent A loses
        ckpt1_total = a_wins_1 + b_wins_2
        ckpt2_total = b_wins_1 + a_wins_2
        avg_turns = (turns_1 + turns_2) / 2.0
        
        ckpt1_wins.append(ckpt1_total)
        ckpt2_wins.append(ckpt2_total)
        avg_turns_list.append(avg_turns)
        
        print(f"  Ckpt1 wins: {ckpt1_total}, Ckpt2 wins: {ckpt2_total}, Avg turns: {avg_turns:.1f}")
    
    # Compute statistics
    ckpt1_mean = np.mean(ckpt1_wins)
    ckpt2_mean = np.mean(ckpt2_wins)
    ckpt1_std = np.std(ckpt1_wins)
    ckpt2_std = np.std(ckpt2_wins)
    avg_turns_mean = np.mean(avg_turns_list)
    
    # Print summary
    print("\n" + "=" * 70)
    print("PERFORMANCE COMPARISON SUMMARY")
    print("=" * 70)
    print(f"Checkpoint 1 ({ckpt1}):")
    print(f"  Average wins: {ckpt1_mean:.1f} ± {ckpt1_std:.1f} (out of {games} games)")
    print(f"  Win rate: {100.0 * ckpt1_mean / games:.2f}%")
    print(f"\nCheckpoint 2 ({ckpt2}):")
    print(f"  Average wins: {ckpt2_mean:.1f} ± {ckpt2_std:.1f} (out of {games} games)")
    print(f"  Win rate: {100.0 * ckpt2_mean / games:.2f}%")
    print(f"\nAverage game length: {avg_turns_mean:.2f} turns")
    
    # Statistical test (paired t-test on win differences)
    if HAS_SCIPY and num_runs > 1:
        win_diffs = [w1 - w2 for w1, w2 in zip(ckpt1_wins, ckpt2_wins)]
        t_stat, p_value = stats.ttest_1samp(win_diffs, 0.0)
        print(f"\nStatistical test (paired t-test):")
        print(f"  t-statistic: {t_stat:.3f}")
        print(f"  p-value: {p_value:.4f}")
        
        if p_value < 0.05:
            if ckpt1_mean > ckpt2_mean:
                print(f"  → Checkpoint 1 is significantly stronger (p < 0.05)")
            else:
                print(f"  → Checkpoint 2 is significantly stronger (p < 0.05)")
        else:
            print(f"  → No significant difference (p >= 0.05)")
    else:
        # Simple comparison without statistical test
        diff = ckpt1_mean - ckpt2_mean
        if abs(diff) > ckpt1_std + ckpt2_std:
            if diff > 0:
                print(f"\n→ Checkpoint 1 appears stronger (difference: {diff:.1f} wins)")
            else:
                print(f"\n→ Checkpoint 2 appears stronger (difference: {abs(diff):.1f} wins)")
        else:
            print(f"\n→ Checkpoints appear similar in strength")
    
    print("=" * 70)


def compare_against_random(ckpt: str, games: int = 500, verbose: bool = False) -> None:
    """Compare a checkpoint against random baseline."""
    print("=" * 70)
    print(f"EVALUATING CHECKPOINT AGAINST RANDOM BASELINE")
    print("=" * 70)
    print(f"Checkpoint: {ckpt}")
    print(f"Games: {games}")
    print("=" * 70)
    
    print("\nLoading checkpoint...")
    try:
        params = load_params(ckpt)
        print(f"✓ Loaded {ckpt}")
    except Exception as e:
        print(f"✗ Error loading {ckpt}: {e}", file=sys.stderr)
        sys.exit(1)
    
    print(f"\nRunning evaluation (checkpoint vs random)...")
    a_wins, b_wins, avg_turns = run_evaluation(params, None, games=games, verbose=verbose)
    
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Checkpoint wins: {a_wins} / {games} ({100.0 * a_wins / games:.2f}%)")
    print(f"Random wins: {b_wins} / {games} ({100.0 * b_wins / games:.2f}%)")
    print(f"Average game length: {avg_turns:.2f} turns")
    
    if a_wins > b_wins:
        print(f"\n✓ Checkpoint is stronger than random baseline")
    elif a_wins < b_wins:
        print(f"\n✗ Checkpoint is weaker than random baseline (this is unusual!)")
    else:
        print(f"\n→ Checkpoint performs similarly to random baseline")
    print("=" * 70)


def tournament_mode(checkpoints: List[str], games: int = 200) -> None:
    """Run a round-robin tournament between multiple checkpoints."""
    print("=" * 70)
    print("TOURNAMENT MODE")
    print("=" * 70)
    print(f"Checkpoints: {len(checkpoints)}")
    for i, ckpt in enumerate(checkpoints, 1):
        print(f"  {i}. {ckpt}")
    print(f"Games per match: {games}")
    print("=" * 70)
    
    # Load all checkpoints
    print("\nLoading checkpoints...")
    params_list = []
    for ckpt in checkpoints:
        try:
            params = load_params(ckpt)
            params_list.append(params)
            print(f"✓ Loaded {ckpt}")
        except Exception as e:
            print(f"✗ Error loading {ckpt}: {e}", file=sys.stderr)
            sys.exit(1)
    
    # Round-robin tournament
    n = len(checkpoints)
    results = np.zeros((n, n), dtype=int)  # results[i][j] = wins for checkpoint i against checkpoint j
    
    print(f"\nRunning round-robin tournament ({n * (n - 1) // 2} matches)...")
    for i in range(n):
        for j in range(i + 1, n):
            print(f"\nMatch: {checkpoints[i]} vs {checkpoints[j]}")
            a_wins, b_wins, _ = run_evaluation(params_list[i], params_list[j], games=games, verbose=False)
            results[i, j] = a_wins
            results[j, i] = b_wins
            print(f"  {checkpoints[i]}: {a_wins} wins")
            print(f"  {checkpoints[j]}: {b_wins} wins")
    
    # Print tournament standings
    print("\n" + "=" * 70)
    print("TOURNAMENT STANDINGS")
    print("=" * 70)
    total_wins = results.sum(axis=1)
    win_rates = total_wins / (games * (n - 1))
    
    # Sort by total wins
    standings = sorted(enumerate(zip(checkpoints, total_wins, win_rates)), key=lambda x: x[1][1], reverse=True)
    
    print(f"{'Rank':<6} {'Checkpoint':<40} {'Wins':<8} {'Win Rate':<10}")
    print("-" * 70)
    for rank, (idx, (ckpt, wins, rate)) in enumerate(standings, 1):
        ckpt_name = os.path.basename(ckpt) if len(ckpt) > 40 else ckpt
        print(f"{rank:<6} {ckpt_name:<40} {wins:<8} {100.0 * rate:.2f}%")
    
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Compare checkpoint performance by playing games",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare two checkpoints
  python compare_checkpoint_performance.py --checkpoint1 checkpoints --checkpoint2 checkpoints_2ply --games 1000

  # Evaluate against random baseline
  python compare_checkpoint_performance.py --checkpoint1 checkpoints --games 500

  # Tournament mode (multiple checkpoints)
  python compare_checkpoint_performance.py --checkpoints ckpt1 ckpt2 ckpt3 --games 200
        """,
    )
    parser.add_argument("--checkpoint1", type=str, help="First checkpoint directory")
    parser.add_argument("--checkpoint2", type=str, default=None, help="Second checkpoint directory (optional)")
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        help="Multiple checkpoints for tournament mode (overrides --checkpoint1/--checkpoint2)",
    )
    parser.add_argument("--games", type=int, default=1000, help="Number of games per comparison (default: 1000)")
    parser.add_argument(
        "--runs", type=int, default=5, help="Number of evaluation runs for two-checkpoint comparison (default: 5)"
    )
    parser.add_argument("--verbose", action="store_true", help="Print per-game results")
    
    args = parser.parse_args()
    
    if args.checkpoints:
        # Tournament mode
        if len(args.checkpoints) < 2:
            print("Error: Tournament mode requires at least 2 checkpoints", file=sys.stderr)
            sys.exit(1)
        tournament_mode(args.checkpoints, games=args.games)
    elif args.checkpoint1 and args.checkpoint2:
        # Compare two checkpoints
        compare_two_checkpoints(args.checkpoint1, args.checkpoint2, games=args.games, num_runs=args.runs, verbose=args.verbose)
    elif args.checkpoint1:
        # Compare against random
        compare_against_random(args.checkpoint1, games=args.games, verbose=args.verbose)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()

