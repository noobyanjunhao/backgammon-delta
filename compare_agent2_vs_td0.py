"""
Quick comparison script: Agent 2 (TD-λ value network) vs TD(0) value network

Usage:
    python compare_agent2_vs_td0.py \
        --agent2 checkpoints_agent2_vectorized/checkpoint_2000 \
        --td0 checkpoints_td0/checkpoint_5000 \
        --games 50
"""

import os
import sys
import argparse
import numpy as np
import jax
import jax.numpy as jnp
from flax.training import checkpoints

from backgammon_engine import _new_game, _actions, _roll_dice
from agent2.agent2_value_net import Agent2ValueNet, BOARD_LENGTH as AGENT2_BOARD_LENGTH, CONV_INPUT_CHANNELS as AGENT2_CONV_INPUT_CHANNELS, AUX_INPUT_SIZE as AGENT2_AUX_INPUT_SIZE
from agent2.agent2_encode import encode_agent2
from backgammon_value_net import BackgammonValueNet, BOARD_LENGTH, CONV_INPUT_CHANNELS, AUX_INPUT_SIZE
from train_value_td0 import encode, py_reward

# Dice outcomes for 2-ply
def dice21():
    outs = []
    probs = []
    for i in range(1, 7):
        for j in range(i, 7):
            outs.append((j, i))
            probs.append((1.0 / 36.0) if i == j else (2.0 / 36.0))
    return np.array(outs, dtype=np.int8), np.array(probs, dtype=np.float32)

DICE21, PROBS21 = dice21()

# Agent 2: 2-ply value-based action selection
_agent2_net = Agent2ValueNet()

@jax.jit
def agent2_value_apply(params, board, aux):
    v = _agent2_net.apply({"params": params}, board_state=board, aux_features=aux)
    return v.squeeze(-1)

def agent2_choose_action_2ply(state, player, dice, params, eval_batch=4096):
    """Agent 2: 2-ply search with value network"""
    moves_a, after_a = _actions(state, np.int8(player), dice)
    moves_a = list(moves_a)
    after_a = [np.asarray(s, dtype=np.int8) for s in after_a]
    nA = len(after_a)
    if nA == 0:
        return None, state
    
    opp_player = -player
    uniq_index = {}
    uniq_states = []
    
    # Pass 1: collect unique afterstates
    for a_idx, s_a in enumerate(after_a):
        for j_idx, d in enumerate(DICE21):
            moves_o, after_o = _actions(s_a, np.int8(opp_player), d)
            after_o = list(after_o)
            if len(after_o) == 0:
                k = memoryview(np.asarray(s_a, dtype=np.int8)).tobytes()
                if k not in uniq_index:
                    uniq_index[k] = len(uniq_states)
                    uniq_states.append(np.asarray(s_a, dtype=np.int8))
            else:
                for s in after_o:
                    s = np.asarray(s, dtype=np.int8)
                    k = memoryview(s).tobytes()
                    if k not in uniq_index:
                        uniq_index[k] = len(uniq_states)
                        uniq_states.append(s)
    
    # Evaluate unique states
    U = len(uniq_states)
    boards = np.empty((U, AGENT2_BOARD_LENGTH, AGENT2_CONV_INPUT_CHANNELS), dtype=np.float32)
    auxs = np.empty((U, AGENT2_AUX_INPUT_SIZE), dtype=np.float32)
    for i, s in enumerate(uniq_states):
        b, a = encode_agent2(s, player)  # IMPORTANT: V is from current player's perspective
        boards[i] = b
        auxs[i] = a
    
    # Batched evaluation
    vals_u = []
    for i in range(0, U, eval_batch):
        b_batch = boards[i:i+eval_batch]
        a_batch = auxs[i:i+eval_batch]
        if b_batch.shape[0] < eval_batch:
            pad = eval_batch - b_batch.shape[0]
            b_batch = np.pad(b_batch, ((0, pad), (0, 0), (0, 0)), mode="constant")
            a_batch = np.pad(a_batch, ((0, pad), (0, 0)), mode="constant")
        v = agent2_value_apply(
            params,
            jnp.asarray(b_batch, dtype=jnp.float32),
            jnp.asarray(a_batch, dtype=jnp.float32),
        )
        v = np.array(v)[:min(eval_batch, U - i)]
        vals_u.extend(v)
    vals_u = np.array(vals_u)
    
    # Pass 2: compute expected values
    min_per_aj = np.full((nA, 21), np.inf, dtype=np.float32)
    for a_idx, s_a in enumerate(after_a):
        for j_idx, d in enumerate(DICE21):
            moves_o, after_o = _actions(s_a, np.int8(opp_player), d)
            after_o = list(after_o)
            if len(after_o) == 0:
                k = memoryview(np.asarray(s_a, dtype=np.int8)).tobytes()
                u = uniq_index[k]
                vmin = vals_u[u]
            else:
                vmin = np.inf
                for s in after_o:
                    s = np.asarray(s, dtype=np.int8)
                    k = memoryview(s).tobytes()
                    u = uniq_index[k]
                    v = vals_u[u]
                    if v < vmin:
                        vmin = v
            min_per_aj[a_idx, j_idx] = vmin
    
    expected = (min_per_aj * PROBS21[None, :]).sum(axis=1)
    best = int(np.argmax(expected))
    return moves_a[best], after_a[best]

# TD(0): 1-ply value-based action selection
_td0_net = BackgammonValueNet()

@jax.jit
def td0_value_apply(params, board, aux):
    v = _td0_net.apply({"params": params}, board_state=board, aux_features=aux)
    return v.squeeze(-1)

def td0_choose_action_1ply(state, player, dice, params, eval_batch=4096):
    """TD(0): 1-ply greedy action selection"""
    moves, afterstates = _actions(state, np.int8(player), dice)
    moves = list(moves)
    afterstates = [np.asarray(s, dtype=np.int8) for s in afterstates]
    
    if len(moves) == 0:
        return None, state
    
    # Evaluate all afterstates from next player's perspective (zero-sum)
    boards = []
    auxs = []
    for ns in afterstates:
        b, a = encode(ns, -player)  # Value net predicts for side-to-move
        boards.append(b)
        auxs.append(a)
    
    boards = np.asarray(boards, dtype=np.float32)
    auxs = np.asarray(auxs, dtype=np.float32)
    
    # Batched evaluation
    vals = []
    n = len(boards)
    for i in range(0, n, eval_batch):
        b_batch = boards[i:i+eval_batch]
        a_batch = auxs[i:i+eval_batch]
        if b_batch.shape[0] < eval_batch:
            pad = eval_batch - b_batch.shape[0]
            b_batch = np.pad(b_batch, ((0, pad), (0, 0), (0, 0)), mode="constant")
            a_batch = np.pad(a_batch, ((0, pad), (0, 0)), mode="constant")
        v = td0_value_apply(
            params,
            jnp.asarray(b_batch, dtype=jnp.float32),
            jnp.asarray(a_batch, dtype=jnp.float32),
        )
        v = np.array(v)[:min(eval_batch, n - i)]
        vals.extend(v)
    vals = np.array(vals)
    
    # Choose move that minimizes opponent's value (zero-sum)
    best = int(np.argmin(vals))
    return moves[best], afterstates[best]

def play_game(white_params, black_params, white_is_agent2=True):
    """Play one game between two agents
    
    Args:
        white_params: Parameters for White player (+1)
        black_params: Parameters for Black player (-1)
        white_is_agent2: True if White is Agent 2, False if White is TD(0)
    """
    player, dice, state = _new_game()
    turns = 0
    
    while True:
        turns += 1
        dice = _roll_dice()
        
        if player == 1:  # White's turn
            if white_params is None:
                # Random
                moves, afterstates = _actions(state, np.int8(player), dice)
                moves = list(moves)
                afterstates = [np.asarray(s, dtype=np.int8) for s in afterstates]
                if len(afterstates) == 0:
                    next_state = state
                else:
                    next_state = afterstates[np.random.randint(len(afterstates))]
            else:
                if white_is_agent2:
                    move, next_state = agent2_choose_action_2ply(state, player, dice, white_params)
                else:
                    move, next_state = td0_choose_action_1ply(state, player, dice, white_params)
                if move is None:
                    next_state = state
        else:  # Black's turn
            if black_params is None:
                # Random
                moves, afterstates = _actions(state, np.int8(player), dice)
                moves = list(moves)
                afterstates = [np.asarray(s, dtype=np.int8) for s in afterstates]
                if len(afterstates) == 0:
                    next_state = state
                else:
                    next_state = afterstates[np.random.randint(len(afterstates))]
            else:
                if white_is_agent2:
                    # Black is TD(0)
                    move, next_state = td0_choose_action_1ply(state, player, dice, black_params)
                else:
                    # Black is Agent 2
                    move, next_state = agent2_choose_action_2ply(state, player, dice, black_params)
                if move is None:
                    next_state = state
        
        r = py_reward(next_state, player)
        if r != 0:
            # Game over
            if player == 1:
                return 1, abs(r), turns  # White wins
            else:
                return -1, abs(r), turns  # Black wins
        
        state = next_state
        player = -player

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent2", type=str, required=True, help="Agent 2 checkpoint path")
    ap.add_argument("--td0", type=str, required=True, help="TD(0) checkpoint path")
    ap.add_argument("--games", type=int, default=50, help="Number of games (default: 50)")
    args = ap.parse_args()
    
    print("=" * 70)
    print("AGENT 2 (TD-λ) vs TD(0) COMPARISON")
    print("=" * 70)
    print(f"Agent 2 checkpoint: {args.agent2}")
    print(f"TD(0) checkpoint: {args.td0}")
    print(f"Games: {args.games}")
    print("=" * 70)
    
    # Convert to absolute paths
    agent2_ckpt = os.path.abspath(args.agent2)
    td0_ckpt = os.path.abspath(args.td0)
    
    # Load Agent 2
    print("\nLoading Agent 2 checkpoint...")
    key = jax.random.PRNGKey(0)
    dummy_board_agent2 = jnp.zeros((1, AGENT2_BOARD_LENGTH, AGENT2_CONV_INPUT_CHANNELS), dtype=jnp.float32)
    dummy_aux_agent2 = jnp.zeros((1, AGENT2_AUX_INPUT_SIZE), dtype=jnp.float32)
    agent2_params_init = Agent2ValueNet().init(key, board_state=dummy_board_agent2, aux_features=dummy_aux_agent2)["params"]
    agent2_params = checkpoints.restore_checkpoint(agent2_ckpt, target=agent2_params_init)
    print("✓ Agent 2 loaded")
    
    # Load TD(0)
    print("Loading TD(0) checkpoint...")
    dummy_board_td0 = jnp.zeros((1, BOARD_LENGTH, CONV_INPUT_CHANNELS), dtype=jnp.float32)
    dummy_aux_td0 = jnp.zeros((1, AUX_INPUT_SIZE), dtype=jnp.float32)
    td0_params_init = BackgammonValueNet().init(key, board_state=dummy_board_td0, aux_features=dummy_aux_td0)["params"]
    td0_params = checkpoints.restore_checkpoint(td0_ckpt, target=td0_params_init)
    print("✓ TD(0) loaded")
    
    # Play games
    print(f"\nPlaying {args.games} games...")
    agent2_wins = 0
    td0_wins = 0
    agent2_gammons = 0
    td0_gammons = 0
    agent2_backgammons = 0
    td0_backgammons = 0
    total_turns = 0
    
    for i in range(args.games):
        # Alternate who goes first (swap colors)
        if i % 2 == 0:
            # Agent 2 as White (+1), TD(0) as Black (-1)
            winner, score, turns = play_game(agent2_params, td0_params, white_is_agent2=True)
            # winner: +1 = White (Agent 2) wins, -1 = Black (TD(0)) wins
        else:
            # TD(0) as White (+1), Agent 2 as Black (-1)
            temp_winner, score, turns = play_game(td0_params, agent2_params, white_is_agent2=False)
            # Flip winner: +1 (White/TD(0) wins) -> -1 (TD(0) wins), -1 (Black/Agent2 wins) -> +1 (Agent2 wins)
            winner = -temp_winner
        
        total_turns += turns
        
        if winner == 1:  # Agent 2 wins
            agent2_wins += 1
            if score == 2:
                agent2_gammons += 1
            elif score == 3:
                agent2_backgammons += 1
        else:  # TD(0) wins
            td0_wins += 1
            if score == 2:
                td0_gammons += 1
            elif score == 3:
                td0_backgammons += 1
        
        if (i + 1) % 10 == 0:
            print(f"  Game {i+1}/{args.games}: Agent2={agent2_wins}, TD(0)={td0_wins}")
    
    # Results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Agent 2 (TD-λ) wins: {agent2_wins} / {args.games} ({100.0 * agent2_wins / args.games:.1f}%)")
    print(f"TD(0) wins: {td0_wins} / {args.games} ({100.0 * td0_wins / args.games:.1f}%)")
    print(f"Average game length: {total_turns / args.games:.1f} turns")
    print(f"\nGammons: Agent2={agent2_gammons}, TD(0)={td0_gammons}")
    print(f"Backgammons: Agent2={agent2_backgammons}, TD(0)={td0_backgammons}")
    
    if agent2_wins > td0_wins:
        print(f"\n✓ Agent 2 (TD-λ) is stronger")
    elif td0_wins > agent2_wins:
        print(f"\n✓ TD(0) is stronger")
    else:
        print(f"\n→ Agents are evenly matched")
    print("=" * 70)

if __name__ == "__main__":
    main()

