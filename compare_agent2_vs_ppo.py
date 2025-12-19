"""
Quick comparison script: Agent 2 (TD-λ value network) vs PPO (policy network)

Usage:
    python compare_agent2_vs_ppo.py \
        --agent2 checkpoints_agent2_vectorized/checkpoint_2000 \
        --ppo checkpoints_ppo/checkpoint_200 \
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
from agent2.agent2_value_net import Agent2ValueNet, BOARD_LENGTH, CONV_INPUT_CHANNELS, AUX_INPUT_SIZE
from agent2.agent2_encode import encode_agent2
from backgammon_ppo_net import BackgammonPPONet
from train_value_td0 import py_reward

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
    boards = np.empty((U, BOARD_LENGTH, CONV_INPUT_CHANNELS), dtype=np.float32)
    auxs = np.empty((U, AUX_INPUT_SIZE), dtype=np.float32)
    for i, s in enumerate(uniq_states):
        b, a = encode_agent2(s, player)
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

# PPO: Policy-based action selection
_ppo_net = BackgammonPPONet()

@jax.jit
def ppo_forward(params, board, aux):
    v, pol_logits = _ppo_net.apply(
        {"params": params},
        board_state=board,
        aux_features=aux,
    )
    return v.squeeze(-1), pol_logits

def ppo_choose_action(state, player, dice, params, max_moves=128):
    """PPO: Policy-based action selection"""
    moves, afterstates = _actions(state, np.int8(player), dice)
    moves = list(moves)
    afterstates = [np.asarray(s, dtype=np.int8) for s in afterstates]
    
    if len(moves) == 0:
        return None, state
    
    # Encode state
    from train_value_td0 import encode
    b, a = encode(state, player)
    
    # Get policy logits
    _, pol_logits_625 = ppo_forward(
        params,
        jnp.asarray(b[None, ...], dtype=jnp.float32),
        jnp.asarray(a[None, ...], dtype=jnp.float32),
    )
    pol_logits_625 = np.array(pol_logits_625[0], dtype=np.float32)
    
    # Score moves by sum of submove logits
    from ppo import move_to_submove_action_indices
    scores = []
    for mv in moves:
        mv_py = [(int(f), int(r)) for (f, r) in mv]
        subidx = move_to_submove_action_indices(mv_py, player)
        score = sum(float(pol_logits_625[idx]) for idx in subidx if idx >= 0)
        scores.append(score)
    
    scores = np.array(scores, dtype=np.float32)
    
    # Prune to top-K
    if len(scores) > max_moves:
        top = np.argpartition(-scores, max_moves)[:max_moves]
        top = top[np.argsort(-scores[top])]
        moves = [moves[i] for i in top]
        afterstates = [afterstates[i] for i in top]
        scores = scores[top]
    
    # Sample from softmax
    scores_shift = scores - scores.max()
    probs = np.exp(scores_shift)
    probs = probs / probs.sum()
    act = int(np.random.choice(len(probs), p=probs))
    
    return moves[act], afterstates[act]

def play_game(agent2_params, ppo_params):
    """Play one game between Agent 2 and PPO
    
    Args:
        agent2_params: Agent 2 parameters (plays as +1/White)
        ppo_params: PPO parameters (plays as -1/Black)
    """
    player, dice, state = _new_game()
    turns = 0
    
    while True:
        turns += 1
        dice = _roll_dice()
        
        if player == 1:  # Agent 2's turn (White)
            if agent2_params is None:
                # Random
                moves, afterstates = _actions(state, np.int8(player), dice)
                moves = list(moves)
                afterstates = [np.asarray(s, dtype=np.int8) for s in afterstates]
                if len(afterstates) == 0:
                    next_state = state
                else:
                    next_state = afterstates[np.random.randint(len(afterstates))]
            else:
                move, next_state = agent2_choose_action_2ply(state, player, dice, agent2_params)
                if move is None:
                    next_state = state
        else:  # PPO's turn (Black)
            if ppo_params is None:
                # Random
                moves, afterstates = _actions(state, np.int8(player), dice)
                moves = list(moves)
                afterstates = [np.asarray(s, dtype=np.int8) for s in afterstates]
                if len(afterstates) == 0:
                    next_state = state
                else:
                    next_state = afterstates[np.random.randint(len(afterstates))]
            else:
                move, next_state = ppo_choose_action(state, player, dice, ppo_params)
                if move is None:
                    next_state = state
        
        r = py_reward(next_state, player)
        if r != 0:
            # Game over
            if player == 1:
                return 1, abs(r), turns  # Agent 2 wins
            else:
                return -1, abs(r), turns  # PPO wins
        
        state = next_state
        player = -player

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent2", type=str, required=True, help="Agent 2 checkpoint path")
    ap.add_argument("--ppo", type=str, required=True, help="PPO checkpoint path")
    ap.add_argument("--games", type=int, default=50, help="Number of games (default: 50)")
    args = ap.parse_args()
    
    print("=" * 70)
    print("AGENT 2 (TD-λ) vs PPO COMPARISON")
    print("=" * 70)
    print(f"Agent 2 checkpoint: {args.agent2}")
    print(f"PPO checkpoint: {args.ppo}")
    print(f"Games: {args.games}")
    print("=" * 70)
    
    # Load Agent 2
    print("\nLoading Agent 2 checkpoint...")
    key = jax.random.PRNGKey(0)
    dummy_board = jnp.zeros((1, BOARD_LENGTH, CONV_INPUT_CHANNELS), dtype=jnp.float32)
    dummy_aux = jnp.zeros((1, AUX_INPUT_SIZE), dtype=jnp.float32)
    agent2_params_init = Agent2ValueNet().init(key, board_state=dummy_board, aux_features=dummy_aux)["params"]
    agent2_params = checkpoints.restore_checkpoint(args.agent2, target=agent2_params_init)
    print("✓ Agent 2 loaded")
    
    # Load PPO
    print("Loading PPO checkpoint...")
    dummy_board_ppo = jnp.zeros((1, BOARD_LENGTH, CONV_INPUT_CHANNELS), dtype=jnp.float32)
    dummy_aux_ppo = jnp.zeros((1, AUX_INPUT_SIZE), dtype=jnp.float32)
    ppo_params_init = BackgammonPPONet().init(key, board_state=dummy_board_ppo, aux_features=dummy_aux_ppo)["params"]
    ppo_params = checkpoints.restore_checkpoint(args.ppo, target=ppo_params_init)
    print("✓ PPO loaded")
    
    # Play games
    print(f"\nPlaying {args.games} games...")
    agent2_wins = 0
    ppo_wins = 0
    agent2_gammons = 0
    ppo_gammons = 0
    agent2_backgammons = 0
    ppo_backgammons = 0
    total_turns = 0
    
    for i in range(args.games):
        # Alternate who goes first (swap colors)
        if i % 2 == 0:
            # Agent 2 as White (+1), PPO as Black (-1)
            winner, score, turns = play_game(agent2_params, ppo_params)
        else:
            # PPO as White (+1), Agent 2 as Black (-1) - swap by passing params in reverse
            # But we need to swap the logic - let's just swap the params and flip the result
            temp_winner, score, turns = play_game(ppo_params, agent2_params)
            winner = -temp_winner  # Flip winner since we swapped colors
        
        total_turns += turns
        
        if winner == 1:  # Agent 2 wins
            agent2_wins += 1
            if score == 2:
                agent2_gammons += 1
            elif score == 3:
                agent2_backgammons += 1
        else:  # PPO wins
            ppo_wins += 1
            if score == 2:
                ppo_gammons += 1
            elif score == 3:
                ppo_backgammons += 1
        
        if (i + 1) % 10 == 0:
            print(f"  Game {i+1}/{args.games}: Agent2={agent2_wins}, PPO={ppo_wins}")
    
    # Results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Agent 2 (TD-λ) wins: {agent2_wins} / {args.games} ({100.0 * agent2_wins / args.games:.1f}%)")
    print(f"PPO wins: {ppo_wins} / {args.games} ({100.0 * ppo_wins / args.games:.1f}%)")
    print(f"Average game length: {total_turns / args.games:.1f} turns")
    print(f"\nGammons: Agent2={agent2_gammons}, PPO={ppo_gammons}")
    print(f"Backgammons: Agent2={agent2_backgammons}, PPO={ppo_backgammons}")
    
    if agent2_wins > ppo_wins:
        print(f"\n✓ Agent 2 (TD-λ) is stronger")
    elif ppo_wins > agent2_wins:
        print(f"\n✓ PPO is stronger")
    else:
        print(f"\n→ Agents are evenly matched")
    print("=" * 70)

if __name__ == "__main__":
    main()

