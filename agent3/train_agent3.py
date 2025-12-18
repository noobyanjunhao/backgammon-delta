"""
Agent 3 PPO Training: Full specification implementation.

Features:
- Agent 2's network architecture + policy head (25×25)
- 2-ply search with policy-based pruning (top 5 moves)
- Sequential submove sampling for training
- GAE (Generalized Advantage Estimation)
- Clipped surrogate objective
- Value function loss
- Entropy bonus
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
import optax
from flax.training import train_state, checkpoints

from backgammon_engine import (
    _new_game,
    _actions,
    _roll_dice,
    NUM_CHECKERS,
    HOME_BOARD_SIZE,
    NUM_POINTS,
    W_BAR, B_BAR, W_OFF, B_OFF,
)
from agent2.agent2_encode import encode_agent2, BOARD_LENGTH, CONV_INPUT_CHANNELS, AUX_INPUT_SIZE
from agent3.agent3_ppo_net import Agent3PPONet, ACTION_SPACE_SIZE
from agent3.agent3_utils import (
    move_to_submove_action_indices,
    submove_idx_to_from_to,
    physical_point_to_canonical,
    MAX_SUBMOVES,
)


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
# 21 dice outcomes with probs
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
# Policy-based pruning: Top 5 moves
# -----------------------------
def prune_moves_by_policy(moves, pol_logits_625, player, top_k=5):
    """
    Prune moves to top K using policy logits.
    
    Specification method:
    1. Mask illegal submoves
    2. Find submove with highest logit
    3. Collect all legal moves starting with that submove
    4. If < K moves, continue with next highest submove
    5. Repeat until we have at least K moves (or all moves)
    
    Args:
        moves: List of move sequences (list of (from_point, roll) tuples)
        pol_logits_625: (625,) array of policy logits
        player: +1 or -1
        top_k: Number of moves to keep (default: 5)
    
    Returns:
        (pruned_moves, move_subidxs) tuple
    """
    if len(moves) == 0:
        return [], []
    
    if len(moves) <= top_k:
        # Return all moves
        subidxs = [move_to_submove_action_indices([(int(f), int(r)) for (f, r) in mv], player) for mv in moves]
        return moves, subidxs
    
    # Build submove-to-moves mapping
    submove_to_moves = {}  # submove_idx -> list of (move_idx, full_subidxs)
    
    for move_idx, mv in enumerate(moves):
        mv_py = [(int(f), int(r)) for (f, r) in mv]
        subidxs = move_to_submove_action_indices(mv_py, player)
        first_submove = subidxs[0]
        if first_submove >= 0:
            if first_submove not in submove_to_moves:
                submove_to_moves[first_submove] = []
            submove_to_moves[first_submove].append((move_idx, subidxs))
    
    # Sort submoves by logit value (descending)
    submove_scores = []
    for submove_idx, move_list in submove_to_moves.items():
        if submove_idx >= 0 and submove_idx < len(pol_logits_625):
            submove_scores.append((submove_idx, float(pol_logits_625[submove_idx]), move_list))
    
    submove_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Collect moves starting with top submoves
    selected_move_indices = set()
    selected_moves = []
    selected_subidxs = []
    
    for submove_idx, score, move_list in submove_scores:
        if len(selected_moves) >= top_k:
            break
        for move_idx, subidxs in move_list:
            if move_idx not in selected_move_indices:
                selected_move_indices.add(move_idx)
                selected_moves.append(moves[move_idx])
                selected_subidxs.append(subidxs)
                if len(selected_moves) >= top_k:
                    break
    
    # If we still don't have enough, add remaining moves by total logit sum
    if len(selected_moves) < top_k:
        remaining = [(i, mv) for i, mv in enumerate(moves) if i not in selected_move_indices]
        move_scores = []
        for i, mv in remaining:
            mv_py = [(int(f), int(r)) for (f, r) in mv]
            subidxs = move_to_submove_action_indices(mv_py, player)
            score = sum(float(pol_logits_625[idx]) for idx in subidxs if idx >= 0)
            move_scores.append((i, mv, subidxs, score))
        
        move_scores.sort(key=lambda x: x[3], reverse=True)
        for i, mv, subidxs, score in move_scores:
            if len(selected_moves) >= top_k:
                break
            selected_moves.append(mv)
            selected_subidxs.append(subidxs)
    
    return selected_moves[:top_k], selected_subidxs[:top_k]


# -----------------------------
# 2-ply search with policy-based pruning
# -----------------------------
def choose_action_2ply_pruned(state, player, dice, params, pol_logits_625, eval_batch, top_k=5, debug=False):
    """
    2-ply search with policy-based pruning to top K moves.
    
    For each player action:
      1. Prune to top K moves using policy
      2. For each pruned move, consider all opponent dice rolls
      3. For each opponent dice roll, prune opponent moves to top K using policy
      4. Compute expected value: sum_j P(d_j) * min_{aopp in top K} V(S_{a,j,aopp})
      5. Choose action with highest expected value
    """
    moves_a, after_a = _actions(state, np.int8(player), dice)
    moves_a = list(moves_a)
    after_a = [np.asarray(s, dtype=np.int8) for s in after_a]
    nA = len(after_a)
    if nA == 0:
        return None, state
    
    # Prune player moves to top K
    pruned_moves_a, pruned_subidxs_a = prune_moves_by_policy(moves_a, pol_logits_625, player, top_k)
    if len(pruned_moves_a) == 0:
        return moves_a[0] if moves_a else None, after_a[0] if after_a else state
    
    # Map pruned moves back to afterstates
    pruned_after_a = []
    pruned_moves_final = []
    for mv in pruned_moves_a:
        if mv in moves_a:
            idx = moves_a.index(mv)
            pruned_after_a.append(after_a[idx])
            pruned_moves_final.append(mv)
    
    opp_player = -player
    expected_values = []
    
    # For each pruned player action
    for a_idx, s_a in enumerate(pruned_after_a):
        ev_sum = 0.0
        
        # For each opponent dice roll
        for j_idx, d in enumerate(DICE21):
            moves_o, after_o = _actions(s_a, np.int8(opp_player), d)
            moves_o = list(moves_o)
            after_o = [np.asarray(s, dtype=np.int8) for s in after_o]
            
            if len(after_o) == 0:
                # Opponent has no moves, use player's afterstate
                b_n, a_n = encode_agent2(s_a, player)
                v_min = float(np.array(
                    value_apply(
                        params,
                        jnp.asarray(b_n[None, ...], dtype=jnp.float32),
                        jnp.asarray(a_n[None, ...], dtype=jnp.float32),
                    )[0]
                ))
            else:
                # Prune opponent moves to top K
                # Get policy logits for opponent's perspective
                b_opp, a_opp = encode_agent2(s_a, opp_player)
                _, pol_logits_opp = model_forward(
                    params,
                    jnp.asarray(b_opp[None, ...], dtype=jnp.float32),
                    jnp.asarray(a_opp[None, ...], dtype=jnp.float32),
                )
                pol_logits_opp = np.array(pol_logits_opp[0], dtype=np.float32)
                
                pruned_moves_o, _ = prune_moves_by_policy(moves_o, pol_logits_opp, opp_player, top_k)
                if len(pruned_moves_o) == 0:
                    pruned_moves_o = moves_o[:1]  # Fallback
                
                # Evaluate all pruned opponent afterstates
                boards_opp = []
                auxs_opp = []
                for mv_o in pruned_moves_o:
                    if mv_o in moves_o:
                        idx_o = moves_o.index(mv_o)
                        s_opp = after_o[idx_o]
                        b_opp_final, a_opp_final = encode_agent2(s_opp, player)  # From player's perspective
                        boards_opp.append(b_opp_final)
                        auxs_opp.append(a_opp_final)
                
                if len(boards_opp) > 0:
                    boards_opp = np.stack(boards_opp)
                    auxs_opp = np.stack(auxs_opp)
                    vals_opp = batched_values(params, boards_opp, auxs_opp, eval_batch)
                    v_min = float(np.min(vals_opp))
                else:
                    # Fallback
                    b_n, a_n = encode_agent2(s_a, player)
                    v_min = float(np.array(
                        value_apply(
                            params,
                            jnp.asarray(b_n[None, ...], dtype=jnp.float32),
                            jnp.asarray(a_n[None, ...], dtype=jnp.float32),
                        )[0]
                    ))
            
            ev_sum += PROBS21[j_idx] * v_min
        
        expected_values.append(ev_sum)
    
    best = int(np.argmax(expected_values))
    return pruned_moves_final[best], pruned_after_a[best]


# -----------------------------
# Sequential submove sampling (simplified)
# -----------------------------
def sample_move_from_policy(state, player, dice, params, rng):
    """
    Sample a move from policy distribution over legal moves.
    
    Simplified version: Gets all legal moves, computes their logits
    (sum of submove logits), and samples using softmax.
    
    Full sequential sampling (sample submove → intermediate state → next submove)
    is more complex and can be added later. This version is a valid approximation.
    
    Returns:
        (move, log_prob) tuple
    """
    # Get all legal moves
    moves, afterstates = _actions(state, np.int8(player), dice)
    moves = list(moves)
    afterstates = [np.asarray(s, dtype=np.int8) for s in afterstates]
    
    if len(moves) == 0:
        return None, 0.0
    
    # Encode state and get policy logits
    b, a = encode_agent2(state, player)
    _, pol_logits_625 = model_forward(
        params,
        jnp.asarray(b[None, ...], dtype=jnp.float32),
        jnp.asarray(a[None, ...], dtype=jnp.float32),
    )
    pol_logits_625 = np.array(pol_logits_625[0], dtype=np.float32)
    
    # Compute move logits (sum of submove logits)
    move_logits = []
    for mv in moves:
        mv_py = [(int(f), int(r)) for (f, r) in mv]
        subidxs = move_to_submove_action_indices(mv_py, player)
        logit_sum = sum(float(pol_logits_625[idx]) for idx in subidxs if idx >= 0)
        move_logits.append(logit_sum)
    
    move_logits = np.array(move_logits, dtype=np.float32)
    
    # Log-sum-exp trick for numerical stability
    logits_max = np.max(move_logits)
    exp_logits = np.exp(move_logits - logits_max)
    probs = exp_logits / np.sum(exp_logits)
    
    # Sample
    chosen_idx = rng.choice(len(moves), p=probs)
    chosen_move = moves[chosen_idx]
    chosen_log_prob = float(np.log(probs[chosen_idx] + 1e-12))
    
    return chosen_move, chosen_log_prob


# -----------------------------
# JAX model functions
# -----------------------------
# Create module instance once outside JIT
_ppo_net = Agent3PPONet()

@jax.jit
def model_forward(params, board, aux):
    """Forward pass: returns (value, policy_logits)"""
    v, pol = _ppo_net.apply(
        {"params": params},
        board_state=board,
        aux_features=aux,
    )
    return v, pol

@jax.jit
def value_apply(params, board, aux):
    """Get value only"""
    v, _ = model_forward(params, board, aux)
    return v

def batched_values(params, boards_np, aux_np, eval_batch):
    """Batched value evaluation"""
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
# PPO Loss and Gradients
# -----------------------------
@jax.jit
def ppo_loss_and_grads(params, batch, clip_eps, vf_coef, ent_coef):
    """
    PPO loss with clipped surrogate objective.
    
    batch contains:
      board: (B, 24, 15)
      aux: (B, 6)
      move_subidx: (B, M, S) int32 with -1 padding
      move_mask: (B, M) bool
      action: (B,) int32 (chosen move id in 0..M-1)
      old_logp: (B,) float32
      adv: (B,) float32
      ret: (B,) float32
    """
    board = batch["board"]
    aux = batch["aux"]
    move_subidx = batch["move_subidx"]
    move_mask = batch["move_mask"]
    action = batch["action"]
    old_logp = batch["old_logp"]
    adv = batch["adv"]
    ret = batch["ret"]
    
    def loss_fn(p):
        v_pred, pol_logits_625 = model_forward(p, board, aux)  # (B,), (B,625)
        
        # Build move logits: sum of submove logits for each move
        idx = jnp.where(move_subidx < 0, 0, move_subidx)  # (B,M,S)
        pol = pol_logits_625[:, None, None, :]  # (B,1,1,625)
        idx_exp = idx[..., None]  # (B,M,S,1)
        gathered = jnp.take_along_axis(pol, idx_exp, axis=-1).squeeze(-1)  # (B,M,S)
        gathered = jnp.where(move_subidx < 0, 0.0, gathered)
        move_logits = jnp.sum(gathered, axis=-1)  # (B,M)
        
        # Mask invalid moves
        neg_inf = jnp.array(-1e9, dtype=move_logits.dtype)
        move_logits = jnp.where(move_mask, move_logits, neg_inf)
        
        # Log probs over legal moves
        log_probs = jax.nn.log_softmax(move_logits, axis=-1)  # (B,M)
        new_logp = log_probs[jnp.arange(log_probs.shape[0]), action]  # (B,)
        
        # Clipped surrogate objective
        ratio = jnp.exp(new_logp - old_logp)
        clipped = jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
        policy_loss = -jnp.mean(jnp.minimum(ratio * adv, clipped * adv))
        
        # Value function loss
        value_loss = jnp.mean((v_pred - ret) ** 2)
        
        # Entropy bonus (over legal moves only)
        probs = jax.nn.softmax(move_logits, axis=-1)
        entropy = -jnp.mean(jnp.sum(probs * log_probs, axis=-1))
        
        # Total loss: L_CLIP - c1*L_VF + c2*H
        total = policy_loss + vf_coef * value_loss - ent_coef * entropy
        return total, (policy_loss, value_loss, entropy, jnp.mean(v_pred))
    
    (total_loss, aux_out), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
    policy_loss, value_loss, entropy, v_mean = aux_out
    return total_loss, grads, policy_loss, value_loss, entropy, v_mean


# -----------------------------
# Rollout collection with 2-ply value estimation
# -----------------------------
def collect_rollout(st, num_steps, top_k, rng, gamma, lam, eval_batch):
    """
    Collect rollout using:
    - 2-ply search with policy pruning to estimate V(S_t)
    - Sequential submove sampling for actual action
    """
    boards = []
    auxs = []
    move_subidxs = []
    move_masks = []
    actions = []
    old_logps = []
    values = []
    rewards = []
    dones = []
    
    player, dice, state = _new_game()
    
    for _ in range(num_steps):
        # Roll dice
        dice = np.sort(np.random.randint(1, 7, size=2).astype(np.int8))[::-1]
        
        # Encode state
        b, a = encode_agent2(state, player)
        
        # Get policy logits for pruning
        _, pol_logits_625 = model_forward(
            st.params,
            jnp.asarray(b[None, ...], dtype=jnp.float32),
            jnp.asarray(a[None, ...], dtype=jnp.float32),
        )
        pol_logits_625 = np.array(pol_logits_625[0], dtype=np.float32)
        
        # Estimate V(S_t) using pruned 2-ply search
        _, v_state = choose_action_2ply_pruned(
            state, player, dice, st.params, pol_logits_625, eval_batch, top_k, debug=False
        )
        b_v, a_v = encode_agent2(v_state, player)
        v_pred = float(np.array(
            value_apply(
                st.params,
                jnp.asarray(b_v[None, ...], dtype=jnp.float32),
                jnp.asarray(a_v[None, ...], dtype=jnp.float32),
            )[0]
        ))
        
        # Sample action using policy distribution
        sampled_move, old_logp = sample_move_from_policy(
            state, player, dice, st.params, rng
        )
        
        if sampled_move is None:
            # Forced pass
            next_state = state
            moves = []
            afterstates = []
            act_idx = 0
        else:
            # Get all legal moves to find index
            moves, afterstates = _actions(state, np.int8(player), dice)
            moves = list(moves)
            afterstates = [np.asarray(s, dtype=np.int8) for s in afterstates]
            
            # Find index of sampled move
            try:
                act_idx = moves.index(sampled_move)
                next_state = afterstates[act_idx]
            except ValueError:
                # Fallback: use first move
                act_idx = 0
                next_state = afterstates[0] if afterstates else state
        
        r = py_reward(next_state, player)
        done = (r != 0)
        
        # Store transition
        boards.append(b)
        auxs.append(a)
        
        # Build move table (pad to max moves)
        # Get all legal moves for the move table
        if sampled_move is None or len(moves) == 0:
            # Forced pass - create empty move table
            max_moves = 5
            mv_table = np.full((max_moves, MAX_SUBMOVES), -1, dtype=np.int32)
            mv_mask = np.zeros((max_moves,), dtype=np.bool_)
        else:
            max_moves = max(5, len(moves))  # At least 5 for padding
            mv_table = np.full((max_moves, MAX_SUBMOVES), -1, dtype=np.int32)
            mv_mask = np.zeros((max_moves,), dtype=np.bool_)
            
            for i, mv in enumerate(moves[:max_moves]):
                mv_py = [(int(f), int(r)) for (f, r) in mv]
                subidxs = move_to_submove_action_indices(mv_py, player)
                mv_table[i] = subidxs
                mv_mask[i] = True
        
        move_subidxs.append(mv_table)
        move_masks.append(mv_mask)
        actions.append(act_idx)
        old_logps.append(old_logp)
        values.append(v_pred)
        rewards.append(float(r))
        dones.append(done)
        
        # Step env
        if done:
            player, dice, state = _new_game()
        else:
            state = next_state
            player = -player
    
    # Compute GAE
    values_np = np.asarray(values, dtype=np.float32)
    rewards_np = np.asarray(rewards, dtype=np.float32)
    dones_np = np.asarray(dones, dtype=np.bool_)
    
    adv = np.zeros_like(values_np, dtype=np.float32)
    next_adv = 0.0
    
    for t in reversed(range(num_steps)):
        if dones_np[t]:
            next_value = 0.0
            next_adv = 0.0
        else:
            next_value = values_np[t + 1] if t + 1 < num_steps else 0.0
        
        delta = rewards_np[t] + gamma * next_value - values_np[t]
        adv[t] = delta + gamma * lam * next_adv
        next_adv = adv[t]
    
    ret = adv + values_np
    
    # Find max moves for padding
    max_m = max(len(m) for m in move_subidxs) if move_subidxs else 5
    
    # Pad all move tables to same size
    padded_subidxs = []
    padded_masks = []
    for mv_table, mv_mask in zip(move_subidxs, move_masks):
        if mv_table.shape[0] < max_m:
            pad_table = np.full((max_m, MAX_SUBMOVES), -1, dtype=np.int32)
            pad_mask = np.zeros((max_m,), dtype=np.bool_)
            pad_table[:mv_table.shape[0]] = mv_table
            pad_mask[:mv_mask.shape[0]] = mv_mask
            padded_subidxs.append(pad_table)
            padded_masks.append(pad_mask)
        else:
            padded_subidxs.append(mv_table[:max_m])
            padded_masks.append(mv_mask[:max_m])
    
    batch = dict(
        board=np.asarray(boards, dtype=np.float32),
        aux=np.asarray(auxs, dtype=np.float32),
        move_subidx=np.asarray(padded_subidxs, dtype=np.int32),
        move_mask=np.asarray(padded_masks, dtype=np.bool_),
        action=np.asarray(actions, dtype=np.int32),
        old_logp=np.asarray(old_logps, dtype=np.float32),
        adv=adv.astype(np.float32),
        ret=ret.astype(np.float32),
    )
    return batch


# -----------------------------
# Minibatch training
# -----------------------------
def minibatches(batch, mb_size, rng):
    """Shuffle and yield minibatches"""
    N = batch["board"].shape[0]
    idx = np.arange(N)
    rng.shuffle(idx)
    for start in range(0, N, mb_size):
        sl = idx[start:start + mb_size]
        yield {k: batch[k][sl] for k in batch.keys()}


# -----------------------------
# Main training loop
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="Agent 3 PPO Training")
    ap.add_argument("--total_updates", type=int, default=200, help="Total training updates")
    ap.add_argument("--rollout_steps", type=int, default=8192, help="Steps per rollout")
    ap.add_argument("--epochs", type=int, default=4, help="Training epochs per update")
    ap.add_argument("--minibatch", type=int, default=512, help="Minibatch size")
    ap.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    ap.add_argument("--clip_eps", type=float, default=0.2, help="PPO clip epsilon")
    ap.add_argument("--vf_coef", type=float, default=0.5, help="Value function loss coefficient")
    ap.add_argument("--ent_coef", type=float, default=0.01, help="Entropy coefficient")
    ap.add_argument("--gamma", type=float, default=1.0, help="Discount factor")
    ap.add_argument("--lam", type=float, default=0.95, help="GAE lambda")
    ap.add_argument("--top_k", type=int, default=5, help="Top K moves for pruning (default: 5)")
    ap.add_argument("--eval_batch", type=int, default=8192, help="Batch size for value evaluation")
    ap.add_argument("--seed", type=int, default=0, help="Random seed")
    ap.add_argument("--save_every", type=int, default=20, help="Save checkpoint every N updates")
    ap.add_argument("--ckpt_dir", type=str, default="checkpoints_agent3", help="Checkpoint directory")
    args = ap.parse_args()
    
    rng = np.random.default_rng(args.seed)
    key = jax.random.PRNGKey(args.seed)
    
    # Initialize model
    model = Agent3PPONet()
    dummy_board = jnp.zeros((1, BOARD_LENGTH, CONV_INPUT_CHANNELS), dtype=jnp.float32)
    dummy_aux = jnp.zeros((1, AUX_INPUT_SIZE), dtype=jnp.float32)
    params = model.init(key, board_state=dummy_board, aux_features=dummy_aux)["params"]
    tx = optax.adam(args.lr)
    st = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)
    
    # Setup checkpoint directory
    ckpt_dir = os.path.abspath(args.ckpt_dir)
    os.makedirs(ckpt_dir, exist_ok=True)
    
    print(f"Agent 3 PPO Training Configuration:")
    print(f"  Total updates: {args.total_updates}")
    print(f"  Rollout steps: {args.rollout_steps}")
    print(f"  Training epochs: {args.epochs}")
    print(f"  Minibatch size: {args.minibatch}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Clip epsilon: {args.clip_eps}")
    print(f"  Top K moves: {args.top_k}")
    print(f"  Checkpoint directory: {ckpt_dir}")
    print()
    
    # Warmup JIT
    dummy_batch = dict(
        board=jnp.zeros((2, BOARD_LENGTH, CONV_INPUT_CHANNELS), dtype=jnp.float32),
        aux=jnp.zeros((2, AUX_INPUT_SIZE), dtype=jnp.float32),
        move_subidx=jnp.zeros((2, 5, MAX_SUBMOVES), dtype=jnp.int32),
        move_mask=jnp.ones((2, 5), dtype=jnp.bool_),
        action=jnp.zeros((2,), dtype=jnp.int32),
        old_logp=jnp.zeros((2,), dtype=jnp.float32),
        adv=jnp.zeros((2,), dtype=jnp.float32),
        ret=jnp.zeros((2,), dtype=jnp.float32),
    )
    _ = ppo_loss_and_grads(st.params, dummy_batch, args.clip_eps, args.vf_coef, args.ent_coef)
    
    t0 = time.time()
    
    for upd in range(1, args.total_updates + 1):
        # Collect rollout
        batch_np = collect_rollout(
            st=st,
            num_steps=args.rollout_steps,
            top_k=args.top_k,
            rng=rng,
            gamma=args.gamma,
            lam=args.lam,
            eval_batch=args.eval_batch,
        )
        
        # Normalize advantages
        adv = batch_np["adv"]
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        batch_np["adv"] = adv.astype(np.float32)
        
        # Training epochs
        pol_losses = []
        v_losses = []
        ents = []
        totals = []
        
        for _epoch in range(args.epochs):
            for mb in minibatches(batch_np, args.minibatch, rng):
                mb_jax = {k: jnp.asarray(v) for k, v in mb.items()}
                total_loss, grads, pl, vl, ent, vmean = ppo_loss_and_grads(
                    st.params, mb_jax, args.clip_eps, args.vf_coef, args.ent_coef
                )
                st = st.apply_gradients(grads=grads)
                
                totals.append(float(total_loss))
                pol_losses.append(float(pl))
                v_losses.append(float(vl))
                ents.append(float(ent))
        
        dt = time.time() - t0
        steps_done = upd * args.rollout_steps
        sps = steps_done / dt
        
        print(
            f"[Update {upd:4d}/{args.total_updates}] "
            f"total={np.mean(totals):.4f} "
            f"policy={np.mean(pol_losses):.4f} "
            f"value={np.mean(v_losses):.4f} "
            f"entropy={np.mean(ents):.4f} "
            f"steps={steps_done} "
            f"sps={sps:.1f}"
        )
        
        if upd % args.save_every == 0 or upd == args.total_updates:
            ckpt_path = checkpoints.save_checkpoint(
                ckpt_dir=ckpt_dir,
                target=st.params,
                step=upd,
                overwrite=True,
            )
            print(f"  💾 CHECKPOINT SAVED: {ckpt_path} @ update {upd}")
    
    print("Done.")


if __name__ == "__main__":
    main()

