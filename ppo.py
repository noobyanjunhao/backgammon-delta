import os
# Set JAX memory settings BEFORE importing jax to avoid OOM errors
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.7'  # Use only 70% of GPU memory
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'  # Don't preallocate all memory
os.environ['XLA_FLAGS'] = '--xla_gpu_enable_command_buffer='  # Disable CUDA graphs

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
    _to_canonical,
    _get_target_index,
    NUM_CHECKERS,
    HOME_BOARD_SIZE,
    NUM_POINTS,
    W_BAR, B_BAR, W_OFF, B_OFF,
)
from backgammon_ppo_net import BackgammonPPONet, AUX_INPUT_SIZE

# ----------------------------
# Hyperparameters / constants
# ----------------------------
MAX_SUBMOVES = 4  # doubles can create up to 4 submoves
# Canonical submove action space (from 0..24) x (to 0..24) => 625
# We'll use:
#   from_idx: 0=bar, 1..24=points
#   to_idx:   0=off, 1..24=points
ACTION_SPACE = 25 * 25

BOARD_LENGTH = 24
CONV_INPUT_CHANNELS = 15  # as in your nets (board reshaped to (24,15))


# ----------------------------
# Reward (pure python)
# ----------------------------
def py_reward(state, player):
    s = np.asarray(state, dtype=np.int8)
    if s.shape[0] < 28:
        return 0
    p = int(player)

    # White wins
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

    # Black wins
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


# ----------------------------
# Encoding (same style as TD script)
# ----------------------------
def encode(state, player):
    """
    Encode engine state into board planes (24,15) + aux features (6,).
    Uses same simple placeholder scheme:
      - board[:,0] = point occupancy / 15
      - aux = [1, bar, opp_bar, off, opp_off, cube_placeholder]
    Canonicalizes so "current player is white".
    """
    state_arr = np.asarray(state, dtype=np.int8)
    s = _to_canonical(state_arr, np.int8(player)).astype(np.float32).ravel()

    board = np.zeros((BOARD_LENGTH, CONV_INPUT_CHANNELS), dtype=np.float32)
    pts = s[1 : 1 + BOARD_LENGTH]
    if pts.shape[0] == BOARD_LENGTH:
        board[:, 0] = pts / 15.0

    aux = np.zeros((AUX_INPUT_SIZE,), dtype=np.float32)
    aux[0] = 1.0
    if AUX_INPUT_SIZE > 1:
        aux[1] = float(s[0]) / 15.0     # bar
    if AUX_INPUT_SIZE > 2:
        aux[2] = float(s[25]) / 15.0    # opp bar (canonical)
    if AUX_INPUT_SIZE > 3:
        aux[3] = float(s[26]) / 15.0    # off
    if AUX_INPUT_SIZE > 4:
        aux[4] = float(s[27]) / 15.0    # opp off
    # aux[5] reserved (cube) left at 0

    return board, aux


# ----------------------------
# Canonical submove mapping
# ----------------------------
def physical_point_to_canonical(point, player):
    """
    Convert a physical point index (0..27) for 'player' into canonical point
    indices (bar=0, points 1..24). Off is handled separately.
    """
    if player == 1:
        # White: bar is 0, points are 1..24 already
        return int(point)
    else:
        # Black: canonical flips using 25 - point for indices 0..25 (bar+points+oppbar)
        # Physical black bar is 25 => 25-25=0 (canonical bar)
        # Physical point 1 => 24; point 24 => 1
        return int(NUM_POINTS + 1 - point)  # 25 - point


def physical_target_to_canonical_to_idx(target, player):
    """
    Convert physical target index into canonical 'to_idx' in 0..24 where:
      0 = off
      1..24 = board points
    """
    if player == 1:
        if target == W_OFF:
            return 0
        if 1 <= target <= 24:
            return int(target)
        # anything else: treat as off (rare)
        return 0
    else:
        if target == B_OFF:
            return 0
        if 1 <= target <= 24:
            return int(NUM_POINTS + 1 - target)  # 25 - target
        return 0


def move_to_submove_action_indices(move_seq, player):
    """
    move_seq is a list of (from_point, roll) pairs from engine.
    Convert to list of canonical submove action indices in 0..624.
    Each submove index corresponds to (from_idx, to_idx):
      idx = from_idx * 25 + to_idx
    where from_idx in [0..24], to_idx in [0..24] (0=off, 1..24=points).
    """
    idxs = []
    for (from_point, roll) in move_seq:
        from_point = int(from_point)
        roll = int(roll)
        target = int(_get_target_index(np.int8(from_point), np.int8(roll), np.int8(player)))

        from_idx = physical_point_to_canonical(from_point, player)  # 0..24
        to_idx = physical_target_to_canonical_to_idx(target, player)  # 0..24

        # clamp safety
        from_idx = max(0, min(24, from_idx))
        to_idx = max(0, min(24, to_idx))

        idxs.append(from_idx * 25 + to_idx)

    # pad to MAX_SUBMOVES with -1
    out = np.full((MAX_SUBMOVES,), -1, dtype=np.int32)
    for i in range(min(len(idxs), MAX_SUBMOVES)):
        out[i] = np.int32(idxs[i])
    return out


# ----------------------------
# JAX model helpers
# ----------------------------
def create_train_state(rng, lr):
    model = BackgammonPPONet()
    dummy_board = jnp.zeros((1, BOARD_LENGTH, CONV_INPUT_CHANNELS), dtype=jnp.float32)
    dummy_aux = jnp.zeros((1, AUX_INPUT_SIZE), dtype=jnp.float32)
    params = model.init(rng, board_state=dummy_board, aux_features=dummy_aux)["params"]
    tx = optax.adam(lr)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)


@jax.jit
def model_forward(params, board, aux):
    # returns (value_pred (B,), policy_logits_625 (B,625))
    value_pred, policy_logits = BackgammonPPONet().apply(
        {"params": params},
        board_state=board,
        aux_features=aux,
    )
    return value_pred.squeeze(-1), policy_logits


@jax.jit
def ppo_loss_and_grads(params, batch, clip_eps, vf_coef, ent_coef):
    """
    batch contains:
      board: (B,24,15)
      aux: (B,6)
      move_subidx: (B,M,S) int32 with -1 padding
      move_mask: (B,M) bool
      action: (B,) int32  (chosen move id in 0..M-1)
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

        # Build move logits:
        # move_subidx: (B,M,S). We want sum over S of pol_logits_625[b, idx]
        # where idx is 0..624. For idx=-1, contribute 0.
        idx = jnp.where(move_subidx < 0, 0, move_subidx)  # (B,M,S)
        # Expand pol_logits_625 to (B,1,1,625) then gather
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

        ratio = jnp.exp(new_logp - old_logp)
        clipped = jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
        policy_loss = -jnp.mean(jnp.minimum(ratio * adv, clipped * adv))

        value_loss = jnp.mean((v_pred - ret) ** 2)

        probs = jax.nn.softmax(move_logits, axis=-1)
        entropy = -jnp.mean(jnp.sum(probs * log_probs, axis=-1))

        total = policy_loss + vf_coef * value_loss - ent_coef * entropy
        return total, (policy_loss, value_loss, entropy, jnp.mean(v_pred))

    (total_loss, aux_out), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
    policy_loss, value_loss, entropy, v_mean = aux_out
    return total_loss, grads, policy_loss, value_loss, entropy, v_mean


# ----------------------------
# Rollout collection
# ----------------------------
def collect_rollout(st, num_steps, max_moves, rng, gamma, lam):
    """
    Collect on-policy rollout from self-play.

    Returns dict of numpy arrays:
      board, aux, move_subidx, move_mask, action, old_logp, adv, ret
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

    player, _, state = _new_game()

    for _ in range(num_steps):
        # roll dice (engine uses sorted dice high->low); we mimic
        dice = np.sort(np.random.randint(1, 7, size=2).astype(np.int8))[::-1]

        # legal moves + afterstates from engine
        moves_nb, after_nb = _actions(state, np.int8(player), dice)
        # convert to python lists
        moves = list(moves_nb)
        afters = list(after_nb)

        # Encode state for net (canonical)
        b, a = encode(state, player)

        # Forward pass
        v_pred, pol_logits_625 = model_forward(
            st.params,
            jnp.asarray(b[None, ...], dtype=jnp.float32),
            jnp.asarray(a[None, ...], dtype=jnp.float32),
        )
        v_pred = float(np.array(v_pred[0]))
        pol_logits_625 = np.array(pol_logits_625[0], dtype=np.float32)

        # Build move_subidx matrix for all legal moves
        subidx_list = []
        score_list = []
        for mv in moves:
            mv_py = [(int(f), int(r)) for (f, r) in mv]
            subidx = move_to_submove_action_indices(mv_py, player)  # (S,)
            subidx_list.append(subidx)

            # score = sum logits for its submoves
            s = 0.0
            for idx in subidx:
                if idx >= 0:
                    s += float(pol_logits_625[idx])
            score_list.append(s)

        subidx_arr = np.stack(subidx_list, axis=0)  # (L,S)
        scores = np.asarray(score_list, dtype=np.float32)  # (L,)

        # Prune to top-K moves
        L = len(scores)
        if L > max_moves:
            top = np.argpartition(-scores, max_moves)[:max_moves]
            # Keep stable order by score descending
            top = top[np.argsort(-scores[top])]
            subidx_arr = subidx_arr[top]
            scores = scores[top]
            afters = [afters[i] for i in top]

        # Softmax sample among pruned legal moves
        scores_shift = scores - scores.max()
        probs = np.exp(scores_shift)
        probs = probs / probs.sum()
        act = int(rng.choice(len(probs), p=probs))
        old_logp = float(np.log(probs[act] + 1e-12))

        next_state = np.array(afters[act], dtype=np.int8)

        r = py_reward(next_state, player)
        done = (r != 0)

        # Store transition (state-based)
        boards.append(b)
        auxs.append(a)

        # Pad move table to (max_moves, S)
        M = max_moves
        S = MAX_SUBMOVES
        mv_table = np.full((M, S), -1, dtype=np.int32)
        mv_mask = np.zeros((M,), dtype=np.bool_)
        mv_count = subidx_arr.shape[0]
        mv_table[:mv_count] = subidx_arr
        mv_mask[:mv_count] = True

        move_subidxs.append(mv_table)
        move_masks.append(mv_mask)

        actions.append(act)
        old_logps.append(old_logp)
        values.append(v_pred)
        rewards.append(float(r))
        dones.append(done)

        # Step env
        if done:
            player, _, state = _new_game()
        else:
            state = next_state
            player = -player

    # Bootstrap values for GAE
    values_np = np.asarray(values, dtype=np.float32)
    rewards_np = np.asarray(rewards, dtype=np.float32)
    dones_np = np.asarray(dones, dtype=np.bool_)

    adv = np.zeros_like(values_np, dtype=np.float32)
    ret = np.zeros_like(values_np, dtype=np.float32)

    next_adv = 0.0
    next_value = 0.0

    # We do not store next state's value; use 0 at terminal, else use next value_pred from next step
    # For simplicity, approximate V(s_{t+1}) as values[t+1] when not terminal.
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

    batch = dict(
        board=np.asarray(boards, dtype=np.float32),
        aux=np.asarray(auxs, dtype=np.float32),
        move_subidx=np.asarray(move_subidxs, dtype=np.int32),
        move_mask=np.asarray(move_masks, dtype=np.bool_),
        action=np.asarray(actions, dtype=np.int32),
        old_logp=np.asarray(old_logps, dtype=np.float32),
        adv=adv.astype(np.float32),
        ret=ret.astype(np.float32),
    )
    return batch


def minibatches(batch, mb_size, rng):
    N = batch["board"].shape[0]
    idx = np.arange(N)
    rng.shuffle(idx)
    for start in range(0, N, mb_size):
        sl = idx[start:start + mb_size]
        yield {k: batch[k][sl] for k in batch.keys()}


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--total_updates", type=int, default=200)
    ap.add_argument("--rollout_steps", type=int, default=8192)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--minibatch", type=int, default=512)

    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--clip_eps", type=float, default=0.2)
    ap.add_argument("--vf_coef", type=float, default=0.5)
    ap.add_argument("--ent_coef", type=float, default=0.01)

    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--lam", type=float, default=0.95)

    ap.add_argument("--max_moves", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save_every", type=int, default=20,
                    help="Save checkpoint every N updates (default: 20)")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    jax_rng = jax.random.PRNGKey(args.seed)

    st = create_train_state(jax_rng, args.lr)
    
    # Use consistent checkpoint directory format: checkpoints_ppo
    ckpt_dir = os.path.abspath("checkpoints_ppo")
    os.makedirs(ckpt_dir, exist_ok=True)

    # Warmup JIT compile quickly
    dummy_batch = dict(
        board=jnp.zeros((2, BOARD_LENGTH, CONV_INPUT_CHANNELS), dtype=jnp.float32),
        aux=jnp.zeros((2, AUX_INPUT_SIZE), dtype=jnp.float32),
        move_subidx=jnp.zeros((2, args.max_moves, MAX_SUBMOVES), dtype=jnp.int32),
        move_mask=jnp.ones((2, args.max_moves), dtype=jnp.bool_),
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
            max_moves=args.max_moves,
            rng=rng,
            gamma=args.gamma,
            lam=args.lam,
        )

        # Normalize advantages (important for PPO)
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
