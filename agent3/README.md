# Agent 3: PPO with 2-Ply Search and Policy Pruning

## Overview

Agent 3 implements Proximal Policy Optimization (PPO) with:
- **Agent 2's network architecture** + policy head (25×25 grid)
- **2-ply search with policy-based pruning** (top 5 moves)
- **GAE (Generalized Advantage Estimation)**
- **Clipped surrogate objective**
- **Sequential submove sampling** (simplified version)

## Files

- `agent3_ppo_net.py`: Network architecture (Agent 2's backbone + policy head)
- `agent3_utils.py`: Utilities for submove indexing and conversion
- `train_agent3.py`: Main training script

## Key Features

### ✅ Implemented

1. **Network Architecture**: 
   - Uses Agent 2's exact ResNet backbone
   - Value head (critic): 134 → 64 → 1 (Tanh scaled by 3)
   - Policy head (actor): 134 → 64 → 625 (25×25 source-destination pairs)

2. **2-Ply Search with Policy Pruning**:
   - Prunes player moves to top 5 using policy logits
   - Prunes opponent moves to top 5 using policy logits
   - Computes expected value over all dice rolls

3. **PPO Algorithm**:
   - GAE computation (Generalized Advantage Estimation)
   - Clipped surrogate objective
   - Value function loss
   - Entropy bonus

4. **Training**:
   - Replay buffer with (S_t, A_sampled, π_old, R_{t+1}, V(S_t))
   - Minibatch training with shuffled indices
   - Multiple epochs over replay buffer

### ⚠️ Simplified

1. **Sequential Submove Sampling**: 
   - Currently uses policy distribution over complete legal moves
   - Full sequential sampling (sample submove → intermediate state → next submove) is partially implemented
   - Can be enhanced later for exact specification compliance

## Usage

### Quick Test

```bash
python agent3/train_agent3.py \
    --total_updates 10 \
    --rollout_steps 100 \
    --top_k 5 \
    --eval_batch 256
```

### Full Training

```bash
python agent3/train_agent3.py \
    --total_updates 200 \
    --rollout_steps 8192 \
    --epochs 4 \
    --minibatch 512 \
    --top_k 5 \
    --eval_batch 8192 \
    --ckpt_dir checkpoints_agent3
```

## Command-Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--total_updates` | 200 | Total training updates |
| `--rollout_steps` | 8192 | Steps per rollout |
| `--epochs` | 4 | Training epochs per update |
| `--minibatch` | 512 | Minibatch size |
| `--lr` | 3e-4 | Learning rate |
| `--clip_eps` | 0.2 | PPO clip epsilon |
| `--vf_coef` | 0.5 | Value function loss coefficient |
| `--ent_coef` | 0.01 | Entropy coefficient |
| `--gamma` | 1.0 | Discount factor |
| `--lam` | 0.95 | GAE lambda |
| `--top_k` | 5 | **Top K moves for pruning (matches spec)** |
| `--eval_batch` | 8192 | Batch size for value evaluation |
| `--ckpt_dir` | `checkpoints_agent3` | Checkpoint directory |

## Differences from Original PPO

1. **Uses Agent 2's network**: Exact same ResNet backbone and encoding
2. **2-ply search**: Implements 2-ply search with policy pruning (not just 1-ply)
3. **Top 5 moves**: Prunes to top 5 moves (not 128)
4. **Policy-based pruning**: Uses policy logits to select top moves (as specified)

## Checkpoints

Checkpoints are saved to `checkpoints_agent3/` by default (configurable with `--ckpt_dir`).

## Notes

- The sequential submove sampling is simplified - it samples from the policy distribution over complete legal moves rather than strictly sampling submoves sequentially. This is a valid approximation and can be enhanced later.
- 2-ply search with pruning is computationally expensive but matches the specification.
- The network architecture exactly matches Agent 2's specification.

