# PPO Training Guide - Starting Small

This guide shows you how to train the PPO (Proximal Policy Optimization) agent for Backgammon, starting with a small example and scaling up.

## What is PPO?

PPO is a policy gradient method that:
- Learns both a **policy** (which moves to make) and a **value function** (how good positions are)
- Uses self-play: the agent plays against itself
- Collects rollouts (game trajectories), then updates the policy/value network
- Uses clipping to prevent large policy updates

## Quick Start: Minimal Test Run

### Step 1: Very Small Test (Verify It Works)

```bash
# Test with minimal settings (should complete in ~1-2 minutes)
python ppo.py \
    --total_updates 5 \
    --rollout_steps 256 \
    --epochs 2 \
    --minibatch 64 \
    --max_moves 32 \
    --save_every 5
```

**What this does:**
- `--total_updates 5`: Only 5 training updates (very quick)
- `--rollout_steps 256`: Collect 256 game steps per update (small)
- `--epochs 2`: Train for 2 epochs on each rollout
- `--minibatch 64`: Small minibatches
- `--max_moves 32`: Only consider top 32 moves (faster)

**Expected output:**
```
[Update    1/5] total=0.1234 policy=0.0567 value=0.0123 entropy=2.3456 steps=256 sps=45.2
[Update    2/5] total=0.1123 policy=0.0456 value=0.0112 entropy=2.2345 steps=512 sps=48.1
...
💾 CHECKPOINT SAVED: /path/to/checkpoints_ppo/checkpoint_5 @ update 5
Done.
```

### Step 2: Small Training Run (10-15 minutes)

```bash
# Small but meaningful training run
python ppo.py \
    --total_updates 20 \
    --rollout_steps 1024 \
    --epochs 4 \
    --minibatch 128 \
    --max_moves 64 \
    --save_every 10
```

### Step 3: Medium Training Run (1-2 hours)

```bash
# Medium training run
python ppo.py \
    --total_updates 50 \
    --rollout_steps 4096 \
    --epochs 4 \
    --minibatch 256 \
    --max_moves 128 \
    --save_every 10
```

### Step 4: Full Training Run (Recommended)

```bash
# Full training with default hyperparameters
python ppo.py \
    --total_updates 200 \
    --rollout_steps 8192 \
    --epochs 4 \
    --minibatch 512 \
    --max_moves 128 \
    --save_every 20
```

## Understanding the Hyperparameters

### Core Training Parameters

| Parameter | Default | Description | Small Example |
|-----------|---------|-------------|---------------|
| `--total_updates` | 200 | Number of PPO update cycles | 5-20 |
| `--rollout_steps` | 8192 | Game steps collected per update | 256-1024 |
| `--epochs` | 4 | Training epochs per rollout | 2-4 |
| `--minibatch` | 512 | Minibatch size for gradient updates | 64-128 |

### PPO-Specific Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--lr` | 3e-4 | Learning rate (Adam optimizer) |
| `--clip_eps` | 0.2 | PPO clipping epsilon (prevents large updates) |
| `--vf_coef` | 0.5 | Value function loss coefficient |
| `--ent_coef` | 0.01 | Entropy bonus coefficient (encourages exploration) |

### GAE (Generalized Advantage Estimation)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--gamma` | 1.0 | Discount factor (1.0 = no discounting) |
| `--lam` | 0.95 | GAE lambda (bias-variance tradeoff) |

### Performance Parameters

| Parameter | Default | Description | Small Example |
|-----------|---------|-------------|---------------|
| `--max_moves` | 128 | Max legal moves to consider (pruning) | 32-64 |
| `--save_every` | 20 | Save checkpoint every N updates | 5-10 |

## Training Progress Interpretation

### What to Look For

**Good signs:**
- **Policy loss decreasing**: Agent is learning better moves
- **Value loss decreasing**: Agent is learning position evaluation
- **Entropy decreasing slowly**: Agent is becoming more confident (but not too fast)
- **Steps/second (sps) increasing**: JIT compilation finished, now running fast

**Warning signs:**
- **Policy loss = 0**: Agent stopped exploring (entropy too low)
- **Value loss exploding**: Learning rate too high
- **No improvement after 50+ updates**: May need more exploration or different hyperparameters

### Example Output Interpretation

```
[Update   10/200] total=0.0456 policy=0.0234 value=0.0123 entropy=1.2345 steps=81920 sps=125.3
```

- `total=0.0456`: Combined loss (lower is better)
- `policy=0.0234`: Policy gradient loss
- `value=0.0123`: Value function prediction error
- `entropy=1.2345`: Exploration level (should decrease slowly)
- `steps=81920`: Total game steps so far
- `sps=125.3`: Steps per second (speed indicator)

## Training Workflow

### 1. Local Quick Test (CPU/GPU)

```bash
# Test on your local machine first
python ppo.py --total_updates 5 --rollout_steps 256 --epochs 2 --minibatch 64
```

### 2. Server Training (GPU Recommended)

```bash
# On Delta/Jetstream2, run in background
nohup python ppo.py \
    --total_updates 200 \
    --rollout_steps 8192 \
    --epochs 4 \
    --minibatch 512 \
    > ppo_training.log 2>&1 &

# Monitor progress
tail -f ppo_training.log
```

### 3. Monitor Training

```bash
# Watch the log file
tail -f ppo_training.log

# Check GPU usage (if on GPU node)
nvidia-smi

# Check if process is still running
ps aux | grep ppo.py
```

## Checkpoint Management

### Checkpoints Location

All checkpoints are saved to: `checkpoints_ppo/`

```
checkpoints_ppo/
  ├── checkpoint_20.orbax-checkpoint/
  ├── checkpoint_40.orbax-checkpoint/
  ├── checkpoint_60.orbax-checkpoint/
  └── ...
```

### Loading Checkpoints

To load a checkpoint for evaluation:

```python
from flax.training import checkpoints
from backgammon_ppo_net import BackgammonPPONet
import jax
import jax.numpy as jnp

# Load checkpoint
key = jax.random.PRNGKey(0)
dummy_board = jnp.zeros((1, 24, 15), dtype=jnp.float32)
dummy_aux = jnp.zeros((1, 6), dtype=jnp.float32)
params = BackgammonPPONet().init(key, board_state=dummy_board, aux_features=dummy_aux)["params"]

restored = checkpoints.restore_checkpoint("checkpoints_ppo", target=params, step=20)
```

## Troubleshooting

### Issue: "Out of Memory" Error

**Solution:** Reduce batch sizes
```bash
python ppo.py --rollout_steps 4096 --minibatch 256 --max_moves 64
```

### Issue: Training is Very Slow

**Causes:**
1. JIT compilation (first update takes 5-30 minutes) - **This is normal!**
2. Too many moves being considered - reduce `--max_moves`
3. CPU instead of GPU - check `jax.devices()`

**Solution:**
```bash
# Check if GPU is available
python -c "import jax; print(jax.devices())"

# Reduce max_moves for faster training
python ppo.py --max_moves 64
```

### Issue: No Learning (Loss Not Decreasing)

**Possible causes:**
1. Learning rate too low - try `--lr 1e-3`
2. Not enough exploration - increase `--ent_coef 0.02`
3. Need more training - run more updates

## Recommended Training Schedule

### Week 1: Exploration
```bash
# High exploration, quick iterations
python ppo.py \
    --total_updates 50 \
    --rollout_steps 4096 \
    --ent_coef 0.02 \
    --max_moves 128
```

### Week 2: Refinement
```bash
# Standard training
python ppo.py \
    --total_updates 200 \
    --rollout_steps 8192 \
    --ent_coef 0.01 \
    --max_moves 128
```

### Week 3+: Advanced
```bash
# Longer rollouts, more updates
python ppo.py \
    --total_updates 500 \
    --rollout_steps 16384 \
    --ent_coef 0.005 \
    --max_moves 128
```

## Evaluation

After training, evaluate your PPO agent:

```bash
# Compare against random baseline
python compare_checkpoint_steps.py checkpoints_ppo --random --games 500

# Compare two PPO checkpoints
python compare_checkpoint_steps.py checkpoints_ppo checkpoints_ppo --games 1000
```

## Next Steps

1. **Start small**: Run the minimal test (Step 1) to verify everything works
2. **Scale up gradually**: Move to Step 2, then Step 3, then full training
3. **Monitor progress**: Watch the loss values and entropy
4. **Save frequently**: Use `--save_every 10` for frequent checkpoints
5. **Evaluate regularly**: Test your agent against random baseline

## Example: Complete Small Training Session

```bash
# 1. Quick test (1-2 minutes)
python ppo.py --total_updates 5 --rollout_steps 256 --epochs 2 --minibatch 64 --max_moves 32

# 2. Small training (10-15 minutes)
python ppo.py --total_updates 20 --rollout_steps 1024 --epochs 4 --minibatch 128 --max_moves 64 --save_every 10

# 3. Evaluate the result
python compare_checkpoint_steps.py checkpoints_ppo --random --games 100

# 4. If results look good, scale up to full training
python ppo.py --total_updates 200 --rollout_steps 8192 --epochs 4 --minibatch 512 --max_moves 128
```

Good luck with your PPO training! 🎲

