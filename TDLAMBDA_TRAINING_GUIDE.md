# TD(λ) Training Guide - Starting Simple

This guide shows you how to train the TD(λ) agent with 2-ply lookahead, starting with small examples and scaling up.

## What is TD(λ)?

TD(λ) (Temporal Difference with eligibility traces) is a value-based reinforcement learning method that:
- Learns a **value function** (how good positions are)
- Uses **eligibility traces** to update past states based on future rewards
- Uses **2-ply lookahead** during action selection (considers opponent responses)
- Updates parameters using gradient descent on TD errors

## Quick Start: Minimal Test Run

### Step 1: Very Small Test (Verify It Works)

```bash
# Test with minimal settings (should complete in ~2-5 minutes)
python train_agent2_tdlambda_2ply.py \
    --steps 100 \
    --log_every 20 \
    --save_every 50 \
    --eval_batch 512
```

**What this does:**
- `--steps 100`: Only 100 training steps (very quick)
- `--log_every 20`: Print progress every 20 steps
- `--save_every 50`: Save checkpoint every 50 steps (frequent saves)
- `--eval_batch 512`: Smaller batch for 2-ply evaluation (faster, less memory)

**Expected output:**
```
[step     20/100] td_err2=0.123456  steps/s=2.34  progress=20.0%  elapsed=8.5min
  ETA: 34.0 min
[step     40/100] td_err2=0.112345  steps/s=2.45  progress=40.0%  elapsed=16.3min
  ETA: 24.5 min
  💾 CHECKPOINT SAVED: /path/to/checkpoints_tdlambda_2ply/checkpoint_50 @ step 50
...
Training complete. Saved final checkpoint to: /path/to/checkpoints_tdlambda_2ply/checkpoint_100
```

### Step 2: Small Training Run (30-60 minutes)

```bash
# Small but meaningful training run
python train_agent2_tdlambda_2ply.py \
    --steps 1000 \
    --log_every 100 \
    --save_every 200 \
    --eval_batch 1024
```

### Step 3: Medium Training Run (2-4 hours)

```bash
# Medium training run
python train_agent2_tdlambda_2ply.py \
    --steps 10000 \
    --log_every 500 \
    --save_every 2000 \
    --eval_batch 2048
```

### Step 4: Full Training Run (Recommended)

```bash
# Full training with default hyperparameters
python train_agent2_tdlambda_2ply.py \
    --steps 200000 \
    --log_every 200 \
    --save_every 2000 \
    --eval_batch 2048
```

## Understanding the Hyperparameters

### Core Training Parameters

| Parameter | Default | Description | Small Example |
|-----------|---------|-------------|---------------|
| `--steps` | 200000 | Number of training steps | 100-1000 |
| `--alpha` | 1e-3 | Learning rate (step size) | 1e-3 |
| `--gamma` | 1.0 | Discount factor | 1.0 |
| `--lam` | 0.85 | TD(λ) lambda (trace decay) | 0.85 |

### Performance Parameters

| Parameter | Default | Description | Small Example |
|-----------|---------|-------------|---------------|
| `--eval_batch` | 2048 | Batch size for 2-ply evaluation | 512-1024 |
| `--log_every` | 200 | Print progress every N steps | 20-100 |
| `--save_every` | 2000 | Save checkpoint every N steps | 50-200 |
| `--seed` | 0 | Random seed | 0 |

## Training Progress Interpretation

### What to Look For

**Good signs:**
- **td_err2 decreasing**: TD error squared is decreasing (agent learning)
- **steps/s increasing**: JIT compilation finished, now running fast
- **Progress percentage increasing**: Training is advancing

**Warning signs:**
- **td_err2 not decreasing**: Learning rate may be too low or too high
- **Very slow steps/s (< 1)**: May be stuck in JIT compilation or OOM
- **No progress after many steps**: May need to adjust hyperparameters

### Example Output Interpretation

```
[step    200/200000] td_err2=0.123456  steps/s=2.34  progress=0.1%  elapsed=85.5min
  ETA: 142.3 hr (8538.0 min)
```

- `td_err2=0.123456`: TD error squared (lower is better)
- `steps/s=2.34`: Steps per second (speed indicator)
- `progress=0.1%`: Training progress percentage
- `elapsed=85.5min`: Time elapsed so far
- `ETA: 142.3 hr`: Estimated time remaining

## Training Workflow

### 1. Local Quick Test (CPU/GPU)

```bash
# Test on your local machine first
python train_agent2_tdlambda_2ply.py --steps 100 --log_every 20 --save_every 50
```

### 2. Server Training (GPU Recommended)

```bash
# On Delta/Jetstream2, run in background
nohup python train_agent2_tdlambda_2ply.py \
    --steps 200000 \
    --log_every 200 \
    --save_every 2000 \
    --eval_batch 2048 \
    > tdlambda_training.log 2>&1 &

# Monitor progress
tail -f tdlambda_training.log

# Check GPU usage (if on GPU node)
nvidia-smi

# Check if process is still running
ps aux | grep train_agent2_tdlambda_2ply
```

### 3. Monitor Training

```bash
# Watch the log file
tail -f tdlambda_training.log

# Check latest checkpoint
ls -lh checkpoints_tdlambda_2ply/

# Check training progress
grep "progress=" tdlambda_training.log | tail -5
```

## Checkpoint Management

### Checkpoints Location

All checkpoints are saved to: `checkpoints_tdlambda_2ply/`

```
checkpoints_tdlambda_2ply/
  ├── checkpoint_2000.orbax-checkpoint/
  ├── checkpoint_4000.orbax-checkpoint/
  ├── checkpoint_6000.orbax-checkpoint/
  └── ...
```

### Checkpoint Frequency Recommendations

**For short runs (< 1 hour):**
```bash
--save_every 50  # Save every 50 steps
```

**For medium runs (1-4 hours):**
```bash
--save_every 200  # Save every 200 steps
```

**For long runs (> 4 hours):**
```bash
--save_every 2000  # Save every 2000 steps (default)
```

### Loading Checkpoints

To load a checkpoint for evaluation:

```python
from flax.training import checkpoints
from backgammon_value_net import BackgammonValueNet
import jax
import jax.numpy as jnp

# Load checkpoint
key = jax.random.PRNGKey(0)
dummy_board = jnp.zeros((1, 24, 15), dtype=jnp.float32)
dummy_aux = jnp.zeros((1, 6), dtype=jnp.float32)
params = BackgammonValueNet().init(key, board_state=dummy_board, aux_features=dummy_aux)["params"]

restored = checkpoints.restore_checkpoint("checkpoints_tdlambda_2ply", target=params, step=2000)
```

## Troubleshooting

### Issue: "Out of Memory" Error

**Solution:** Reduce eval_batch size
```bash
python train_agent2_tdlambda_2ply.py --eval_batch 512 --steps 1000
```

### Issue: Training is Very Slow

**Causes:**
1. JIT compilation (first step takes 5-30 minutes) - **This is normal!**
2. 2-ply lookahead is computationally expensive
3. CPU instead of GPU - check `jax.devices()`

**Solution:**
```bash
# Check if GPU is available
python -c "import jax; print(jax.devices())"

# Reduce eval_batch for faster training (but may use more memory)
python train_agent2_tdlambda_2ply.py --eval_batch 1024
```

### Issue: No Learning (td_err2 Not Decreasing)

**Possible causes:**
1. Learning rate too low - try `--alpha 5e-3`
2. Learning rate too high - try `--alpha 5e-4`
3. Need more training - run more steps
4. Lambda too high/low - try `--lam 0.7` or `--lam 0.9`

## Recommended Training Schedule

### Week 1: Exploration (Small Runs)

```bash
# Day 1: Quick test
python train_agent2_tdlambda_2ply.py --steps 100 --log_every 20 --save_every 50

# Day 2-3: Small training
python train_agent2_tdlambda_2ply.py --steps 1000 --log_every 100 --save_every 200

# Day 4-7: Medium training
python train_agent2_tdlambda_2ply.py --steps 10000 --log_every 500 --save_every 2000
```

### Week 2+: Full Training

```bash
# Full training run
python train_agent2_tdlambda_2ply.py \
    --steps 200000 \
    --log_every 200 \
    --save_every 2000 \
    --eval_batch 2048
```

## Evaluation

After training, evaluate your TD(λ) agent:

```bash
# Compare against random baseline
python compare_checkpoint_steps.py checkpoints_tdlambda_2ply --random --games 500

# Compare two TD(λ) checkpoints
python compare_checkpoint_steps.py checkpoints_tdlambda_2ply checkpoints_2ply --games 1000

# Compare against other methods
python compare_checkpoint_steps.py checkpoints_tdlambda_2ply checkpoints_ppo --games 1000
```

## Example: Complete Small Training Session

```bash
# 1. Quick test (2-5 minutes)
python train_agent2_tdlambda_2ply.py \
    --steps 100 \
    --log_every 20 \
    --save_every 50 \
    --eval_batch 512

# 2. Small training (30-60 minutes)
python train_agent2_tdlambda_2ply.py \
    --steps 1000 \
    --log_every 100 \
    --save_every 200 \
    --eval_batch 1024

# 3. Evaluate the result
python compare_checkpoint_steps.py checkpoints_tdlambda_2ply --random --games 100

# 4. If results look good, scale up to full training
python train_agent2_tdlambda_2ply.py \
    --steps 200000 \
    --log_every 200 \
    --save_every 2000 \
    --eval_batch 2048
```

## Important Notes

1. **First step is slow**: JIT compilation takes 5-30 minutes - this is normal!
2. **2-ply is expensive**: Each action selection evaluates many states (your move × 21 dice × opponent moves)
3. **Checkpoint frequently**: Use `--save_every` to save often, especially for long runs
4. **Monitor progress**: Watch `td_err2` to see if learning is happening
5. **GPU recommended**: 2-ply lookahead benefits greatly from GPU acceleration

## Quick Reference

### Minimal Test (2-5 min)
```bash
python train_agent2_tdlambda_2ply.py --steps 100 --log_every 20 --save_every 50
```

### Small Run (30-60 min)
```bash
python train_agent2_tdlambda_2ply.py --steps 1000 --log_every 100 --save_every 200
```

### Medium Run (2-4 hours)
```bash
python train_agent2_tdlambda_2ply.py --steps 10000 --log_every 500 --save_every 2000
```

### Full Run (days)
```bash
python train_agent2_tdlambda_2ply.py --steps 200000 --log_every 200 --save_every 2000
```

Good luck with your TD(λ) training! 🎲

