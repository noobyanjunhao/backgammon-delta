# How to Run `2ply_rev.py`

## 🚀 **Running the Script**

### Basic Usage:
```bash
# Run with defaults (1-ply, 200k steps, batch 4096)
python 2ply_rev.py

# Run with 1-ply search (fast)
python 2ply_rev.py --train_search 1ply

# Run with 2-ply search (slow but stronger)
python 2ply_rev.py --train_search 2ply

# Custom parameters
python 2ply_rev.py \
    --train_search 2ply \
    --steps 50000 \
    --batch 256 \
    --eval_batch 4096 \
    --lr 3e-4 \
    --eps_greedy 0.05 \
    --seed 0
```

### On HPC (Delta) with Slurm:
```bash
# Create a Slurm script (e.g., train_2ply.slurm)
sbatch train_2ply.slurm
```

---

## 📊 **Progress Messages**

The script prints progress updates:

### 1. **Startup Messages:**
```
======================================================================
Training mode: 2ply
Steps: 200000, Batch: 4096, Eval batch: 4096
Learning rate: 0.0003, Epsilon-greedy: 0.05
======================================================================

⚠️  WARNING: 2-ply search is much slower but produces stronger play.
   First run will trigger JAX JIT compilation (5-30 minutes).
```

### 2. **Progress Updates (every 5000 steps):**
```
[5000/200000] loss=0.123456 rate=12.34 steps/s ETA: 13.2 min
[10000/200000] loss=0.098765 rate=12.45 steps/s ETA: 12.8 min
[15000/200000] loss=0.087654 rate=12.50 steps/s ETA: 12.3 min
...
```

**Progress includes:**
- Current step / total steps
- Latest loss value
- Training rate (steps per second)
- Estimated time remaining (ETA)

### 3. **Completion Message:**
```
Training complete. Saved checkpoint to: /path/to/checkpoints/checkpoint_200000.orbax-checkpoint
```

---

## 💾 **Where Results Are Saved**

### Checkpoint Location:
- **Folder**: `checkpoints/` (in the current working directory)
- **Full path**: Absolute path to `checkpoints/` directory
- **Format**: Orbax checkpoint format
- **Filename**: `checkpoint_<step_number>.orbax-checkpoint`

### Example:
```bash
# If you run from /Users/junhaoyan/backgammon/
# Checkpoints saved to:
/Users/junhaoyan/backgammon/checkpoints/checkpoint_200000.orbax-checkpoint
```

### ⚠️ **Important Notes:**

1. **Only saves at the END** - The script currently only saves checkpoints when training completes (line 354-359). It does NOT save periodically during training.

2. **No intermediate checkpoints** - If training crashes or is interrupted, you lose progress.

3. **Overwrites by default** - Uses `overwrite=True`, so it will overwrite existing checkpoints.

---

## 🔧 **Recommended Improvements**

To add periodic checkpoint saving (like `train_value_td0_2ply.py`), you could add:

```python
# Inside the training loop, after line 337:
if step % 1000 == 0 and step > 0:
    ckpt_dir = os.path.abspath("checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = checkpoints.save_checkpoint(
        ckpt_dir,
        st.params,
        step=step,
        overwrite=True,
    )
    print(f"💾 Checkpoint saved at step {step}: {ckpt_path}")
```

---

## 📝 **Command Line Arguments**

| Argument | Default | Description |
|----------|---------|-------------|
| `--steps` | 200000 | Number of training steps |
| `--batch` | 4096 | Batch size for gradient updates |
| `--lr` | 3e-4 | Learning rate |
| `--train_search` | 1ply | Search depth: `1ply` or `2ply` |
| `--eval_batch` | 4096 | Batch size for 2-ply evaluation (reduce if OOM) |
| `--eps_greedy` | 0.05 | Epsilon for random exploration |
| `--seed` | 0 | Random seed |

---

## 🎯 **Example Commands**

### Quick test (1-ply, 1000 steps):
```bash
python 2ply_rev.py --train_search 1ply --steps 1000 --batch 256
```

### Full training (2-ply, 50k steps):
```bash
python 2ply_rev.py \
    --train_search 2ply \
    --steps 50000 \
    --batch 256 \
    --eval_batch 4096 \
    --eps_greedy 0.05
```

### Save output to log file:
```bash
python 2ply_rev.py --train_search 2ply > training.log 2>&1
```

### Run in background (local):
```bash
nohup python 2ply_rev.py --train_search 2ply > training.log 2>&1 &
```

---

## 📂 **Checkpoint Structure**

The checkpoint contains:
- **Model parameters**: All neural network weights
- **Step number**: Training step when saved
- **Format**: Orbax checkpoint (Flax's recommended format)

To load checkpoints later, use:
```python
from flax.training import checkpoints
import orbax.checkpoint as ocp

# Load checkpoint
ckpt_dir = "checkpoints"
restored = checkpoints.restore_checkpoint(ckpt_dir, target=params)
```

