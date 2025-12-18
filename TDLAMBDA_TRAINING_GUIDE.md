# TD(λ) Training Guide


##Minimal Test Run

### Step 1: Very Small Test (Verify It Works)

```bash
# Test with minimal settings (should complete in ~2-5 minutes)
python agent2/train_agent2.py \
    --steps 100 \
    --log_every 20 \
    --save_every 50 \
    --eval_batch 512


### Monitor Training

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

### Loading Checkpoints

To load a checkpoint for evaluation:

```python
from flax.training import checkpoints
from agent2.agent2_value_net import Agent2ValueNet
import jax
import jax.numpy as jnp

# Load checkpoint
key = jax.random.PRNGKey(0)
dummy_board = jnp.zeros((1, 24, 15), dtype=jnp.float32)
dummy_aux = jnp.zeros((1, 6), dtype=jnp.float32)
params = Agent2ValueNet().init(key, board_state=dummy_board, aux_features=dummy_aux)["params"]

restored = checkpoints.restore_checkpoint("checkpoints_tdlambda_2ply", target=params, step=2000)
```

## Evaluation

```bash
# Compare against random baseline
python compare_checkpoint_steps.py checkpoints_tdlambda_2ply --random --games 500

## Important Notes

1. **First step is slow**: JIT compilation takes 5-30 minutes - this is normal!
2. **2-ply is expensive**: Each action selection evaluates many states (your move × 21 dice × opponent moves)
3. **Checkpoint frequently**: Use `--save_every` to save often, especially for long runs
4. **Monitor progress**: Watch `td_err2` to see if learning is happening


