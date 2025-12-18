# How to Compare Checkpoints on Server

## Quick Start

### Option 1: Use the helper script
```bash
# Compare step 5000 vs 200000 from 1-ply training
./compare_on_server.sh 1ply 5000 200000 1000

# Compare step 5000 vs 200000 from 2-ply training
./compare_on_server.sh 2ply 5000 200000 1000
```

### Option 2: Direct Python command
```bash
# Compare step 5000 vs 200000 from 1-ply training
python compare_checkpoint_steps.py --method 1ply --step1 5000 --step2 200000 --games 1000

# Compare step 5000 vs 200000 from 2-ply training
python compare_checkpoint_steps.py --method 2ply --step1 5000 --step2 200000 --games 1000
```

## Check What Checkpoints Exist

First, check what checkpoints you have:

```bash
# List checkpoint directories
ls -d checkpoints_*

# List checkpoints in a directory
ls -lh checkpoints_1ply/
ls -lh checkpoints_2ply/
```

## Examples

### Compare two specific steps:
```bash
python compare_checkpoint_steps.py \
    --method 1ply \
    --step1 5000 \
    --step2 200000 \
    --games 1000 \
    --runs 3
```

### Compare against random baseline:
```bash
python compare_checkpoint_steps.py \
    --method 1ply \
    --step1 5000 \
    --games 500 \
    --vs_random
```

### Compare with more games (more accurate):
```bash
python compare_checkpoint_steps.py \
    --method 1ply \
    --step1 5000 \
    --step2 200000 \
    --games 5000 \
    --runs 5
```

## Output

The script will show:
- Win rates for each checkpoint
- Average game length
- Statistical significance (if scipy available)
- Which checkpoint is stronger

## Notes

- Make sure you're in the directory containing the checkpoints
- The script looks for checkpoints in `checkpoints_1ply/` or `checkpoints_2ply/`
- More games = more accurate but slower
- Multiple runs help account for randomness

