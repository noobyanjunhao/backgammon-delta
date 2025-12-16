#!/bin/bash
# Automated setup script for Delta with account bfxm-delta-gpu
# Run this on Delta after cloning the repo

set -euo pipefail

ACCOUNT="bfxm-delta-gpu"
echo "=== Delta Backgammon Setup ==="
echo "Using account: $ACCOUNT"
echo ""

# Check if we're in the right directory
if [ ! -f "smoke_jax_gpu.py" ]; then
    echo "ERROR: smoke_jax_gpu.py not found. Are you in the backgammon directory?"
    exit 1
fi

# Update Slurm script with account
if [ -f "smoke_gpu.slurm" ]; then
    echo "Updating smoke_gpu.slurm with account name..."
    sed -i "s/--account=ACCOUNT_NAME/--account=$ACCOUNT/" smoke_gpu.slurm
    sed -i "s/#SBATCH --account=ACCOUNT_NAME/#SBATCH --account=$ACCOUNT/" smoke_gpu.slurm
    echo "✓ Updated smoke_gpu.slurm"
else
    echo "WARNING: smoke_gpu.slurm not found"
fi

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Ready to submit: sbatch smoke_gpu.slurm"

