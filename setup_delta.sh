#!/bin/bash
# Setup script to run on Delta after cloning the repo
# This script will:
# 1. Find your account name
# 2. Update the Slurm script with your account
# 3. Verify the setup

set -euo pipefail

echo "=== Delta Backgammon Setup ==="
echo ""

# Find account name
echo "Finding your account name..."
ACCOUNT=$(accounts | head -n 1 | awk '{print $1}')
if [ -z "$ACCOUNT" ]; then
    echo "ERROR: No account found. Run 'accounts' manually to see available accounts."
    exit 1
fi

echo "Found account: $ACCOUNT"
echo ""

# Update Slurm script
if [ -f "smoke_gpu.slurm" ]; then
    echo "Updating smoke_gpu.slurm with account name..."
    sed -i "s/--account=ACCOUNT_NAME/--account=$ACCOUNT/" smoke_gpu.slurm
    echo "✓ Updated smoke_gpu.slurm"
else
    echo "WARNING: smoke_gpu.slurm not found in current directory"
fi

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Next steps:"
echo "1. Review smoke_gpu.slurm to ensure account is set correctly"
echo "2. Submit the job: sbatch smoke_gpu.slurm"
echo "3. Monitor: squeue -u \$USER"
echo "4. Check output: tail -f slurm-<job_id>.out"

