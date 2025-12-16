#!/bin/bash
# Local script to prepare files for Delta transfer
# Run this on your local machine before transferring to Delta

set -euo pipefail

echo "=== Preparing files for Delta ==="
echo ""

# Check if required files exist
REQUIRED_FILES=("smoke_jax_gpu.py" "smoke_gpu.slurm" "setup_delta.sh" "backgammon_value_net.py" "backgammon_ppo_net.py" "backgammon_engine.py")

echo "Checking required files..."
for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "✓ $file"
    else
        echo "✗ $file (MISSING)"
        exit 1
    fi
done

echo ""
echo "All required files present!"
echo ""
echo "Next steps:"
echo "1. SSH to Delta: ssh jyan10@login.delta.ncsa.illinois.edu"
echo "2. Navigate to your project directory"
echo "3. Clone the repo OR transfer these files"
echo "4. Run: chmod +x setup_delta.sh && ./setup_delta.sh"
echo "5. Submit: sbatch smoke_gpu.slurm"

