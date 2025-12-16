#!/bin/bash
# Verification script to run on Delta after setup
# Checks that everything is configured correctly

set -euo pipefail

echo "=== Checking Delta Setup ==="
echo ""

# Check account
echo "1. Checking account configuration..."
if grep -q "ACCOUNT_NAME" smoke_gpu.slurm 2>/dev/null; then
    echo "   ✗ WARNING: smoke_gpu.slurm still contains ACCOUNT_NAME placeholder"
    echo "   Run ./setup_delta.sh to fix this"
else
    ACCOUNT=$(grep -oP '--account=\K\S+' smoke_gpu.slurm | head -1)
    echo "   ✓ Account set to: $ACCOUNT"
fi

# Check required files
echo ""
echo "2. Checking required files..."
FILES=("smoke_jax_gpu.py" "smoke_gpu.slurm" "backgammon_value_net.py" "backgammon_ppo_net.py" "backgammon_engine.py")
ALL_PRESENT=true
for file in "${FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "   ✓ $file"
    else
        echo "   ✗ $file (MISSING)"
        ALL_PRESENT=false
    fi
done

# Check Python
echo ""
echo "3. Checking Python availability..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version)
    echo "   ✓ $PYTHON_VERSION"
else
    echo "   ✗ Python3 not found"
    ALL_PRESENT=false
fi

# Check if we're on a compute node or login node
echo ""
echo "4. Checking node type..."
if [ -n "${SLURM_JOB_ID:-}" ]; then
    echo "   ✓ Running on compute node (Job ID: $SLURM_JOB_ID)"
    echo "   ✓ Partition: ${SLURM_JOB_PARTITION:-unknown}"
else
    echo "   ℹ Running on login node (this is expected before submitting jobs)"
fi

echo ""
if [ "$ALL_PRESENT" = true ]; then
    echo "=== Setup looks good! ==="
    echo ""
    echo "Ready to submit: sbatch smoke_gpu.slurm"
else
    echo "=== Setup incomplete ==="
    echo "Please fix the issues above before submitting jobs"
fi

