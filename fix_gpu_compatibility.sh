#!/bin/bash
# Comprehensive GPU compatibility fix script for Delta
# This script tries multiple approaches to get JAX GPU working

set -euo pipefail

echo "=== GPU Compatibility Diagnostic ==="
echo ""

# 1. Check CUDA modules
echo "1. Checking CUDA modules..."
module list | grep -i cuda || echo "  No CUDA modules loaded"
echo ""

# 2. Check CUDA installation
echo "2. Checking CUDA installation..."
if command -v nvcc &> /dev/null; then
    echo "  nvcc found: $(which nvcc)"
    nvcc --version | head -3
    CUDA_ROOT=$(dirname $(dirname $(which nvcc)))
    echo "  CUDA_ROOT: $CUDA_ROOT"
else
    echo "  nvcc not found in PATH"
fi
echo ""

# 3. Check NVIDIA driver
echo "3. Checking NVIDIA driver..."
nvidia-smi --query-gpu=driver_version,cuda_version --format=csv,noheader || echo "  nvidia-smi failed"
echo ""

# 4. Check LD_LIBRARY_PATH
echo "4. Current LD_LIBRARY_PATH:"
echo "  $LD_LIBRARY_PATH" | tr ':' '\n' | head -10
echo ""

# 5. Try to find cuDNN
echo "5. Searching for cuDNN libraries..."
find /opt/cray /usr/local /opt -name "*cudnn*" -type f 2>/dev/null | head -5 || echo "  No cuDNN found in common locations"
echo ""

echo "=== Recommendations ==="
echo ""
echo "Based on the diagnostics above, try:"
echo ""
echo "Option A: Unload cudatoolkit and use cuda/12.8 module"
echo "  module unload cudatoolkit"
echo "  module load cuda/12.8"
echo "  # Then reinstall JAX"
echo ""
echo "Option B: Use JAX with system CUDA libraries"
echo "  # Set CUDA paths before installing JAX"
echo "  export CUDA_PATH=\$CUDA_ROOT"
echo "  export LD_LIBRARY_PATH=\$CUDA_ROOT/lib64:\$LD_LIBRARY_PATH"
echo ""
echo "Option C: Try JAX nightly build (might have better compatibility)"
echo "  pip install --upgrade 'jax[cuda12] @ git+https://github.com/google/jax.git'"
echo ""

