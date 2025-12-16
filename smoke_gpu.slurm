#!/bin/bash
#SBATCH --job-name=bg_smoke
#SBATCH --account=bfxm-delta-gpu
#SBATCH --partition=gpuA100x4-interactive
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:10:00

#SBATCH --gpus-per-node=1
#SBATCH --gpus-per-task=1

set -euo pipefail
echo "Running on $(hostname)"
nvidia-smi || true

# Check and use system CUDA modules
echo "Checking CUDA modules..."
module list | grep -i cuda || echo "No CUDA modules loaded"
echo "Available CUDA modules:"
module avail cuda 2>&1 | grep -E "cuda|^--" | head -10 || true

# Set CUDA library paths (cudatoolkit is already loaded)
# Try to find CUDA path dynamically and add cuDNN paths
if command -v nvcc &> /dev/null; then
    CUDA_ROOT=$(dirname $(dirname $(which nvcc)))
    export CUDA_ROOT
    # Add CUDA lib64 and common cuDNN locations
    export LD_LIBRARY_PATH=${CUDA_ROOT}/lib64:${LD_LIBRARY_PATH:-}
    # Try common cuDNN locations
    for cudnn_path in ${CUDA_ROOT}/lib64 ${CUDA_ROOT}/targets/x86_64-linux/lib /usr/local/cuda/lib64; do
        if [ -d "$cudnn_path" ]; then
            export LD_LIBRARY_PATH=${cudnn_path}:${LD_LIBRARY_PATH}
        fi
    done
    echo "CUDA paths set: CUDA_ROOT=$CUDA_ROOT"
    echo "LD_LIBRARY_PATH includes: $(echo $LD_LIBRARY_PATH | tr ':' '\n' | head -5)"
elif [ -d "/opt/cray/pe/cuda" ]; then
    export CUDA_ROOT=/opt/cray/pe/cuda/12.8
    export LD_LIBRARY_PATH=${CUDA_ROOT}/lib64:${LD_LIBRARY_PATH:-}
    echo "CUDA paths set (default): CUDA_ROOT=$CUDA_ROOT"
else
    echo "Warning: Could not find CUDA installation"
fi

# Use system Python3 (should be available by default)
echo "Python version:"
python3 --version

# (1) Create env once interactively, OR do it here for a first smoke test:
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip

# Check CUDA version
echo "Checking CUDA version..."
nvcc --version 2>/dev/null || echo "nvcc not in PATH"
nvidia-smi --query-gpu=driver_version --format=csv,noheader || true

# Install compatible NumPy first (JAX 0.4.23 needs NumPy < 2.0)
echo "Installing compatible NumPy version..."
pip install "numpy<2.0"

# JAX GPU install - Try latest JAX first (better cuDNN handling)
# Latest JAX has better cuDNN initialization and error handling
echo "Installing latest JAX (better cuDNN support)..."
pip install -U "numpy>=2.0"
pip install -U "jax[cuda12]" "flax" "optax" || {
    echo "Latest JAX failed, trying JAX 0.4.23 with NumPy < 2.0..."
    pip install "numpy<2.0"
    pip install "jax[cuda12]==0.4.23" "jaxlib==0.4.23" "flax" "optax"
}

# Try GPU version first
echo "Attempting GPU test..."
python smoke_jax_gpu.py || {
    echo "GPU test failed, trying CPU fallback..."
    python smoke_jax_cpu.py
}

