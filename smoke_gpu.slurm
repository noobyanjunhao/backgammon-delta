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
export CUDA_ROOT=/opt/cray/pe/cuda/12.8
export LD_LIBRARY_PATH=${CUDA_ROOT}/lib64:${LD_LIBRARY_PATH:-}
echo "CUDA paths set: CUDA_ROOT=$CUDA_ROOT"

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

# JAX GPU install - The driver is CUDA 12.8, JAX CUDA 12.9 has version mismatch
# Try older JAX version that might be more compatible, with compatible NumPy
echo "Installing JAX 0.4.23 with compatible NumPy..."
pip install "jax[cuda12]==0.4.23" "jaxlib==0.4.23" "flax" "optax" || {
    echo "JAX 0.4.23 failed, trying latest with NumPy 2.x..."
    pip install -U "numpy>=2.0"
    pip install -U "jax[cuda12]" "flax" "optax"
}

# Try GPU version first
echo "Attempting GPU test..."
python smoke_jax_gpu.py || {
    echo "GPU test failed, trying CPU fallback..."
    python smoke_jax_cpu.py
}

