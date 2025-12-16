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

# Use system Python3 (should be available by default)
echo "Python version:"
python3 --version

# (1) Create env once interactively, OR do it here for a first smoke test:
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip

# JAX GPU install - try CUDA 11 first for better driver compatibility
# The driver is CUDA 12.8, but JAX CUDA 12.9 causes segfaults
# CUDA 11 should work with forward compatibility
echo "Installing JAX with CUDA 11 (better compatibility with CUDA 12.8 driver)..."
pip install -U "jax[cuda11_local]" "flax" "optax" || {
    echo "CUDA 11 install failed, trying CUDA 12..."
    pip install -U "jax[cuda12]" "flax" "optax"
}

# Try GPU version first
echo "Attempting GPU test..."
python smoke_jax_gpu.py || {
    echo "GPU test failed, trying CPU fallback..."
    python smoke_jax_cpu.py
}

