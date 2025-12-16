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
echo "=== CUDA Module Diagnostics ==="
module list | grep -i cuda || echo "No CUDA modules loaded"
echo "Available CUDA modules:"
module avail cuda 2>&1 | grep -E "cuda|^--" | head -10 || true

# Try to use cuda/12.8 module if cudatoolkit is causing issues
echo ""
echo "Attempting to configure CUDA modules for compatibility..."
if module list 2>&1 | grep -q "cudatoolkit"; then
    echo "cudatoolkit is loaded. Trying to switch to cuda/12.8 module..."
    module unload cudatoolkit 2>/dev/null || true
    module load cuda/12.8 2>/dev/null || echo "Could not load cuda/12.8, continuing with current setup"
fi

# Set CUDA library paths - try multiple methods
echo "=== Setting CUDA Library Paths ==="
CUDA_FOUND=false

# Method 1: Use nvcc to find CUDA
if command -v nvcc &> /dev/null; then
    CUDA_ROOT=$(dirname $(dirname $(which nvcc)))
    export CUDA_ROOT
    export CUDA_PATH=$CUDA_ROOT
    export LD_LIBRARY_PATH=${CUDA_ROOT}/lib64:${LD_LIBRARY_PATH:-}
    CUDA_FOUND=true
    echo "✓ Found CUDA via nvcc: $CUDA_ROOT"
fi

# Method 2: Check module-provided CUDA
if [ -n "${CUDA_HOME:-}" ]; then
    export CUDA_ROOT=$CUDA_HOME
    export CUDA_PATH=$CUDA_HOME
    export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}
    CUDA_FOUND=true
    echo "✓ Found CUDA via CUDA_HOME: $CUDA_HOME"
fi

# Method 3: Try default Delta CUDA location
if [ "$CUDA_FOUND" = false ] && [ -d "/opt/cray/pe/cuda/12.8" ]; then
    export CUDA_ROOT=/opt/cray/pe/cuda/12.8
    export CUDA_PATH=$CUDA_ROOT
    export LD_LIBRARY_PATH=${CUDA_ROOT}/lib64:${LD_LIBRARY_PATH:-}
    CUDA_FOUND=true
    echo "✓ Using default CUDA path: $CUDA_ROOT"
fi

# Add common library paths
for lib_path in /usr/local/cuda/lib64 /opt/cuda/lib64; do
    if [ -d "$lib_path" ]; then
        export LD_LIBRARY_PATH=${lib_path}:${LD_LIBRARY_PATH}
        echo "  Added to LD_LIBRARY_PATH: $lib_path"
    fi
done

if [ "$CUDA_FOUND" = true ]; then
    echo "CUDA configuration:"
    echo "  CUDA_ROOT=$CUDA_ROOT"
    echo "  CUDA_PATH=$CUDA_PATH"
    echo "  LD_LIBRARY_PATH (first 5): $(echo $LD_LIBRARY_PATH | tr ':' '\n' | head -5 | tr '\n' ':')"
else
    echo "⚠ Warning: Could not find CUDA installation"
fi
echo ""

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

# JAX GPU install - Try multiple strategies
echo "=== Installing JAX with GPU Support ==="
pip install -U "numpy>=2.0"

# Strategy 1: Try JAX with system CUDA (uses system libraries)
echo "Strategy 1: JAX with system CUDA libraries..."
export JAX_PLATFORMS=cuda
pip install -U "jax[cuda12]" "flax" "optax" && JAX_INSTALLED=true || JAX_INSTALLED=false

# Strategy 2: If that fails, try CUDA 11 (forward compatible)
if [ "$JAX_INSTALLED" = false ]; then
    echo "Strategy 2: JAX with CUDA 11 (forward compatible)..."
    pip install -U "jax[cuda11_local]" "flax" "optax" && JAX_INSTALLED=true || JAX_INSTALLED=false
fi

# Strategy 3: Try older JAX version with compatible NumPy
if [ "$JAX_INSTALLED" = false ]; then
    echo "Strategy 3: JAX 0.4.23 with NumPy < 2.0..."
    pip install "numpy<2.0"
    pip install "jax[cuda12]==0.4.23" "jaxlib==0.4.23" "flax" "optax" && JAX_INSTALLED=true || JAX_INSTALLED=false
fi

# Strategy 4: Try JAX nightly (might have better compatibility)
if [ "$JAX_INSTALLED" = false ]; then
    echo "Strategy 4: JAX nightly build (experimental)..."
    pip install -U "numpy>=2.0"
    pip install --upgrade "jax[cuda12] @ git+https://github.com/google/jax.git" "flax" "optax" && JAX_INSTALLED=true || JAX_INSTALLED=false
fi

if [ "$JAX_INSTALLED" = true ]; then
    echo "✓ JAX installed successfully"
else
    echo "⚠ All JAX installation strategies failed"
fi
echo ""

# Try GPU version first
echo "Attempting GPU test..."
python smoke_jax_gpu.py || {
    echo "GPU test failed, trying CPU fallback..."
    python smoke_jax_cpu.py
}

