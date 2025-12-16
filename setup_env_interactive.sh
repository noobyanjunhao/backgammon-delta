#!/bin/bash
# Setup script to run in an interactive GPU session (srun)
# This builds the environment ONCE, then batch jobs reuse it

set -euo pipefail

echo "=== Building JAX GPU Environment ==="
echo "This should be run in an interactive GPU session (srun)"
echo ""

# Reset modules and load Python
module reset
module load anaconda3_gpu || {
    echo "Warning: anaconda3_gpu not found, trying alternatives..."
    module load python || module load python/3.11 || echo "Using system Python"
}
module list

# Use SCRATCH for the environment (faster I/O)
ENV_DIR=${SCRATCH:-$HOME}/bg
mkdir -p $ENV_DIR
cd $ENV_DIR

echo "Environment directory: $ENV_DIR"
echo ""

# Create venv
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

source venv/bin/activate
pip install -U pip

echo "Installing JAX with CUDA 12 support..."
# Clean install: GPU JAX + dependencies
pip install -U "jax[cuda12]" flax optax numba

echo ""
echo "=== Testing Installation ==="
python -c "import jax; print('JAX version:', jax.__version__); print('Devices:', jax.devices())"

echo ""
echo "=== Setup Complete ==="
echo "Environment location: $ENV_DIR/venv"
echo ""
echo "Next steps:"
echo "1. Test GPU: python smoke_jax_gpu_min.py"
echo "2. Use this env in batch jobs: source $ENV_DIR/venv/bin/activate"

