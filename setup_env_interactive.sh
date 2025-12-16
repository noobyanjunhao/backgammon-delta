#!/bin/bash
# Setup script to run in an interactive GPU session (srun)
# This builds the environment ONCE, then batch jobs reuse it

set -euo pipefail

echo "=== Building JAX GPU Environment ==="
echo "This should be run in an interactive GPU session (srun)"
echo ""

# Reset modules and load Python ONLY
# DO NOT load CUDA modules - JAX uses its bundled CUDA libraries
module reset

# Load Python module only
module load python 2>/dev/null || {
    echo "Warning: python module not found, trying alternatives..."
    module load python/3.11 2>/dev/null || module load anaconda3_gpu 2>/dev/null || echo "Using system Python"
}
module list

# Use SCRATCH for the environment (faster I/O)
# Try to find SCRATCH, fallback to home
if [ -z "${SCRATCH:-}" ]; then
    # Try common Delta scratch locations
    if [ -d "/scratch/bfxm/${USER}" ]; then
        export SCRATCH="/scratch/bfxm/${USER}"
    elif [ -d "/scratch/${USER}" ]; then
        export SCRATCH="/scratch/${USER}"
    else
        export SCRATCH="$HOME"
    fi
    echo "Set SCRATCH to: $SCRATCH"
else
    echo "SCRATCH is already set to: $SCRATCH"
fi

ENV_DIR=$SCRATCH/bg
echo "Environment will be created at: $ENV_DIR"
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
# Use --no-cache-dir to avoid issues
pip install -U --no-cache-dir "jax[cuda12]" flax optax numba

echo ""
echo "Verifying CUDA libraries are accessible..."
python -c "import os; print('CUDA_HOME:', os.environ.get('CUDA_HOME', 'Not set')); print('LD_LIBRARY_PATH (first 3):', ':'.join(os.environ.get('LD_LIBRARY_PATH', '').split(':')[:3]))"

echo ""
echo "=== Testing Installation ==="
# Check CUDA library paths
echo "CUDA environment check:"
python << 'EOF'
import os
print("CUDA_HOME:", os.environ.get('CUDA_HOME', 'Not set'))
print("LD_LIBRARY_PATH:", os.environ.get('LD_LIBRARY_PATH', 'Not set')[:200] if os.environ.get('LD_LIBRARY_PATH') else 'Not set')
# Check for cuSPARSE library
import ctypes
import sys
lib_paths = os.environ.get('LD_LIBRARY_PATH', '').split(':')
lib_paths.extend(['/usr/lib64', '/lib64', '/opt/cray/pe/cuda/12.8/lib64'])
for path in lib_paths:
    if path and os.path.exists(path):
        cusparse = os.path.join(path, 'libcusparse.so')
        if os.path.exists(cusparse):
            print(f"Found cuSPARSE at: {cusparse}")
            break
else:
    print("cuSPARSE library not found in LD_LIBRARY_PATH")
EOF

python -c "import jax; print('JAX version:', jax.__version__); print('Devices:', jax.devices())"

# Copy smoke test to env directory
if [ -f "${SLURM_SUBMIT_DIR:-$HOME/backgammon-delta}/smoke_jax_gpu_min.py" ]; then
    cp "${SLURM_SUBMIT_DIR:-$HOME/backgammon-delta}/smoke_jax_gpu_min.py" .
    echo "Copied smoke_jax_gpu_min.py to $ENV_DIR"
fi

echo ""
echo "=== Setup Complete ==="
echo "Environment location: $ENV_DIR/venv"
echo ""
echo "Next steps:"
echo "1. Test GPU: python smoke_jax_gpu_min.py"
echo "2. Exit interactive session: exit"
echo "3. Submit batch job: sbatch bg_smoke.slurm"

