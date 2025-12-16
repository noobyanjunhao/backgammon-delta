#!/bin/bash
#SBATCH --job-name=bg_smoke
#SBATCH --account=bfxm-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --gpus-per-node=1
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err

set -euo pipefail
echo "Running on $(hostname)"
nvidia-smi

module reset

# Load Python module ONLY
# DO NOT load CUDA modules - JAX uses its bundled CUDA libraries
module load python 2>/dev/null || {
    echo "Warning: python module not found, trying alternatives..."
    module load python/3.11 2>/dev/null || module load anaconda3_gpu 2>/dev/null || echo "Using system Python"
}

# Activate the pre-built environment (built in interactive session)
# Try multiple possible locations
ENV_DIR=""
for dir in "${SCRATCH}/bg" "${HOME}/bg" "/scratch/bfxm/${USER}/bg"; do
    if [ -d "$dir/venv" ]; then
        ENV_DIR="$dir"
        break
    fi
done

if [ -z "$ENV_DIR" ] || [ ! -d "$ENV_DIR/venv" ]; then
    echo "ERROR: Environment not found"
    echo "Searched in:"
    echo "  ${SCRATCH}/bg"
    echo "  ${HOME}/bg"
    echo "  /scratch/bfxm/${USER}/bg"
    echo ""
    echo "Please run setup_env_interactive.sh in an interactive GPU session first:"
    echo "  srun -A bfxm-delta-gpu --partition=gpuA100x4-interactive --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=32G --gpus-per-node=1 --time=00:30:00 --pty bash"
    echo "  ./setup_env_interactive.sh"
    exit 1
fi

echo "Using environment at: $ENV_DIR/venv"
source $ENV_DIR/venv/bin/activate

# KEY: Unset CUDA paths before running JAX
# JAX uses its bundled CUDA libraries - don't override them
unset LD_LIBRARY_PATH
unset CUDA_HOME CUDA_ROOT CUDA_PATH

# Copy smoke test if not already there, or use from repo
if [ -f "$ENV_DIR/smoke_jax_gpu_min.py" ]; then
    cd $ENV_DIR
    python smoke_jax_gpu_min.py
elif [ -f "${SLURM_SUBMIT_DIR}/smoke_jax_gpu_min.py" ]; then
    python ${SLURM_SUBMIT_DIR}/smoke_jax_gpu_min.py
else
    echo "ERROR: smoke_jax_gpu_min.py not found"
    exit 1
fi

