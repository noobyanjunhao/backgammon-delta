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
module load anaconda3_gpu || {
    echo "Warning: anaconda3_gpu not found, trying alternatives..."
    module load python || module load python/3.11 || echo "Using system Python"
}

# Activate the pre-built environment (built in interactive session)
ENV_DIR=${SCRATCH:-$HOME}/bg
if [ ! -d "$ENV_DIR/venv" ]; then
    echo "ERROR: Environment not found at $ENV_DIR/venv"
    echo "Please run setup_env_interactive.sh in an interactive GPU session first"
    exit 1
fi

source $ENV_DIR/venv/bin/activate

# Run smoke test
cd $ENV_DIR
python smoke_jax_gpu_min.py

