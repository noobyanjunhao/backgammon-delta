# Running Backgammon Training on Delta HPC

This guide walks you through setting up and running the backgammon training code on Delta.

## Quick Start

### 1. Login to Delta
```bash
ssh jyan10@login.delta.ncsa.illinois.edu
# Enter password, then approve Duo push
```

### 2. Navigate to Project Directory
```bash
# Find your account name first
accounts

# Create and navigate to project directory (replace ACCOUNT_NAME with your actual account)
mkdir -p /projects/ACCOUNT_NAME/$USER/backgammon
cd /projects/ACCOUNT_NAME/$USER/backgammon
```

### 3. Clone Repository
```bash
git clone https://cs.bc.edu/mctague/t/2025/rl/backgammon.git
cd backgammon
```

### 4. Run Setup Script
```bash
chmod +x setup_delta.sh
./setup_delta.sh
```

This will automatically:
- Find your account name
- Update `smoke_gpu.slurm` with your account

### 5. Submit First Test Job
```bash
sbatch smoke_gpu.slurm
```

### 6. Monitor Job
```bash
# Check job status
squeue -u $USER

# View output (replace JOBID with actual job ID)
tail -f slurm-JOBID.out
```

## Files

- `smoke_jax_gpu.py` - Smoke test script that verifies JAX/GPU setup
- `smoke_gpu.slurm` - Slurm job script for the smoke test
- `setup_delta.sh` - Automated setup script (run on Delta)

## Expected Output

If everything works, you should see:
```
JAX devices: [gpu:0, ...]
Default backend: gpu
Forward OK. v.shape = (32, 1) v[:5] = ...
```

## Troubleshooting

### JAX CUDA Version Issues
If JAX installation fails, Delta might use a different CUDA version. Try:
- `pip install -U "jax[cuda11_local]"` for CUDA 11
- `pip install -U "jax[cuda12]"` for CUDA 12 (default in script)
- Check available CUDA modules: `module avail cuda`

### No GPU Visible
- Verify you're on the correct partition: `echo $SLURM_JOB_PARTITION`
- Check GPU: `nvidia-smi`
- Ensure `--gpus-per-node=1` is set in Slurm script

### Account Not Found
- Run `accounts` to see available accounts
- Contact NCSA support if no accounts are listed

## Next Steps After Smoke Test

Once the smoke test passes, you can proceed to implement:
1. `train_td0_linear.py` - TD(0) with linear features
2. `train_td_lambda_resnet.py` - TD(λ) with ResNet value net
3. `train_ppo.py` - PPO with actor-critic net

See the main project instructions for details on implementing the training loops.

