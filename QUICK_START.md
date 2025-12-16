# Quick Start - Running Smoke Test on Delta

## Step-by-Step Commands

### 1. Connect to Delta
```bash
ssh jyan10@login.delta.ncsa.illinois.edu
```
(Enter password, then approve Duo push)

### 2. Find Your Account Name
```bash
accounts
```
**Copy the account name** (first column, usually something like `abc123` or `def456`)

### 3. Navigate to Project Directory
```bash
# Replace ACCOUNT_NAME with the account you found above
mkdir -p /projects/ACCOUNT_NAME/$USER/backgammon
cd /projects/ACCOUNT_NAME/$USER/backgammon
```

### 4. Clone the Repository
```bash
git clone https://cs.bc.edu/mctague/t/2025/rl/backgammon.git
cd backgammon
```

### 5. Run Automated Setup
```bash
chmod +x setup_delta.sh
./setup_delta.sh
```
This will automatically update the Slurm script with your account name.

### 6. (Optional) Verify Setup
```bash
chmod +x check_setup.sh
./check_setup.sh
```

### 7. Submit the Job
```bash
sbatch smoke_gpu.slurm
```
You'll see output like: `Submitted batch job 12345678`

### 8. Monitor the Job
```bash
# Check job status
squeue -u $USER

# View output (replace JOBID with your job ID)
tail -f slurm-JOBID.out
```

### 9. Check Results
Once the job completes, view the full output:
```bash
cat slurm-JOBID.out
```

## Expected Success Output
```
JAX devices: [gpu:0, ...]
Default backend: gpu
Forward OK. v.shape = (32, 1) v[:5] = [array([...])]
```

## Troubleshooting

If JAX installation fails, try:
```bash
# Check CUDA version
module avail cuda

# Try different JAX install
pip install -U "jax[cuda11_local]" "flax" "optax"
```

