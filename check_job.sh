#!/bin/bash
# Quick script to check job status and output on Delta

JOBID=${1:-"14118045"}

echo "=== Checking Job $JOBID ==="
echo ""

# Check job status
echo "Job Status:"
squeue -j $JOBID 2>/dev/null || echo "Job not in queue (likely completed)"

echo ""
echo "=== Job Output ==="
if [ -f "slurm-${JOBID}.out" ]; then
    echo "Last 50 lines of output:"
    tail -50 "slurm-${JOBID}.out"
else
    echo "Output file slurm-${JOBID}.out not found yet"
    echo "Waiting for job to start writing output..."
fi

echo ""
echo "=== Job Error Log (if exists) ==="
if [ -f "slurm-${JOBID}.err" ]; then
    cat "slurm-${JOBID}.err"
else
    echo "No error file found"
fi

