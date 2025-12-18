#!/bin/bash
# TD(λ) Quick Start Script
# Run this to test TD(λ) training with progressively larger examples

set -e  # Exit on error

echo "=========================================="
echo "TD(λ) Training Quick Start"
echo "=========================================="
echo ""

# Step 1: Minimal test (2-5 minutes)
echo "Step 1: Running minimal test (2-5 minutes)..."
echo "This verifies everything works correctly."
echo "Note: First step will be slow due to JIT compilation (5-30 min)."
echo ""
python train_agent2_tdlambda_2ply.py \
    --steps 100 \
    --log_every 20 \
    --save_every 50 \
    --eval_batch 512

echo ""
echo "✓ Minimal test completed!"
echo ""

# Ask if user wants to continue
read -p "Continue to small training run? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Stopping here. You can run the next steps manually."
    exit 0
fi

# Step 2: Small training (30-60 minutes)
echo ""
echo "Step 2: Running small training (30-60 minutes)..."
echo "This trains the agent for a meaningful amount of time."
echo ""
python train_agent2_tdlambda_2ply.py \
    --steps 1000 \
    --log_every 100 \
    --save_every 200 \
    --eval_batch 1024

echo ""
echo "✓ Small training completed!"
echo ""

# Ask if user wants to evaluate
read -p "Evaluate the trained agent against random? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "Evaluating agent against random baseline..."
    python compare_checkpoint_steps.py checkpoints_tdlambda_2ply --random --games 100
fi

echo ""
echo "=========================================="
echo "Quick start completed!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Review the training output above"
echo "2. Check checkpoints in: checkpoints_tdlambda_2ply/"
echo "3. For full training, run:"
echo "   python train_agent2_tdlambda_2ply.py --steps 200000 --save_every 2000"
echo ""

