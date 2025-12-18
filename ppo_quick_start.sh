#!/bin/bash
# PPO Quick Start Script
# Run this to test PPO training with progressively larger examples

set -e  # Exit on error

echo "=========================================="
echo "PPO Training Quick Start"
echo "=========================================="
echo ""

# Step 1: Minimal test (1-2 minutes)
echo "Step 1: Running minimal test (1-2 minutes)..."
echo "This verifies everything works correctly."
echo ""
python ppo.py \
    --total_updates 5 \
    --rollout_steps 256 \
    --epochs 2 \
    --minibatch 64 \
    --max_moves 32 \
    --save_every 5

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

# Step 2: Small training (10-15 minutes)
echo ""
echo "Step 2: Running small training (10-15 minutes)..."
echo "This trains the agent for a meaningful amount of time."
echo ""
python ppo.py \
    --total_updates 20 \
    --rollout_steps 1024 \
    --epochs 4 \
    --minibatch 128 \
    --max_moves 64 \
    --save_every 10

echo ""
echo "✓ Small training completed!"
echo ""

# Ask if user wants to evaluate
read -p "Evaluate the trained agent against random? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "Evaluating agent against random baseline..."
    python compare_checkpoint_steps.py checkpoints_ppo --random --games 100
fi

echo ""
echo "=========================================="
echo "Quick start completed!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Review the training output above"
echo "2. Check checkpoints in: checkpoints_ppo/"
echo "3. For full training, run:"
echo "   python ppo.py --total_updates 200 --rollout_steps 8192"
echo ""

