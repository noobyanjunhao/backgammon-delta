#!/bin/bash
# Script to compare checkpoints on the server
# Usage: ./compare_on_server.sh [method] [step1] [step2] [games]

METHOD=${1:-1ply}  # Default to 1ply
STEP1=${2:-5000}   # Default to step 5000
STEP2=${3:-200000} # Default to step 200000
GAMES=${4:-1000}   # Default to 1000 games

echo "Comparing checkpoints on server..."
echo "Method: $METHOD"
echo "Step 1: $STEP1"
echo "Step 2: $STEP2"
echo "Games: $GAMES"
echo ""

# Check if checkpoints exist
CKPT_DIR="checkpoints_${METHOD}"
if [ ! -d "$CKPT_DIR" ]; then
    echo "Error: Checkpoint directory $CKPT_DIR not found!"
    echo "Available checkpoint directories:"
    ls -d checkpoints_* 2>/dev/null || echo "  None found"
    exit 1
fi

echo "Available checkpoints in $CKPT_DIR:"
ls -lh "$CKPT_DIR" 2>/dev/null | grep checkpoint || echo "  No checkpoints found"

echo ""
echo "Running comparison..."
python compare_checkpoint_steps.py \
    --method "$METHOD" \
    --step1 "$STEP1" \
    --step2 "$STEP2" \
    --games "$GAMES" \
    --runs 3

