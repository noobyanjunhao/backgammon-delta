# Backgammon RL Project - Folder Structure and Purpose

This document describes the purpose of each folder and key file in relation to the project requirements.

### Training Scripts (Root Level)

- **`train_value_td0.py`**: Implements **Agent 1** - On-policy TD(0) with linear function approximation.
  - Saves checkpoints to `checkpoints/` or `checkpoints_1ply/` directories

- **`ppo.py`**: Original PPO implementation (baseline/reference). This is my initial PPO code that serves as a comparison point. It implements:
  - Actor-critic architecture with shared ResNet backbone
  - Policy-based pruning for 2-ply search
  - Sequential submove sampling during training
  - Clipped surrogate objective, value function loss, and entropy bonus
  - Generalized Advantage Estimation (GAE)
  - Saves checkpoints to `checkpoints_ppo/`

### Evaluation and Comparison Scripts

- **`compare_agent2_vs_ppo.py`**: Plays games between Agent 2 (TD-λ, ConvNet/ResNet) and PPO, reports win rates.
- **`compare_agent2_vs_td0.py`**: Compares Agent 2 to Agent 1 (TD(0)), showing gains from neural networks and eligibility traces.
- **`compare_checkpoint_steps.py`**: Compares model performance at different checkpoints.

## Agent 2 Folder (`agent2/`)

**Purpose**: Agent 2 — TD(λ) with a modern ConvNet/ResNet v2.

### Key Files:

- **`agent2_encode.py`**: Feature encoder for game state; produces (24, 15) × 2 plus 6 extra features per state for network input.
- **`agent2_value_net.py`**: 1D ConvNet with ResNet v2, global pooling, then dense layers. Only predicts state value (critic head).
- **`train_agent2.py`**: Standard training script using TD(λ), 2-ply search, and eligibility traces. Saves checkpoints to `checkpoints_agent2/`.
- **`train_agent2_vectorized.py`**: Vectorized/batched training for speed—runs many games and updates at once, with chunked 2-ply evaluation to save memory. Saves checkpoints to `checkpoints_agent2_vectorized/`. **Recommended training script.**

## Agent 3 Folder (`agent3/`)

**Purpose**: Implements **Agent 3** - Proximal Policy Optimization (PPO) with policy-based pruning.

### Key Components:

- **`agent3_ppo_net.py`**: Actor-critic network architecture:
  - **Shared Backbone**: Uses Agent 2's ResNet architecture (1D ConvNet + 9 ResNet v2 blocks)
  - **Value Head (Critic)**: Same as Agent 2, predicts state values V(S) ∈ [-3, 3]
  - **Policy Head (Actor)**: 134 inputs → 64 hidden (ReLU) → 625 outputs (25×25 grid of submove logits)
  - The 625 outputs encode logits for each possible source-destination pair for a single submove

- **`agent3_utils.py`**: Utility functions for Agent 3:
  - Submove indexing and conversion (physical moves ↔ canonical submove indices)
  - Helper functions for policy-based move pruning

- **`train_agent3.py`**: PPO training script using Agent 2's features. 
  - Implements policy-based pruning for 2-ply, sequential submove sampling, and PPO-style advantage/losses.
  - Stores data with a replay buffer and trains offline using GAE.
  - Saves checkpoints to `checkpoints_agent3/` (when implemented).

## Checkpoint Directories

- **`checkpoints/`**: Checkpoints from original TD(0) training (Agent 1)
- **`checkpoints_1ply/`**: Checkpoints from 1-ply TD(0) training (faster training variant)
- **`checkpoints_ppo/`**: Checkpoints from original PPO implementation (`ppo.py`)
- **`checkpoints_agent2/`**: Checkpoints from non-vectorized Agent 2 training
- **`checkpoints_agent2_vectorized/`**: Checkpoints from vectorized Agent 2 training (recommended)
- **`checkpoints_agent2_test/`**: Test checkpoints for Agent 2 (short training runs)
- **`checkpoints_tdlambda_2ply/`**: Checkpoints from TD(λ) training with 2-ply search
- **`checkpoints_agent3/`**: Checkpoints from Agent 3 (PPO) training (when implemented)

## Project Requirements Alignment

### Agent 1 (TD(0)) ✓
- corresponding checkpoints the checkpoints.checkpoint_5000(trained with 20000 step though naming issue)


### Agent 2 (TD(λ)) ✓
- Best on so far
- Corresponding checkpoints_agent2/checkpoint_2000

### Agent 3 (PPO) ✓
-  Uses Agent 2's feature encoding
-  However 2ply added took too long, we have the other model with slight less feature added (ppo.py). This generates checkpoint in checkpoints_ppo

## Training Guidance
- To train Agent 1 (TD(0)): `python train_value_td0.py --checkpoint_dir checkpoints/ --steps 20000`
- To train Agent 2 (TD(λ), 2-ply): `python train_value_tdlambda_2ply.py --checkpoint_dir checkpoints_agent2/ --steps 2000 --vectorized`
- To train Agent 3 (PPO): `python train_ppo.py --checkpoint_dir checkpoints_ppo/ --total_steps 100000 --batch_size 128`


